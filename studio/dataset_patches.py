"""Materialize spatially registered PNG patch pairs in a new private directory."""

from __future__ import annotations

import ctypes as ct
import json
import os
import stat
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, TextIO

from trainner_data.file_version import artifact_version

from .dataset_pack import _publish
from .dataset_split import paired_disk_rows
from .native import executable
from .pair_audit import pair_source_options
from .paths import local_path

_ROOT = Path(__file__).resolve().parents[1]
_NOTIFY = ct.CFUNCTYPE(ct.c_int, ct.c_int, ct.c_uint32, ct.c_uint32, ct.c_char_p)
Json = dict[str, Any]


class DatasetPatchesCancelled(RuntimeError):
    """Extraction stopped before publication."""


class _Format(ct.Structure):
    _fields_ = [
        (key, ct.c_uint32) for key in ("width", "height", "channels", "bitDepth")
    ]


@lru_cache(maxsize=1)
def _kernel() -> Any:
    library = executable("paired_patches", "patches.so", source="patches.c")
    func = ct.CDLL(str(library)).studio_paired_patches_v1
    func.argtypes = (
        [ct.c_int] * 4
        + [ct.c_uint32] * 3
        + [ct.POINTER(_Format), _NOTIFY, ct.POINTER(ct.c_char), ct.c_size_t]
    )
    func.restype = ct.c_int
    return func


def validate_patch_options(body: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return scale, square LQ crop and LQ stride for the job."""
    values = []
    for key in ("scale", "cropSize", "stride"):
        value = body.get(key)
        if type(value) is not int or not 1 <= value <= 0x7FFFFFFF:
            raise ValueError(f"{key} must be a positive PNG-grid integer.")
        values.append(value)
    scale, crop, stride = values
    if stride > crop:
        raise ValueError("stride must be at most cropSize to cover every source pixel.")
    if scale * crop > 0x7FFFFFFF:
        raise ValueError("The scaled crop exceeds the PNG dimension limit.")
    return scale, crop, stride


@dataclass(slots=True)
class _Source:
    root_id: int
    gt: Path
    lq: Path


def _version(path: Path) -> str:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular source file: {path}")
    return artifact_version(info)


def _extract_pair(
    source: _Source,
    index: int,
    roots: dict[str, list[Path]],
    versions: dict[Path, str],
    staged: Path,
    parameters: tuple[int, int, int],
    manifest: TextIO,
    meta: TextIO,
    tick: Callable[[str, int, int], None],
) -> int:
    scale, crop, stride = parameters
    subdir = f"source-{index:08d}"
    formats = (_Format * 2)()
    source_record: Json = {"row": index, "rootId": source.root_id}
    count = 0
    role_counts = [0, 0]
    failure: BaseException | None = None

    def notify(role: int, x: int, y: int, name: bytes | None) -> int:
        nonlocal failure, count
        try:
            if not count:
                # The kernel validated both source headers before its first event.
                lq = formats[0]
                cols = 1 + (lq.width - crop + stride - 1) // stride
                rows = 1 + (lq.height - crop + stride - 1) // stride
                count = cols * rows
                for i, key in enumerate(("lq", "gt")):
                    source_record[key].update(
                        {
                            field: getattr(formats[i], field)
                            for field, _ in _Format._fields_
                        }
                    )
            if name is not None:
                role_counts[role] += 1
                if role == 1:
                    key = f"{subdir}/{name.decode('ascii')}"
                    record = {
                        "key": key,
                        "source": source_record,
                        "lqBox": [x, y, crop, crop],
                        "gtBox": [x * scale, y * scale, crop * scale, crop * scale],
                    }
                    manifest.write(
                        json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                        + "\n"
                    )
                    meta.write(key + "\n")
            tick("lq" if role == 0 else "gt", role_counts[role], count)
            return 0
        except BaseException as exc:
            # ctypes callbacks return normally, then the worker re-raises.
            failure = exc
            return 1

    with ExitStack() as stack:
        fds, dirs = [], []
        for role, path in (("lq", source.lq), ("gt", source.gt)):
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            stack.callback(os.close, fd)
            if artifact_version(os.fstat(fd)) != versions[path]:
                raise ValueError(f"Source changed before extraction: {path}")
            fds.append(fd)
            folder = staged / role / subdir
            folder.mkdir()
            dirfd = os.open(folder, os.O_DIRECTORY)
            stack.callback(os.close, dirfd)
            dirs.append(dirfd)
            source_record[role] = {
                "path": path.relative_to(roots[role][source.root_id - 1]).as_posix(),
                "version": versions[path],
            }
        error = ct.create_string_buffer(512)
        status = _kernel()(
            *fds,
            *dirs,
            scale,
            crop,
            stride,
            formats,
            _NOTIFY(notify),
            error,
            len(error),
        )
        if failure is not None:
            raise failure
        if status:
            detail = error.value.decode(errors="replace")
            raise ValueError(
                f"Source pair {index} ({source.lq}, {source.gt}): {detail}"
            )
        if not count or role_counts != [count, count]:
            raise RuntimeError(f"Incomplete native patch grid for source pair {index}.")
        for path, fd in zip((source.lq, source.gt), fds, strict=True):
            if artifact_version(os.fstat(fd)) != versions[path]:
                raise ValueError(f"Source changed during PNG extraction: {path}")
    return count


def extract_paired_patches(
    body: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    root: Path = _ROOT,
    progress: Callable[[Json], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Json:
    """Publish two paired PNG roots, trainer membership, coordinates and portable settings."""
    parameters = validate_patch_options(body)
    scale, crop, stride = parameters
    dataset = body.get("dataset")
    if (
        body.get("task", "image_restoration") != "image_restoration"
        or not isinstance(dataset, dict)
        or str(dataset.get("type", "")).lower() != "pairedimagedataset"
    ):
        raise ValueError("Choose an image_restoration PairedImageDataset.")
    backend = dataset.get("io_backend", {"type": "disk"})
    if not isinstance(backend, Mapping) or backend.get("type") != "disk":
        raise ValueError("Paired PNG extraction requires the disk backend.")
    root = root.absolute()
    options = pair_source_options(
        {"datasets": {"source": dataset}, "scale": scale}, "source", root
    )
    template = options.get("filename_tmpl", "{}")
    if not isinstance(template, str) or "\0" in template:
        raise ValueError("filename_tmpl must be a string without a NUL character.")
    source_meta = (
        Path(options["meta_info"]) if options.get("meta_info") is not None else None
    )
    roots = {
        role: [Path(p) for p in options[f"dataroot_{role}"]] for role in ("gt", "lq")
    }
    requested = local_path(os.fspath(destination), root)
    if requested.name in ("", ".", ".."):
        raise ValueError("Choose a new destination directory.")
    output = requested.parent.resolve(strict=True) / requested.name
    if os.path.lexists(output):
        raise FileExistsError(f"Patch destination already exists: {output}")
    for paths in roots.values():
        for path in paths:
            if output.is_relative_to(path.resolve(strict=True)):
                raise ValueError(
                    f"Place the patch destination outside the source root: {path}"
                )
            if path.suffix.lower() == ".lmdb" or (path / "data.mdb").is_file():
                raise ValueError(f"LMDB sources are unsupported: {path}")

    sources: list[_Source] = []
    versions = {source_meta: _version(source_meta)} if source_meta is not None else {}
    last_event = 0.0
    current = patch_pairs = 0

    def report(
        stage: str, done: int, total: int | None, *, force: bool = False
    ) -> None:
        nonlocal last_event
        now = monotonic()
        if not force and now - last_event < 0.25:
            return
        last_event = now
        if progress is not None:
            progress(
                {
                    "stage": stage,
                    "completed": done,
                    "total": total,
                    "sourceRow": current,
                    "completedPatchPairs": patch_pairs + (done if stage == "gt" else 0),
                }
            )
        if cancelled is not None and cancelled():
            raise DatasetPatchesCancelled("Paired patch extraction was cancelled.")

    report("selecting", 0, None, force=True)
    for group, (gt, lq) in enumerate(zip(roots["gt"], roots["lq"], strict=True), 1):
        for row in paired_disk_rows(gt, lq, template, source_meta):
            left, right = Path(row["gt_path"]), Path(row["lq_path"])
            for path, base in ((left, gt), (right, lq)):
                relative = path.relative_to(base)
                if ".." in relative.parts:
                    raise ValueError(
                        f"Selected source escapes its declared root: {path}"
                    )
                if path not in versions:
                    versions[path] = _version(path)
            sources.append(_Source(group, left, right))
            report("selecting", len(sources), None)
    if not sources:
        raise ValueError("The paired provider selected no source images.")
    report("selecting", len(sources), len(sources), force=True)

    with TemporaryDirectory(dir=output.parent, prefix=".paired-patches-") as temporary:
        staged = Path(temporary) / "patches"
        staged.mkdir(mode=0o700)
        for role in ("gt", "lq"):
            (staged / role).mkdir()
        with (
            (staged / "patches.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as manifest,
            (staged / "meta_info.txt").open(
                "x", encoding="utf-8", newline="\n"
            ) as meta,
        ):
            for current, source in enumerate(sources, 1):
                report("extracting", current - 1, len(sources), force=True)
                patch_pairs += _extract_pair(
                    source,
                    current,
                    roots,
                    versions,
                    staged,
                    parameters,
                    manifest,
                    meta,
                    report,
                )
        selection = {
            "name": f"{dataset.get('name') or 'Paired images'} patches",
            "type": "PairedImageDataset",
            "io_backend": {"type": "disk"},
            "filename_tmpl": "{}",
            "dataroot_gt": ["gt"],
            "dataroot_lq": ["lq"],
            "meta_info": "meta_info.txt",
        }
        summary = {
            "scale": scale,
            "cropSize": crop,
            "stride": stride,
            "sourcePairs": len(sources),
            "patchPairs": patch_pairs,
            "sourceRoots": {k: [str(p) for p in paths] for k, paths in roots.items()},
            "sourceMetaInfo": str(source_meta) if source_meta is not None else None,
            "sourceFilenameTemplate": template,
            "dataset": selection,
            "coordinates": "patches.jsonl",
        }
        (staged / "dataset.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        report("verifying", 0, len(versions), force=True)
        for i, (path, version) in enumerate(versions.items(), 1):
            if _version(path) != version:
                raise ValueError(f"Source changed before publication: {path}")
            report("verifying", i, len(versions))
        report("verifying", len(versions), len(versions), force=True)
        _publish(staged, output)

    gt_path, lq_path, meta_path = (
        str(output / name) for name in ("gt", "lq", "meta_info.txt")
    )
    return {
        **summary,
        "destination": str(output),
        "gtPath": gt_path,
        "lqPath": lq_path,
        "metaInfoPath": meta_path,
        "manifestPath": str(output / "patches.jsonl"),
        "reportPath": str(output / "dataset.json"),
        "dataset": {
            **selection,
            "dataroot_gt": [gt_path],
            "dataroot_lq": [lq_path],
            "meta_info": meta_path,
        },
    }

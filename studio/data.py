"""Headless dataset inspection and previews for Studio."""

from __future__ import annotations

import io
import os
import stat
import subprocess
import threading
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from string import Formatter
from typing import Any

from PIL import Image, UnidentifiedImageError

from studio.paths import local_path

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE = _ROOT / "native" / "studio" / "data"
_INSPECTOR = _NATIVE / "build" / "clang" / "inspect"
_TOOLCHAIN = _INSPECTOR.with_name("toolchain.txt")
_BUILD_LOCK = threading.Lock()
_toolchain_checked = False
_PREVIEW_EDGE = 1536
_PREVIEW_FORMATS = (
    "AVIF",
    "BMP",
    "DDS",
    "GIF",
    "ICO",
    "JPEG",
    "JPEG2000",
    "PNG",
    "PPM",
    "PSD",
    "QOI",
    "TGA",
    "TIFF",
    "WEBP",
)


def _run(
    args: list[str],
    operation: str,
    timeout: float | None = None,
    *,
    input_data: bytes | None = None,
) -> bytes:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL if input_data is None else None,
            input=input_data,
            timeout=timeout,
        ).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{operation} failed ({exc.returncode}). {detail}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{operation} failed. {exc}") from exc


def _inspector() -> Path:
    global _toolchain_checked

    with _BUILD_LOCK:
        newest_source = max(
            (_NATIVE / filename).stat().st_mtime_ns
            for filename in ("Makefile", "inspect.cpp")
        )
        if (
            not _toolchain_checked
            or not _INSPECTOR.is_file()
            or not _TOOLCHAIN.is_file()
            or _INSPECTOR.stat().st_mtime_ns
            < max(newest_source, _TOOLCHAIN.stat().st_mtime_ns)
        ):
            _run(
                ["make", "--no-print-directory", "-C", str(_NATIVE)],
                "Building the native dataset inspector",
                timeout=60,
            )
            _toolchain_checked = True
    return _INSPECTOR


def _read_report(output: bytes, multiple_roots: bool) -> dict[str, Any]:
    fields = iter(os.fsdecode(output).split("\0"))
    issues: list[dict[str, str]] = []
    samples: list[dict[str, int | str]] = []
    roots: list[dict[str, Any]] = []
    try:
        for kind in fields:
            if kind == "issue":
                root_id, level, path, message = islice(fields, 4)
                issue = {"level": level, "path": path, "message": message}
                if root_id != "0":
                    issue["rootId"] = root_id
                issues.append(issue)
            elif kind == "root":
                root_id, gt, lq, g, low, paired = islice(fields, 6)
                root = {"id": root_id, "gt": gt, "counts": _counts(g, low, paired)}
                if lq:
                    root["lq"] = lq
                roots.append(root)
            elif kind == "sample":
                root_id, filename, gt, lq, gw, gh, lw, lh, alignment = islice(fields, 9)
                sample: dict[str, int | str] = {
                    "id": f"{root_id}:{filename}" if multiple_roots else filename,
                    "rootId": root_id,
                    "name": filename,
                    "gt": gt,
                }
                if lq:
                    sample["lq"] = lq
                    sample["geometry"] = alignment
                for key, value in zip(
                    ("gtWidth", "gtHeight", "lqWidth", "lqHeight"),
                    (gw, gh, lw, lh),
                    strict=True,
                ):
                    if value != "0":
                        sample[key] = int(value)
                samples.append(sample)
            elif kind == "counts":
                (
                    gt_count,
                    lq_count,
                    paired,
                    issue_count,
                    errors,
                    sample_limit,
                    issue_limit,
                ) = map(int, islice(fields, 7))
                if next(fields, None) != "" or next(fields, None) is not None:
                    break
                return {
                    "source": "folder",
                    "ok": errors == 0,
                    "counts": _counts(gt_count, lq_count, paired),
                    "roots": roots,
                    "issues": issues,
                    "samples": samples,
                    "truncated": len(samples) < gt_count or len(issues) < issue_count,
                    "limits": {"samples": sample_limit, "issues": issue_limit},
                    "totals": {"samples": gt_count, "issues": issue_count},
                }
            else:
                break
    except ValueError as exc:
        raise RuntimeError(
            "The native dataset inspector returned an invalid report."
        ) from exc
    raise RuntimeError("The native dataset inspector returned an incomplete report.")


def _counts(gt: int | str, lq: int | str, paired: int | str) -> dict[str, int]:
    return {"gt": int(gt), "lq": int(lq), "paired": int(paired)}


def _path(value: str | os.PathLike[str]) -> str:
    return str(
        local_path(os.fspath(value) if isinstance(value, os.PathLike) else value, _ROOT)
    )


def _roots(
    scalar: str | os.PathLike[str] | None,
    multiple: list[str | os.PathLike[str]] | None,
    name: str,
) -> list[str]:
    if multiple is None:
        return [] if scalar is None or scalar == "" else [_path(scalar)]
    if scalar is not None and scalar != "":
        raise ValueError(f"Use {name} or {name}Roots, not both.")
    if not isinstance(multiple, (list, tuple)):
        raise ValueError(f"{name}Roots must be an array of directory paths.")
    return [_path(value) for value in multiple]


def _positive(value: int | None, label: str) -> None:
    if value is not None and (type(value) is not int or not 1 <= value < 2**64):
        raise ValueError(f"{label} must be a positive integer that fits in 64 bits.")


def _format_stem(template: str, filename: str) -> str:
    folder, basename = os.path.split(filename)
    expanded = template.format(os.path.splitext(basename)[0])
    return os.path.join(folder, expanded) if folder else expanded


def _native_template(
    inspector: Path, template: str, gt_paths: list[str]
) -> tuple[str, list[str], bytes | None]:
    if template == "{}":
        return "stem", [], None
    try:
        parsed = list(Formatter().parse(template))
        if all(
            field in {None, "", "0"} and not spec and conversion is None
            for _, field, spec, conversion in parsed
        ):
            # NUL cannot occur in a filename or in the supplied literal text.
            return "format", template.format("\0").split("\0"), None

        raw = _run(
            [str(inspector), "--list-gt", *gt_paths],
            "Reading GT names for the filename template",
        )
        fields = iter(os.fsdecode(raw).split("\0")[:-1])
        mapping = bytearray()
        for tag in fields:
            root_id, filename = islice(fields, 2)
            if tag != "file":
                raise RuntimeError(
                    "The native inspector returned an invalid GT inventory."
                )
            expected = _format_stem(template, filename)
            mapping.extend(os.fsencode(f"{root_id}\0{filename}\0{expected}\0"))
        return "mapped", [], bytes(mapping)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid filenameTemplate {template!r}. {exc}") from exc


def scan(
    gt: str | os.PathLike[str] | None = None,
    lq: str | os.PathLike[str] | None = None,
    scale: int | None = None,
    *,
    gt_roots: list[str | os.PathLike[str]] | None = None,
    lq_roots: list[str | os.PathLike[str]] | None = None,
    filename_template: str = "{}",
    gt_patch_size: int | None = None,
    phase: str = "train",
    source: str = "folder",
    meta_info: str | os.PathLike[str] | None = None,
    validation_roots: list[str | os.PathLike[str]] | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Inspect the roots or explicit records that paired-image training would select."""
    _positive(scale, "Scale")
    _positive(gt_patch_size, "GT patch size")
    gt_paths = _roots(gt, gt_roots, "gt")
    lq_paths = _roots(lq, lq_roots, "lq")
    validation = _roots(None, validation_roots, "validation")
    if not gt_paths:
        raise ValueError("Provide at least one GT directory.")
    if lq_paths and len(lq_paths) != len(gt_paths):
        raise ValueError(
            "Provide one LQ directory per GT directory, in the same root order."
        )
    if phase not in ("train", "val", "test"):
        raise ValueError("Phase must be train, val, or test.")
    if source not in ("folder", "meta_info", "lmdb"):
        raise ValueError("Source must be folder, meta_info, or lmdb.")
    if not isinstance(filename_template, str) or "\0" in filename_template:
        raise ValueError("filenameTemplate must be a string without NUL characters.")
    if task == "semantic_segmentation":
        return _segmentation_notice(gt_paths, lq_paths, source)
    if (
        phase == "train"
        and lq_paths
        and gt_patch_size
        and scale
        and gt_patch_size % scale
    ):
        raise ValueError(
            "GT patch size must be divisible by scale for paired training."
        )

    if source != "folder":
        if not lq_paths:
            raise ValueError(f"The {source} source needs both GT and LQ directories.")
        if validation:
            raise ValueError("validationRoots applies to folder scans.")
        if source == "meta_info" and meta_info is None:
            raise ValueError("Provide metaInfo for the meta_info source.")
        if source == "lmdb" and meta_info is not None:
            raise ValueError("LMDB uses each database's own meta_info.txt.")
        from studio.data_sources import scan_explicit

        return scan_explicit(
            source,
            gt_paths,
            lq_paths,
            filename_template,
            _path(meta_info) if meta_info is not None else None,
            scale,
            gt_patch_size,
            phase,
        )
    if meta_info is not None:
        raise ValueError("Use source meta_info to scan a folder manifest.")
    for root in gt_paths + lq_paths:
        path = Path(root)
        if path.suffix.lower() == ".lmdb" and (path / "data.mdb").is_file():
            raise ValueError(f"Use source lmdb for the database at {root}.")

    inspector = _inspector()
    mode, parts, mapping = _native_template(
        inspector, filename_template if lq_paths else "{}", gt_paths
    )
    groups = [
        path
        for gt_path, lq_path in zip(
            gt_paths, lq_paths or [""] * len(gt_paths), strict=True
        )
        for path in (gt_path, lq_path)
    ]
    output = _run(
        [
            str(inspector),
            "--scan",
            str(scale or 0),
            str(gt_patch_size or 0),
            phase,
            mode,
            str(len(parts)),
            str(len(validation)),
            *parts,
            *validation,
            *groups,
        ],
        "Scanning the dataset",
        input_data=mapping,
    )
    return _read_report(output, len(gt_paths) > 1)


def _segmentation_notice(
    gt_paths: list[str], lq_paths: list[str], source: str
) -> dict[str, Any]:
    issue = {
        "level": "warning",
        "path": gt_paths[0],
        "message": "Segmentation uses integer mask IDs and image/mask matching. This scan covers restoration pairs.",
    }
    counts = {"gt": None, "lq": None, "paired": None}
    roots = [
        {
            "id": str(index + 1),
            "gt": path,
            "counts": counts,
            **({"lq": lq_paths[index]} if lq_paths else {}),
        }
        for index, path in enumerate(gt_paths)
    ]
    return {
        "source": source,
        "ok": False,
        "counts": counts,
        "roots": roots,
        "issues": [issue],
        "samples": [],
        "truncated": False,
        "limits": {"samples": 100, "issues": 200},
        "totals": {"samples": None, "issues": 1},
    }


def scan_request(body: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the dataset scan request for the HTTP and command-line adapters."""
    return scan(
        body.get("gt"),
        body.get("lq"),
        body.get("scale"),
        gt_roots=body.get("gtRoots"),
        lq_roots=body.get("lqRoots"),
        filename_template=body.get("filenameTemplate", "{}"),
        gt_patch_size=body.get("gtPatchSize"),
        phase=body.get("phase", "train"),
        source=body.get("source", "folder"),
        meta_info=body.get("metaInfo"),
        validation_roots=body.get("validationRoots"),
        task=body.get("task"),
    )


def preview(path: str | os.PathLike[str], max_size: int = _PREVIEW_EDGE,
            *, version: str | None = None) -> tuple[str, bytes]:
    """Return a PNG thumbnail of the first image or frame at the given path."""
    _check_size(max_size)
    image_path = local_path(
        os.fspath(path) if isinstance(path, os.PathLike) else path, _ROOT
    )
    try:
        with open(
            image_path,
            "rb",
            opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK),
        ) as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Preview requires a regular image file.")
            from trainner_data.file_version import artifact_version
            if version and artifact_version(info) != version:
                raise FileExistsError("The recorded image has been replaced.")
            result = _thumbnail(source,max_size)
            if version and artifact_version(os.fstat(source.fileno())) != version:
                raise FileExistsError("The recorded image changed while reading its preview.")
            return result
    except FileExistsError:
        raise
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Unsupported or unreadable image for preview: {image_path}"
        ) from exc
    except (Image.DecompressionBombError, OSError, SyntaxError) as exc:
        raise ValueError(f"Could not preview {image_path}. {exc}") from exc


def _check_size(max_size: int) -> None:
    if type(max_size) is not int or not 16 <= max_size <= _PREVIEW_EDGE:
        raise ValueError(f"Preview size must be an integer from 16 to {_PREVIEW_EDGE}.")


def _thumbnail(source, max_size: int) -> tuple[str, bytes]:
    with Image.open(source, formats=_PREVIEW_FORMATS) as image:
        if image.format == "JPEG":
            image.draft(image.mode, (max_size, max_size))
        if image.mode.startswith("I;16"):
            thumbnail = (
                image.convert("I")
                .point(lambda value: value * (1 / 257) + 0.5)
                .convert("RGB")
            )
        else:
            thumbnail = image.convert(
                "RGBA" if image.has_transparency_data else "RGB"
            )
        thumbnail.thumbnail(
            (max_size, max_size), Image.Resampling.LANCZOS
        )
        thumbnail.info.clear()
        output = io.BytesIO()
        thumbnail.save(output, format="PNG")
    return "image/png", output.getvalue()


def preview_bytes(encoded: bytes, max_size: int) -> tuple[str,bytes]:
    _check_size(max_size)
    try:
        return _thumbnail(io.BytesIO(encoded),max_size)
    except (Image.DecompressionBombError, OSError, SyntaxError) as exc:
        raise ValueError(f"Could not preview encoded validation source. {exc}") from exc


def preview_lmdb(
    path: str | os.PathLike[str], key: str, max_size: int = _PREVIEW_EDGE
) -> tuple[str, bytes]:
    """Preview an existing encoded record by its database path and key."""
    _check_size(max_size)
    if not isinstance(key, str) or not key or not key.isascii() or "\0" in key:
        raise ValueError("Provide a nonempty ASCII LMDB record key.")
    try:
        from lmdb import Error as LmdbError
    except ImportError as exc:
        raise ValueError("The lmdb reader is unavailable in this trainer environment.") from exc
    from trainner_data.lmdb import PairedLmdbReader

    try:
        data = PairedLmdbReader({"type": "lmdb"}).read(_path(path), key)
    except LmdbError as exc:
        raise ValueError(f"Could not read LMDB record {key!r}. {exc}") from exc
    return preview_bytes(data, max_size)

"""Audit encoded image and prepared frame identity across trainer-selected dataset splits."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, NamedTuple

from studio.paths import local_path

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE = _ROOT / "native/studio/split_audit"
_BLOCK = 1 << 20
_MAX_GROUPS = 200
_MAX_ISSUES = 200
_EXAMPLES = 8
_HASH_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int)
_PAIRED = "pairedimagedataset"
_VIDEO = "pairedvideodataset"
_CLIP = "videoclipdataset"
_SINGLE = "singleimagedataset"
_OTF = "realesrgandataset"
_HYBRID = "realesrganpaireddataset"
_SR_TASKS = ("image_restoration", "recurrent_restoration")
_SUPERVISED = {
    "image_classification": "classfolderdataset",
    "semantic_segmentation": "imagemaskdataset",
    "object_detection": "cocodetectiondataset",
}
_SELECT_ERRORS = (
    OSError,
    ValueError,
    AssertionError,
    KeyError,
    IndexError,
    ImportError,
    OverflowError,
)
Json = dict[str, Any]

_SCHEMA = """
PRAGMA cache_size = -4096;
PRAGMA temp_store = FILE;
PRAGMA synchronous = OFF;
CREATE TABLE sources (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, path BLOB NOT NULL, key BLOB NOT NULL,
    size INTEGER, stamp TEXT, sha BLOB, state TEXT NOT NULL DEFAULT 'pending',
    UNIQUE(kind, path, key)
);
CREATE TABLE inputs (
    split INTEGER NOT NULL, sample INTEGER NOT NULL, role TEXT NOT NULL,
    root INTEGER NOT NULL, relative BLOB NOT NULL, label_context TEXT, first_window INTEGER, last_window INTEGER,
    source INTEGER NOT NULL,
    PRIMARY KEY(split, sample, role)
) WITHOUT ROWID;
CREATE INDEX input_source ON inputs(source);
CREATE TABLE verified_splits (split INTEGER PRIMARY KEY);
CREATE TABLE sizes (size INTEGER PRIMARY KEY);
CREATE TABLE hashes (
    kind TEXT NOT NULL, stamp TEXT NOT NULL, key BLOB NOT NULL, sha BLOB NOT NULL,
    PRIMARY KEY(kind, stamp, key)
) WITHOUT ROWID;
"""


class SplitAuditCancelled(RuntimeError):
    """The requested split audit stopped before producing a report."""


class _Changed(ValueError):
    pass


class _NotRegular(ValueError):
    pass


class _Input(NamedTuple):
    role: str
    root: int
    relative: bytes
    path: bytes
    key: bytes
    label_context: Json | None = None
    windows: tuple[int, int] | None = None

    def selection_fields(self) -> list[Any]:
        fields = [self.role, self.root, os.fsdecode(self.relative)]
        if self.label_context is not None:
            fields.append(self.label_context)
        if self.windows is not None:
            fields.append(self.windows)
        return fields


@dataclass(frozen=True, slots=True)
class _Split:
    id: str
    name: str
    provider: str
    backend: dict[str, Any]
    roots: dict[str, list[str]]
    manifest: str | None
    template: str
    categories: dict[int, str] | None = None
    clip_size: int | None = None

    @property
    def kind(self) -> str:
        return self.backend["type"]


def _configured_split(
    split: str,
    value: Mapping[str, Any],
    root: Path,
    task: str,
    settings: Mapping[str, Any],
) -> _Split:
    kind = value.get("type")
    if not isinstance(kind, str):
        raise ValueError("Provide the selected dataset type.")
    kind = kind.lower()
    if task in _SUPERVISED:
        from studio.task_data import _category_names, _roots

        if kind != _SUPERVISED[task]:
            raise ValueError(
                f"{task} split audits support {_SUPERVISED[task]}, got {kind!r}."
            )
        backend = value.get("io_backend", {"type": "disk"})
        if not isinstance(backend, Mapping) or backend.get("type") != "disk":
            raise ValueError(f"{kind} requires the disk backend.")
        root_keys = ["dataroot_lq"]
        if task == "semantic_segmentation":
            root_keys.append("dataroot_gt")
        roots = {key: [str(p) for p in _roots(value, key, root)] for key in root_keys}
        manifest = None
        categories = None
        if task == "object_detection":
            if len(roots["dataroot_lq"]) != 1:
                raise ValueError("CocoDetectionDataset requires one image root.")
            manifest = str(local_path(value.get("annotation_file"), root))
            categories = _category_names(settings)
        elif task == "semantic_segmentation" and len(roots["dataroot_lq"]) != len(
            roots["dataroot_gt"]
        ):
            raise ValueError("Provide one mask root per image root.")
        if task != "object_detection" and value.get("meta_info") is not None:
            manifest = str(local_path(value["meta_info"], root))
        name = value.get("name")
        return _Split(
            split,
            name if isinstance(name, str) else split,
            kind,
            dict(backend),
            roots,
            manifest,
            "{}",
            categories,
        )
    if kind not in {_PAIRED, _SINGLE, _OTF, _HYBRID, _VIDEO, _CLIP}:
        raise ValueError(
            f"Dataset provider {kind!r} is outside the supported split leakage audit."
        )
    backend = value.get("io_backend", {"type": "disk"})
    width = None
    if kind == _CLIP:
        if split != "train" or task != "recurrent_restoration":
            raise ValueError("VideoClipDataset supplies on-the-fly recurrent training clips only.")
        # The provider opens indexed video containers directly from dataroot_gt.
        backend = {"type": "disk"}
    if kind == _VIDEO:
        if task == "recurrent_restoration" and split == "train":
            raise ValueError(
                "Recurrent training requires OTF sequences. Prepared PairedVideoDataset is supported for evaluation."
            )
        if not isinstance(backend, Mapping) or backend.get("type", "disk") != "disk":
            raise ValueError("PairedVideoDataset reads prepared frame folders on disk.")
        backend = {**backend, "type": "disk"}
        width = value.get("clip_size")
        if type(width) is not int or width < 1:
            raise ValueError(
                "Provide a positive integer clip_size for PairedVideoDataset."
            )
    if not isinstance(backend, Mapping) or backend.get("type") not in ("disk", "lmdb"):
        raise ValueError("The leakage audit supports disk and read-only LMDB.")
    if backend["type"] == "lmdb" and (
        backend.get("readonly", True) is not True
        or backend.get("subdir", True) is not True
    ):
        raise ValueError(
            "The paired and single-image audit requires read-only LMDB directories."
        )

    needed = ("dataroot_lq",) if kind == _SINGLE else ("dataroot_gt",)
    if kind in {_PAIRED, _HYBRID, _VIDEO}:
        needed += ("dataroot_lq",)
    optional = (
        ("target_dataroot_gt", "paired_dataroot_gt")
        if kind == _HYBRID
        else ("target_dataroot_gt",)
        if kind == _OTF
        else ()
    )
    roots = {}
    for key in needed + optional:
        values = value.get(key)
        if values is None and key in optional:
            continue
        if isinstance(values, str) and key not in optional:
            values = [values]
        if not isinstance(values, list) or not values:
            form = (
                "a nonempty path array"
                if key in optional
                else "a path or nonempty path array"
            )
            raise ValueError(f"Provide {key} as {form}.")
        roots[key] = [str(local_path(path, root)) for path in values]
    if kind in {_PAIRED, _HYBRID, _VIDEO}:
        paired_gt = roots.get("paired_dataroot_gt", roots["dataroot_gt"])
        if len(paired_gt) != len(roots["dataroot_lq"]):
            raise ValueError("Provide one paired GT root per LQ root.")
    if "target_dataroot_gt" in roots and len(roots["target_dataroot_gt"]) != len(
        roots["dataroot_gt"]
    ):
        raise ValueError("Provide one target GT root per OTF source root.")

    if backend["type"] == "lmdb" and "target_dataroot_gt" in roots:
        raise ValueError("target_dataroot_gt is not supported with OTF LMDB datasets.")

    # Container sampling uses its own per-video index, even when a recipe
    # carries legacy image options or a paired validation scene manifest.
    meta = None if kind == _CLIP else value.get("video_scene_manifest" if kind == _VIDEO else "meta_info")
    manifest = (
        str(local_path(meta, root))
        if meta is not None and backend["type"] != "lmdb"
        else None
    )
    template = "{}" if kind in (_VIDEO, _CLIP) else value.get("filename_tmpl", "{}")
    if not isinstance(template, str):
        raise ValueError("filename_tmpl must be a string.")
    name = value.get("name")
    return _Split(
        split,
        name if isinstance(name, str) else split,
        kind,
        dict(backend),
        roots,
        manifest,
        template,
        clip_size=width,
    )


def _locate(role: str, group: int, root: str, selected: str, kind: str) -> _Input:
    if kind == "lmdb":
        key = os.fsencode(selected)
        return _Input(role, group, key, os.fsencode(root), key)
    return _Input(
        role,
        group,
        os.fsencode(os.path.relpath(selected, root)),
        os.fsencode(selected),
        b"",
    )


def _paired(split: _Split) -> Iterator[tuple[_Input, ...]]:
    from trainner_data.manifests import (
        paired_paths_from_lmdb,
        paired_paths_from_meta_info_file,
    )

    gt_roots = split.roots.get("paired_dataroot_gt", split.roots["dataroot_gt"])
    for group, (lq, gt) in enumerate(
        zip(split.roots["dataroot_lq"], gt_roots, strict=True), 1
    ):
        folders, roles = ([lq], [gt]), ("lq", "gt")
        if split.kind == "lmdb":
            rows = paired_paths_from_lmdb(folders, roles, include_storage=True)
        elif split.manifest is not None:
            rows = paired_paths_from_meta_info_file(
                folders, roles, split.manifest, split.template
            )
        else:
            from traiNNer.data.data_util import paired_paths_from_folder

            rows = paired_paths_from_folder(folders, roles, split.template)
        gt_role = "pairedGt" if split.provider == _HYBRID else "gt"
        for row in rows:
            yield (
                _locate(gt_role, group, gt, row["gt_path"], split.kind),
                _locate("lq", group, lq, row["lq_path"], split.kind),
            )
        del rows


def _raise_unreadable(_path: Path, error: OSError) -> None:
    raise error


def _supervised_images(
    split: _Split, check: Callable[[], None]
) -> Iterator[tuple[_Input, ...]]:
    from trainner_data.task_folders import (
        IMAGE_EXTENSIONS,
        MASK_EXTENSIONS,
        class_entries,
        files_by_stem,
        image_files,
    )

    def image(group: int, root: Path, path: Path, context: Json) -> tuple[_Input]:
        return (
            _Input(
                "image",
                group,
                os.fsencode(path.relative_to(root)),
                os.fsencode(path),
                b"",
                context,
            ),
        )

    if split.provider != "cocodetectiondataset" and split.manifest is not None:
        from trainner_data.supervised_manifest import folder_members

        roots = [Path(p) for p in split.roots["dataroot_lq"]]
        masks = (
            [Path(p) for p in split.roots["dataroot_gt"]]
            if "dataroot_gt" in split.roots
            else None
        )
        for group, relative, mask in folder_members(Path(split.manifest), roots, masks):
            check()
            context = (
                {"maskRelativePath": str(mask)}
                if mask is not None
                else {"className": relative.parts[0]}
            )
            root = roots[group - 1]
            yield image(group, root, root / relative, context)
        return

    for group, directory in enumerate(split.roots["dataroot_lq"], 1):
        root = Path(directory)
        if split.provider == "classfolderdataset":
            for folder, is_class in class_entries(root):
                check()
                if not is_class:
                    raise ValueError(f"Image {folder} has no class label.")
                paths = image_files(folder, check=check, unreadable=_raise_unreadable)
                if not paths:
                    raise ValueError(
                        f"Class folder {folder} contains no supported images."
                    )
                for path in paths:
                    yield image(group, root, path, {"className": folder.name})
                del paths
        elif split.provider == "imagemaskdataset":
            mask_root = Path(split.roots["dataroot_gt"][group - 1])
            inputs = files_by_stem(
                root, IMAGE_EXTENSIONS, check=check, unreadable=_raise_unreadable
            )
            masks = files_by_stem(
                mask_root, MASK_EXTENSIONS, check=check, unreadable=_raise_unreadable
            )
            for stem, path in inputs.items():
                mask = masks.pop(stem, None)
                if mask is None:
                    raise ValueError(f"Missing mask for {stem} in {mask_root}.")
                yield image(
                    group,
                    root,
                    path,
                    {"maskRelativePath": str(mask.relative_to(mask_root))},
                )
            if masks:
                raise ValueError(f"Missing image for {min(masks)} in {root}.")
            del inputs, masks
        else:
            from trainner_data.coco import image_path, index_coco, read_coco

            from studio.task_data import _box

            resolved_root = root.resolve()
            if not resolved_root.is_dir():
                raise FileNotFoundError(f"COCO image root does not exist: {root}")
            assert split.manifest is not None
            manifest = Path(split.manifest)
            index = index_coco(
                read_coco(manifest), manifest, split.categories, check=check
            )
            for record in index.images.values():
                # The provider includes empty and ignored-only images. Invalid boxes
                # reject construction even if their image would otherwise be selected.
                for ann in index.annotations.get(record.id, ()):
                    check()
                    _box(ann, record)
                yield image(
                    group,
                    root,
                    root
                    / image_path(resolved_root, record.file_name).relative_to(
                        resolved_root
                    ),
                    {"cocoImageId": record.id, "fileName": record.file_name},
                )


def _video_frames(
    split: _Split, check: Callable[[], None]
) -> Iterator[tuple[_Input, ...]]:
    from trainner_data.frame_sequences import FrameWindowIndex

    from .video_sources import selected_frame_positions

    assert split.clip_size is not None
    index = FrameWindowIndex(
        split.roots["dataroot_lq"],
        split.clip_size,
        split.roots["dataroot_gt"],
        scene_manifest=split.manifest,
        check=check,
    )
    offset = 0
    for scene in index.scenes:
        context = {"key": scene.key[1], "numbered": scene.key[0]}
        last_context = 0
        for role, folder, pos, first in selected_frame_positions(scene, index.width):
            if role == "lq":
                # LQ positions arrive in order. Advance each eligible start at
                # most once to find the last window which still uses this frame.
                last_context = max(first, last_context)
                while (
                    last_context < len(scene.starts)
                    and scene.starts[last_context] <= pos
                ):
                    if last_context % 4096 == 0:
                        check()
                    last_context += 1
                last = last_context
                label = "lqContext"
            else:
                last = first
                label = "gtCenter"
            name = os.fsencode(scene.frames[pos])
            yield (
                _Input(
                    label,
                    scene.slot,
                    name,
                    os.path.join(os.fsencode(folder), name),
                    b"",
                    context,
                    (offset + first, offset + last),
                ),
            )
        offset += len(scene.starts)


def _selected(split: _Split, check: Callable[[], None]) -> Iterator[tuple[_Input, ...]]:
    if split.provider == _CLIP:
        raise ValueError(
            "VideoClipDataset draws a clip length per batch and a start per item from indexed "
            "video containers. Its GT is generated from decoded YUV crops. There are zero "
            "independently encoded training GT images to compare with prepared validation "
            "files. Train/validation overlap is unknown."
        )
    if split.provider == _VIDEO:
        yield from _video_frames(split, check)
        return
    if split.provider in _SUPERVISED.values():
        yield from _supervised_images(split, check)
        return

    if split.provider == _PAIRED:
        yield from _paired(split)
        return

    if split.provider == _SINGLE:
        for group, lq in enumerate(split.roots["dataroot_lq"], 1):
            if split.kind == "lmdb":
                from trainner_data.manifests import lmdb_keys

                paths = lmdb_keys(lq)
            else:
                from traiNNer.data.single_image_dataset import SingleImageDataset
                from trainner_config.options import DatasetOptions

                paths = SingleImageDataset(
                    DatasetOptions(
                        name=split.name,
                        type=split.provider,
                        phase=split.id.split("_", 1)[0],
                        dataroot_lq=[lq],
                        meta_info=split.manifest,
                        io_backend=split.backend.copy(),
                    )
                ).paths
            for path in paths:
                yield (_locate("lq", group, lq, path, split.kind),)
            del paths
        return

    from traiNNer.data.realesrgan_dataset import RealESRGANDataset
    from trainner_config.options import DatasetOptions

    otf_selected = 0
    source_roots = split.roots["dataroot_gt"]
    targets = split.roots.get("target_dataroot_gt")
    for group, gt in enumerate(source_roots, 1):
        target = targets[group - 1] if targets is not None else None
        provider = RealESRGANDataset(
            DatasetOptions(
                name=split.name,
                type=split.provider,
                phase=split.id.split("_", 1)[0],
                dataroot_gt=[gt],
                target_dataroot_gt=[target] if target is not None else None,
                meta_info=split.manifest,
                io_backend=split.backend.copy(),
            )
        )
        for index, path in enumerate(provider.paths):
            otf_selected += 1
            storage = provider.storage[index] if split.kind == "lmdb" else gt
            source = _locate("gt", group, storage, path, split.kind)
            if provider.target_paths is None:
                yield (source,)
            else:
                assert target is not None
                yield (
                    source,
                    _locate(
                        "targetGt", group, target, provider.target_paths[index], "disk"
                    ),
                )
        del provider
    if split.provider == _HYBRID:
        paired_selected = False
        for index, pair in enumerate(_paired(split)):
            paired_selected = True
            # The hybrid provider takes paired[i % len(paired)] for each OTF row.
            if index < otf_selected:
                yield pair
        if not paired_selected:
            raise ValueError("The hybrid provider has no paired rows to index.")


def _stamp(info: os.stat_result) -> str:
    return f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}"


def _file_info(path: bytes) -> tuple[int, str]:
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise _NotRegular("Expected a regular file.")
    return info.st_size, _stamp(info)


def _manifest_stamps(split: _Split) -> tuple[str, ...]:
    if split.kind == "lmdb":
        roots = sorted({path for dirs in split.roots.values() for path in dirs})
        paths = [os.path.join(root, "meta_info.txt") for root in roots]
    else:
        paths = [split.manifest] if split.manifest is not None else []
    return tuple(_file_info(os.fsencode(path))[1] for path in paths)


def _commit_hash(digest: Any, fields: Any) -> None:
    digest.update(
        json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\n")


@lru_cache(maxsize=1)
def _kernel() -> Callable[..., int] | None:
    if sys.platform != "linux":
        return None
    try:
        subprocess.run(
            ["make", "--no-print-directory", "-C", str(_NATIVE)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=True,
        )
        native = ctypes.CDLL(
            str(_NATIVE / "build/clang/sha256.so")
        ).studio_split_sha256_v1
    except (OSError, AttributeError, subprocess.SubprocessError):
        return None
    native.argtypes = (ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, _HASH_CALLBACK)
    native.restype = ctypes.c_int
    return native


def disk_file_digest(
    path: bytes, size: int, stamp: str, check: Callable[[], None] | None = None
) -> tuple[bytes, str]:
    with open(path, "rb", buffering=0, opener=_open_read) as stream:
        fd = stream.fileno()
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or _stamp(info) != stamp:
            raise _Changed("File changed before hashing.")
        kernel = _kernel()
        if kernel is None:
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                if check is not None:
                    check()
                block = stream.read(min(remaining, _BLOCK))
                if not block:
                    raise _Changed("File shrank while hashing.")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1):
                raise _Changed("File grew while hashing.")
            sha, engine = digest.digest(), "hashlibFile"
        else:
            output = (ctypes.c_uint8 * 32)()
            failure = None

            def interrupt() -> int:
                nonlocal failure
                try:
                    assert check is not None
                    check()
                except Exception as exc:
                    failure = exc
                    return 1
                return 0

            callback = (
                _HASH_CALLBACK(interrupt) if check is not None else _HASH_CALLBACK()
            )
            status = kernel(fd, size, output, callback)
            if failure is not None:
                raise failure
            if status == 1:
                raise _Changed("File length changed while hashing.")
            if status < 0:
                raise OSError(-status, os.strerror(-status))
            if status != 0:
                raise RuntimeError(f"The SHA-256 kernel returned {status}.")
            sha, engine = bytes(output), "clangFile"
        if _stamp(os.fstat(fd)) != stamp or _file_info(path)[1] != stamp:
            raise _Changed("File changed while hashing.")
        return sha, engine


def _lmdb_options(backend: Mapping[str, Any]) -> Json:
    options = {"readahead": False}
    options.update(
        {
            k: v
            for k, v in backend.items()
            if k not in ("type", "db_paths", "client_keys")
        }
    )
    # An audit never creates or writes the LMDB lock region.
    options.update(readonly=True, create=False, lock=False)
    return options


@contextmanager
def _mapped_values(
    lmdb: Any, path: bytes, options: Json, expected: str | None
) -> Iterator[tuple[Any, str]]:
    data_path = os.path.join(path, b"data.mdb")
    _, before = _file_info(data_path)
    with lmdb.open(os.fsdecode(path), **options) as env, env.begin(buffers=True) as txn:
        txn_id = txn.id()
        stamp = f"{before}:{txn_id}"
        if expected is not None and expected != stamp:
            raise _Changed("LMDB data or transaction changed after inventory.")
        yield txn, stamp

        try:
            unchanged = (
                _file_info(data_path)[1] == before
                and env.info()["last_txnid"] == txn_id
            )
        except (OSError, _NotRegular) as exc:
            raise _Changed(
                "LMDB data became unavailable during the read transaction."
            ) from exc
        if not unchanged:
            raise _Changed("LMDB data changed during the read transaction.")


class _Audit:
    def __init__(
        self,
        db: sqlite3.Connection,
        progress: Callable[[Json], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        self.db = db
        self.progress = progress
        self.cancelled = cancelled
        self.next_check = self.next_progress = 0.0
        self.failure: Exception | None = None
        self.stores: dict[bytes, Json] = {}
        self.store_stamps: dict[bytes, str] = {}
        self.engines: dict[str, int] = {}
        self.issues: list[Json] = []
        self.issue_total = 0
        if cancelled is not None:
            db.set_progress_handler(self._interrupt, 20_000)

    def check(self, *, force: bool = False) -> None:
        if self.cancelled is not None and (force or monotonic() >= self.next_check):
            self.next_check = monotonic() + 0.15
            try:
                stop = self.cancelled()
            except Exception as exc:
                raise RuntimeError(
                    "The split audit cancellation callback failed."
                ) from exc
            if stop:
                raise SplitAuditCancelled("Dataset split audit was cancelled.")

    def _interrupt(self) -> int:
        try:
            self.check()
        except Exception as exc:
            self.failure = exc
            return 1
        return 0

    def update(
        self,
        stage: str,
        completed: int,
        total: int | None,
        *,
        split: str | None = None,
        force: bool = False,
    ) -> None:
        self.check(force=force)
        if self.progress is not None and (force or monotonic() >= self.next_progress):
            self.next_progress = monotonic() + 0.5
            try:
                self.progress(
                    {
                        "stage": stage,
                        "completed": completed,
                        "total": total,
                        "splitId": split,
                    }
                )
            except Exception as exc:
                raise RuntimeError("The split audit progress callback failed.") from exc
            self.check(force=True)

    def issue(self, code: str, path: str, message: str, **where: Any) -> None:
        self.issue_total += 1
        if len(self.issues) < _MAX_ISSUES:
            self.issues.append(
                {
                    "level": "error",
                    "code": code,
                    "path": path,
                    "message": message,
                    **where,
                }
            )

    def selection(
        self, ordinal: int, split: _Split, *, record: bool
    ) -> tuple[str, int, int]:
        stage = "selecting" if record else "verifying-selection"
        self.update(stage, 0, None, split=split.id, force=True)
        digest = hashlib.sha256()
        _commit_hash(
            digest, ("split-selection-v1", split.id, split.provider, split.kind)
        )
        samples = inputs = 0
        if split.provider == _VIDEO:
            _commit_hash(digest, ("window-ranges-v1", split.clip_size))
        for member, row in enumerate(_selected(split, self.check), 1):
            windows = row[0].windows
            samples = member if windows is None else max(samples, windows[1])
            self.update(stage, samples, None, split=split.id)
            _commit_hash(digest, [item.selection_fields() for item in row])
            inputs += len(row)
            if not record:
                continue
            for item in row:
                locator = (split.kind, item.path, item.key)
                self.db.execute(
                    "INSERT OR IGNORE INTO sources(kind, path, key) VALUES (?,?,?)",
                    locator,
                )
                self.db.execute(
                    """INSERT INTO inputs
                       SELECT ?,?,?,?,?,?,?,?,id FROM sources WHERE kind=? AND path=? AND key=?""",
                    (
                        ordinal,
                        member,
                        item.role,
                        item.root,
                        item.relative,
                        json.dumps(item.label_context, ensure_ascii=True)
                        if item.label_context is not None
                        else None,
                        *(item.windows or (None, None)),
                        *locator,
                    ),
                )
        if not samples:
            raise ValueError("The provider selected no samples.")
        self.update(stage, samples, samples, split=split.id, force=True)
        return digest.hexdigest(), samples, inputs

    def fail_source(
        self,
        source_id: int,
        path: bytes,
        key: bytes | None,
        exc: Exception,
        *,
        observed: bool,
    ) -> None:
        state = _error_state(exc, observed=observed)
        self.db.execute("UPDATE sources SET state=? WHERE id=?", (state, source_id))
        details = {"key": os.fsdecode(key)} if key is not None else {}
        self.issue(state, os.fsdecode(path), str(exc), **details)

    def fail_store(self, path: bytes, exc: Exception, *, observed: bool) -> None:
        state = _error_state(exc, observed=observed)
        self.db.execute(
            """UPDATE sources SET state=?
               WHERE kind='lmdb' AND path=? AND state IN ('pending','ready')""",
            (state, path),
        )
        self.issue(state, os.fsdecode(path), str(exc), storage="lmdb")

    def _cached(self, kind: str, stamp: str, key: bytes) -> bytes | None:
        row = self.db.execute(
            "SELECT sha FROM hashes WHERE kind=? AND stamp=? AND key=?",
            (kind, stamp, key),
        ).fetchone()
        return row[0] if row is not None else None

    def _remember(
        self, kind: str, stamp: str, key: bytes, digest: bytes, engine: str
    ) -> None:
        self.db.execute(
            "INSERT INTO hashes VALUES (?,?,?,?)", (kind, stamp, key, digest)
        )
        self.engines[engine] = self.engines.get(engine, 0) + 1

    def disk_metadata(self) -> None:
        total = self.db.execute(
            "SELECT COUNT(*) FROM sources WHERE kind='disk'"
        ).fetchone()[0]
        self.update("disk-metadata", 0, total, force=True)
        for done, (source_id, path) in enumerate(
            self.db.execute(
                "SELECT id, path FROM sources WHERE kind='disk' ORDER BY id"
            ),
            1,
        ):
            self.update("disk-metadata", done, total)
            try:
                size, stamp = _file_info(path)
            except (OSError, ValueError) as exc:
                self.fail_source(source_id, path, None, exc, observed=False)
                continue
            self.db.execute(
                "UPDATE sources SET size=?, stamp=?, state='ready' WHERE id=?",
                (size, stamp, source_id),
            )

    def disk_hashes(self) -> None:
        query = """FROM sources JOIN sizes USING(size)
                   WHERE kind='disk' AND state='ready'"""
        total = self.db.execute("SELECT COUNT(*) " + query).fetchone()[0]
        self.update("disk-hashing", 0, total, force=True)
        for done, (source_id, path, size, stamp) in enumerate(
            self.db.execute("SELECT id, path, size, stamp " + query + " ORDER BY id"), 1
        ):
            self.update("disk-hashing", done, total)
            try:
                sha = self._cached("disk", stamp, b"")
                if sha is None:
                    sha, engine = disk_file_digest(path, size, stamp, self.check if self.cancelled is not None else None)
                    self._remember("disk", stamp, b"", sha, engine)
            except (OSError, ValueError) as exc:
                self.fail_source(source_id, path, None, exc, observed=True)
                continue
            self.db.execute("UPDATE sources SET sha=? WHERE id=?", (sha, source_id))

    def disk_verify(self) -> None:
        total = self.db.execute(
            "SELECT COUNT(*) FROM sources WHERE kind='disk' AND state='ready'"
        ).fetchone()[0]
        self.update("disk-verifying", 0, total, force=True)
        for done, (source_id, path, stamp) in enumerate(
            self.db.execute(
                "SELECT id,path,stamp FROM sources WHERE kind='disk' AND state='ready' ORDER BY id"
            ),
            1,
        ):
            self.update("disk-verifying", done, total)
            try:
                if _file_info(path)[1] != stamp:
                    raise _Changed("File changed after inventory.")
            except (OSError, ValueError) as exc:
                self.fail_source(source_id, path, None, exc, observed=True)

    def lmdb_pass(self, stage: str) -> None:
        query = (
            "FROM sources JOIN sizes USING(size) WHERE kind='lmdb' AND state='ready'"
            if stage == "hashing"
            else "FROM sources WHERE kind='lmdb'"
        )
        paths = self.db.execute("SELECT DISTINCT path " + query + " ORDER BY path")
        for done, (path,) in enumerate(paths, 1):
            self.update(f"lmdb-{stage}", done, None)
            expected = self.store_stamps.get(path)
            if stage != "metadata" and expected is None:
                continue
            try:
                import lmdb
            except ImportError:
                self.fail_store(
                    path,
                    ValueError("The installed trainer environment cannot import lmdb."),
                    observed=False,
                )
                continue
            try:
                with _mapped_values(lmdb, path, self.stores[path], expected) as (
                    txn,
                    stamp,
                ):
                    if stage != "verifying":
                        self._lmdb_values(
                            lmdb, txn, path, stamp, hashing=stage == "hashing"
                        )
                self.store_stamps[path] = stamp
            except (lmdb.Error, OSError, TypeError, ValueError) as exc:
                self.fail_store(path, exc, observed=stage != "metadata")
                self.store_stamps.pop(path, None)

    def _lmdb_values(
        self, lmdb: Any, txn: Any, path: bytes, stamp: str, *, hashing: bool
    ) -> None:
        if hashing:
            rows = self.db.execute(
                """SELECT id,key,size FROM sources JOIN sizes USING(size)
                   WHERE kind='lmdb' AND path=? AND state='ready' ORDER BY key""",
                (path,),
            )
        else:
            rows = self.db.execute(
                "SELECT id,key,size FROM sources WHERE kind='lmdb' AND path=? ORDER BY key",
                (path,),
            )
        for source_id, key, size in rows:
            self.check()
            try:
                if not key.isascii():
                    raise ValueError("The trainer's LMDB reader requires an ASCII key.")
                value = txn.get(key)
                if value is None:
                    raise FileNotFoundError("Selected LMDB key is missing.")
                if not hashing:
                    self.db.execute(
                        "UPDATE sources SET size=?,stamp=?,state='ready' WHERE id=?",
                        (len(value), stamp, source_id),
                    )
                else:
                    if len(value) != size:
                        raise _Changed("LMDB value length changed after inventory.")
                    sha = self._cached("lmdb", stamp, key)
                    if sha is None:
                        digest = hashlib.sha256()
                        view = memoryview(value)
                        for start in range(0, size, _BLOCK):
                            self.check()
                            digest.update(view[start : start + _BLOCK])
                        sha = digest.digest()
                        self._remember("lmdb", stamp, key, sha, "lmdbBuffer")
                    self.db.execute(
                        "UPDATE sources SET sha=? WHERE id=?", (sha, source_id)
                    )
            except (lmdb.Error, OSError, ValueError) as exc:
                self.fail_source(source_id, path, key, exc, observed=hashing)


def _open_read(path: bytes, flags: int) -> int:
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def _error_state(exc: Exception, *, observed: bool) -> str:
    if isinstance(exc, _Changed) or (
        observed
        and isinstance(exc, (FileNotFoundError, NotADirectoryError, _NotRegular))
    ):
        return "changed"
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
        return "missing"
    return "unreadable"


_VERIFIED_SPLITS = """
    JOIN verified_splits v ON v.split=i.split
    WHERE s.state='ready'
"""
_MEMBERS = " FROM sources s JOIN inputs i ON i.source=s.id " + _VERIFIED_SPLITS


def _window_matches(db: sqlite3.Connection) -> None:
    """Aggregate confirmed multi-member digests without expanding frame reuse.

    In start order only the interval above prior_hi adds new windows. For a
    GT singleton, a matching LQ is present if an interval starting at or
    before its window reaches it. Process LQ first at equal starts.
    """
    db.execute(
        """CREATE TEMP TABLE match_splits AS
           WITH repeated AS (
               SELECT s.size,s.sha """
        + _MEMBERS
        + """
               AND s.sha IS NOT NULL GROUP BY s.size,s.sha HAVING COUNT(*)>1
           ), members AS (
               SELECT s.size,s.sha,i.split,i.sample,i.role,i.first_window IS NOT NULL AS video,
                      COALESCE(i.first_window,i.sample) AS lo,
                      COALESCE(i.last_window,i.sample) AS hi
               """
        + _MEMBERS
        + """
               AND (s.size,s.sha) IN (SELECT size,sha FROM repeated)
           ), spans AS (
               SELECT *,
                 MAX(hi) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_hi,
                 MAX(CASE WHEN role='lqContext' THEN hi END)
                     OVER (w ROWS UNBOUNDED PRECEDING) AS lq_hi
               FROM members WINDOW w AS (
                   PARTITION BY size,sha,split ORDER BY lo,(role='lqContext') DESC,sample,role
               )
           )
           SELECT size,sha,split,COUNT(*) AS inputs,
                  SUM(MAX(0, hi-MAX(lo-1,COALESCE(prior_hi,0)))) AS samples,
                  CASE WHEN MAX(video) THEN MAX(CASE WHEN role='gtCenter' AND lq_hi>=lo
                                                     THEN 1 ELSE 0 END)
                       ELSE COUNT(*)>COUNT(DISTINCT sample) END AS same_sample
           FROM spans GROUP BY size,sha,split"""
    )
    db.execute("CREATE UNIQUE INDEX match_splits_key ON match_splits(sha,size,split)")
    db.execute(
        """CREATE TEMP TABLE matches AS
           SELECT size,sha,SUM(inputs) AS inputs,SUM(samples) AS samples,
                  MIN(split)=0 AND MAX(split)>0 AS train_validation,
                  SUM(split>0)>1 AS cross_validation,
                  MAX(inputs>1 AND samples>1) AS within_split,
                  MAX(same_sample) AS same_sample
           FROM match_splits GROUP BY size,sha"""
    )


def _groups(work: _Audit, splits: list[Json], limit: int) -> tuple[Json, list[Json]]:
    db = work.db
    db.execute("CREATE INDEX sources_digest ON sources(sha,size) WHERE state='ready'")
    temporal = any(split.get("provider") == _VIDEO for split in splits)
    if temporal:
        _window_matches(db)
    else:
        db.execute(
            """CREATE TEMP TABLE matches AS
               WITH samples AS (
                   SELECT s.size,s.sha,i.split,i.sample,COUNT(*) AS inputs
                   """
            + _MEMBERS
            + """
                   AND s.sha IS NOT NULL
                   GROUP BY s.size,s.sha,i.split,i.sample
               )
               SELECT size,sha,SUM(inputs) AS inputs,COUNT(*) AS samples,
                      MIN(split)=0 AND MAX(split)>0 AS train_validation,
                      COUNT(DISTINCT CASE WHEN split>0 THEN split END)>1 AS cross_validation,
                      COUNT(*)>COUNT(DISTINCT split) AS within_split,
                      MAX(inputs>1) AS same_sample
               FROM samples GROUP BY size,sha HAVING SUM(inputs)>1"""
        )
    totals = db.execute(
        """SELECT COUNT(*),COALESCE(SUM(train_validation),0),
                  COALESCE(SUM(cross_validation),0),COALESCE(SUM(within_split),0),
                  COALESCE(SUM(same_sample),0) FROM matches"""
    ).fetchone()
    counts = dict(
        zip(
            (
                "groups",
                "trainValidationGroups",
                "crossValidationGroups",
                "withinSplitGroups",
                "sameSampleRoleGroups",
            ),
            totals,
            strict=True,
        )
    )
    items = []
    # For one digest, look up its sources before their occurrences. With an
    # ordinary join SQLite can walk the entire ordered training split to find
    # four examples, once per displayed group, even when only one row matches.
    query = (
        " FROM sources s CROSS JOIN inputs i ON i.source=s.id "
        + _VERIFIED_SPLITS + " AND s.sha=? AND s.size=?"
    )
    for size, digest, inputs, samples, leakage, val, within, same in db.execute(
        """SELECT * FROM matches
           ORDER BY train_validation DESC,cross_validation DESC,within_split DESC,sha,size
           LIMIT ?""",
        (limit,),
    ):
        work.check()
        by_split: dict[int, Json] = {}
        for ordinal, role, count in db.execute(
            "SELECT i.split,i.role,COUNT(*) " + query + " GROUP BY i.split,i.role",
            (digest, size),
        ):
            item = by_split.setdefault(
                ordinal, {"splitId": splits[ordinal]["id"], "inputs": 0, "roles": {}}
            )
            item["inputs"] += count
            item["roles"][role] = count

        if temporal:
            for ordinal, affected in db.execute(
                "SELECT split,samples FROM match_splits WHERE sha=? AND size=?",
                (digest, size),
            ):
                by_split[ordinal]["samples"] = affected
        evidence = []
        projection = "SELECT i.split,i.sample,i.role,i.root,i.relative,s.path,s.key,s.kind,i.label_context,i.first_window,i.last_window "
        # Keep evidence on each side of a train/validation overlap even when
        # one side accounts for nearly all occurrences.
        sides = (
            ((" AND i.split=0", _EXAMPLES // 2), (" AND i.split>0", _EXAMPLES // 2))
            if leakage
            else (("", _EXAMPLES),)
        )
        for condition, capacity in sides:
            for (
                split,
                sample,
                role,
                root,
                relative,
                path,
                key,
                kind,
                context,
                first_window,
                last_window,
            ) in db.execute(
                projection
                + query
                + condition
                + " ORDER BY i.split,i.sample,i.role LIMIT ?",
                (digest, size, capacity),
            ):
                split_id = splits[split]["id"]
                record = {
                    "splitId": split_id,
                    "sampleId": f"{split_id}:{first_window if first_window is not None else sample}",
                    "role": role,
                    "rootId": str(root),
                    "path": os.fsdecode(path),
                    "storage": kind,
                }
                if kind == "disk":
                    record["relativePath"] = os.fsdecode(relative)
                else:
                    record["key"] = os.fsdecode(key)
                if first_window is not None:
                    record["windowRange"] = [first_window, last_window]
                    record["scene"] = json.loads(context)
                elif context is not None:
                    record["labelContext"] = json.loads(context)
                evidence.append(record)

        items.append(
            {
                "sha256": digest.hex(),
                "encodedBytes": size,
                "inputs": inputs,
                "samples": samples,
                "trainValidation": bool(leakage),
                "crossValidation": bool(val),
                "withinSplit": bool(within),
                "sameSampleRoles": bool(same),
                "bySplit": list(by_split.values()),
                "examples": evidence,
                "examplesTruncated": len(evidence) < inputs,
            }
        )
    return counts, items


def _report(work: _Audit, splits: list[Json], limit: int) -> Json:
    db = work.db
    work.update("reporting", 0, None, force=True)
    sources = dict(db.execute("SELECT state,COUNT(*) FROM sources GROUP BY state"))
    candidate = db.execute(
        "SELECT COUNT(*) FROM sources JOIN sizes USING(size)"
    ).fetchone()[0]
    hashed = db.execute(
        "SELECT COUNT(*) FROM sources WHERE state='ready' AND sha IS NOT NULL"
    ).fetchone()[0]
    pending = db.execute(
        """SELECT COUNT(*) FROM sources JOIN sizes USING(size)
           WHERE state='ready' AND sha IS NULL"""
    ).fetchone()[0]
    if pending:
        raise RuntimeError(
            f"{pending} candidate sources did not produce a digest or a read status."
        )

    for split, state, count, distinct_sources in db.execute(
        """SELECT i.split,s.state,COUNT(*),COUNT(DISTINCT i.source) FROM inputs i JOIN sources s ON s.id=i.source
           GROUP BY i.split,s.state"""
    ):
        counts = splits[split]["counts"]
        counts["inputStates"][state] = count
        counts["sources"] += distinct_sources
    for split in splits:
        counts = split["counts"]
        if counts is not None:
            split["complete"] = split["selectionComplete"] and counts[
                "inputs"
            ] == counts["inputStates"].get("ready", 0)

    complete = all(split["complete"] for split in splits)
    match_counts, groups = _groups(work, splits, limit)
    for split, samples, sources_count in db.execute(
        "SELECT i.split,COUNT(DISTINCT i.sample),COUNT(DISTINCT i.source) "
        + _MEMBERS
        + """ AND (s.size,s.sha) IN (SELECT size,sha FROM matches WHERE train_validation)
              GROUP BY i.split"""
    ):
        splits[split]["counts"].update(
            trainValidationSamples=samples, trainValidationSources=sources_count
        )
    if any(split.get("provider") == _VIDEO for split in splits):
        # Groups may touch the same window through different files or roles.
        # Count the union across *all* confirmed train/evaluation groups.
        previous_split = None
        high = 0
        for ordinal, lo, hi in db.execute(
            "SELECT i.split,i.first_window,i.last_window "
            + _MEMBERS
            + """ AND i.first_window IS NOT NULL
                  AND (s.size,s.sha) IN (SELECT size,sha FROM matches WHERE train_validation)
                  ORDER BY i.split,i.first_window,i.last_window"""
        ):
            work.check()
            counts = splits[ordinal]["counts"]
            if ordinal != previous_split:
                previous_split, high = ordinal, 0
                counts["trainValidationSamples"] = 0
            counts["trainValidationSamples"] += max(0, hi - max(high, lo - 1))
            high = max(high, hi)
    if complete:
        identity = hashlib.sha256()
        _commit_hash(
            identity, ("split-audit-v1", [split["selectionId"] for split in splits])
        )
        for split, sample, role, root, relative, size, sha in db.execute(
            """SELECT i.split,i.sample,i.role,i.root,i.relative,s.size,s.sha """
            + _MEMBERS
            + " ORDER BY i.split,i.sample,i.role"
        ):
            work.check()
            _commit_hash(
                identity,
                (
                    split,
                    sample,
                    role,
                    root,
                    os.fsdecode(relative),
                    size,
                    sha.hex() if sha else None,
                ),
            )
        report_id = identity.hexdigest()
    else:
        report_id = None

    work.check(force=True)
    return {
        "algorithm": "sha256-encoded",
        "complete": complete,
        "reportId": report_id,
        "hasTrainValidationOverlap": True
        if match_counts["trainValidationGroups"]
        else False
        if complete
        else None,
        "counts": {
            **match_counts,
            "sources": sum(sources.values()),
            "hashedSources": hashed,
            "sizeOnlySources": sources.get("ready", 0) - hashed,
            "candidateSources": candidate,
            "sourceStates": sources,
        },
        "splits": splits,
        "groups": groups,
        "groupsTruncated": len(groups) < match_counts["groups"],
        "hashEngines": work.engines,
        "limits": {
            "groups": limit,
            "examplesPerGroup": _EXAMPLES,
            "issues": _MAX_ISSUES,
        },
        "issues": work.issues,
        "issueTotal": work.issue_total,
        "issuesTruncated": len(work.issues) < work.issue_total,
    }


def _inventory(
    work: _Audit,
    datasets: Mapping[str, Any],
    names: list[str],
    root: Path,
    task: str,
    settings: Mapping[str, Any],
) -> tuple[list[Json], dict[int, tuple[_Split, tuple[str, ...]]]]:
    reports = []
    selections = {}
    store_options = {}
    for ordinal, name in enumerate(names):
        value = datasets[name]
        result = {
            "id": name,
            "name": name,
            "selectionId": None,
            "counts": None,
            "selectionComplete": False,
            "complete": False,
        }
        reports.append(result)
        work.db.execute("SAVEPOINT dataset")
        try:
            if not isinstance(value, Mapping):
                raise ValueError("The split must contain dataset options.")
            split = _configured_split(name, value, root, task, settings)
            result.update(
                name=split.name,
                provider=split.provider,
                storage=split.kind,
                roots=split.roots,
                manifest=split.manifest,
            )
            snapshot = _manifest_stamps(split)
            if split.kind == "lmdb":
                options = _lmdb_options(split.backend)
                locations = {
                    os.fsencode(path)
                    for paths in split.roots.values()
                    for path in paths
                }
                for path in locations:
                    real = os.path.realpath(path)
                    if store_options.get(real, options) != options:
                        raise ValueError(
                            f"Conflicting LMDB reader options for {os.fsdecode(real)}."
                        )
            selection_id, samples, inputs = work.selection(ordinal, split, record=True)
        except _SELECT_ERRORS as exc:
            work.db.execute("ROLLBACK TO dataset")
            work.db.execute("RELEASE dataset")
            work.issue("selectionError", f"datasets.{name}", str(exc), splitId=name)
            continue
        work.db.execute("RELEASE dataset")
        if split.kind == "lmdb":
            for path in locations:
                store_options[os.path.realpath(path)] = options
                work.stores[path] = options
        result.update(
            selectionId=selection_id,
            counts={
                "samples": samples,
                "inputs": inputs,
                "sources": 0,
                "inputStates": {},
                "trainValidationSamples": 0,
                "trainValidationSources": 0,
            },
        )
        if split.provider == _VIDEO:
            assert split.clip_size is not None
            result.update(
                clipSize=split.clip_size,
                sampleUnit="window",
                inputUnit="selectedFrameRole",
            )
            result["counts"]["windowInputUses"] = samples * (split.clip_size + 1)
        selections[ordinal] = (split, snapshot)
    return reports, selections


def _verify_selection(
    work: _Audit,
    reports: list[Json],
    selections: dict[int, tuple[_Split, tuple[str, ...]]],
) -> None:
    for ordinal, (split, stamp) in selections.items():
        work.check()
        try:
            if _manifest_stamps(split) != stamp:
                raise _Changed("A selection manifest changed during the audit.")
            # LMDB keys depend only on the manifests whose stamps were checked.
            # Disk selections can change through nested paths and flat fallbacks.
            if split.kind == "disk":
                selection_id, _, _ = work.selection(ordinal, split, record=False)
                if selection_id != reports[ordinal]["selectionId"]:
                    raise _Changed(
                        "The provider selected different rows during verification."
                    )
                if _manifest_stamps(split) != stamp:
                    raise _Changed("A selection manifest changed during verification.")
        except _SELECT_ERRORS as exc:
            work.issue(
                "selectionChanged", f"datasets.{split.id}", str(exc), splitId=split.id
            )
            continue
        reports[ordinal]["selectionComplete"] = True
        work.db.execute("INSERT INTO verified_splits VALUES (?)", (ordinal,))


def split_selection(
    body: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], Mapping[str, Any], list[str]]:
    datasets = body.get("datasets")
    if not isinstance(datasets, Mapping) or "train" not in datasets:
        raise ValueError("Provide the recipe datasets mapping with a train split.")
    task = body.get("task")
    if task is None:
        task = "image_restoration"
        if any(
            isinstance(options, Mapping)
            and str(options.get("type", "")).lower() in (_VIDEO, _CLIP)
            for options in datasets.values()
        ):
            # Legacy recurrent recipes select the task through network_g.type.
            from .recipes import task_from_config

            task = task_from_config(dict(body))
    settings = {}
    if task not in _SR_TASKS:
        if not isinstance(task, str) or task not in _SUPERVISED:
            raise ValueError(f"Unsupported task for split leakage: {task!r}.")
        settings = body.get("task_options", {})
        if not isinstance(settings, Mapping):
            raise ValueError("task_options must be an object.")
    validation = sorted(
        name
        for name in datasets
        if isinstance(name, str) and name.split("_", 1)[0] == "val"
    )
    if not validation:
        raise ValueError("Provide at least one val or val_* split.")
    return task, settings, datasets, validation


def audit_split_leakage(
    body: Mapping[str, Any],
    *,
    root: Path = _ROOT,
    scratch_dir: Path | None = None,
    progress: Callable[[Json], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Json:
    """Compare encoded image sources across the recipe's train and validation splits."""
    task, settings, datasets, validation = split_selection(body)
    limit = body.get("groupLimit", 100)
    if type(limit) is not int or not 0 <= limit <= _MAX_GROUPS:
        raise ValueError(f"groupLimit must be an integer from 0 to {_MAX_GROUPS}.")

    with TemporaryDirectory(prefix="split-audit-", dir=scratch_dir) as temporary:
        with closing(sqlite3.connect(Path(temporary) / "index.sqlite3")) as db:
            db.executescript(_SCHEMA)
            work = _Audit(db, progress, cancelled)
            try:
                work.check(force=True)
                splits, selections = _inventory(
                    work,
                    datasets,
                    ["train", *validation],
                    root.absolute(),
                    task,
                    settings,
                )
                work.disk_metadata()
                work.lmdb_pass("metadata")
                db.execute(
                    """INSERT INTO sizes SELECT s.size FROM sources s JOIN inputs i ON s.id=i.source
                       WHERE s.state='ready' GROUP BY s.size HAVING COUNT(*)>1"""
                )
                work.disk_hashes()
                work.lmdb_pass("hashing")
                _verify_selection(work, splits, selections)
                work.disk_verify()
                work.lmdb_pass("verifying")
                db.commit()
                return {"task": task, **_report(work, splits, limit)}
            except sqlite3.OperationalError:
                if work.failure is not None:
                    raise work.failure from None
                raise

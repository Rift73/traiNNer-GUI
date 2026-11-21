"""Prepare one byte-preserving pair of LMDB databases in a private job directory."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CHUNK_BYTES = 1 << 20
_BATCH_BYTES = 32 << 20
_BATCH_PAIRS = 256

Progress = Callable[[dict[str, Any]], None]
Cancelled = Callable[[], bool]
Signature = tuple[int, int, int, int, int]


class DatasetPackCancelled(RuntimeError):
    """The worker stopped before publishing a dataset."""


@dataclass(frozen=True, slots=True)
class _File:
    path: Path
    relative: str
    signature: Signature

    @property
    def size(self) -> int:
        return self.signature[2]


@dataclass(frozen=True, slots=True)
class _Root:
    path: Path
    identity: tuple[int, int]
    files: dict[Path, _File]


@dataclass(frozen=True, slots=True)
class _Pair:
    root_id: str
    gt: _File
    lq: _File

    @property
    def size(self) -> int:
        return self.gt.size + self.lq.size


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled is not None and cancelled():
        raise DatasetPackCancelled("Dataset preparation was cancelled.")


def _signature(status: os.stat_result) -> Signature:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _snapshot(
    path: Path,
    scan_names: Callable[..., Iterable[str]],
    cancelled: Cancelled | None,
) -> _Root:
    _check_cancelled(cancelled)
    root_status = path.stat()
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"Source root is not a directory: {path}")
    files = {}
    for relative in scan_names(str(path), recursive=True):
        _check_cancelled(cancelled)
        file_path = path / relative
        status = file_path.stat()
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"Source changed during inventory: {file_path}")
        files[file_path] = _File(file_path, relative, _signature(status))
    return _Root(path, (root_status.st_dev, root_status.st_ino), files)


def _plan(gt: list[_Root], lq: list[_Root], template: str) -> list[_Pair]:
    from traiNNer.data.data_util import paired_paths_from_folder

    sizes = []
    for gt_root, lq_root in zip(gt, lq, strict=True):
        count_gt, count_lq = len(gt_root.files), len(lq_root.files)
        if count_gt != count_lq:
            raise ValueError(
                f"Packing requires every visible file to have a partner. "
                f"{gt_root.path} contains {count_gt} files and {lq_root.path} contains {count_lq}."
            )
        sizes.append(count_gt)
    rows = paired_paths_from_folder(
        ([str(root.path) for root in lq], [str(root.path) for root in gt]),
        ("lq", "gt"),
        template,
    )
    expected = sum(sizes)
    if len(rows) != expected:
        raise ValueError(
            f"The filename template selected {len(rows)} of {expected} pairs. "
            "Packing requires a unique partner for every trainer-visible file."
        )
    if not rows:
        raise ValueError("The source folders contain no paired images.")

    pairs = []
    offset = 0
    for index, (gt_root, lq_root, count) in enumerate(
        zip(gt, lq, sizes, strict=True), 1
    ):
        matched_gt, matched_lq = set(), set()
        for row in rows[offset : offset + count]:
            gt_path, lq_path = Path(row["gt_path"]), Path(row["lq_path"])
            if gt_path in matched_gt or lq_path in matched_lq:
                raise ValueError(
                    f"More than one pair uses a source path in root group {index}."
                )
            if gt_path not in gt_root.files or lq_path not in lq_root.files:
                raise ValueError(f"Source selection changed in root group {index}.")
            matched_gt.add(gt_path)
            matched_lq.add(lq_path)
            pairs.append(
                _Pair(str(index), gt_root.files[gt_path], lq_root.files[lq_path])
            )
        offset += count
    return pairs


def _read_source(file: _File, cancelled: Cancelled | None) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    remaining = file.size
    try:
        with (
            open(
                file.path,
                "rb",
                opener=lambda path, flags: os.open(path, flags | os.O_NONBLOCK),
            ) as stream,
            io.BytesIO() as encoded,
        ):
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or _signature(status) != file.signature:
                raise ValueError(f"Source changed before copying: {file.path}")
            while remaining:
                _check_cancelled(cancelled)
                block = stream.read(min(remaining, _CHUNK_BYTES))
                if not block:
                    raise ValueError(f"Source shrank during copying: {file.path}")
                remaining -= len(block)
                encoded.write(block)
                digest.update(block)
            if (
                stream.read(1)
                or _signature(os.fstat(stream.fileno())) != file.signature
                or _signature(file.path.stat()) != file.signature
            ):
                raise ValueError(f"Source changed during copying: {file.path}")
            return encoded.getvalue(), digest.hexdigest()
    except OSError as exc:
        raise ValueError(f"Could not read source image {file.path}. {exc}") from exc


def _batches(pairs: list[_Pair]) -> Iterator[list[_Pair]]:
    start = 0
    size = 0
    for index, pair in enumerate(pairs):
        if index > start and (
            size + pair.size > _BATCH_BYTES or index - start == _BATCH_PAIRS
        ):
            yield pairs[start:index]
            start, size = index, 0
        size += pair.size
    yield pairs[start:]


def _map_size(encoded_bytes: int, pairs: int) -> int:
    # Reserving address space leaves room for value page rounding and indices.
    return encoded_bytes + (4 * pairs + 4096) * os.sysconf("SC_PAGE_SIZE")


def _put_record(
    txn: Any,
    key: bytes,
    file: _File,
    cancelled: Cancelled | None,
) -> dict[str, Any]:
    import pyvips
    from traiNNer.data.transforms import get_vips_dtype
    from traiNNer.utils.img_util import vipsimfrombuffer

    encoded, digest = _read_source(file, cancelled)
    _check_cancelled(cancelled)
    try:
        image = vipsimfrombuffer(encoded)
        width, height, bands = int(image.width), int(image.height), int(image.bands)
        get_vips_dtype(image)
    except (pyvips.Error, ValueError) as exc:
        raise ValueError(
            f"The trainer could not read the encoded header of {file.path}. {exc}"
        ) from exc
    if width <= 0 or height <= 0 or bands < 1 or bands == 2:
        raise ValueError(
            f"Unsupported trainer image geometry in {file.path}: {width} x {height}, {bands} bands."
        )
    del image
    if not txn.put(key, encoded, overwrite=False):
        raise ValueError(f"Duplicate LMDB key {key.decode('ascii')!r} for {file.path}.")
    return {
        "path": str(file.path),
        "relativePath": file.relative,
        "encodedBytes": file.size,
        "sha256": digest,
        "width": width,
        "height": height,
        "bands": bands,
    }


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(staged: Path, destination: Path) -> None:
    # A regular POSIX rename can replace a concurrent job's empty directory.
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "This runtime does not support atomic no-replace publication."
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if renameat2(
        at_fdcwd,
        os.fsencode(staged),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    ):
        error = ctypes.get_errno()
        if error in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL):
            raise OSError(
                error,
                "The output filesystem does not support atomic no-replace publication.",
                destination,
            )
        raise OSError(error, os.strerror(error), destination)


def pack_dataset(
    body: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    """Build paired LMDBs from a folder scan registration.

    Accepts gt/lq or gtRoots/lqRoots, scale, filenameTemplate, phase, and
    optional gtPatchSize. The caller supplies a new destination under an
    existing private output parent. The progress callback receives stage,
    completedPairs, totalPairs, and encodedBytes. The stages are validating,
    packing, and verifying. The cancelled predicate can stop the worker up
    to the atomic rename, raising DatasetPackCancelled.

    A successful result contains destination, gtPath, lqPath, inventoryPath,
    pairs, encodedBytes, scale, phase, and the dataset options fragment. An
    optional directorySyncWarning reports a post-publication fsync failure.
    """
    from studio.data import _positive, _roots
    from studio.data_sources import _geometry

    if body.get("source", "folder") != "folder" or body.get("metaInfo") is not None:
        raise ValueError("Dataset packing takes explicit source folders.")
    if body.get("task") not in (None, "image_restoration"):
        raise ValueError(
            "Dataset packing requires paired restoration images."
        )

    scale, patch = body.get("scale"), body.get("gtPatchSize")
    _positive(scale, "Scale")
    _positive(patch, "GT patch size")
    if scale is None:
        raise ValueError("Provide the scale used by the paired image registration.")
    phase = body.get("phase", "train")
    if phase not in ("train", "val", "test"):
        raise ValueError("Phase must be train, val, or test.")
    if phase == "train" and patch is not None and patch % scale:
        raise ValueError(
            "GT patch size must be divisible by scale for paired training."
        )
    template = body.get("filenameTemplate", "{}")
    if not isinstance(template, str) or "\0" in template:
        raise ValueError("filenameTemplate must be a string without NUL characters.")

    gt_paths = list(map(Path, _roots(body.get("gt"), body.get("gtRoots"), "gt")))
    lq_paths = list(map(Path, _roots(body.get("lq"), body.get("lqRoots"), "lq")))
    if not gt_paths or len(gt_paths) != len(lq_paths):
        raise ValueError(
            "Provide one LQ source directory per GT directory in the same order."
        )

    requested = Path(destination).expanduser()
    if not requested.name:
        raise ValueError("Provide a new destination directory.")
    output = requested.parent.resolve(strict=True) / requested.name
    if os.path.lexists(output):
        raise FileExistsError(
            errno.EEXIST, "Dataset destination already exists.", output
        )
    for root in gt_paths + lq_paths:
        if output.is_relative_to(root.resolve(strict=True)):
            raise ValueError(f"Place the new dataset outside the source root {root}.")

    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError(
            "The lmdb package is unavailable in this trainer environment."
        ) from exc
    from traiNNer.utils import scandir

    def report(stage: str, completed: int, total: int | None, written: int) -> None:
        _check_cancelled(cancelled)
        if progress is not None:
            progress(
                {
                    "stage": stage,
                    "completedPairs": completed,
                    "totalPairs": total,
                    "encodedBytes": written,
                }
            )
        _check_cancelled(cancelled)

    report("validating", 0, None, 0)
    gt_roots = [_snapshot(root, scandir, cancelled) for root in gt_paths]
    lq_roots = [_snapshot(root, scandir, cancelled) for root in lq_paths]
    pairs = _plan(gt_roots, lq_roots, template)
    expected_gt = sum(pair.gt.size for pair in pairs)
    expected_lq = sum(pair.lq.size for pair in pairs)
    result = {
        "destination": str(output),
        "gtPath": str(output / "gt.lmdb"),
        "lqPath": str(output / "lq.lmdb"),
        "inventoryPath": str(output / "inventory.json"),
        "pairs": len(pairs),
        "encodedBytes": expected_gt + expected_lq,
        "scale": scale,
        "phase": phase,
        "dataset": {
            "name": output.name,
            "type": "PairedImageDataset",
            "dataroot_gt": [str(output / "gt.lmdb")],
            "dataroot_lq": [str(output / "lq.lmdb")],
            "io_backend": {"type": "lmdb"},
            **({"gt_size": patch} if patch is not None and phase == "train" else {}),
        },
    }
    inventory = {
        "formatVersion": 1,
        "source": {
            "filenameTemplate": template,
            "scale": scale,
            "phase": phase,
            "gtPatchSize": patch,
            "roots": [
                {
                    "id": str(index),
                    "gt": str(gt_root.path),
                    "lq": str(lq_root.path),
                    "pairs": len(gt_root.files),
                }
                for index, (gt_root, lq_root) in enumerate(
                    zip(gt_roots, lq_roots, strict=True), 1
                )
            ],
        },
        "output": {
            "gt": "gt.lmdb",
            "lq": "lq.lmdb",
            "pairs": len(pairs),
            "encodedBytes": expected_gt + expected_lq,
        },
    }

    def geometry_issue(level: str, path: str, message: str) -> None:
        if level == "error":
            raise ValueError(f"{path}. {message}")

    report("packing", 0, len(pairs), 0)
    with tempfile.TemporaryDirectory(
        prefix=".trainner-pack-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir(mode=0o700)
        gt_db, lq_db = staged / "gt.lmdb", staged / "lq.lmdb"
        gt_db.mkdir(mode=0o700)
        lq_db.mkdir(mode=0o700)
        try:
            with ExitStack() as stack:
                gt_env = stack.enter_context(
                    lmdb.open(
                        str(gt_db),
                        map_size=_map_size(expected_gt, len(pairs)),
                        mode=0o600,
                    )
                )
                lq_env = stack.enter_context(
                    lmdb.open(
                        str(lq_db),
                        map_size=_map_size(expected_lq, len(pairs)),
                        mode=0o600,
                    )
                )
                gt_meta = stack.enter_context(
                    (gt_db / "meta_info.txt").open("x", encoding="ascii")
                )
                lq_meta = stack.enter_context(
                    (lq_db / "meta_info.txt").open("x", encoding="ascii")
                )
                manifest = stack.enter_context(
                    (staged / "inventory.json").open("x", encoding="utf-8")
                )
                manifest.write(
                    json.dumps(inventory, ensure_ascii=True)[:-1] + ',"records":[\n'
                )
                completed, written = 0, 0
                digits = len(str(len(pairs)))
                for batch in _batches(pairs):
                    with (
                        gt_env.begin(write=True) as gt_txn,
                        lq_env.begin(write=True) as lq_txn,
                    ):
                        for pair in batch:
                            _check_cancelled(cancelled)
                            key = f"p{completed + 1:0{digits}d}"
                            key_bytes = key.encode("ascii")
                            row_gt = _put_record(gt_txn, key_bytes, pair.gt, cancelled)
                            row_lq = _put_record(lq_txn, key_bytes, pair.lq, cancelled)
                            geometry = _geometry(
                                (row_gt["width"], row_gt["height"]),
                                (row_lq["width"], row_lq["height"]),
                                row_gt["path"],
                                row_lq["path"],
                                scale,
                                patch,
                                phase,
                                geometry_issue,
                            )
                            for meta, row in ((gt_meta, row_gt), (lq_meta, row_lq)):
                                meta.write(
                                    f"{key}.png ({row['height']},{row['width']},{row['bands']})\n"
                                )
                            if completed:
                                manifest.write(",\n")
                            json.dump(
                                {
                                    "key": key,
                                    "rootId": pair.root_id,
                                    "gt": row_gt,
                                    "lq": row_lq,
                                    "geometry": geometry,
                                },
                                manifest,
                                ensure_ascii=True,
                            )
                            completed += 1
                            written += pair.size
                    report("packing", completed, len(pairs), written)
                manifest.write("\n]}\n")
                for artifact in (gt_meta, lq_meta, manifest):
                    artifact.flush()
                    os.fsync(artifact.fileno())
                for environment in (gt_env, lq_env):
                    if environment.stat()["entries"] != len(pairs):
                        raise RuntimeError(
                            "The number of LMDB records differs from the paired selection."
                        )
                    environment.sync(True)
        except lmdb.Error as exc:
            raise RuntimeError(
                f"Could not write the private LMDB dataset. {exc}"
            ) from exc

        report("verifying", len(pairs), len(pairs), expected_gt + expected_lq)
        for directory in (gt_db, lq_db, staged):
            _sync_directory(directory)
        for original in gt_roots + lq_roots:
            if _snapshot(original.path, scandir, cancelled) != original:
                raise ValueError(
                    f"Source inventory changed before publication: {original.path}"
                )
        _check_cancelled(cancelled)
        _publish(staged, output)

    try:
        _sync_directory(output.parent)
    except OSError as exc:
        result["directorySyncWarning"] = str(exc)
    return result

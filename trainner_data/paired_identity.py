"""Startup identity of the ordered sources used by the built-in paired loader."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from traiNNer.data.paired_image_dataset import PairedImageDataset
    from trainner_config.options import DatasetOptions

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE = _ROOT / "native/studio/data"
_PROVIDER = "pairedimagedataset"
_STAT = struct.Struct("<7Q")
_U64 = struct.Struct("<Q")
_MAX_PATHS = 2048
_PATH_BYTES = 1 << 20
_NANOS = 1_000_000_000
_MASK = (1 << 64) - 1


class PairedTrainingIdentityError(ValueError):
    """A paired training source could not be captured or matched for replay."""

    def __init__(self, message: str, path: str = "datasets.train") -> None:
        super().__init__(message)
        self.path = path


class PairedReplaySettings(TypedDict):
    scale: int
    gt_size: int
    lq_size: int
    use_hflip: bool
    use_rot: bool
    native_paired_crop: bool
    color: str | None
    mean: list[float] | None
    std: list[float] | None


class PairedTrainingIdentity(TypedDict):
    version: Literal[1]
    provider: Literal["pairedimagedataset"]
    storage: Literal["folder", "meta_info", "lmdb"]
    pairs: int
    selection_sha256: str
    source_metadata_sha256: str
    settings: PairedReplaySettings


def _normalization(values: list[float] | None, name: str) -> list[float] | None:
    if values is None:
        return None
    if any(not math.isfinite(value) for value in values):
        raise PairedTrainingIdentityError(
            "Paired replay requires finite normalization values.",
            f"datasets.train.{name}",
        )
    return list(values)


def paired_replay_settings(
    opt: DatasetOptions, *, scale: int | None = None
) -> PairedReplaySettings:
    """Validate and resolve the input settings without opening a dataset."""
    if opt.type.casefold() != _PROVIDER or opt.phase not in (None, "train"):
        raise PairedTrainingIdentityError(
            "Ordered paired identity requires the built-in PairedImageDataset for training.",
            "datasets.train.type",
        )
    backend = opt.io_backend.get("type")
    if backend not in ("disk", "lmdb") or (
        backend == "lmdb"
        and (
            opt.io_backend.get("subdir", True) is not True
            or opt.io_backend.get("readonly", True) is not True
        )
    ):
        raise PairedTrainingIdentityError(
            "Paired replay requires disk or read-only LMDB directory storage.",
            "datasets.train.io_backend",
        )

    scale = opt.scale if scale is None else scale
    if type(scale) is not int or scale <= 0:
        raise PairedTrainingIdentityError(
            "Paired replay requires a positive integer scale.", "scale"
        )
    high, low = opt.gt_size, opt.lq_size
    if high is None and type(low) is int:
        high = low * scale
    if low is None and type(high) is int:
        low = high // scale
    if type(high) is not int or type(low) is not int or low <= 0 or high != low * scale:
        raise PairedTrainingIdentityError(
            "Paired replay requires a positive aligned crop with gt_size equal to lq_size times scale."
        )
    return {
        "scale": scale,
        "gt_size": high,
        "lq_size": low,
        "use_hflip": opt.use_hflip,
        "use_rot": opt.use_rot,
        "native_paired_crop": opt.native_paired_crop,
        "color": opt.color,
        "mean": _normalization(opt.mean, "mean"),
        "std": _normalization(opt.std, "std"),
    }


def _join_path(cwd: bytes, value: str) -> bytes:
    encoded = os.fsencode(value)
    if not encoded or b"\0" in encoded:
        raise PairedTrainingIdentityError(f"Invalid paired source path {value!r}.")
    # Preserve the loader's spelling. Collapsing .. can change symlink traversal.
    return os.path.join(cwd, encoded)


def _file_stamp(path: bytes) -> bytes:
    try:
        info = os.stat(path)
    except OSError as exc:
        raise PairedTrainingIdentityError(
            f"Could not stat paired source {os.fsdecode(path)!r}. {exc.strerror}"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise PairedTrainingIdentityError(
            f"Expected a regular paired source at {os.fsdecode(path)!r}."
        )
    msec, mnano = divmod(info.st_mtime_ns, _NANOS)
    csec, cnano = divmod(info.st_ctime_ns, _NANOS)
    return _STAT.pack(
        info.st_dev & _MASK,
        info.st_ino & _MASK,
        info.st_size,
        msec & _MASK,
        mnano,
        csec & _MASK,
        cnano,
    )


@lru_cache(maxsize=1)
def _stat_kernel() -> Callable[..., int] | None:
    if sys.platform != "linux":
        return None
    try:
        subprocess.run(
            ["make", "--no-print-directory", "-C", str(_NATIVE), "paired-identity"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=True,
        )
        kernel = ctypes.CDLL(
            str(_NATIVE / "build/clang/paired_identity.so")
        ).studio_paired_stat_v1
    except (OSError, AttributeError, subprocess.SubprocessError):
        return None
    kernel.argtypes = (
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel.restype = ctypes.c_int
    return kernel


def _disk_chunks(paths: Iterable[bytes]) -> Iterator[tuple[bytes, int]]:
    packed = bytearray()
    count = 0
    for path in paths:
        if count and (count == _MAX_PATHS or len(packed) + len(path) + 1 > _PATH_BYTES):
            yield bytes(packed), count
            packed.clear()
            count = 0
        packed.extend(path)
        packed.append(0)
        count += 1
    if count:
        yield bytes(packed), count


def _disk_identity(rows: Sequence[Mapping[str, str]], cwd: bytes) -> tuple[str, str]:
    ordered = (_join_path(cwd, row[field]) for row in rows for field in ("gt_path", "lq_path"))
    return disk_file_identity(ordered, b"paired-disk")


def disk_file_identity(ordered: Iterable[bytes], domain: bytes) -> tuple[str, str]:
    """Digest ordered file paths and version stamps, using the same LLVM kernel as the paired loader."""
    selection = hashlib.sha256(domain + b"-selection-v1\0")
    metadata = hashlib.sha256(domain + b"-metadata-v1\0")
    kernel = _stat_kernel()
    output = (
        (ctypes.c_uint8 * (_STAT.size * _MAX_PATHS))() if kernel is not None else None
    )
    failed = ctypes.c_size_t()
    for paths, count in _disk_chunks(ordered):
        selection.update(paths)
        if kernel is None:
            for path in paths.split(b"\0")[:-1]:
                metadata.update(_file_stamp(path))
            continue
        status = kernel(paths, len(paths), count, output, ctypes.byref(failed))
        if status:
            if status == -2:
                raise PairedTrainingIdentityError(
                    "Invalid path batch for the paired identity tool."
                )
            end = paths.index(b"\0", failed.value)
            name = os.fsdecode(paths[failed.value : end])
            message = (
                "Expected a regular file." if status == -1 else os.strerror(status)
            )
            raise PairedTrainingIdentityError(
                f"Could not stat paired source {name!r}. {message}"
            )
        assert output is not None
        metadata.update(memoryview(output).cast("B")[: count * _STAT.size])
    return selection.hexdigest(), metadata.hexdigest()


def _lmdb_metadata(folders: list[str]) -> str:
    """Read each database's committed transaction number and backing-file stat."""
    import lmdb

    metadata = hashlib.sha256(b"paired-lmdb-metadata-v1\0")
    for folder in folders:
        path = os.fsencode(os.path.join(folder, "data.mdb"))
        stamp = _file_stamp(path)
        try:
            with lmdb.open(
                folder, readonly=True, create=False, lock=False, readahead=False
            ) as env:
                transaction = env.info()["last_txnid"]
                if (
                    _file_stamp(path) != stamp
                    or env.info()["last_txnid"] != transaction
                ):
                    raise PairedTrainingIdentityError(
                        f"LMDB changed while capturing paired identity: {folder!r}."
                    )
        except lmdb.Error as exc:
            raise PairedTrainingIdentityError(
                f"Could not inspect paired LMDB transaction metadata for {folder!r}. {exc}"
            ) from exc
        metadata.update(stamp)
        metadata.update(_U64.pack(transaction))
    return metadata.hexdigest()


def _lmdb_identity(rows: Sequence[Mapping[str, str]], cwd: bytes) -> tuple[str, str]:
    selection = hashlib.sha256(b"paired-lmdb-selection-v1\0")
    folders: dict[str, bytes] = {}
    packed = bytearray()
    for row in rows:
        for db_field, key_field in (("gt_db", "gt_path"), ("lq_db", "lq_path")):
            name = row[db_field]
            folder = folders.get(name)
            if folder is None:
                folder = folders[name] = _join_path(cwd, name)
            try:
                key = row[key_field].encode("ascii")
            except UnicodeEncodeError as exc:
                raise PairedTrainingIdentityError(
                    f"The trainer requires an ASCII LMDB key, got {row[key_field]!r} in {name!r}."
                ) from exc
            packed.extend(folder)
            packed.append(0)
            packed.extend(_U64.pack(len(key)))
            packed.extend(key)
        if len(packed) >= _PATH_BYTES:
            selection.update(packed)
            packed.clear()
    selection.update(packed)

    # python-lmdb cannot safely open a second environment for the same file in
    # one process. A short-lived probe also works when a trainer reader is open.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "trainner_data.paired_identity",
                "--lmdb-metadata",
            ],
            cwd=_ROOT,
            input=json.dumps([os.fsdecode(path) for path in folders.values()]).encode(
                "ascii"
            ),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr.decode(errors="replace").strip()
            or "The LMDB metadata probe failed."
        )
        raise PairedTrainingIdentityError(detail) from exc
    except OSError as exc:
        raise PairedTrainingIdentityError(
            f"Could not start the LMDB metadata probe. {exc}"
        ) from exc
    digest = probe.stdout.decode("ascii").strip()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PairedTrainingIdentityError(
            "The LMDB metadata probe returned an invalid digest."
        )
    return selection.hexdigest(), digest


def capture_paired_training_identity(
    dataset: PairedImageDataset,
) -> PairedTrainingIdentity:
    """Capture the built loader once before replay or checkpointing begins."""
    from traiNNer.data.paired_image_dataset import PairedImageDataset

    if type(dataset) is not PairedImageDataset or dataset.opt.phase != "train":
        raise PairedTrainingIdentityError(
            "Ordered paired identity requires an instantiated built-in PairedImageDataset for training.",
            "datasets.train.type",
        )
    settings = paired_replay_settings(dataset.opt)
    settings["mean"] = _normalization(dataset.mean, "mean")
    settings["std"] = _normalization(dataset.std, "std")
    # The dataset chooses its actual lane at construction.
    settings["native_paired_crop"] = dataset._native_crop  # noqa: SLF001
    rows = dataset.paths
    if not rows:
        raise PairedTrainingIdentityError(
            "The paired loader selected no training pairs."
        )
    cwd = os.getcwdb()
    storage: Literal["folder", "meta_info", "lmdb"]
    if dataset.lmdb is not None:
        storage = "lmdb"
        selection, metadata = _lmdb_identity(rows, cwd)
    else:
        storage = "meta_info" if dataset.opt.meta_info is not None else "folder"
        selection, metadata = _disk_identity(rows, cwd)
    return {
        "version": 1,
        "provider": "pairedimagedataset",
        "storage": storage,
        "pairs": len(rows),
        "selection_sha256": selection,
        "source_metadata_sha256": metadata,
        "settings": settings,
    }


def require_paired_training_identity(
    saved: object, current: PairedTrainingIdentity
) -> None:
    """Raise when a checkpoint cannot prove the same ordered paired inputs."""
    fresh = "Set checkpoint_rng to false to continue with fresh streams."
    if saved is None:
        raise PairedTrainingIdentityError(
            f"This checkpoint has no paired training-data identity. {fresh}",
            "checkpoint_rng",
        )
    if (
        not isinstance(saved, dict)
        or type(saved.get("version")) is not int
        or saved["version"] != current["version"]
        or saved.get("provider") != current["provider"]
    ):
        raise PairedTrainingIdentityError(
            f"This checkpoint has an incompatible paired training-data identity. {fresh}",
            "checkpoint_rng",
        )
    changed = []
    if (
        saved.get("storage") != current["storage"]
        or type(saved.get("pairs")) is not int
        or saved["pairs"] != current["pairs"]
        or saved.get("selection_sha256") != current["selection_sha256"]
    ):
        changed.append("selection or row order")
    elif saved.get("source_metadata_sha256") != current["source_metadata_sha256"]:
        changed.append("encoded-source metadata")
    if saved.get("settings") != current["settings"]:
        changed.append("crop, augmentation, or normalization settings")
    if changed:
        raise PairedTrainingIdentityError(
            f"Paired training-data identity changed ({'; '.join(changed)}). {fresh}",
            "checkpoint_rng",
        )


def _main() -> None:
    if sys.argv[1:] != ["--lmdb-metadata"]:
        raise SystemExit("Use the paired training identity Python API.")
    try:
        print(_lmdb_metadata(json.load(sys.stdin)))
    except (OSError, ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    _main()

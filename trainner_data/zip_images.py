"""Index and read original encoded images in ordinary ZIP archives."""

from __future__ import annotations

import os
import posixpath
import stat
from collections import deque
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from functools import cache
from multiprocessing.util import Finalize
from ntpath import splitdrive
from struct import Struct
from typing import TYPE_CHECKING
from zipfile import ZIP_STORED, BadZipFile, ZipFile, ZipInfo, _get_decompressor

from trainner_data.file_version import artifact_version

if TYPE_CHECKING:
    from trainner_config.options import DatasetOptions


class ZipConfigError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


def validate_zip_dataset(dataset: DatasetOptions) -> list[str]:
    """Validate ZIP options before opening either the manifest or the archives."""
    if dataset.type.lower() != "realesrgandataset":
        raise ZipConfigError(
            "io_backend.type", "ZIP storage requires RealESRGANDataset."
        )
    if dataset.target_dataroot_gt is not None:
        raise ZipConfigError(
            "target_dataroot_gt",
            "target_dataroot_gt is not supported with ZIP datasets.",
        )
    extra = sorted(dataset.io_backend.keys() - {"type"})
    if extra:
        raise ZipConfigError(
            "io_backend",
            f"ZIP storage accepts only io_backend.type. Remove {', '.join(extra)}.",
        )
    roots = dataset.dataroot_gt
    if isinstance(roots, str):
        roots = [roots]
    if not roots or any(not root.strip() or "\x00" in root for root in roots):
        raise ZipConfigError(
            "dataroot_gt", "Select one or more ZIP archives in dataroot_gt."
        )
    return roots


def _require_zip_suffix(archive: str) -> None:
    if not archive.lower().endswith(".zip"):
        raise ValueError(f"Expected a .zip image archive, got {archive!r}.")


def zip_archive_stat(archive: str) -> os.stat_result:
    """Check the path without opening a directory or special file as an archive."""
    _require_zip_suffix(archive)
    info = os.stat(archive)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular .zip archive file: {archive}")
    return info


def _manifest_zip_name(archive: str, name: str) -> str:
    # ZIP names use forward slashes. Accept the ordinary relative aliases a
    # loose manifest can spell, while keeping the lookup inside the archive.
    relative = name.replace(os.sep, "/")
    parts = relative.split("/")
    if (
        not parts[0]
        or parts[-1] in ("", ".")
        or "\\" in relative
        or "\x00" in relative
        or ".." in parts
    ):
        raise ValueError(f"Unsafe ZIP manifest name {archive}!/{name!r}.")
    canonical = posixpath.normpath(relative)
    if splitdrive(canonical)[0]:
        raise ValueError(f"Unsafe ZIP manifest name {archive}!/{name!r}.")
    return canonical


def _selected_zip_names(
    archive: str, source: ZipFile, manifest: list[str] | None
) -> Iterator[str]:
    _require_zip_suffix(archive)
    if manifest is not None and not manifest:
        raise ValueError(f"No image names in the manifest for ZIP archive {archive!r}.")
    explicit = (
        [_manifest_zip_name(archive, name) for name in manifest]
        if manifest is not None
        else None
    )
    missing = set(explicit) if explicit is not None else None
    checked_methods = {ZIP_STORED}
    found = False
    for member in source.infolist():
        name = member.orig_filename
        parts = name.removesuffix("/").split("/")
        if (
            "\\" in name
            or "\x00" in name
            or splitdrive(name)[0]
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise ValueError(f"Unsafe ZIP member {archive}!/{name!r}.")
        if member.is_dir() or (
            name not in missing
            if missing is not None
            else any(part.startswith(".") for part in parts)
        ):
            continue
        if source.getinfo(name) is not member:
            raise ValueError(f"Duplicate ZIP image {archive}!/{name}.")
        if member.external_attr & 0x10 or (
            member.create_system == 3
            and stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG)
        ):
            raise ValueError(f"Expected a regular ZIP image: {archive}!/{name}.")
        # The ZIP flags mark encryption, compressed patches and strong encryption.
        if member.flag_bits & (1 | (1 << 5) | (1 << 6)):
            raise ValueError(
                f"Expected an unencrypted, ordinary ZIP image: {archive}!/{name}."
            )
        method = member.compress_type
        if method not in checked_methods:
            # Check the same decoder ZipFile.open would use, without reading
            # each member's local header or creating a decoder for each image.
            try:
                _get_decompressor(method)
            except RuntimeError as exc:
                raise ValueError(
                    f"Unsupported ZIP compression method {method} for {archive}!/{name}: {exc}"
                ) from exc
            checked_methods.add(method)
        if missing is None:
            found = True
            yield name
        else:
            missing.remove(name)

    if explicit is not None:
        if missing:
            name = next(name for name in explicit if name in missing)
            raise ValueError(f"Manifest image missing from ZIP: {archive}!/{name}.")
        yield from explicit
    elif not found:
        raise ValueError(f"No image members in ZIP archive {archive!r}.")


def selected_zip_names(
    archive: str, source: ZipFile, manifest: list[str] | None = None
) -> list[str]:
    """Select paths from an open ZIP in loose discovery or manifest order."""
    names = _selected_zip_names(archive, source, manifest)
    if manifest is not None:
        return list(names)
    return sorted(
        names, key=None if os.sep == "/" else lambda name: name.replace("/", os.sep)
    )


def validate_zip_archive(archive: str, manifest: list[str] | None = None) -> None:
    """Validate selection without collecting or sorting unmanifested members."""
    zip_archive_stat(archive)
    with ZipFile(archive) as source:
        for _ in _selected_zip_names(archive, source, manifest):
            pass


@cache
def _worker_pool() -> ThreadPoolExecutor:
    # Created only inside a DataLoader worker, never by a parent doing a sample read.
    pool = ThreadPoolExecutor(max_workers=1)
    Finalize(None, pool.shutdown, exitpriority=10)
    return pool


# Keep only the central-directory fields needed to open an individual member.
# All sizes and offsets include ZIP64. The end offset preserves zipfile's
# overlap check even when the next directory entry is outside the manifest.
_MEMBER = Struct("<QQQQIHH")
_MAX_AHEAD_BYTES = 16 * 1024 * 1024


class _IndexedZipFile(ZipFile):
    def _RealGetContents(self) -> None:  # noqa: N802 - CPython ZipFile hook
        # The parent already parsed the directory. Workers use ZipFile.read(ZipInfo)
        # on independent handles, retaining stdlib header, decompression and CRC checks.
        pass


def _index_archive(
    archive: str, manifest: list[str] | None, records: bytearray
) -> tuple[list[str], str]:
    zip_archive_stat(archive)
    with open(archive, "rb") as raw:
        version = artifact_version(os.fstat(raw.fileno()))
        with ZipFile(raw) as source:
            names = selected_zip_names(archive, source, manifest)
            for name in names:
                info = source.getinfo(name)
                records.extend(
                    _MEMBER.pack(
                        info.file_size,
                        info.compress_size,
                        info.header_offset,
                        info._end_offset,  # noqa: SLF001
                        info.CRC,
                        info.flag_bits,
                        info.compress_type,
                    )
                )
        if artifact_version(os.fstat(raw.fileno())) != version:
            raise ValueError(f"ZIP archive changed during indexing: {archive}")
    return names, version


class ZipImageReader:
    def __init__(self, roots: Iterable[str], manifest: list[str] | None = None) -> None:
        # Studio imports the selection helpers without loading the training runtime.
        import torch

        self.paths: list[str] = []
        self.storage: list[str] = []
        self._identities: dict[str, str] = {}
        records = bytearray()
        for root in roots:
            archive = os.path.abspath(root)
            names, version = _index_archive(archive, manifest, records)
            if self._identities.setdefault(archive, version) != version:
                raise ValueError(f"ZIP archive changed during indexing: {archive}")
            self.paths.extend(names)
            self.storage.extend([archive] * len(names))

        # Fork inherits these packed bytes; PyTorch also shares CPU tensor storage
        # when it sends a dataset to a spawn/forkserver worker. No index sidecar.
        self._index = torch.frombuffer(records, dtype=torch.uint8)
        self._buffer = records
        self._handles: dict[str, _IndexedZipFile] = {}
        self._pid = os.getpid()

    def __getstate__(self) -> dict:
        # The parent keeps the original bytearray alive even if spawning a
        # DataLoader remaps the tensor. Only send the shareable copy to workers.
        return {**vars(self), "_handles": {}, "_buffer": None}

    def _handle(self, archive: str) -> _IndexedZipFile:
        if self._pid != os.getpid():
            for inherited in self._handles.values():
                inherited.close()
            self._handles.clear()
            self._pid = os.getpid()
        handle = self._handles.get(archive)
        if handle is None:
            handle = _IndexedZipFile(archive, "r")
            try:
                assert handle.fp is not None
                if (
                    artifact_version(os.fstat(handle.fp.fileno()))
                    != self._identities[archive]
                ):
                    raise ValueError(
                        f"ZIP archive changed since dataset indexing: {archive}. Reopen the dataset."
                    )
            except BaseException:
                handle.close()
                raise
            self._handles[archive] = handle
        return handle

    def _member(self, index: int) -> ZipInfo:
        buffer = self._buffer if self._buffer is not None else self._index.numpy()
        info = ZipInfo(self.paths[index])
        (
            info.file_size,
            info.compress_size,
            info.header_offset,
            info._end_offset,  # noqa: SLF001
            info.CRC,
            info.flag_bits,
            info.compress_type,
        ) = _MEMBER.unpack_from(buffer, index * _MEMBER.size)
        return info

    def _read_member(self, archive: str, info: ZipInfo) -> bytes:
        try:
            return self._handle(archive).read(info)
        except (BadZipFile, KeyError, OSError, RuntimeError, ValueError) as exc:
            exc.add_note(f"Reading ZIP image {archive}!/{info.filename}")
            raise

    def read(self, index: int) -> bytes:
        return self._read_member(self.storage[index], self._member(index))

    def read_ahead(
        self, indices: Iterable[int], *, in_worker: bool
    ) -> Generator[bytes, None, None]:
        context = (
            nullcontext(_worker_pool())
            if in_worker
            else ThreadPoolExecutor(max_workers=1)
        )
        with context as pool:
            pending = deque()
            pending_bytes = 0
            try:
                for index in indices:
                    info = self._member(index)
                    # Large images still load, one at a time. Do not queue other
                    # full uncompressed members alongside an oversized member.
                    while pending and (
                        len(pending) >= 5
                        or pending_bytes + info.file_size > _MAX_AHEAD_BYTES
                    ):
                        size, future = pending.popleft()
                        pending_bytes -= size
                        yield future.result()
                    pending.append(
                        (
                            info.file_size,
                            pool.submit(self._read_member, self.storage[index], info),
                        )
                    )
                    pending_bytes += info.file_size
                while pending:
                    yield pending.popleft()[1].result()
            finally:
                outstanding = [future for _, future in pending]
                for future in outstanding:
                    future.cancel()
                if outstanding:
                    wait(outstanding)

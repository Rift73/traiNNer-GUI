"""Identify Windows drive mounts using the current Linux mount namespace."""

import errno
import os
import stat
import sys
from collections.abc import Iterable
from functools import cache, lru_cache


@cache
def _drvfs_devices() -> frozenset[int]:
    if sys.platform != "linux":
        return frozenset()

    devices: set[int] = set()
    try:
        with open(
            "/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape"
        ) as mounts:
            for line in mounts:
                mount, _, filesystem = line.partition(" - ")
                kind, _, options = filesystem.split(maxsplit=2)
                # WSL2 uses 9p for several things, including its driver mounts.
                # Only the Windows drive export has aname=drvfs.
                is_drvfs = kind == "drvfs" or (
                    kind == "9p"
                    and any(
                        option.partition(";")[0] == "aname=drvfs"
                        for option in options.strip().split(",")
                    )
                )
                if is_drvfs:
                    major, minor = map(int, mount.split()[2].split(":"))
                    devices.add(os.makedev(major, minor))
    except OSError:
        return frozenset()
    return frozenset(devices)


@lru_cache(maxsize=4096)
def drvfs_device_for_directory(directory: str) -> int | None:
    """Resolve a directory once per cache entry, following folder symlinks."""
    devices = _drvfs_devices()
    if not devices:
        return None
    try:
        info = os.stat(directory)
    except OSError:
        return None
    if stat.S_ISDIR(info.st_mode) and info.st_dev in devices:
        return info.st_dev
    return None


def is_drvfs_directory(path: str | os.PathLike[str]) -> bool:
    """Return whether a dataset directory resides on DrvFS.

    Workers cache the mount table and directory lookups for stable dataset
    roots and working directories.
    """
    return drvfs_device_for_directory(os.fspath(path)) is not None


def close_drvfs_directories(pins: dict[str, int]) -> None:
    for fd in pins.values():
        os.close(fd)
    pins.clear()


def pin_drvfs_directories(pins: dict[str, int], paths: Iterable[str]) -> None:
    """Keep shard directories warm within the worker's spare FD budget."""
    if sys.platform != "linux":
        return

    from resource import RLIM_INFINITY, RLIMIT_NOFILE, getrlimit

    max_pins = 32
    limit, _ = getrlimit(RLIMIT_NOFILE)
    if limit != RLIM_INFINITY:
        # Leave room for concurrent reads, libvips and DataLoader's IPC.
        # Counting the /proc directory's own FD is deliberately conservative.
        reserve = 32
        try:
            spare = limit - len(os.listdir("/proc/self/fd"))
        except OSError:
            # The FD table may already be full.
            spare = 0
        if spare < reserve:
            close_drvfs_directories(pins)
            return
        max_pins = min(max_pins, len(pins) + spare - reserve)

    for path in paths:
        if len(pins) >= max_pins:
            break
        directory = os.path.dirname(path) or "."
        if directory in pins or drvfs_device_for_directory(directory) is None:
            continue
        try:
            pins[directory] = os.open(
                directory, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC
            )
        except OSError as error:
            # Pinning is optional. The actual file open enforces access and reports errors.
            if error.errno in (errno.EMFILE, errno.ENFILE):
                close_drvfs_directories(pins)
                break

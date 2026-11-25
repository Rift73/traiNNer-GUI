"""User path resolution shared by the headless tools on Linux and WSL."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PureWindowsPath

from trainner_data.file_version import (
    artifact_version as artifact_version,  # noqa: PLC0414
)


def local_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Provide a file or folder path.")
    value = value.strip()
    if os.name == "posix" and re.match(r"^[A-Za-z]:[/\\]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    elif os.name == "posix" and value.startswith(("\\\\", "//")):
        parts = value.replace("\\", "/").split("/")
        if len(parts) >= 5 and parts[2].casefold() in ("wsl.localhost", "wsl$"):
            if parts[3].casefold() != os.environ.get("WSL_DISTRO_NAME", "Ubuntu").casefold():
                raise ValueError("The path belongs to a different WSL distribution.")
            value = "/" + "/".join(parts[4:])
        else:
            raise ValueError("Mount the network share in WSL and use its Linux path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _unescape_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _windows_mount_path(path: str) -> str | None:
    # Use the deepest covering mount so a Linux mount below a drive keeps its own path.
    mount_point = ""
    covering_mount = None
    try:
        with open(
            "/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape"
        ) as mounts:
            for line in mounts:
                mount, _, filesystem = line.partition(" - ")
                root, point = mount.split()[3:5]
                point = _unescape_mount_field(point)
                if len(point) >= len(mount_point) and (
                    path == point or path.startswith(point.rstrip("/") + "/")
                ):
                    mount_point = point
                    covering_mount = root, filesystem
    except OSError:
        return None

    if covering_mount is None:
        return None
    root, filesystem = covering_mount
    kind, source, options = filesystem.split(maxsplit=2)
    if kind != "drvfs" and (
        kind != "9p" or not re.search(r"(?:^|,)aname=drvfs(?:[;,]|$)", options)
    ):
        return None

    windows_root = PureWindowsPath(_unescape_mount_field(source) + "\\")
    if not windows_root.is_absolute():
        return None
    suffix = _unescape_mount_field(root).rstrip("/") + path[len(mount_point) :]
    return str(windows_root).rstrip("\\") + "\\" + suffix.lstrip("/").replace("/", "\\")


def clipboard_paths(path: Path) -> dict[str, str]:
    """Return an absolute host path and, on WSL, its Windows clipboard spelling."""
    native = path.absolute()
    paths = {"native": str(native)}
    if sys.platform != "linux" or "microsoft" not in os.uname().release.casefold():
        return paths

    try:
        target = str(native.resolve())
    except (OSError, RuntimeError):
        return paths
    if re.search(r'[<>:"\\|?*\x00-\x1f]', target):
        return paths

    if windows := _windows_mount_path(target):
        paths["windows"] = windows
    elif distro := os.environ.get("WSL_DISTRO_NAME"):
        unc_root = f"\\\\wsl.localhost\\{distro}"
        paths["windows"] = unc_root + target.replace("/", "\\")
    return paths

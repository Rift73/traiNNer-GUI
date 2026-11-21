"""Select and verify the optional Windows-backed Studio project."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

PROJECT_DIR_ENV = "TRAINNER_GUI_PROJECT_DIR"
PROJECT_ID_ENV = "TRAINNER_GUI_PROJECT_ID"
PROJECT_MARKER = ".trainner-gui-project.json"
PROJECT_STATE_DIR = "projects"


class ProjectUnavailable(ValueError):
    """The selected project or its Windows mount is unavailable."""


def _filesystem(fd: int) -> tuple[str, str]:
    mount_id = next(
        line.split()[1]
        for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
        if line.startswith("mnt_id:")
    )
    # Read the live mount namespace. WSL also uses 9p for its own system
    # mounts, so only accept a 9p export explicitly identified as DrvFS.
    with Path("/proc/self/mountinfo").open(errors="surrogateescape") as mounts:
        for line in mounts:
            mount, _, filesystem = line.partition(" - ")
            if mount.split()[0] == mount_id:
                kind, _, options = filesystem.split(maxsplit=2)
                return kind, options.strip()
    raise OSError("Directory is no longer on a mounted filesystem.")


def _directory_fd(directory: Path) -> int:
    return os.open(directory, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)


def require_ext4(directory: Path) -> None:
    """Validate the filesystem of an existing or as yet uncreated state path."""
    if directory.resolve() != directory:
        raise ValueError(f"Studio runtime directory is redirected: {directory}.")
    parent = directory
    while not parent.exists():
        parent = parent.parent
    fd = _directory_fd(parent)
    try:
        if _filesystem(fd)[0] != "ext4":
            raise ValueError(f"Studio project runtime requires an ext4 directory: {directory}.")
    finally:
        os.close(fd)


def _read_marker(directory: Path) -> tuple[str, int]:
    try:
        if directory.resolve(strict=True) != directory:
            raise ProjectUnavailable(f"The Windows project directory is redirected: {directory}.")
        root = _directory_fd(directory)
        try:
            kind, options = _filesystem(root)
            drvfs_export = any(
                option.partition(";")[0] == "aname=drvfs" for option in options.split(",")
            )
            if kind != "drvfs" and (kind != "9p" or not drvfs_export):
                raise ProjectUnavailable(f"The Windows project is not on a mounted WSL Windows drive: {directory}.")
            device = os.fstat(root).st_dev
            marker = os.open(PROJECT_MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root)
            with os.fdopen(marker, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_dev != device:
                    raise ProjectUnavailable(f"{PROJECT_MARKER} must be a regular file on the project drive.")
                raw = stream.read(4097)
        finally:
            os.close(root)
    except (OSError, RuntimeError) as error:
        raise ProjectUnavailable(
            f"Cannot open the Windows project and its {PROJECT_MARKER} marker at {directory}: {error}"
        ) from error
    try:
        if len(raw) > 4096:
            raise ValueError("Project marker exceeds 4 KiB.")
        data = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Expected project marker version 1.")
        project_id = str(UUID(data["project_id"]))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ProjectUnavailable(
            f"Invalid {PROJECT_MARKER} at {directory}. Expected version 1 and a project_id UUID."
        ) from error
    return project_id, device


@dataclass(frozen=True, slots=True)
class WindowsProject:
    root: Path
    id: str

    @classmethod
    def from_environment(cls) -> WindowsProject | None:
        configured = os.environ.get(PROJECT_DIR_ENV)
        expected = os.environ.get(PROJECT_ID_ENV)
        if not configured:
            if expected:
                raise ValueError(f"{PROJECT_ID_ENV} also requires {PROJECT_DIR_ENV}.")
            return None
        directory = Path(configured)
        if directory.anchor != "/" or ".." in directory.parts:
            raise ValueError(f"{PROJECT_DIR_ENV} must be an absolute Linux path on a mounted Windows drive.")
        if expected:
            try:
                expected = str(UUID(expected))
            except ValueError as error:
                raise ValueError(f"{PROJECT_ID_ENV} must be a UUID.") from error
        project_id, _ = _read_marker(directory)
        if expected and project_id != expected:
            raise ProjectUnavailable(f"The Windows project marker at {directory} does not match {PROJECT_ID_ENV}.")
        return cls(directory, project_id)

    def require_directory(self, directory: Path) -> None:
        """Recheck the drive, marker and any existing managed directory components."""
        project_id, device = _read_marker(self.root)
        if project_id != self.id:
            raise ProjectUnavailable(f"The Windows project marker changed at {self.root}.")
        path = self.root
        for part in directory.relative_to(self.root).parts:
            path /= part
            try:
                info = path.lstat()
            except FileNotFoundError:
                break
            if not stat.S_ISDIR(info.st_mode) or info.st_dev != device:
                raise ProjectUnavailable(f"The project directory is redirected or is not a folder: {path}.")

    def environment(self) -> dict[str, str]:
        return {PROJECT_DIR_ENV: str(self.root), PROJECT_ID_ENV: self.id}

    def status(self) -> dict[str, str]:
        return {"projectDir": str(self.root), "projectId": self.id}


@dataclass(frozen=True, slots=True)
class Workspace:
    state: Path
    project: WindowsProject | None

    @classmethod
    def from_environment(cls, base_state: Path) -> Workspace:
        state = base_state.expanduser().resolve()
        project = WindowsProject.from_environment()
        if project is not None:
            state = state / PROJECT_STATE_DIR / project.id
            require_ext4(state)
        return cls(state, project)

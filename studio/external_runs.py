"""Read-only experiment archives with a local, metadata-only index."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from . import run_process as process
from .paths import local_path

ARCHIVE_PREFIX = "archive-"
ARCHIVE_ID = re.compile(r"archive-[0-9a-f]{64}")
ARCHIVE_RUN = re.compile(r"(archive-[0-9a-f]{64})\.([0-9a-f]{2,510})")
REFRESH_SECONDS = 30


def _valid_name(name: str) -> bool:
    if not name or name.startswith(".") or "/" in name or "\0" in name:
        return False
    try:
        return len(name.encode()) <= 255
    except UnicodeEncodeError:
        return False


def _identity(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


def _experiment_files(
    folder: Path, *, include_recipe: bool
) -> tuple[Path | None, Path | None, int] | None:
    # Only top-level entries matter. Model and training-state payloads are
    # never opened by discovery, and their directories are never traversed.
    log: tuple[int, str] | None = None
    recipe: tuple[int, str] | None = None
    outputs = False
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name in {"models", "training_states"} and entry.is_dir(
                follow_symlinks=False
            ):
                outputs = True
                continue
            is_log = entry.name.startswith("train_") and entry.name.endswith(".log")
            is_recipe = include_recipe and entry.name.lower().endswith((".yaml", ".yml", ".toml"))
            if (is_log or is_recipe) and entry.is_file(follow_symlinks=False):
                try:
                    key = (entry.stat(follow_symlinks=False).st_mtime_ns, entry.name)
                except FileNotFoundError:
                    continue
                if is_log and (log is None or key > log):
                    log = key
                if is_recipe and (recipe is None or key > recipe):
                    recipe = key
    if log is None and not outputs:
        return None
    updated = max(log[0] if log else 0, recipe[0] if recipe else 0)
    return (
        folder / log[1] if log else None,
        folder / recipe[1] if recipe else None,
        updated,
    )


class ExternalRuns:
    def __init__(self, checkout: Path, state: Path) -> None:
        self.checkout = checkout.resolve()
        self.state = state.resolve()
        self.directory = self.state / "archives"
        self.registry = self.directory / "registry.json"
        self._lock = Lock()
        self._refreshing: set[str] = set()

    def _roots(self) -> dict[str, dict[str, Any]]:
        record = process.read_record(self.registry)
        return record["roots"] if record is not None else {}

    @contextmanager
    def _edit(self) -> Iterator[dict[str, dict[str, Any]]]:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.resolve() != self.directory:
            raise ValueError("The archive registry directory is redirected.")
        with (self.directory / "registry.lock").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield self._roots()

    def _index_path(self, archive_id: str) -> Path:
        return self.directory / f"{archive_id}.json"

    def _refresh_path(self, record: dict[str, Any]) -> Path:
        return self.directory / f"{record['id']}.{record['generation']}.lock"

    @staticmethod
    def _public(
        record: dict[str, Any], snapshot: dict[str, Any] | None
    ) -> dict[str, Any]:
        result = {key: record[key] for key in ("id", "path", "addedAt")}
        if snapshot is None:
            return {**result, "status": "checking"}
        result["checkedAt"] = process.timestamp(snapshot["checked"])
        if error := snapshot.get("error"):
            return {**result, "status": "unavailable", "error": error}
        return {**result, "status": "available", "runCount": len(snapshot["runs"])}

    @staticmethod
    def _scan(root: Path) -> list[dict[str, Any]]:
        if root.resolve(strict=True) != root:
            raise OSError(
                "The registered archive directory now resolves to a different path."
            )
        runs = []
        with os.scandir(root) as entries:
            for entry in entries:
                if not _valid_name(entry.name):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    observed = _experiment_files(Path(entry.path), include_recipe=True)
                except OSError:
                    # An individual experiment disappearing or becoming
                    # unreadable does not hide the rest of the archive.
                    continue
                if observed:
                    runs.append({"name": entry.name, "modifiedNs": observed[2]})
        return runs

    def _refresh(self, record: dict[str, Any]) -> None:
        archive_id = record["id"]
        guard = self._refresh_path(record)
        with guard.open("ab") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if self._roots().get(archive_id) != record:
                return
            try:
                runs = self._scan(Path(record["path"]))
                snapshot: dict[str, Any] = {"runs": runs}
            except (OSError, RuntimeError) as error:
                snapshot = {"runs": [], "error": str(error)}
            snapshot.update(generation=record["generation"], checked=time.time())
            with self._edit() as roots:
                if roots.get(archive_id) == record:
                    process.write_record(self._index_path(archive_id), snapshot)

    def _background_refresh(self, record: dict[str, Any]) -> None:
        try:
            self._refresh(record)
        finally:
            with self._lock:
                self._refreshing.discard(record["generation"])

    def _snapshot(
        self, record: dict[str, Any], *, refresh: bool = False
    ) -> dict[str, Any] | None:
        snapshot = process.read_record(self._index_path(record["id"]))
        if snapshot is not None and snapshot.get("generation") != record["generation"]:
            snapshot = None
        if refresh and (
            snapshot is None
            or not 0 <= time.time() - snapshot["checked"] < REFRESH_SECONDS
        ):
            generation = record["generation"]
            with self._lock:
                if generation not in self._refreshing:
                    self._refreshing.add(generation)
                    Thread(
                        target=self._background_refresh, args=(record,), daemon=True
                    ).start()
        return snapshot

    def list_archives(self) -> list[dict[str, Any]]:
        roots = sorted(self._roots().values(), key=lambda record: record["path"])
        return [
            self._public(record, self._snapshot(record, refresh=True))
            for record in roots
        ]

    def add_archive(self, value: str | Path) -> dict[str, Any]:
        supplied = str(value) if isinstance(value, Path) else value
        requested = local_path(supplied, self.checkout)
        # A previously registered canonical path remains removable and
        # idempotent even while its storage is disconnected.
        for existing in self._roots().values():
            if existing["path"] == str(requested):
                return self._public(existing, self._snapshot(existing, refresh=True))
        try:
            canonical = requested.resolve(strict=True)
            info = canonical.stat()
        except RuntimeError as error:
            raise ValueError("The archive path contains a symlink loop.") from error
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Select an existing experiments directory.")
        if canonical.is_relative_to(self.state) or self.state.is_relative_to(canonical):
            raise ValueError(
                "Choose an archive separate from Studio's state directory."
            )
        identity = (info.st_dev, info.st_ino)
        legacy = self.checkout / "experiments"
        try:
            is_legacy = (
                legacy.resolve(strict=True) == legacy and _identity(legacy) == identity
            )
        except (OSError, RuntimeError):
            is_legacy = False
        if canonical == legacy or is_legacy:
            raise ValueError("The checkout's experiments directory is already listed.")

        archive_id = ARCHIVE_PREFIX + hashlib.sha256(os.fsencode(canonical)).hexdigest()
        with self._edit() as roots:
            if existing := roots.get(archive_id):
                if existing["path"] != str(canonical):
                    raise ValueError(
                        "The archive ID collides with a different registered path."
                    )
                return self._public(existing, self._snapshot(existing, refresh=True))
            for existing in roots.values():
                if tuple(existing["identity"]) != identity:
                    continue
                # A case or mount alias can resolve to the same directory.
                # Verify the registered path still denotes it before reuse.
                try:
                    same = _identity(Path(existing["path"])) == identity
                except OSError:
                    same = False
                if same:
                    return self._public(
                        existing, self._snapshot(existing, refresh=True)
                    )
            record = {
                "id": archive_id,
                "path": str(canonical),
                "addedAt": process.timestamp(),
                "identity": list(identity),
                "generation": uuid4().hex,
            }
            roots[archive_id] = record
            process.write_record(self.registry, {"roots": roots})
        # Seed a small usable index for CLI callers that exit after add.
        # Archive I/O never runs under the registry edit lock.
        self._refresh(record)
        return self._public(record, self._snapshot(record))

    def remove_archive(self, archive_id: str) -> bool:
        if not isinstance(archive_id, str) or ARCHIVE_ID.fullmatch(archive_id) is None:
            raise ValueError("Provide a registered archive ID.")
        if not self.registry.is_file():
            return False
        with self._edit() as roots:
            if (record := roots.pop(archive_id, None)) is None:
                return False
            process.write_record(self.registry, {"roots": roots})
            self._index_path(archive_id).unlink(missing_ok=True)
            self._refresh_path(record).unlink(missing_ok=True)
        return True

    def runs(self) -> Iterator[dict[str, Any]]:
        for record in self._roots().values():
            snapshot = self._snapshot(record, refresh=True)
            if snapshot is None:
                continue
            for cached in snapshot["runs"]:
                name = cached["name"]
                result = {
                    "id": record["id"] + "." + name.encode().hex(),
                    "name": name,
                    "status": "imported",
                    "source": "archive",
                    "sourceRootId": record["id"],
                    "sourceRoot": record["path"],
                }
                if modified := cached["modifiedNs"]:
                    result["updatedAt"] = process.timestamp(modified / 1_000_000_000)
                yield result

    def run(
        self, run_id: str, *, include_recipe: bool
    ) -> tuple[dict[str, Any], Path, Path | None, Path | None]:
        match = ARCHIVE_RUN.fullmatch(run_id)
        if match is None:
            raise FileNotFoundError("Archive run was not found.")
        try:
            name = bytes.fromhex(match[2]).decode()
        except (ValueError, UnicodeError) as error:
            raise FileNotFoundError("Archive run was not found.") from error
        if not _valid_name(name):
            raise FileNotFoundError("Archive run was not found.")
        record = self._roots().get(match[1])
        if record is None:
            raise FileNotFoundError("The archive root is no longer registered.")
        experiment = Path(record["path"]) / name
        try:
            if experiment.resolve(strict=True) != experiment or not experiment.is_dir():
                raise FileNotFoundError(
                    "The archive experiment directory is unavailable."
                )
            observed = _experiment_files(experiment, include_recipe=include_recipe)
        except (OSError, RuntimeError) as error:
            raise FileNotFoundError(
                "The archive experiment directory is unavailable."
            ) from error
        if observed is None:
            raise FileNotFoundError("Archive run was not found.")
        return record, experiment, observed[0], observed[1]

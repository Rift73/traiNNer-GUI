"""Serialize managed Studio access with a Windows release activation.

The activation CLI runs from the staged release using only the standard library.
It takes the same distro-local directory flock as managed Studio commands.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import pwd
import re
import select
import signal
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

MANAGED_ROOT = Path("/opt/trainner-gui")
MANAGED_RELEASES = Path("/opt/trainner-gui-releases")
READY = "TRAINNER_GUI_UPDATE_GATE_READY"
DEFERRED = 75
_STATE_REGISTRY = ".managed-state-bases"
_RELEASE_ID = re.compile(r"[0-9a-f]{64}")
_QUEUE_WAITING = {"held", "pending", "starting", "running", "stopping"}
_QUEUE_FINISHED = {"cancelled", "completed", "failed", "stopped"}


class UpdateDeferred(RuntimeError):
    """The runtime is in use or activation cannot prove it is idle."""


def _managed_location(root: Path) -> bool:
    return root == MANAGED_ROOT or (
        root.parent == MANAGED_RELEASES and _RELEASE_ID.fullmatch(root.name) is not None
    )


def is_managed_runtime(root: Path) -> bool:
    # An arbitrary checkout can use Studio without registering any state or
    # participating in the managed activation protocol.
    if not _managed_location(root):
        return False
    try:
        directory = root.lstat()
        marker = (root / ".managed-ready").lstat()
    except FileNotFoundError:
        return False
    return (
        stat.S_ISDIR(directory.st_mode)
        and directory.st_uid == 0
        and not directory.st_mode & 0o022
        and root.resolve() == root
        and stat.S_ISREG(marker.st_mode)
        and marker.st_uid == 0
        and not marker.st_mode & 0o022
    )


def _require_trusted_directory(path: Path, info: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
        or path.resolve(strict=True) != path
    ):
        raise UpdateDeferred(f"Cannot verify the managed directory at {path}.")


@contextmanager
def _release_lock(*, exclusive: bool) -> Iterator[None]:
    # The original tree's Studio directory survives side-by-side releases.
    # Provision separately locks the checkout directory during staging.
    anchor = MANAGED_ROOT / "studio"
    fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _require_trusted_directory(anchor, os.fstat(fd))
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = (
                "Studio is starting or handling an operation. Retry the update when it is idle."
                if exclusive
                else "A managed release update is in progress. Retry when it finishes."
            )
            raise UpdateDeferred(message) from error
        yield
    finally:
        os.close(fd)


@contextmanager
def studio_admission(root: Path) -> Iterator[bool]:
    """Exclude release activation during one managed Studio operation."""
    managed = is_managed_runtime(root)
    if managed:
        with _release_lock(exclusive=False):
            yield managed
    else:
        yield managed


def _state_base(state_home: Path, identity: Path) -> Path:
    digest = hashlib.sha256(os.fsencode(identity)).hexdigest()[:12]
    return state_home / "traiNNer-GUI" / digest


def default_state_path(root: Path, state_home: Path) -> Path:
    return _state_base(state_home, MANAGED_ROOT if is_managed_runtime(root) else root)


def _canonical_base() -> Path:
    home = Path(pwd.getpwuid(os.geteuid()).pw_dir) / ".local/state"
    return _state_base(home, MANAGED_ROOT).resolve()


def _record_name(base: Path) -> str:
    return hashlib.sha256(os.fsencode(base)).hexdigest()


def remember_state_base(base: Path) -> None:
    """Register custom state before letting Studio create any work there."""
    canonical = _canonical_base()
    if base == canonical:
        return
    registry = canonical / _STATE_REGISTRY
    registry.mkdir(mode=0o700, parents=True, exist_ok=True)
    if registry.resolve() != registry:
        raise UpdateDeferred(f"The managed workspace registry is redirected at {registry}.")
    base.mkdir(parents=True, exist_ok=True)
    record = registry / _record_name(base)
    # Use Studio's atomic writer for competing registrations for the same base.
    from .recipes import atomic_text

    data = json.dumps(str(base)) + "\n"
    with _regular_file(record) as stream:
        previous = stream.read() if stream is not None else None
    if previous is None:
        atomic_text(record, data)
    elif previous != data.encode():
        raise UpdateDeferred(f"Cannot verify the managed workspace registration at {record}.")
    fd = os.open(registry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _regular_file(path: Path, *, create: bool = False) -> Iterator[BinaryIO | None]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags | (os.O_CREAT if create else 0), 0o600)
    except FileNotFoundError:
        if create:
            raise
        yield None
        return
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise UpdateDeferred(f"Expected a regular Studio state file at {path}.")
        yield stream


def _directory(path: Path, *, required: bool = False) -> bool:
    if path.resolve() != path:
        raise UpdateDeferred(f"Studio state is redirected at {path}.")
    try:
        info = path.lstat()
    except FileNotFoundError:
        if required:
            raise UpdateDeferred(f"Cannot find the registered Studio state directory at {path}.") from None
        return False
    if not stat.S_ISDIR(info.st_mode):
        raise UpdateDeferred(f"Expected a Studio state directory at {path}.")
    return True


def _try_exclusive(stream: BinaryIO) -> bool:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _reserve(locks: ExitStack, path: Path, *, create: bool = False) -> bool:
    stream = locks.enter_context(_regular_file(path, create=create))
    if stream is None:
        return False
    if not _try_exclusive(stream):
        raise UpdateDeferred(f"Studio is still using {path}. Let the operation finish and retry.")
    return True


def _check_queue(path: Path) -> None:
    with _regular_file(path) as stream:
        if stream is None:
            return
        queue = json.load(stream)
    if (
        not isinstance(queue, dict)
        or type(queue.get("paused")) is not bool
        or type(queue.get("nextOrder")) is not int
        or queue["nextOrder"] < 0
        or not isinstance(queue.get("entries"), dict)
        or "activeId" not in queue
    ):
        raise UpdateDeferred(f"Cannot verify the persisted Studio queue at {path}.")
    active = queue["activeId"]
    if active is not None:
        raise UpdateDeferred(f"The persisted Studio queue at {path} still has a claimed entry.")
    for entry_id, entry in queue["entries"].items():
        if not isinstance(entry, dict) or entry.get("id") != entry_id:
            raise UpdateDeferred(f"Cannot verify entry {entry_id!r} in the persisted Studio queue at {path}.")
        status = entry.get("status")
        if not isinstance(status, str) or status not in _QUEUE_WAITING | _QUEUE_FINISHED:
            raise UpdateDeferred(f"Unknown queue entry status in the persisted Studio queue at {path}.")
        if status in _QUEUE_WAITING:
            raise UpdateDeferred(f"The persisted Studio queue at {path} still has a {status} entry.")


def _registered_bases(canonical: Path) -> Iterator[Path]:
    registry = canonical / _STATE_REGISTRY
    if not _directory(registry):
        return
    for record in registry.iterdir():
        if record.name.startswith(".writing-"):
            continue  # A registration has no effect before its atomic rename.
        with _regular_file(record) as stream:
            if stream is None:
                raise UpdateDeferred(f"The managed workspace registration disappeared at {record}.")
            name = json.load(stream)
        if not isinstance(name, str):
            raise UpdateDeferred(f"Invalid managed workspace registration at {record}.")
        base = Path(name)
        if not base.is_absolute() or _record_name(base) != record.name:
            raise UpdateDeferred(f"Invalid managed workspace registration at {record}.")
        _directory(base, required=True)
        yield base


def _project_id(name: str) -> bool:
    try:
        return str(UUID(name)) == name
    except ValueError:
        return False


def _workspaces(base: Path) -> Iterator[Path]:
    if not _directory(base):
        return
    yield base
    projects = base / "projects"
    if _directory(projects):
        for state in projects.iterdir():
            if not _project_id(state.name):
                raise UpdateDeferred(f"Invalid Studio project state directory at {state}.")
            _directory(state, required=True)
            yield state


def _start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _open_lock_paths(pid: int) -> Iterator[tuple[Path, Path]]:
    proc = Path(f"/proc/{pid}")
    for fd in (proc / "fd").iterdir():
        try:
            target = Path(os.readlink(fd))
        except FileNotFoundError:
            continue  # A request socket or the service itself may have closed.
        if target.is_absolute() and target.name == "service-live.lock":
            yield target, proc / "fdinfo" / fd.name


def _owns_live_lock(pid: int, state: Path) -> bool:
    """Match the service's open lock path, inode and exclusive flock."""
    path = state.resolve() / "service-live.lock"
    info = path.stat()
    inode = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    expected = ["ADVISORY", "WRITE", str(pid), inode, "0", "EOF"]
    for target, fdinfo in _open_lock_paths(pid):
        if target != path:
            continue
        try:
            with fdinfo.open() as details:
                for line in details:
                    fields = line.split()
                    # WSL DrvFS represents flock as a whole-file POSIX lock.
                    if fields[:1] == ["lock:"] and len(fields) == 9 and fields[2] in ("FLOCK", "POSIX") and fields[3:] == expected:
                        return True
        except FileNotFoundError:
            continue
    return False


def _serving(args: list[bytes]) -> bool:
    for index, pair in enumerate(itertools.pairwise(args)):
        if pair in {(b"-m", b"studio"), (b"-m", b"studio.__main__")}:
            command = args[index + 2:]
            if command[:1] == [b"--state-dir"]:
                command = command[2:]
            elif command and command[0].startswith(b"--state-dir="):
                command = command[1:]
            return command[:1] == [b"serve"]
    return False


def _running_services(root: Path) -> dict[Path, int]:
    """Find live legacy workspaces, including alternate and project state dirs."""
    services = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.geteuid():
                continue
            args = (process / "cmdline").read_bytes().split(b"\0")
            modules = set(itertools.pairwise(args))
            training = any(arg == b"-opt" or arg.startswith(b"-opt=") for arg in args) and (
                (b"-m", b"train") in modules
                or any(arg.rsplit(b"/", 1)[-1] == b"train.py" for arg in args)
            )
            if training or any(
                (b"-m", module) in modules for module in (b"studio.run_process", b"studio.analysis_jobs")
            ):
                raise UpdateDeferred(f"A trainer or analysis worker is still running (PID {process.name}).")
            if not _serving(args) or Path(os.readlink(process / "cwd")) != root:
                continue
            pid = int(process.name)
            states = {target.parent for target, _ in _open_lock_paths(pid)}
            if len(states) != 1:
                raise UpdateDeferred(f"Cannot verify the workspace of Studio process {pid}. Retry after startup or shutdown.")
            state = states.pop()
            if state in services:
                raise UpdateDeferred(f"Multiple Studio processes claim the workspace at {state}.")
            services[state] = pid
        except (FileNotFoundError, ProcessLookupError):
            continue
    return services


def _service_pidfd(locks: ExitStack, state: Path, root: Path, expected_pid: int | None) -> int:
    with _regular_file(state / "service.json") as stream:
        identity = json.load(stream) if stream is not None else None
    error = f"Cannot verify the managed Studio service that owns {state / 'service-live.lock'}."
    if not isinstance(identity, dict):
        raise UpdateDeferred(error)
    pid = identity.get("pid")
    if (
        type(pid) is not int or pid <= 1 or pid != expected_pid
        or identity.get("boot") != _boot_id() or identity.get("root") != str(root)
    ):
        raise UpdateDeferred(error)
    fd = os.pidfd_open(pid)
    locks.callback(os.close, fd)
    if _start_ticks(pid) != identity.get("startTicks") or not _owns_live_lock(pid, state):
        raise UpdateDeferred(error)
    return fd


def _reserve_workspace(
    locks: ExitStack, state: Path, root: Path, pid: int | None, *, activating: bool
) -> int | None:
    launch_path = state / "service-launch.lock"
    launch_reserved = _reserve(locks, launch_path)
    live = locks.enter_context(_regular_file(state / "service-live.lock"))
    service = None
    if live is not None and not _try_exclusive(live):
        service = _service_pidfd(locks, state, root, pid)
        if activating and not launch_reserved:
            _reserve(locks, launch_path, create=True)
    elif pid is not None:
        raise UpdateDeferred(f"The managed Studio service changed at {state}. Retry the update.")

    # Preview1's dispatcher and analysis API use these same locks. Reserve them
    # even in a live workspace that has not yet created its first job.
    create = activating and service is not None
    for folder in ("queue", "analysis"):
        directory = state / folder
        if not _directory(directory):
            if not create:
                continue
            directory.mkdir(mode=0o700, exist_ok=True)
        _reserve(locks, directory / "control.lock", create=create)
        if folder == "queue":
            _check_queue(directory / "queue.json")
        else:
            _reserve(locks, directory / "worker.lock", create=create)

    if service is None:
        _reserve(locks, state / "tailscale-worker.lock")
    else:
        _reserve(locks, state / "tailscale-control.lock", create=create)
    return service


def _stop_idle_services(locks: ExitStack, services: dict[Path, int]) -> None:
    poll = select.poll()
    waiting = set(services.values())
    for fd in waiting:
        poll.register(fd, select.POLLIN)
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The verified instance closed on its own.
    deadline = time.monotonic() + 10
    while waiting:
        ready = poll.poll(max(0, round((deadline - time.monotonic()) * 1000)))
        if not ready:
            raise UpdateDeferred("An idle Studio service is still closing. Retry the update.")
        for fd, _ in ready:
            waiting.remove(fd)
            poll.unregister(fd)
    phone_deadline = time.monotonic() + 5
    for state in services:
        _reserve(locks, state / "service-live.lock")
        # The phone worker follows its service's pidfd and can clean up routes
        # only after the service exits. Wait briefly without sending it a signal.
        phone = locks.enter_context(_regular_file(state / "tailscale-worker.lock"))
        if phone is not None:
            while not _try_exclusive(phone):
                if time.monotonic() >= phone_deadline:
                    raise UpdateDeferred(f"The Studio phone gateway is still closing at {state}.")
                time.sleep(0.1)


def _idle_services(
    locks: ExitStack, old_root: Path, state_dirs: list[Path], *, activating: bool
) -> dict[Path, int]:
    if not is_managed_runtime(old_root):
        raise UpdateDeferred(f"Cannot identify a managed runtime at {old_root}.")
    _require_trusted_directory(old_root, old_root.lstat())
    with _regular_file(old_root / ".managed-ready") as stream:
        if stream is None:
            raise UpdateDeferred(f"The managed runtime is no longer ready at {old_root}.")
        account = pwd.getpwuid(os.geteuid())
        ready = stream.read().decode().split()
        if (
            len(ready) != 4
            or not _RELEASE_ID.fullmatch(ready[0])
            or ready[1:3] != [account.pw_name, str(account.pw_uid)]
            or ready[3] not in {"cpu", "cuda"}
        ):
            raise UpdateDeferred(f"Run the activation gate as the managed trainer for {old_root}.")

    cache = old_root / ".cache"
    _directory(cache, required=True)
    # Supervisors and trainers inherit this lease. The gate holds it even
    # after the old service has exited and while Windows commits its selection.
    _reserve(locks, cache / "studio-trainer.lock", create=activating)
    discovered = _running_services(old_root)
    canonical = _canonical_base()
    bases = {canonical, *_registered_bases(canonical)}
    if configured := os.environ.get("TRAINNER_GUI_STATE_DIR"):
        state_dirs = [*state_dirs, Path(configured)]
    if xdg_home := os.environ.get("XDG_STATE_HOME"):
        state_dirs = [*state_dirs, _state_base(Path(xdg_home), MANAGED_ROOT)]
    bases.update(base.expanduser().resolve() for base in state_dirs)
    bases.update(
        state.parent.parent for state in discovered
        if state.parent.name == "projects" and _project_id(state.name)
    )
    workspaces = set(discovered)
    for base in bases:
        workspaces.update(_workspaces(base))
    services = {}
    for state in sorted(workspaces):
        _directory(state, required=True)
        service = _reserve_workspace(locks, state, old_root, discovered.get(state), activating=activating)
        if service is not None:
            services[state] = service
    if _running_services(old_root) != discovered:
        raise UpdateDeferred("A managed Studio service changed during the idle check. Retry the update.")
    return services


def _require_activation_pipe() -> None:
    if select.select([sys.stdin], [], [], 0)[0]:
        raise UpdateDeferred("The Windows launcher did not leave the activation pipe open.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hold the managed Studio activation gate until stdin closes.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("probe", "activate"):
        options = commands.add_parser(command)
        options.add_argument("--old-root", type=Path, required=True)
        options.add_argument("--state-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    activating = args.command == "activate"
    try:
        if os.geteuid() == 0:
            raise UpdateDeferred("Run the managed Studio activation gate as the trainer account.")
        if activating:
            _require_activation_pipe()
        with _release_lock(exclusive=True), ExitStack() as locks:
            services = _idle_services(locks, args.old_root, args.state_dir, activating=activating)
            if not activating:
                return 0
            _require_activation_pipe()
            _stop_idle_services(locks, services)
            _require_activation_pipe()
            print(READY, flush=True)
            if sys.stdin.buffer.read(1):
                raise UpdateDeferred("Unexpected input on the Windows activation pipe.")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"traiNNer-GUI update deferred: {error}", file=sys.stderr)
        return DEFERRED
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Detached ownership of one CLI attempt.

A Linux flock survives server exit because the supervisor and trainer each
inherit its original open file description. A stop request is scoped to an
attempt directory; only the supervisor signals its own unreaped child.
"""

from __future__ import annotations

import fcntl
import itertools
import json
import logging
import mmap
import os
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Any

from .recipes import atomic_text
from .run_checkpoints import stage_import, verify_phase_output
from .workspace import PROJECT_DIR_ENV, PROJECT_ID_ENV, WindowsProject, require_ext4

LOG = logging.getLogger(__name__)
PROGRESS_ENV = "TRAINNER_STUDIO_PROGRESS"
_PROGRESS_FILE = "progress.bin"
_PROGRESS = struct.Struct("<q")


def timestamp(value: float | None = None) -> str:
    return (
        datetime.fromtimestamp(time.time() if value is None else value, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def write_record(path: Path, record: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(record, ensure_ascii=False) + "\n")


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def lock_held(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        pass
    return False


def _remove_progress(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        LOG.warning("Could not remove the live iteration counter at %s. %s", path, error)


def _create_progress(control: Path) -> Path | None:
    path = control / _PROGRESS_FILE
    created = False
    try:
        require_ext4(control)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created = True
        try:
            if os.write(fd, _PROGRESS.pack(-1)) != _PROGRESS.size:
                raise OSError("Incomplete live iteration counter initialization.")
        finally:
            os.close(fd)
    except (OSError, ValueError) as error:
        if created:
            _remove_progress(path)
        LOG.warning("Live iteration is unavailable for attempt %s. %s", control.name, error)
        return None
    return path


def read_progress(control: Path) -> int | None:
    """Sample the current attempt's fixed-size counter without retaining a mapping."""
    try:
        fd = os.open(control / _PROGRESS_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != _PROGRESS.size:
                return None
            # The writer never truncates this inode. If the supervisor unlinks it
            # during this read, the open descriptor still refers to the same bytes.
            with mmap.mmap(fd, _PROGRESS.size, access=mmap.ACCESS_READ) as counter:
                (iteration,) = _PROGRESS.unpack(counter)
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None
    return iteration if iteration >= 0 else None


def cli_trainer_running() -> bool:
    """Observe ordinary WSL training commands without taking ownership of them."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if b"--help" in args or b"-h" in args:
            continue
        if not any(arg == b"-opt" or arg.startswith(b"-opt=") for arg in args):
            continue
        if any(arg.rsplit(b"/", 1)[-1] == b"train.py" for arg in args):
            return True
        if any(
            first == b"-m" and second == b"train"
            for first, second in itertools.pairwise(args)
        ):
            return True
    return False


def invocation(root: Path, run: Path, attempt: Path, seed: int) -> dict[str, Any]:
    return {
        "argv": [
            sys.executable,
            "-B",
            "-u",
            str(root / "train.py"),
            "-opt",
            str(attempt / "options.yml"),
            "--no-auto-recover",
            "--manual_seed",
            str(seed),
        ],
        "cwd": str(root),
        "env": {
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRAINNER_MANAGED_RUN_ROOT": str(run),
            "TRAINNER_STUDIO_ATTEMPT": str(attempt),
        },
        "unsetEnv": [
            "TRAINNER_NONFINITE_REPORT",
            "TRAINNER_RECOVERY_RESUME_STATE",
            "TRAINNER_RECOVERY_MANUAL_SEED",
        ],
        "stdin": "/dev/null",
    }


def launch(
    attempt: Path,
    lease_fd: int,
    experiment: Path,
    command: dict[str, Any],
    resource_fd: int,
    *,
    control: Path,
    state: Path,
) -> None:
    env = {**os.environ, **command["env"]}
    for key in command["unsetEnv"]:
        env.pop(key, None)
    with (attempt / "console.log").open("ab") as log:
        supervisor = subprocess.Popen(
            [
                command["argv"][0],
                "-B",
                "-u",
                "-m",
                "studio.run_process",
                str(attempt),
                str(lease_fd),
                str(experiment),
                str(resource_fd),
                str(control),
                str(state),
            ],
            cwd=command["cwd"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lease_fd, resource_fd),
        )
    # Reap the detached supervisor if this server is still alive at exit.
    Thread(target=supervisor.wait, daemon=True).start()


def _run_trainer(
    control: Path, leases: tuple[int, ...], command: dict[str, Any], progress: Path | None
) -> tuple[int, str | None]:
    # Ensure waitid can pin the child's identity until its group is drained.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    env = os.environ.copy()
    env.pop(PROGRESS_ENV, None)
    if progress is not None:
        env[PROGRESS_ENV] = str(progress)
    trainer = subprocess.Popen(
        command["argv"],
        cwd=command["cwd"],
        env=env,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=leases,
    )
    (control / "launched").touch()

    stop_signal = None
    while (
        os.waitid(os.P_PID, trainer.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
    ):
        if stop_signal != "forceStopped" and (control / "force-stop").is_file():
            # The unreaped leader pins this attempt's private process group, including loaders.
            os.killpg(trainer.pid, signal.SIGKILL)
            stop_signal = "forceStopped"
        elif stop_signal is None and (control / "stop").is_file():
            os.kill(trainer.pid, signal.SIGINT)
            stop_signal = "interrupted"
        time.sleep(0.25)

    # The trainer has now exited. Clean up only its private group, with its
    # unreaped leader still pinning the group ID. Never signal loaders during
    # a graceful checkpoint save.
    try:
        os.killpg(trainer.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return trainer.wait(), stop_signal


def _supervise(
    attempt: Path, leases: tuple[int, ...], experiment: Path, control: Path, state: Path
) -> None:
    from .run_observe import engine_log, tail

    project = None
    if control != attempt:
        project = WindowsProject(Path(os.environ[PROJECT_DIR_ENV]), os.environ[PROJECT_ID_ENV])
    with (control / "supervisor.lock").open("ab") as owner, ExitStack() as cleanup:
        fcntl.flock(owner, fcntl.LOCK_EX)
        started_ns = time.time_ns()
        result: dict[str, Any] = {}
        try:
            if project is not None:
                project.require_directory(attempt)
            command = read_record(attempt / "invocation.json")
            assert command is not None
            imported = read_record(attempt / "import.json")
            import_target = (
                attempt / "inputs" if imported and imported.get("inputs") else experiment
            )
            if (control / "stop").is_file() or (
                imported and not stage_import(imported["files"], control, import_target, project=project)
            ):
                result["status"] = "stopped"
            elif (control / "stop").is_file():
                result["status"] = "stopped"
            elif command.get("queueEntryId") and cli_trainer_running():
                result.update(
                    status="failed",
                    error="An ordinary WSL trainer started during queue launch. "
                    "Retry after it finishes.",
                )
            else:
                from .provenance import capture_attempt
                capture_attempt(Path(command["cwd"]), state, attempt)
                if project is not None:
                    project.require_directory(experiment)
                progress = _create_progress(control)
                if progress is not None:
                    cleanup.callback(_remove_progress, progress)
                code, stop_signal = _run_trainer(control, leases, command, progress)
                result["returncode"] = code
                if code == 0:
                    outcome = read_record(attempt / "outcome.json")
                    phase = read_record(attempt / "curriculum.json")
                    if outcome is None:
                        # Compatibility with an older trainer finishing during an upgrade.
                        ended = any("End of training. Time consumed:" in line
                                    for line in tail(engine_log(experiment, since=started_ns)))
                        if not ended and stop_signal is None:
                            raise ValueError("Trainer exited without completion or interruption evidence.")
                        result["status"] = "completed" if ended else "stopped"
                    else:
                        reason, step = outcome.get("completionReason"), outcome.get("completedIteration")
                        if type(step) is not int or step < 0 or reason not in {"targetReached", "earlyStopping", "interrupted"}:
                            raise ValueError("Invalid trainer completion receipt.")
                        result.update(completionReason=reason, completedIteration=step)
                        if "earlyStopping" in outcome:
                            result["earlyStopping"] = outcome["earlyStopping"]
                        result["status"] = "stopped" if reason == "interrupted" else "completed"
                        if reason == "earlyStopping":
                            verify_phase_output(experiment, (attempt / "options.yml").read_text(encoding="utf-8"), step)
                            if phase and step < phase["targetStep"]:
                                result["status"] = "stopped"

                    if result["status"] == "completed" and phase:
                        runtime = read_record(attempt / "runtime.json")
                        if (runtime is None or runtime.get("initialStep") != phase["startStep"]
                                or (outcome and outcome["completedIteration"] != phase["targetStep"])):
                            raise ValueError("Trainer did not finish the curriculum phase at its declared clock.")
                        verify_phase_output(
                            experiment, (attempt / "options.yml").read_text(encoding="utf-8"), phase["targetStep"]
                        )
                elif stop_signal == "forceStopped" and code == -signal.SIGKILL:
                    result.update(status="stopped", completionReason="forceStopped")
                elif stop_signal and code in (130, -signal.SIGINT):
                    result["status"] = "stopped"
                else:
                    result.update(
                        status="failed", error=f"Trainer exited with status {code}."
                    )
        except (OSError, ValueError) as error:
            traceback.print_exc()
            result.update(status="failed", error=str(error))
        if project is not None:
            project.require_directory(attempt)
        if result.get("status") == "stopped" and "returncode" not in result and (control / "force-stop").is_file():
            result["completionReason"] = "forceStopped"
        write_record(attempt / "result.json", {**result, "updatedAt": timestamp()})


if __name__ == "__main__":
    attempt_dir, descriptor, experiment_dir, *runtime = sys.argv[1:]
    attempt = Path(attempt_dir)
    resource = runtime[:1]
    control, state = map(Path, runtime[1:]) if len(runtime) == 3 else (attempt, attempt.parents[3])
    _supervise(
        attempt,
        tuple(int(fd) for fd in (descriptor, *resource)),
        Path(experiment_dir),
        control,
        state,
    )

"""Run the phone gateway for the lifetime of one local Studio instance."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

from .recipes import atomic_text
from .update_gate import default_state_path
from .workspace import PROJECT_STATE_DIR


def default_state(root: Path) -> Path:
    home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return default_state_path(root, home)


def _enabled(root: Path, state: Path) -> bool:
    try:
        return (state / "tailscale.preference").read_text().strip() == "on"
    except FileNotFoundError:
        default = default_state(root)
        return state == default or state.parent == default / PROJECT_STATE_DIR


def _report(state: Path, phase: str, **details) -> dict:
    result = {"state": phase, **details}
    atomic_text(state / "tailscale.json", json.dumps(result))
    return result


def status(state: Path) -> dict:
    try:
        result = json.loads((state / "tailscale.json").read_bytes())
        if result["state"] in ("starting", "ready"):
            with (state / "tailscale-worker.lock").open("rb") as worker:
                try:
                    fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return result
                return {"state": "stopped"}
        return result
    except FileNotFoundError:
        return {"state": "stopped"}


def _begin(root: Path, state: Path) -> None:
    if not _enabled(root, state):
        return
    # The child inherits this file description. No interval exists in which
    # a second desktop launch can attach a second gateway to the same state.
    with (state / "tailscale-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # An immediate relaunch may find the previous Studio's gateway
            # still closing its foreground route. Only wait for that handover.
            current = json.loads((state / "service.json").read_bytes())
            if status(state).get("servicePid") in (None, current["pid"]):
                return
            deadline = time.monotonic() + 4
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        return
                    time.sleep(0.1)
        current = json.loads((state / "service.json").read_bytes())
        _report(state, "starting", servicePid=current["pid"])
        try:
            with (state / "tailscale.log").open("ab") as log:
                log.truncate(0)
                # A detached service outlives the Windows console that ran
                # wsl.exe. Use the distro's persistent interop server when present.
                worker_env = os.environ.copy()
                distro_interop = Path("/run/WSL/1_interop")
                if worker_env.get("WSL_INTEROP") and distro_interop.is_socket():
                    worker_env["WSL_INTEROP"] = str(distro_interop)
                subprocess.Popen(
                    [sys.executable, "-m", "studio.tailnet", str(root), str(state), str(lock.fileno())],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    pass_fds=(lock.fileno(),), start_new_session=True, env=worker_env,
                )
        except OSError as exc:
            _report(state, "unavailable", detail=str(exc))


def auto_start(root: Path, state: Path) -> None:
    """Keep an optional phone setup failure from interrupting a healthy desktop launch."""
    try:
        configure(root, state)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).warning("Phone access could not start: %s", exc)


def configure(root: Path, state: Path, mode: str = "auto") -> dict:
    """Automatic launch is nonblocking; explicit 'on' waits for a usable Serve route."""
    state.mkdir(parents=True, exist_ok=True)
    with (state / "tailscale-control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if mode == "off":
            atomic_text(state / "tailscale.preference", "off\n")
            # Worker observes the preference and tears down just its foreground route.
            with (state / "tailscale-worker.lock").open("a") as worker:
                deadline = time.monotonic() + 18
                while True:
                    try:
                        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        previous = status(state)
                        if previous.get("state") == "cleanup-failed":
                            return previous
                        return _report(state, "off")
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("The phone gateway is still stopping; check tailscale status.") from None
                        time.sleep(0.15)
        if mode == "on":
            atomic_text(state / "tailscale.preference", "on\n")
        _begin(root, state)
    if mode == "on":
        deadline = time.monotonic() + 22
        while status(state)["state"] == "starting" and time.monotonic() < deadline:
            time.sleep(0.12)
    return status(state)


def run(root: Path, state: Path, lock_fd: int) -> None:
    from .tailnet_cli import discover, foreground_serve, is_serving, stop_foreground
    from .tailnet_gateway import make_gateway

    # Only the launcher hands us the held lock, so a manual module invocation
    # cannot publish competing status or remove another instance's registration.
    if not os.path.samestat(os.fstat(lock_fd), (state / "tailscale-worker.lock").stat()):
        raise ValueError("Unexpected phone gateway lock.")
    stopping = False

    def stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    service = json.loads((state / "service.json").read_bytes())
    pid = service["pid"]
    with os.fdopen(os.pidfd_open(pid), "rb", buffering=0) as owner:
        info = discover()
        if stopping or not _enabled(root, state) or select.select([owner], [], [], 0)[0]:
            _report(state, "stopped")
            return
        with make_gateway(root, state, service, info.host, info.port, info.login) as gateway:
            port = gateway.server_port
            with (state / "tailscale.log").open("ab") as log:
                proxy = foreground_serve(info, port, log)
                try:
                    deadline = time.monotonic() + 12
                    ready = False
                    while not stopping and _enabled(root, state) and not select.select([owner], [], [], 0)[0]:
                        if proxy.poll() is not None:
                            raise RuntimeError("Tailscale Serve exited. See tailscale.log and the PC's Tailscale status.")
                        if not ready:
                            if is_serving(info, port):
                                _report(state, "ready", url=f"https://{info.host}:{info.port}/", servicePid=pid)
                                ready = True
                            elif time.monotonic() >= deadline:
                                raise RuntimeError("The private Serve route did not start. See tailscale.log.")
                        if ready:
                            gateway.timeout = 0.5
                            gateway.handle_request()
                        else:
                            time.sleep(0.2)
                finally:
                    try:
                        stop_foreground(proxy)
                    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                        _report(state, "cleanup-failed", detail=str(exc))
                        raise SystemExit(1) from exc
    _report(state, "stopped")


if __name__ == "__main__":
    root_arg, state_arg, fd_arg = sys.argv[1:]
    try:
        run(Path(root_arg), Path(state_arg), int(fd_arg))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        _report(Path(state_arg), "unavailable", detail=str(error))
        print(error, file=sys.stderr)
        sys.exit(1)

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
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Thread
from types import FrameType

from .recipes import atomic_text
from .tailnet_cli import TailnetDisconnected, TailnetError, daemon_unreachable
from .update_gate import _boot_id, _owns_live_lock, _start_ticks, default_state_path
from .workspace import PROJECT_STATE_DIR

_RETRY_MIN = 5
_RETRY_MAX = 30
_PROBE_INTERVAL = 8
_PROBE_FILE = "tailscale-probe"
_CONNECT_DETAIL = "Open Tailscale on the PC and connect or sign in."


def default_state(root: Path) -> Path:
    home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return default_state_path(root, home)


def _enabled(root: Path, state: Path) -> bool:
    try:
        return (state / "tailscale.preference").read_text().strip() == "on"
    except FileNotFoundError:
        default = default_state(root)
        return state == default or state.parent == default / PROJECT_STATE_DIR


def _report(state: Path, phase: str, **details: object) -> dict:
    result = {"state": phase, **details}
    atomic_text(state / "tailscale.json", json.dumps(result))
    return result


def _cooldown(delay: int) -> dict:
    return {"retryDelay": delay, "retryAt": time.monotonic() + delay}


def _read_status(state: Path) -> dict:
    try:
        result = json.loads((state / "tailscale.json").read_bytes())
        if isinstance(result, dict) and isinstance(result.get("state"), str):
            return result
    except (FileNotFoundError, ValueError):
        pass
    return {"state": "stopped"}


def _worker_running(state: Path) -> bool:
    try:
        with (state / "tailscale-worker.lock").open("rb") as worker:
            fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except FileNotFoundError:
        pass
    return False


def status(state: Path) -> dict:
    """Read the status for the CLI, including any cleanup failure."""
    result = _read_status(state)
    if result["state"] in ("starting", "ready") and not _worker_running(state):
        return {"state": "stopped"}
    return {key: result[key] for key in ("state", "url", "detail", "servicePid") if key in result}


def _current_service(root: Path, state: Path) -> dict | None:
    try:
        service = json.loads((state / "service.json").read_bytes())
        if isinstance(service, dict):
            pid = service.get("pid")
            if (
                type(pid) is int and pid > 1
                and service.get("root") == str(root.resolve())
                and service.get("boot") == _boot_id()
                and service.get("startTicks") == _start_ticks(pid)
            ):
                return service
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return None


def _worker_identity(service: dict) -> dict:
    return {
        "servicePid": service["pid"],
        "serviceStartTicks": service["startTicks"],
        "serviceBoot": service["boot"],
    }


def _service_status(state: Path, service: dict) -> dict:
    result = _read_status(state)
    if any(result.get(key, value) != value for key, value in _worker_identity(service).items()):
        return {"state": "stopped"}
    return result


def _phone_status(result: dict, worker_running: bool) -> dict:
    phase = result["state"]
    url, detail = result.get("url"), result.get("detail")
    detail = detail if isinstance(detail, str) else ""
    if worker_running:
        if phase == "ready" and isinstance(url, str) and url.startswith("https://"):
            return {"state": "ready", "url": url}
        if phase == "starting":
            return {"state": "starting"}

    if phase == "unavailable":
        # Earlier workers reported every startup failure as unavailable.
        message = detail.casefold()
        if (
            message.startswith((
                "tailscale is disconnected or signed out",
                "the local tailscale device is offline",
                "tailscale serve exited.",
            ))
            or (message.startswith("tailscale ") and message.endswith(" timed out."))
            or daemon_unreachable(detail)
        ):
            phase, detail = "disconnected", _CONNECT_DETAIL
        elif message.startswith("no installed tailscale cli"):
            detail = "Install Tailscale on the PC, sign in, then select Try again."
        elif message.startswith("tailscale https certificates are unavailable"):
            detail = "Enable HTTPS certificates on the DNS page of the Tailscale admin console, then select Try again."
        elif message.startswith("magicdns must be enabled"):
            detail = "Enable MagicDNS in the Tailscale admin console, then select Try again."

    if phase in ("unavailable", "cleanup-failed"):
        return {"state": "unavailable", "detail": detail or "Check tailscale.log for the phone gateway error."}
    # Clients keep polling through cooldown and foreground route cleanup.
    if phase == "disconnected":
        return {"state": phase, "detail": detail or _CONNECT_DETAIL, "retryable": True}
    return {"state": "stopped", "retryable": True}


def _probe_stamp(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def phone_access(root: Path, state: Path) -> dict:
    """Read the phone state and request a background check of a ready URL."""
    if not _enabled(root, state):
        return {"state": "off"}
    service = _current_service(root, state)
    if service is None:
        return {"state": "stopped"}
    running = _worker_running(state)
    result = _phone_status(_service_status(state, service), running)
    if result["state"] == "ready":
        probe = state / _PROBE_FILE
        age = time.time_ns() - _probe_stamp(probe)
        if not 0 <= age < _PROBE_INTERVAL * 1_000_000_000:
            probe.touch()
    return result


def request_phone_access(root: Path, state: Path, action: str) -> dict:
    """Opt in or retry an enabled connection without waiting for Serve."""
    if action not in ("on", "check", "retry"):
        raise ValueError("Phone access action must be 'on', 'check', or 'retry'.")
    state.mkdir(parents=True, exist_ok=True)
    with (state / "tailscale-control.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if action == "on":
                raise FileExistsError("Phone access is busy. Try again.") from exc
        else:
            if action == "on":
                atomic_text(state / "tailscale.preference", "on\n")
            _begin(root, state, retry=action == "retry")
    return phone_access(root, state)


def _begin(
    root: Path, state: Path, *, retry: bool = False, wait_for_handover: bool = False
) -> None:
    if not _enabled(root, state):
        return
    service = _current_service(root, state)
    if service is None:
        return
    # The child inherits this file description. No interval exists in which
    # a second desktop launch can attach a second gateway to the same state.
    with (state / "tailscale-worker.lock").open("a") as lock:
        deadline = time.monotonic() + 4
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # Only CLI launches wait for an earlier desktop to finish closing.
                if (
                    not wait_for_handover
                    or _read_status(state).get("servicePid") in (None, service["pid"])
                    or time.monotonic() >= deadline
                ):
                    return
                time.sleep(0.1)
        previous = _service_status(state, service)
        retry_at = previous.get("retryAt")
        if retry and type(retry_at) in (int, float) and time.monotonic() < retry_at:
            return
        try:
            if not _owns_live_lock(service["pid"], state):
                return
        except FileNotFoundError:
            return
        identity = _worker_identity(service)
        previous_delay = previous.get("retryDelay", 0) if retry else 0
        delay = _RETRY_MIN
        if type(previous_delay) is int:
            delay = min(max(_RETRY_MIN, 2 * previous_delay), _RETRY_MAX)
        _report(state, "starting", **identity, **_cooldown(delay))
        try:
            with (state / "tailscale.log").open("ab") as log:
                log.truncate(0)
                # A detached service outlives the Windows console that ran
                # wsl.exe. Use the distro's persistent interop server when present.
                worker_env = os.environ.copy()
                distro_interop = Path("/run/WSL/1_interop")
                if worker_env.get("WSL_INTEROP") and distro_interop.is_socket():
                    worker_env["WSL_INTEROP"] = str(distro_interop)
                child = subprocess.Popen(
                    [sys.executable, "-m", "studio.tailnet", str(root), str(state), str(lock.fileno())],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    pass_fds=(lock.fileno(),), start_new_session=True, env=worker_env,
                )
                Thread(target=child.wait, daemon=True).start()
        except OSError as exc:
            _report(state, "unavailable", **identity, detail=str(exc), **_cooldown(delay))


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
        _begin(root, state, wait_for_handover=True)
    if mode == "on":
        deadline = time.monotonic() + 22
        while status(state)["state"] == "starting" and time.monotonic() < deadline:
            time.sleep(0.12)
    return status(state)


def run(root: Path, state: Path, lock_fd: int) -> None:
    from .tailnet_cli import (
        check_connection,
        discover,
        foreground_serve,
        is_serving,
        stop_foreground,
    )
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
    service = _current_service(root, state)
    if service is None or (launch := _service_status(state, service))["state"] != "starting":
        _report(state, "stopped")
        return
    identity = _worker_identity(service)
    delay = launch.get("retryDelay", _RETRY_MIN)

    def close_proxy(proxy: subprocess.Popen[bytes]) -> None:
        try:
            stop_foreground(proxy)
        except TailnetDisconnected:
            # The exact child was reaped before checking Serve registration.
            # The next discovery still needs a fresh config and an unused port.
            pass
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            _report(state, "cleanup-failed", **identity, detail=str(exc), **_cooldown(delay))
            raise SystemExit(1) from exc

    try:
        owner_fd = os.pidfd_open(service["pid"])
    except ProcessLookupError:
        _report(state, "stopped", **identity)
        return

    with ExitStack() as stack:
        owner = stack.enter_context(os.fdopen(owner_fd, "rb", buffering=0))

        def wanted() -> bool:
            return not stopping and _enabled(root, state) and not select.select([owner], [], [], 0)[0]

        try:
            if not wanted():
                _report(state, "stopped", **identity)
                return
            info = discover()
            if not wanted():
                _report(state, "stopped", **identity)
                return
            gateway = stack.enter_context(make_gateway(root, state, service, info.host, info.port, info.login))
            log = stack.enter_context((state / "tailscale.log").open("ab"))
            probes = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            proxy = foreground_serve(info, gateway.server_port, log)
            stack.callback(close_proxy, proxy)

            deadline = time.monotonic() + 12
            gateway.timeout = 0.5
            ready = False
            probe_path = state / _PROBE_FILE
            probe_stamp = _probe_stamp(probe_path)
            next_probe = 0.0
            probe = None

            while wanted():
                if proxy.poll() is not None:
                    if ready:
                        raise TailnetDisconnected("The private Tailscale Serve session ended.")
                    check_connection(info)
                    raise TailnetError("Tailscale Serve exited before it was ready. See tailscale.log.")
                if not ready:
                    if not is_serving(info, gateway.server_port):
                        if time.monotonic() >= deadline:
                            raise TailnetError("The private Serve route did not start. See tailscale.log.")
                        time.sleep(0.2)
                        continue
                    _report(state, "ready", **identity, url=f"https://{info.host}:{info.port}/")
                    ready = True
                    delay = _RETRY_MIN

                # Only status reads request a probe. A hung CLI never pauses the
                # gateway loop or the desktop's HTTP request handler.
                if probe is not None and probe.done():
                    probe.result()
                    probe = None
                now = time.monotonic()
                if probe is None and now >= next_probe:
                    stamp = _probe_stamp(probe_path)
                    if stamp != probe_stamp:
                        probe_stamp = stamp
                        next_probe = now + _PROBE_INTERVAL
                        probe = probes.submit(check_connection, info)
                    else:
                        next_probe = now + gateway.timeout
                gateway.handle_request()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            disconnected = isinstance(exc, TailnetDisconnected)
            _report(
                state, "disconnected" if disconnected else "unavailable", **identity,
                detail=str(exc), **_cooldown(delay),
            )
            raise
        else:
            _report(state, "stopped", **identity)


if __name__ == "__main__":
    root_arg, state_arg, fd_arg = sys.argv[1:]
    try:
        run(Path(root_arg), Path(state_arg), int(fd_arg))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

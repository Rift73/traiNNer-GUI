"""Smoke-check the isolated Studio desktop service before a managed slot is ready."""

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
from contextlib import closing
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from types import FrameType


def check_status(port: int, root: Path, state: Path) -> None:
    # Open loopback directly and treat redirects as a health failure.
    with closing(HTTPConnection("127.0.0.1", port, timeout=3)) as client:
        client.request("GET", "/api/status")
        with client.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"Studio /api/status returned HTTP {response.status}.")
            status = json.loads(response.read(8192))
    if (
        not isinstance(status, dict)
        or status.get("name") != "traiNNer-GUI"
        or status.get("trainerRoot") != str(root)
        or status.get("stateDir") != str(state)
        or status.get("projectId") is not None
        or status.get("projectDir") is not None
    ):
        raise RuntimeError("Studio /api/status did not identify the isolated candidate service.")


def stop_service(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)


def smoke_studio(root: Path) -> None:
    ready_marker = root / ".managed-ready"
    if ready_marker.exists() or ready_marker.is_symlink():
        raise RuntimeError("Run this check before Studio registers the managed checkout's state.")
    with tempfile.TemporaryDirectory(prefix="trainner-studio-smoke-") as scratch:
        work = Path(scratch).resolve()
        home = work / "home"
        home.mkdir()
        state = work / "state"
        env = {
            name: os.environ[name]
            for name in ("PATH", "LC_ALL", "CUDA_HOME", "LD_LIBRARY_PATH", "CPATH")
            if name in os.environ
        }
        env.update(
            HOME=str(home),
            PYTHONPATH=str(root),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="",
            TRAINNER_GUI_RUNTIME_MODE="cpu",
        )
        with (work / "service.log").open("w+b") as log:
            try:
                read_fd, write_fd = os.pipe()
                with os.fdopen(read_fd, "rb", buffering=0) as startup:
                    child = None
                    try:
                        # The desktop readiness pipe also skips phone gateway startup.
                        with os.fdopen(write_fd, "wb") as announce:
                            fd = announce.fileno()
                            child = subprocess.Popen(
                                [
                                    sys.executable, "-P", "-m", "studio",
                                    "--state-dir", str(state), "serve",
                                    "--port", "0", "--desktop-fd", str(fd),
                                ],
                                cwd=root,
                                env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                pass_fds=(fd,),
                            )
                        if not select.select([startup], [], [], 20)[0]:
                            raise RuntimeError("Studio did not report a listener within 20 seconds.")
                        report = startup.read(32).strip()
                        if not report.isdigit() or not 1 <= (port := int(report)) <= 65535:
                            raise RuntimeError("Studio did not report a valid listener port.")
                        check_status(port, root, state)
                    finally:
                        if child is not None:
                            stop_service(child)
            except (OSError, ValueError, RuntimeError, HTTPException, subprocess.TimeoutExpired):
                log_size = log.seek(0, os.SEEK_END)
                log.seek(max(0, log_size - 4096))
                sys.stderr.write(log.read().decode(errors="replace"))
                raise


def interrupt(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        smoke_studio(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, RuntimeError, HTTPException, subprocess.TimeoutExpired) as error:
        sys.exit(f"Managed Studio HTTP smoke failed: {error}")
    print("Managed Studio /api/status smoke passed.")

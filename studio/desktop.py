"""Start or reconnect to this checkout's local web service from a desktop launcher."""
from __future__ import annotations

import fcntl
import json
import os
import select
import signal
from contextlib import contextmanager
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .recipes import atomic_text
from .workspace import WindowsProject, Workspace

DEFAULT_PORT = 8732


def _start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19]


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


@contextmanager
def exclusive_service(state: Path):
    state.mkdir(parents=True,exist_ok=True)
    with (state/"service-live.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"A local service is already using {state}.") from exc
        yield


def stamp(state: Path, port: int, root: Path, *, project: WindowsProject | None = None) -> None:
    identity = {"port":port,"pid":os.getpid(),"startTicks":_start_ticks(os.getpid()),
                "boot":_boot_id(),"root":str(root.resolve())}
    if project is not None:
        identity.update(project.status())
    atomic_text(state / "service.json",json.dumps(identity))


def _owns_live_lock(pid: int, state: Path) -> bool:
    """Match the service's open lock path, inode and exclusive flock."""
    path = state.resolve() / "service-live.lock"
    info = path.stat()
    inode = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    expected = ["ADVISORY", "WRITE", str(pid), inode, "0", "EOF"]
    proc = Path(f"/proc/{pid}")
    for fd in (proc / "fd").iterdir():
        try:
            if os.readlink(fd) != str(path):
                continue
            with (proc / "fdinfo" / fd.name).open() as details:
                for line in details:
                    if line.startswith("lock:"):
                        fields = line.split()
                        # WSL DrvFS represents flock() as a whole-file POSIX lock.
                        if len(fields) == 9 and fields[2] in ("FLOCK", "POSIX") and fields[3:] == expected:
                            return True
        except FileNotFoundError:
            continue  # A request socket or the service itself may have closed.
    return False


def stop_service(root: Path, state: Path) -> bool:
    if not state.is_dir():
        return False
    with (state/"service-launch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        path = state/"service.json"
        if not path.is_file():
            return False
        instance = json.loads(path.read_bytes())
        pid = instance.get("pid")
        if type(pid) is not int or pid <= 1 or instance.get("boot") != _boot_id() or instance.get("root") != str(root):
            raise ValueError("The service record does not identify a current process from this checkout.")
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            return False
        try:
            if _start_ticks(pid) != instance.get("startTicks"):
                return False
            if not _owns_live_lock(pid, state):
                return False
            poll = select.poll()
            poll.register(fd,select.POLLIN)
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            if not poll.poll(10000):
                raise TimeoutError("The local service is still stopping.")
        except (FileNotFoundError,ProcessLookupError):
            return False
        finally:
            os.close(fd)
        return True


def forget(state: Path) -> None:
    path=state / "service.json"
    try:
        if json.loads(path.read_bytes()).get("pid") == os.getpid():
            path.unlink(missing_ok=True)
    except (ValueError, FileNotFoundError):
        pass


def _matches(port: int, root: Path, workspace: Workspace) -> bool:
    if type(port) is not int or not 1 <= port <= 65535:
        return False
    try:
        # A desktop health check always targets this WSL instance's loopback.
        local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with local.open(f"http://127.0.0.1:{port}/api/status", timeout=0.7) as response:
            info = json.loads(response.read(8192))
            project_status = workspace.project.status() if workspace.project else {"projectId": None}
            return (isinstance(info,dict) and info.get("name") == "traiNNer-GUI"
                    and info.get("trainerRoot") == str(root)
                    and info.get("stateDir") == str(workspace.state)
                    and all(info.get(key) == value for key, value in project_status.items()))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def desktop_url(root: Path, base_state: Path, *, workspace: Workspace | None = None) -> str:
    base_state = base_state.expanduser().resolve()
    if workspace is None:
        workspace = Workspace.from_environment(base_state)
    state = workspace.state
    state.mkdir(parents=True, exist_ok=True)
    with (state/"service-launch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        instance = state/"service.json"
        if instance.exists():
            try:
                port = json.loads(instance.read_bytes()).get("port")
            except (ValueError,FileNotFoundError):
                port = None
            if _matches(port, root, workspace):
                return f"http://127.0.0.1:{port}/"
        # The child binds port zero if the preferred port belongs to another
        # service. The actual listener address comes back through a private pipe.
        read_fd, write_fd = os.pipe()
        try:
            with (state/"service.log").open("ab") as log:
                child = subprocess.Popen(
                    [sys.executable,"-m","studio","--state-dir",str(base_state),"serve","--desktop-fd",str(write_fd)],
                    cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    env={**os.environ, **workspace.project.environment()} if workspace.project else None,
                    pass_fds=(write_fd,), start_new_session=True)
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        try:
            if not select.select([read_fd],[],[],20)[0]:
                raise TimeoutError("The local service did not start within 20 seconds.")
            raw=os.read(read_fd,64)
            if not raw.strip().isdigit():
                raise RuntimeError(f"The local service failed to start. See {state/'service.log'}")
            port=int(raw.strip())
            if not _matches(port, root, workspace):
                raise RuntimeError("The new service did not identify the expected trainer checkout.")
            return f"http://127.0.0.1:{port}/"
        except BaseException:
            # Popen is the exact process started here; no PID-file signal.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            os.close(read_fd)


def hold_windows_client(url: str, root: Path, state: Path) -> None:
    """Keep one Windows WSL client alive until this exact service exits."""
    instance = json.loads((state / "service.json").read_bytes())
    pid = instance.get("pid")
    if type(pid) is not int or pid <= 1 or instance.get("boot") != _boot_id():
        raise RuntimeError("The selected Studio service no longer has a live process.")
    port = instance.get("port")
    if url != f"http://127.0.0.1:{port}/" or instance.get("root") != str(root):
        raise RuntimeError("The selected Studio listener changed during startup.")

    fd = os.pidfd_open(pid)
    try:
        if _start_ticks(pid) != instance.get("startTicks") or not _owns_live_lock(pid, state):
            raise RuntimeError("The Studio process changed during startup.")
        # The lock is for this particular process identity. A replacement
        # service can take its own lease while the old client is leaving.
        lease = state / f"windows-client-{pid}-{instance['startTicks']}.lock"
        with lease.open("ab") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(url, flush=True)
                return
            try:
                poll = select.poll()
                poll.register(fd, select.POLLIN)
                print(url, flush=True)
                print("TRAINNER_GUI_WSL_CLIENT_READY", flush=True)
                poll.poll()
            finally:
                lease.unlink(missing_ok=True)
    finally:
        os.close(fd)

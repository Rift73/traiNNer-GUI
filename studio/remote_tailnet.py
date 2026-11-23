"""Select a native Tailscale client for the Linux remote command."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import select
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from .tailnet_cli import TailnetError, _identity, _read_json

_PACKAGES = "https://pkgs.tailscale.com/stable/"
_CLI_ENV = "TRAINNER_GUI_TAILSCALE_CLI"
_SOCKET_ENV = "TRAINNER_GUI_TAILSCALE_SOCKET"
_SIGN_IN_TIMEOUT = 300
_PENDING_APPROVAL = "NeedsMachineAuth"


class SetupError(Exception):
    """An actionable setup or cleanup failure."""


def _select(cli: str, path: str = "") -> None:
    os.environ[_CLI_ENV] = cli
    os.environ[_SOCKET_ENV] = path


def _state(cli: str) -> str:
    try:
        status = _read_json(cli, "status", "--json", "--peers=false")
    except TailnetError:
        return ""
    if isinstance(status, dict) and isinstance(state := status.get("BackendState"), str):
        return state
    return ""


def _check_serve(cli: str) -> None:
    try:
        _identity(cli)
    except TailnetError as exc:
        if str(exc).startswith(("MagicDNS must be enabled", "Tailscale HTTPS certificates")):
            raise SetupError(
                "Enable MagicDNS and HTTPS Certificates on the tailnet DNS page "
                "(https://login.tailscale.com/admin/dns), then retry. "
                "A tailnet administrator can enable them for you."
            ) from None
        raise SetupError("Studio needs an online, user-owned Tailscale device.") from None

    # Use the supported read-only command. Studio handles write permissions
    # when it opens the foreground Serve route.
    try:
        _read_json(cli, "serve", "status", "--json")
    except TailnetError as exc:
        if any(text in str(exc).lower() for text in ("denied", "403 forbidden")):
            raise SetupError(
                "Ask the Linux host administrator to grant your account Tailscale "
                "operator access for Serve. "
                "https://tailscale.com/docs/reference/troubleshooting/linux/linux-operator-permission"
            ) from None
        raise SetupError("Could not read Tailscale Serve status. Check the local client and retry.") from None


def _fetch(name: str, output: Path | None = None, timeout: int = 20) -> bytes:
    command = [
        "curl", "-q", "--proto", "=https", "-fsS", "--connect-timeout", "8",
        "--max-time", str(timeout), "--retry", "1",
    ]
    if output is not None:
        command.extend(("--output", os.fspath(output)))
    return subprocess.run(
        [*command, _PACKAGES + name], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=timeout + 2, check=True,
    ).stdout


def _install_private(directory: Path) -> None:
    binaries = directory / "bin"
    programs = ("tailscale", "tailscaled")
    if os.path.lexists(binaries):
        if binaries.is_symlink() or any(
            (path := binaries / name).is_symlink() or not path.is_file()
            or not os.access(path, os.X_OK) for name in programs
        ):
            raise SetupError("The private Tailscale binary directory is incomplete or invalid.")
        return
    arch = {
        "x86_64": "amd64", "aarch64": "arm64", "armv7l": "arm",
        "i686": "386", "riscv64": "riscv64",
    }.get(platform.machine())
    if not arch or not shutil.which("curl"):
        raise SetupError(
            "Private Tailscale needs curl and a supported Linux architecture. "
            "For a host installation, see https://tailscale.com/docs/install/linux."
        )
    try:
        archive = json.loads(_fetch("?mode=json"))["Tarballs"][arch]
        if not isinstance(archive, str) or not re.fullmatch(
            rf"tailscale_[0-9]+\.[0-9]+\.[0-9]+_{arch}\.tgz", archive
        ):
            raise ValueError
        digest = _fetch(archive + ".sha256").strip()
        if not re.fullmatch(rb"[0-9a-fA-F]{64}", digest):
            raise ValueError
        with tempfile.TemporaryDirectory(dir=directory) as scratch:
            staging = Path(scratch)
            bundle = staging / archive
            _fetch(archive, bundle, timeout=90)
            sha256 = hashlib.sha256()
            with bundle.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    sha256.update(chunk)
            if sha256.hexdigest().encode() != digest.lower():
                raise SetupError("The official Tailscale archive failed SHA-256 verification.")
            extracted = staging / "bin"
            extracted.mkdir()
            with tarfile.open(bundle, "r:gz") as members:
                for name in programs:
                    member = members.getmember(f"{archive[:-4]}/{name}")
                    if not member.isfile():
                        raise ValueError
                    with members.extractfile(member) as source, (extracted / name).open("xb") as target:
                        shutil.copyfileobj(source, target)
                    (extracted / name).chmod(0o700)
            extracted.rename(binaries)
    except (KeyError, OSError, TypeError, ValueError, tarfile.TarError, subprocess.SubprocessError):
        raise SetupError(
            "Could not fetch and verify the Tailscale release from pkgs.tailscale.com. "
            "Check outbound HTTPS access and retry."
        ) from None


def _private_dir(directory: Path) -> None:
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise SetupError("The private Tailscale directory must be owned by you and private.")


@contextmanager
def _lock(path: Path) -> Iterator[int]:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError("Tailscale setup or cleanup is busy for this checkout. Retry shortly.") from None
        yield handle.fileno()


def _socket_exists(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
        raise SetupError("The private Tailscale socket path has an unexpected owner or type.")
    return True


def _daemon_pidfd(path: Path, daemon: list[str]) -> int | None:
    if not _socket_exists(path):
        return None
    with socket.socket(socket.AF_UNIX) as peer:
        peer.settimeout(1)
        try:
            peer.connect(os.fspath(path))
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        creds = struct.Struct("iII")
        pid, uid, _ = creds.unpack(
            peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, creds.size)
        )
    if uid != os.geteuid():
        raise SetupError("The private socket is held by a different user.")
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return None
    try:
        proc = Path("/proc") / str(pid)
        argv = (proc / "cmdline").read_bytes().split(b"\0")[:-1]
        if (
            argv == list(map(os.fsencode, daemon))
            and os.path.samefile(proc / "exe", daemon[0])
        ):
            return fd
        raise SetupError("The private socket is held by a different process.")
    except BaseException:
        os.close(fd)
        raise


def _terminal() -> IO[bytes]:
    try:
        return open("/dev/tty", "r+b", buffering=0)
    except OSError:
        raise SetupError("Browser sign-in needs a terminal. Reconnect with ssh -t and retry.") from None


def _start_private(directory: Path, daemon: list[str]) -> subprocess.Popen[bytes]:
    # Verify the kernel allows the operations we need to stop an orphan.
    try:
        fd = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
    except OSError:
        raise SetupError("Private Tailscale needs working Linux pidfd support.") from None
    # Upstream tailscaled can clean up host DNS even with a userspace tunnel.
    for path in ("/etc/resolv.pre-tailscale-backup.conf", "/etc/resolv.tailscale.conf",
                 "/run/resolvconf/interface/tun-tailscale.inet",
                 "/run/resolvconf/interfaces/tun-tailscale.inet"):
        if os.path.lexists(path):
            raise SetupError("Linux Tailscale already manages host DNS. Ask the host administrator to restore the existing client.")
    with _terminal():
        pass
    _install_private(directory)
    _private_dir(directory / "state")
    # The inherited lock prevents a second launch even before the socket binds.
    with _lock(directory / "daemon.lock") as lease:
        return subprocess.Popen(
            daemon, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, pass_fds=(lease,),
        )


def _setup_private(directory: Path, path: Path, daemon: list[str], command: list[str]) -> None:
    child, keep = None, False
    try:
        fd = _daemon_pidfd(path, daemon)
        if fd is None:
            child = _start_private(directory, daemon)
        else:
            os.close(fd)
        deadline = time.monotonic() + 12
        while not (state := _state(command[0])) or state == "Starting":
            if (child is not None and child.poll() is not None) or time.monotonic() >= deadline:
                raise SetupError("The private Tailscale daemon did not become ready.")
            time.sleep(0.25)
        if state in ("NoState", "NeedsLogin", "Stopped"):
            with _terminal() as tty:
                try:
                    subprocess.run(
                        [*command, "up", f"--timeout={_SIGN_IN_TIMEOUT}s",
                         "--accept-dns=false", "--accept-routes=false"],
                        stdin=tty, stdout=tty, stderr=tty, timeout=_SIGN_IN_TIMEOUT + 10,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    pass
            state = _state(command[0])
        keep = state in ("Running", _PENDING_APPROVAL)
        if state == _PENDING_APPROVAL:
            raise SetupError(
                "Ask a tailnet administrator to approve this device at "
                "https://login.tailscale.com/admin/machines, then retry."
            )
        if state != "Running":
            raise SetupError("Tailscale sign-in did not finish. Retry setup and complete browser sign-in.")
        _check_serve(command[0])
    finally:
        if child is not None and not keep and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)


def _stop_private(fd: int) -> None:
    exited = select.poll()
    exited.register(fd, select.POLLIN)
    for sig, wait_ms in ((signal.SIGTERM, 8000), (signal.SIGKILL, 2000)):
        if exited.poll(0):
            return
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            return
        if exited.poll(wait_ms):
            return
    raise SetupError("The private Tailscale daemon is still stopping. Retry cleanup.")


def _cleanup_private(directory: Path, path: Path, daemon: list[str], command: list[str], logout: bool) -> None:
    logout_failed = False
    if (fd := _daemon_pidfd(path, daemon)) is not None:
        try:
            if logout:
                try:
                    reply = subprocess.run(
                        [*command, "logout"], stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=8, check=False,
                    )
                    logout_failed = reply.returncode != 0
                except (OSError, subprocess.TimeoutExpired):
                    logout_failed = True
            _stop_private(fd)
        finally:
            os.close(fd)
    with _lock(directory / "daemon.lock"):
        if _socket_exists(path):
            path.unlink()
        state = directory / "state"
        if os.path.lexists(state):
            if state.is_symlink() or not state.is_dir():
                raise SetupError("The private Tailscale state path has an unexpected type.")
            shutil.rmtree(state)
    if logout_failed:
        raise SetupError(
            "Private Tailscale stopped, but explicit logout failed. "
            "The ephemeral device expires after it goes offline."
        )


def _run(action: str) -> None:
    if sys.platform != "linux":
        raise SetupError("Remote Tailscale setup runs on Linux.")
    if action == "setup":
        # Studio usually prefers Windows Tailscale in WSL. Explicitly select
        # the native Linux client for the remote command.
        system = shutil.which("tailscale")
        if system and not Path(system).resolve().name.lower().endswith(".exe"):
            cli = os.path.abspath(system)
            _select(cli)
            if _state(cli) == "Running":
                _check_serve(cli)
                _print_paths(cli)
                return

    state_home = Path(os.environ.get("XDG_STATE_HOME") or "")
    if not state_home.is_absolute():
        state_home = Path.home() / ".local/state"
    checkout = hashlib.sha256(os.getcwdb()).hexdigest()[:16]
    directory = state_home / "trainner-gui" / f"ts-{checkout}"
    if action != "setup" and not os.path.lexists(directory):
        return
    path = directory / "ts.sock"
    if "\n" in str(path) or "\r" in str(path):
        raise SetupError("The private Tailscale path contains a line break.")
    if len(os.fsencode(path)) > 107:
        raise SetupError("The Tailscale socket path is too long. Use a shorter absolute XDG_STATE_HOME.")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise SetupError("Private Tailscale needs a Linux kernel and Python with pidfd support.")
    _private_dir(directory)
    daemon = [
        os.fspath(directory / "bin/tailscaled"), "--tun=userspace-networking", "--state=mem:",
        f"--statedir={directory / 'state'}", f"--socket={path}", "--port=0",
    ]
    cli = os.fspath(directory / "bin/tailscale")
    _select(cli, os.fspath(path))
    command = [cli, f"--socket={path}"]
    with _lock(directory / "setup.lock"):
        if action == "setup":
            _setup_private(directory, path, daemon, command)
            _print_paths(cli, os.fspath(path))
        else:
            _cleanup_private(directory, path, daemon, command, logout=action == "logout")
            _print_paths(os.fspath(path))


def _print_paths(*paths: str) -> None:
    if any("\n" in path or "\r" in path for path in paths):
        raise SetupError("The selected Tailscale paths contain a line break.")
    print(*paths, sep="\n")


def _interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("setup", "cleanup", "logout"):
        print("Usage: remote_tailnet {setup|cleanup|logout}", file=sys.stderr)
        return 2
    os.umask(0o077)
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    try:
        _run(sys.argv[1])
    except SetupError as error:
        print(error, file=sys.stderr)
        return 1
    except (OSError, subprocess.SubprocessError) as error:
        reason = errno.errorcode.get(getattr(error, "errno", None), type(error).__name__)
        print(f"Private Tailscale operation failed ({reason}). Check permissions and retry.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Tailscale setup or cleanup interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

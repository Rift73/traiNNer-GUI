"""Discover and supervise one private, foreground Tailscale Serve session."""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import IO, Any, NamedTuple

_LOG = logging.getLogger(__name__)
_PORTS = range(8732, 8740)
_QUERY_TIMEOUT = 4
_WINDOWS_CLI = "/mnt/c/Program Files/Tailscale/tailscale.exe"
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_JsonObject = dict[str, Any]


class TailnetError(RuntimeError):
    """Tailscale identity, Serve configuration, or child shutdown failed."""


class TailnetInfo(NamedTuple):
    cli: str
    host: str
    login: str
    port: int


def _query(*args: str, check: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=_QUERY_TIMEOUT,
        check=check,
    )


def _windows_cli() -> str | None:
    where = shutil.which("/mnt/c/Windows/System32/where.exe") or shutil.which(
        "where.exe"
    )
    if where:
        try:
            # Restrict the search to Windows PATH, excluding the working directory.
            matches = _query(where, "$PATH:tailscale.exe")
            if matches.returncode == 0:
                for match in matches.stdout.decode("utf-8", "replace").splitlines():
                    path = os.fsdecode(
                        _query("wslpath", "-u", match, check=True).stdout
                    )
                    if cli := shutil.which(path.rstrip("\r\n")):
                        return cli
            elif matches.returncode != 1:
                _LOG.warning(
                    "Windows PATH search for Tailscale exited with %d",
                    matches.returncode,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            _LOG.warning("Could not search Windows PATH for Tailscale: %s", exc)
    return shutil.which(_WINDOWS_CLI)


def _find_cli() -> str:
    if sys.platform == "linux" and (
        os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower()
    ):
        if cli := _windows_cli():
            return cli
    if cli := shutil.which("tailscale"):
        return cli
    raise TailnetError("No installed Tailscale CLI was found.")


def _read_json(cli: str, *args: str) -> Any:
    command = "tailscale " + " ".join(args)
    try:
        result = _query(cli, *args)
    except subprocess.TimeoutExpired as exc:
        raise TailnetError(f"{command} timed out.") from exc
    except OSError as exc:
        raise TailnetError(f"Could not run {command}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise TailnetError(
            f"{command} exited with {result.returncode}. {detail}".strip()
        )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise TailnetError(f"{command} returned invalid JSON.") from exc


def _object(value: Any, label: str) -> _JsonObject:
    if not isinstance(value, dict):
        raise TailnetError(f"Tailscale returned an invalid {label} object.")
    return value


def _map(obj: _JsonObject, key: str) -> _JsonObject:
    value = obj.get(key)
    return {} if value is None else _object(value, key)


def _dns_name(raw: Any) -> str:
    if isinstance(raw, str) and raw.isascii():
        name = raw.removesuffix(".").lower()
        if len(name) <= 253 and all(
            _DNS_LABEL.fullmatch(label) for label in name.split(".")
        ):
            return name
    raise TailnetError("Tailscale returned an invalid MagicDNS name.")


def _identity(cli: str) -> tuple[str, str]:
    status = _object(_read_json(cli, "status", "--json", "--peers=false"), "status")
    state = status.get("BackendState")
    if state != "Running":
        raise TailnetError(
            f"Tailscale is disconnected or signed out (state {state!r})."
        )
    node = _object(status.get("Self"), "Self")
    if node.get("Online") is not True or node.get("Expired", False) is not False:
        raise TailnetError(
            "The local Tailscale device is offline or its key has expired."
        )

    tailnet = _object(status.get("CurrentTailnet"), "CurrentTailnet")
    if tailnet.get("MagicDNSEnabled") is not True:
        raise TailnetError("MagicDNS must be enabled on the current tailnet.")
    suffix = _dns_name(tailnet.get("MagicDNSSuffix"))
    host = _dns_name(node.get("DNSName"))
    if (
        not suffix.endswith(".ts.net")
        or host.partition(".")[2] != suffix
        or _dns_name(status.get("MagicDNSSuffix", suffix)) != suffix
    ):
        raise TailnetError(
            "The device's MagicDNS name does not match the current tailnet."
        )

    user_id = node.get("UserID")
    if (
        type(user_id) is not int
        or user_id <= 0
        or node.get("Tags") not in (None, [])
        or node.get("ShareeNode", False) is not False
        or node.get("AltSharerUserID", 0) != 0
    ):
        raise TailnetError(
            "The local Tailscale device must be owned by a signed-in user."
        )
    profile = _object(_map(status, "User").get(str(user_id)), "local user profile")
    login = profile.get("LoginName")
    if (
        profile.get("ID", user_id) != user_id
        or not isinstance(login, str)
        or not login
        or not login.isprintable()
        or any(char.isspace() for char in login)
        or login == "tagged-devices"
    ):
        raise TailnetError(
            "Tailscale did not report a login for the local device's owner."
        )

    # The CLI can otherwise enter an interactive HTTPS enablement flow.
    cert_domains = status.get("CertDomains")
    if not isinstance(cert_domains, list) or host not in cert_domains:
        raise TailnetError(
            "Tailscale HTTPS certificates are unavailable for this MagicDNS name."
        )
    return host, login


@dataclass
class _Binding:
    tcp: _JsonObject = field(default_factory=dict)
    web: dict[str, _JsonObject] = field(default_factory=dict)
    funnel: bool = False


def _port_number(value: str) -> int:
    if re.fullmatch(r"[0-9]{1,5}", value) and 1 <= (port := int(value)) <= 65535:
        return port
    raise TailnetError(f"Tailscale Serve returned an invalid port {value!r}.")


def _host_port(value: str) -> int:
    host, _, port = value.rpartition(":")
    if not host:
        raise TailnetError(
            f"Tailscale Serve returned an invalid host and port {value!r}."
        )
    return _port_number(port)


def _serve_bindings(cli: str) -> dict[int, list[_Binding]]:
    raw = _read_json(cli, "serve", "status", "--json")
    pending = [] if raw is None else [_object(raw, "Serve configuration")]
    bindings: defaultdict[int, list[_Binding]] = defaultdict(list)
    while pending:
        config = pending.pop()
        pending.extend(
            _object(session, "foreground Serve configuration")
            for session in _map(config, "Foreground").values()
        )
        # Keep each session separate so an unrelated TCP or Web entry cannot
        # complete the HTTPS mapping from another session.
        ports: defaultdict[int, _Binding] = defaultdict(_Binding)
        for key, handler in _map(config, "TCP").items():
            ports[_port_number(key)].tcp = _object(handler, "Serve TCP handler")
        for key, handler in _map(config, "Web").items():
            ports[_host_port(key)].web[key] = _object(handler, "Serve Web handler")
        for key, enabled in _map(config, "AllowFunnel").items():
            if not isinstance(enabled, bool):
                raise TailnetError(
                    "Tailscale Serve returned an invalid AllowFunnel entry."
                )
            if enabled:
                ports[_host_port(key)].funnel = True
        for port, binding in ports.items():
            bindings[port].append(binding)
    return bindings


def discover() -> TailnetInfo:
    """Read the local identity and select the first unused Serve port."""
    cli = _find_cli()
    host, login = _identity(cli)
    used = _serve_bindings(cli)
    for port in _PORTS:
        if port not in used:
            return TailnetInfo(cli, host, login, port)
    raise TailnetError(
        f"No free Tailscale Serve port from {_PORTS.start} through {_PORTS.stop - 1}."
    )


def _target_url(target_port: int) -> str:
    if type(target_port) is not int or not 1 <= target_port <= 65535:
        raise ValueError("The local gateway port must be an integer from 1 to 65535.")
    return f"http://127.0.0.1:{target_port}"


def _windows_serve(cli: str, port: int, target: str) -> list[str]:
    # WSL can reap its interop relay while the Windows CLI and its route live on.
    # A Windows parent retains the exact process handle. Closing the Python
    # parent's pipe (including on abrupt exit) tells it to release that route.
    shell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    windows_cli = os.fsdecode(_query("wslpath", "-w", cli, check=True).stdout).strip()
    quoted_cli = windows_cli.replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = '{quoted_cli}'
$info.Arguments = 'serve --https={port} {target}'
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.RedirectStandardInput = $true
$child = [Diagnostics.Process]::Start($info)
try {{
    $child.StandardInput.Close()
    $inputStream = [Console]::OpenStandardInput()
    $stop = $inputStream.ReadAsync((New-Object 'byte[]' 1), 0, 1)
    while (-not $child.WaitForExit(150)) {{
        if ($stop.IsCompleted) {{ break }}
    }}
    if ($child.HasExited) {{ exit $child.ExitCode }}
}} finally {{
    try {{
        if (-not $child.HasExited) {{ $child.Kill() }}
    }} finally {{
        $child.WaitForExit()
        $child.Dispose()
    }}
}}
"""
    return [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
            base64.b64encode(script.encode("utf-16-le")).decode("ascii")]


class _ServeProcess(subprocess.Popen[bytes]):
    def __init__(
        self, info: TailnetInfo, target: str, log: IO[bytes] | IO[str]
    ) -> None:
        self.info = info
        self.target = target
        windows = info.cli.lower().endswith(".exe")
        command = (_windows_serve(info.cli, info.port, target) if windows
                   else [info.cli, "serve", f"--https={info.port}", target])
        super().__init__(
            command,
            stdin=subprocess.PIPE if windows else subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def foreground_serve(
    info: TailnetInfo, target_port: int, log: IO[bytes] | IO[str]
) -> subprocess.Popen[bytes]:
    """Start private HTTPS Serve to the loopback gateway, logging both outputs."""
    target = _target_url(target_port)
    if type(info.port) is not int or info.port not in _PORTS:
        raise TailnetError(
            "The HTTPS port was not selected from the Studio Serve range."
        )
    if _identity(info.cli) != (info.host, info.login):
        raise TailnetError("The local Tailscale device or signed-in owner changed.")
    if info.port in _serve_bindings(info.cli):
        raise TailnetError(f"Tailscale Serve port {info.port} is already in use.")
    try:
        return _ServeProcess(info, target, log)
    except OSError as exc:
        raise TailnetError(f"Could not start Tailscale Serve: {exc}") from exc


def _handlers(binding: _Binding, host_port: str) -> _JsonObject:
    return _map(binding.web.get(host_port, {}), "Handlers")


def is_serving(info: TailnetInfo, target_port: int) -> bool:
    """Check the exact HTTPS root route and reject Funnel on this port."""
    host_port = f"{info.host}:{info.port}"
    try:
        target = _target_url(target_port)
        bindings = _serve_bindings(info.cli).get(info.port)
        if not bindings:
            return False
        if len(bindings) == 1:
            binding = bindings[0]
            handlers = _handlers(binding, host_port)
            root = _map(handlers, "/")
            if (
                binding.tcp.get("HTTPS") is True
                and binding.tcp.get("HTTP", False) is False
                and not binding.tcp.get("TCPForward")
                and not binding.funnel
                and binding.web.keys() == {host_port}
                and handlers.keys() == {"/"}
                and root.get("Proxy") == target
                and not root.get("Path")
                and not root.get("Text")
            ):
                return True
        _LOG.warning(
            "Tailscale Serve port %d has an unexpected or public route.", info.port
        )
    except (TailnetError, ValueError) as exc:
        _LOG.warning("Could not verify private Tailscale Serve: %s", exc)
    return False


def _wait_for_removal(info: TailnetInfo, target: str) -> None:
    host_port = f"{info.host}:{info.port}"
    deadline = time.monotonic() + _QUERY_TIMEOUT
    while True:
        bindings = _serve_bindings(info.cli).get(info.port, [])
        if not any(
            _map(_handlers(binding, host_port), "/").get("Proxy") == target
            for binding in bindings
        ):
            return
        if time.monotonic() >= deadline:
            raise TailnetError(
                f"The Tailscale Serve route on port {info.port} is still registered."
            )
        time.sleep(0.15)


def stop_foreground(process: subprocess.Popen[bytes]) -> None:
    """Reap our exact child and confirm its private route has gone."""
    if not isinstance(process, _ServeProcess):
        raise ValueError("Expected a process returned by foreground_serve().")
    try:
        if process.stdin is not None:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _LOG.warning("Windows Serve supervisor did not acknowledge shutdown.")
        for sig, timeout in (
            (signal.SIGINT, 3),
            (signal.SIGTERM, 2),
            (signal.SIGKILL, 2),
        ):
            if process.poll() is not None:
                break
            process.send_signal(sig)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _LOG.warning(
                    "Tailscale Serve child %d did not exit after %s.",
                    process.pid,
                    sig.name,
                )
        if process.poll() is None:
            raise TailnetError(f"Tailscale Serve child {process.pid} did not exit.")
        _wait_for_removal(process.info, process.target)
    except (OSError, TailnetError) as exc:
        _LOG.error("Could not stop private Tailscale Serve: %s", exc)
        raise

"""Private Tailscale Serve gateway for one running Studio service."""

from __future__ import annotations

import json
import os
import re
import select
import socket
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException, HTTPMessage, HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import unquote

from .desktop import _boot_id, _owns_live_lock, _start_ticks
from .server import MAX_BODY, is_phone_access_route

_LOOPBACK = "127.0.0.1"
_CONNECT_TIMEOUT = 1.0
_CLIENT_TIMEOUT = 30
_UPSTREAM_TIMEOUT = 180
_MAX_CONNECTIONS = 64
_CHUNK_SIZE = 128 * 1024
_STATUS_LIMIT = 8192
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")
_TARGET = re.compile(r"/(?!/)[\x21-\x7e]*")


class _ServiceUnavailable(RuntimeError):
    """The pinned trainer service is unavailable."""


class _RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns


def _content_length(headers: HTTPMessage) -> int | None:
    values = headers.get_all("Content-Length", [])
    if not values:
        return None
    value = values[0].strip(" \t")
    if len(values) != 1 or not value.isascii() or not value.isdecimal():
        raise ValueError("Invalid Content-Length.")
    return int(value.lstrip("0") or "0")


def _response_length(response: HTTPResponse) -> int:
    length = _content_length(response.headers)
    if (
        length is None
        or response.headers.defects
        or response.headers.get_all("Transfer-Encoding")
        or not 200 <= response.status <= 599
    ):
        raise HTTPException("Invalid trainer response framing.")
    return length


class _PinnedService:
    def __init__(self, root: Path, state: Path, record: dict) -> None:
        port, pid = record.get("port"), record.get("pid")
        if (
            type(port) is not int or not 1 <= port <= 65535
            or type(pid) is not int or pid <= 1
            or record.get("root") != str(root)
            or record.get("boot") != _boot_id()
        ):
            raise ValueError("The service record does not identify this checkout on this boot.")
        self.port = port
        self.authority = f"{_LOOPBACK}:{port}"
        self.pid = pid
        self.start_ticks = record.get("startTicks")
        self.record_path = state / "service.json"
        with self.record_path.open("rb") as stream:
            data = stream.read(_STATUS_LIMIT + 1)
            if len(data) > _STATUS_LIMIT or json.loads(data) != record:
                raise _ServiceUnavailable("The trainer service record has changed.")
            self.record_identity = _file_identity(os.fstat(stream.fileno()))
        self._require_current()
        if not _owns_live_lock(pid, state):
            raise _ServiceUnavailable("The recorded process does not own the live service.")

        with self.connect(timeout=_CONNECT_TIMEOUT) as connection:
            connection.request(
                "GET", "/api/status", headers={"Host": self.authority, "Connection": "close"}
            )
            with connection.getresponse() as response:
                size = _response_length(response)
                if response.status != HTTPStatus.OK or size > _STATUS_LIMIT:
                    raise _ServiceUnavailable("The trainer service returned an invalid status.")
                payload = response.read(size)
                if len(payload) != size:
                    raise _ServiceUnavailable("The trainer status was incomplete.")
                status = json.loads(payload)
        if (
            not isinstance(status, dict)
            or status.get("name") != "traiNNer-GUI"
            or status.get("trainerRoot") != str(root)
            or status.get("stateDir") != str(state)
            or status.get("projectId") != record.get("projectId")
            or status.get("projectDir") != record.get("projectDir")
        ):
            raise _ServiceUnavailable("The trainer status does not match this checkout and state.")
        self._require_current()

    def _require_current(self) -> None:
        try:
            pidfd = os.pidfd_open(self.pid)
            try:
                poll = select.poll()
                poll.register(pidfd, select.POLLIN)
                if (
                    _file_identity(self.record_path.stat()) != self.record_identity
                    or _start_ticks(self.pid) != self.start_ticks
                    or poll.poll(0)
                ):
                    raise _ServiceUnavailable("The trainer service instance has changed.")
            finally:
                os.close(pidfd)
        except OSError as exc:
            raise _ServiceUnavailable("The pinned trainer service is unavailable.") from exc

    @contextmanager
    def connect(self, timeout: float = _UPSTREAM_TIMEOUT) -> Iterator[HTTPConnection]:
        connection = HTTPConnection(_LOOPBACK, self.port, timeout=_CONNECT_TIMEOUT)
        connection.auto_open = False
        try:
            try:
                connection.connect()
            except OSError as exc:
                raise _ServiceUnavailable("The pinned trainer service is unavailable.") from exc
            # Studio unpublishes its record before closing its exclusive listener.
            # Checking after connect closes the port-reuse race. The connected
            # socket cannot migrate to a replacement if Studio exits afterward.
            self._require_current()
            assert connection.sock is not None
            connection.sock.settimeout(timeout)
            yield connection
        finally:
            connection.close()


class _GatewayHandler(BaseHTTPRequestHandler):
    def __init__(
        self, *args: object, upstream: _PinnedService, hosts: frozenset[str],
        origin: str, login: str,
    ) -> None:
        self.upstream = upstream
        self.hosts = hosts
        self.origin = origin
        self.login = login
        self.response_started = False
        super().__init__(*args)

    def setup(self) -> None:
        self.request.settimeout(_CLIENT_TIMEOUT)
        super().setup()

    def log_message(self, format: str, *args: object) -> None:
        # URLs can contain local file paths.
        pass

    def _authorize(self) -> None:
        headers = self.headers
        if (
            self.request_version not in {"HTTP/1.0", "HTTP/1.1"}
            or headers.defects
            or any(_CONTROL.search(value) for _, value in headers.raw_items())
        ):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Malformed HTTP request.")
        if (
            headers.get_all("Tailscale-User-Login", []) != [self.login]
            or headers.get_all("Tailscale-Funnel-Request")
        ):
            raise _RequestError(HTTPStatus.FORBIDDEN, "Access requires the configured tailnet account.")
        hosts = headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.hosts:
            raise _RequestError(HTTPStatus.FORBIDDEN, "This host is not allowed.")
        origins = headers.get_all("Origin", [])
        if origins != [self.origin] and (origins or self.command == "POST"):
            raise _RequestError(HTTPStatus.FORBIDDEN, "This origin is not allowed.")
        if headers.get_all("Sec-Fetch-Site", []) not in ([], ["same-origin"], ["none"]):
            raise _RequestError(HTTPStatus.FORBIDDEN, "Cross-origin requests are disabled.")
        referers = headers.get_all("Referer", [])
        if referers and (len(referers) != 1 or not referers[0].startswith(self.origin + "/")):
            raise _RequestError(HTTPStatus.FORBIDDEN, "Cross-origin requests are disabled.")

    def _request(self) -> tuple[str, int | None]:
        # BaseHTTPRequestHandler normalizes a leading // in self.path.
        target = self.requestline.split()[1]
        path, separator, query = target.partition("?")
        route = unquote(path)
        if (
            not _TARGET.fullmatch(target) or "#" in target
            or route.startswith("//") or "\\" in route or _CONTROL.search(route)
        ):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Use an origin-form request target.")
        if is_phone_access_route(route):
            raise _RequestError(
                HTTPStatus.FORBIDDEN, "Phone access settings are available on the desktop only."
            )
        if self.headers.get_all("Transfer-Encoding") or self.headers.get_all("Trailer"):
            raise _RequestError(HTTPStatus.BAD_REQUEST, "Use a fixed-length request body.")
        if self.headers.get_all("Expect"):
            raise _RequestError(HTTPStatus.EXPECTATION_FAILED, "Expect is unsupported.")
        try:
            length = _content_length(self.headers)
        except ValueError as exc:
            raise _RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if self.command == "POST":
            limit = 4 * MAX_BODY if route == "/api/reproduction/prepare" else MAX_BODY
            if length is None:
                raise _RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required.")
            if length > limit:
                raise _RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid or excessive request length.")
            if len(self.headers.get_all("Content-Type", [])) > 1:
                raise _RequestError(HTTPStatus.BAD_REQUEST, "Use a single Content-Type.")
        elif length:
            raise _RequestError(HTTPStatus.BAD_REQUEST, "A request body is only supported for POST.")
        if route in {"/", "/index.html"}:
            target = "/mobile.html" + separator + query
        return target, length

    def _proxy(self) -> None:
        try:
            self._authorize()
            target, length = self._request()
            with self.upstream.connect() as connection:
                self._send_request(connection, target, length)
                with connection.getresponse() as response:
                    self._relay(response)
        except _RequestError as exc:
            self._error(exc.status, str(exc))
        except _ServiceUnavailable:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "The pinned trainer service is unavailable.")
        except TimeoutError:
            self._error(HTTPStatus.GATEWAY_TIMEOUT, "The trainer connection timed out.")
        except (OSError, HTTPException, ValueError):
            self._error(HTTPStatus.BAD_GATEWAY, "The trainer connection failed.")

    do_HEAD = do_GET = do_POST = _proxy  # noqa: N815 (HTTP handler names)

    def _send_request(
        self, connection: HTTPConnection, target: str, length: int | None
    ) -> None:
        connection.putrequest(self.command, target, skip_host=True, skip_accept_encoding=True)
        connection.putheader("Host", self.upstream.authority)
        connection.putheader("Connection", "close")
        if self.headers.get("Origin") is not None:
            connection.putheader("Origin", "http://" + self.upstream.authority)
        if self.command == "POST":
            assert length is not None
            connection.putheader("Content-Length", str(length))
            if content_type := self.headers.get("Content-Type"):
                connection.putheader("Content-Type", content_type)

        sock = connection.sock
        assert sock is not None
        if length:
            # Studio's read(length) can return a partial JSON document at EOF.
            # Reset an incomplete upload so Studio cannot act on a valid prefix.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        connection.endheaders()
        remaining = length or 0
        while remaining:
            try:
                chunk = self.rfile.read1(min(remaining, _CHUNK_SIZE))
            except TimeoutError as exc:
                raise _RequestError(HTTPStatus.REQUEST_TIMEOUT, "Request body timed out.") from exc
            if not chunk:
                raise _RequestError(HTTPStatus.BAD_REQUEST, "Incomplete request body.")
            try:
                connection.send(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # Studio can reject a request before consuming the body.
                return
            remaining -= len(chunk)
        if length:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 0, 0))

    def _relay(self, response: HTTPResponse) -> None:
        length = _response_length(response)
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        if _CONTROL.search(content_type):
            raise HTTPException("Invalid trainer Content-Type.")
        self._headers(response.status, length, content_type)
        if self.command == "HEAD":
            return
        while length:
            chunk = response.read1(min(length, _CHUNK_SIZE))
            if not chunk:
                raise HTTPException("Incomplete trainer response.")
            self.wfile.write(chunk)
            length -= len(chunk)

    def _headers(self, status: int, length: int, content_type: str) -> None:
        self.response_started = True
        self.send_response_only(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()

    def _error(self, status: HTTPStatus, message: str) -> None:
        if self.response_started:
            return  # Closing the stream signals a truncated response.
        data = json.dumps({"error": message}).encode("utf-8")
        try:
            self._headers(status, len(data), "application/json; charset=utf-8")
            if self.command != "HEAD":
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


class _GatewayServer(ThreadingHTTPServer):
    request_queue_size = _MAX_CONNECTIONS

    def __init__(self, handler: partial[_GatewayHandler]) -> None:
        self._slots = BoundedSemaphore(_MAX_CONNECTIONS)
        super().__init__((_LOOPBACK, 0), handler)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_gateway(
    root: Path, state: Path, service_record: dict, public_host: str, public_port: int, login: str
) -> ThreadingHTTPServer:
    """Verify the existing service and bind a loopback gateway for Tailscale Serve."""
    host = public_host.lower().removesuffix(".")
    if (
        len(host) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in host.split("."))
        or type(public_port) is not int or not 1 <= public_port <= 65535
    ):
        raise ValueError("Provide a valid Serve hostname and HTTPS port.")
    if not re.fullmatch(r"[!-~]+", login) or "," in login:
        raise ValueError("Provide one nonempty Tailscale account login.")

    # For a TCP backend, Serve preserves the original Host.
    authority = f"{host}:{public_port}"
    hosts = frozenset({authority, host} if public_port == 443 else {authority})
    origin = f"https://{host if public_port == 443 else authority}"
    upstream = _PinnedService(root.resolve(), state.resolve(), service_record)
    return _GatewayServer(partial(
        _GatewayHandler, upstream=upstream, hosts=hosts, origin=origin, login=login
    ))

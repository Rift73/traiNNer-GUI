"""Row-streamed PNG sample differences between versioned saved validation images."""

from __future__ import annotations

import ctypes as ct
import os
import stat
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Any

from trainner_data.file_version import artifact_version

from .native import executable
from .paths import local_path

_PROGRESS = ct.CFUNCTYPE(ct.c_int, ct.c_uint32, ct.c_uint32)
_FIELDS = ("x", "y", "width", "height")
PathArg = str | os.PathLike[str]


class ImageDifferenceCancelled(RuntimeError):
    """Comparison was cancelled before publication."""


class _Summary(ct.Structure):
    _fields_ = [
        ("mae", ct.c_double * 3),
        ("rmse", ct.c_double * 3),
        ("maximum", ct.c_uint32 * 3),
        ("changed", ct.c_uint64),
    ]

    def report(self, channels: list[str]) -> dict[str, Any]:
        return {
            "changedPixels": self.changed,
            "channels": {
                name: {"mae": self.mae[i], "rmse": self.rmse[i], "max": self.maximum[i]}
                for i, name in enumerate(channels)
            },
        }


class _Result(ct.Structure):
    _fields_ = [
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("channels", ct.c_uint32),
        ("depth", ct.c_uint32),
        ("pair", _Summary * 3),
        ("closer", ct.c_uint64 * 3),
    ]


@lru_cache(maxsize=1)
def _kernel() -> Any:
    path = executable("image_difference", "difference.so", source="difference.c")
    func = ct.CDLL(str(path)).studio_png_difference_v1
    func.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(_Result),
        _PROGRESS,
        ct.POINTER(ct.c_char),
        ct.c_size_t,
    ]
    func.restype = ct.c_int
    return func


def validate_region(region: Mapping[str, int] | None) -> tuple[int, int, int, int] | None:
    if region is None:
        return None
    if not isinstance(region, Mapping) or set(region) != set(_FIELDS):
        raise ValueError("region requires x, y, width and height.")
    x, y, width, height = (region[key] for key in _FIELDS)
    if (
        any(
            type(v) is not int or not 0 <= v <= 0x7FFFFFFF
            for v in (x, y, width, height)
        )
        or not width
        or not height
    ):
        raise ValueError(
            "Use nonnegative integer PNG coordinates and positive integer dimensions."
        )
    return x, y, width, height


def _open_read(path: str, flags: int) -> int:
    return os.open(path, flags | os.O_NONBLOCK)


def compare_pngs(
    left: PathArg,
    right: PathArg,
    output_dir: PathArg,
    *,
    versions: Mapping[str, str],
    reference: PathArg | None = None,
    region: Mapping[str, int] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Compare opaque PNG integer samples and publish one lossless absolute-difference PNG."""
    cwd = Path.cwd()
    inputs = {
        k: local_path(os.fspath(p), cwd)
        for k, p in (("left", left), ("right", right), ("reference", reference))
        if p is not None
    }
    if not isinstance(versions, Mapping) or set(versions) != set(inputs):
        raise ValueError("Supply an expected artifact version for each input role.")
    expected = dict(versions)
    if any(not isinstance(v, str) or not v for v in expected.values()):
        raise ValueError("Expected artifact versions must be nonempty strings.")
    rectangle = validate_region(region)
    bounds = (ct.c_uint32 * 4)(*rectangle) if rectangle is not None else None
    folder = local_path(os.fspath(output_dir), cwd)
    destination = folder / "absolute-difference.png"

    def check() -> None:
        if cancelled is not None and cancelled():
            raise ImageDifferenceCancelled("PNG comparison was cancelled.")

    callback_error = None
    last_event = 0.0

    def update(done: int, total: int) -> int:
        nonlocal last_event, callback_error
        try:
            check()
            now = monotonic()
            if progress is not None and (
                done in (0, total) or now - last_event >= 0.25
            ):
                progress({"stage": "comparing", "completed": done, "total": total})
                last_event = now
            return 0
        except BaseException as exc:
            # ctypes callbacks must return normally. Propagate on the Python side.
            callback_error = exc
            return 1

    check()
    with ExitStack() as stack:
        fds = {}
        for role, path in inputs.items():
            stream = stack.enter_context(open(path, "rb", opener=_open_read))
            fd = stream.fileno()
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"{role} must be a regular PNG file.")
            if artifact_version(info) != expected[role]:
                raise FileExistsError(
                    f"{role} differs from its recorded version: {path}"
                )
            fds[role] = fd

        kernel = _kernel()
        folder.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=folder, prefix=".image-difference-") as temporary:
            result = _Result()
            error = ct.create_string_buffer(512)
            callback = (
                _PROGRESS(update)
                if progress is not None or cancelled is not None
                else _PROGRESS()
            )
            status = kernel(
                fds["left"],
                fds["right"],
                fds.get("reference", -1),
                temporary.fileno(),
                bounds,
                ct.byref(result),
                callback,
                error,
                len(error),
            )
            if callback_error is not None:
                raise callback_error
            if status:
                raise ValueError(
                    error.value.decode(errors="replace") or "PNG comparison failed."
                )

            channels = ["Y"] if result.channels == 1 else ["R", "G", "B"]
            roi = dict(
                zip(
                    _FIELDS,
                    rectangle or (0, 0, result.width, result.height),
                    strict=True,
                )
            )
            report = {
                "inputs": {
                    name: {"path": str(path), "version": expected[name]}
                    for name, path in inputs.items()
                },
                "width": result.width,
                "height": result.height,
                "region": roi,
                "pixels": roi["width"] * roi["height"],
                "bitDepth": result.depth,
                "channels": channels,
                "units": "stored integer sample levels",
                "leftVsRight": result.pair[0].report(channels),
                "outputs": {"absoluteDifference": str(destination)},
            }
            if reference is not None:
                report.update(
                    leftVsReference=result.pair[1].report(channels),
                    rightVsReference=result.pair[2].report(channels),
                    closerPixels={
                        "left": result.closer[0],
                        "right": result.closer[1],
                        "tie": result.closer[2],
                        "criterion": "sumSquaredChannelError",
                    },
                )
            check()
            for role, fd in fds.items():
                if (
                    artifact_version(os.fstat(fd)) != expected[role]
                    or artifact_version(inputs[role].stat()) != expected[role]
                ):
                    raise FileExistsError(
                        f"{role} changed during comparison: {inputs[role]}"
                    )
            # One same-directory, no-overwrite publication after all checks succeed.
            os.link(temporary.name, destination)
            return report

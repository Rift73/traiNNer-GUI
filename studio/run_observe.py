"""Bounded, CPU-only reads of trainer logs and checkpoint artifacts."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlencode

from .yaml_edit import MAX_RECIPE_BYTES

# The captured console includes startup errors. The engine file log preserves
# unwrapped progress and validation lines even when Rich wraps the console.
TAIL_BYTES = 128 * 1024
LOG_LINES = 160
LOG_PREFIX = (r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:,\d+)?"
              r"|\d\d-\d\d-\d{4} \d\d:\d\d [AP]M \|) INFO: ")
NEO_UPDATE = re.compile(LOG_PREFIX + r"\[ epoch:\s*\d+ \] \[ iter:\s*([\d,]+) \]")
NEO_START = re.compile(LOG_PREFIX + r"Starting at epoch \d+, update ([\d,]+)\.")
NEO_END = re.compile(LOG_PREFIX + r"End of training at update ([\d,]+)\.")
ITERATION = re.compile(
    r"\[epoch:\s*[\d,]+, iter:\s*([\d,]+),|"
    r"(?:Start training from|Resuming training from|for) epoch: \d+, iter: (\d+)\."
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SCALAR = re.compile(r" (l_[_\w]+): (" + NUMBER + r")(?=\s|$)")
SPEED = re.compile(r"\[\s*performance:\s*(" + NUMBER + r") it/s(?:, data (\d+)%)?\s*\]")
DATA_WAIT = re.compile(r"\[data wait:\s*(" + NUMBER + r")%\]")
WALL_SPEED = re.compile(r"\[wall:\s*(" + NUMBER + r") updates/s\]")
ETA = re.compile(r"\[\s*eta: ((?:-?\d{1,9} days?, )?\d{1,2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s*\]")
VRAM = re.compile(
    r"\[peak VRAM:\s*(" + NUMBER + r") GB allocated,\s*(" + NUMBER + r") GB reserved\]"
)
NEO_VRAM = re.compile(r"\[ mem: (" + NUMBER + r") / (" + NUMBER + r") GiB \]")
CURSOR = re.compile(r"([0-9a-f]{24})\.([0-9a-f]{1,16})\.([0-9a-f]{24})")
VALIDATION = re.compile(LOG_PREFIX + r"Validation (.+)")
METRIC = re.compile(r"^\s*# (.+?):\s*(" + NUMBER + r")(?=\s|$|\.{8} Best:)")
ESCAPE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~])")

ARTIFACT_FOLDERS = {
    "models": {".pth", ".safetensors"},
    "models/resume_models": {".pth", ".safetensors"},
    "training_states": {".state"},
}


def files(folder: Path) -> Iterator[Path]:
    if folder.resolve() != folder:
        return
    try:
        yield from folder.iterdir()
    except (FileNotFoundError, NotADirectoryError):
        return


def regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.resolve() == path


def mtime_ns(path: Path | None) -> int:
    try:
        return path.stat().st_mtime_ns if path is not None else 0
    except FileNotFoundError:
        return 0


def engine_log(experiment: Path, since: int = 0) -> Path | None:
    newest = None
    newest_key = (since - 1, "")
    for path in files(experiment):
        if (
            path.name.startswith("train_")
            and path.suffix == ".log"
            and regular_file(path)
        ):
            key = (mtime_ns(path), path.name)
            if key > newest_key:
                newest, newest_key = path, key
    return newest


def cli_recipe(experiment: Path) -> Path | None:
    return max(
        (
            path
            for path in files(experiment)
            if path.suffix in {".yml", ".yaml"} and regular_file(path)
        ),
        key=mtime_ns,
        default=None,
    )


def recipe_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_RECIPE_BYTES + 1)
        return data.decode("utf-8") if len(data) <= MAX_RECIPE_BYTES else None
    except (FileNotFoundError, UnicodeError):
        return None


def _decode_lines(raw: bytes) -> list[str]:
    return ESCAPE.sub("", raw.decode("utf-8", errors="replace")).splitlines()


def tail(path: Path | None, *, complete: bool = False) -> list[str]:
    if path is None:
        return []
    try:
        with path.open("rb") as stream:
            offset = max(0, stream.seek(0, 2) - TAIL_BYTES)
            stream.seek(offset)
            raw = stream.read(TAIL_BYTES)
    except FileNotFoundError:
        return []
    if offset:
        raw = raw.partition(b"\n")[2]
    if complete:
        raw = raw[: raw.rfind(b"\n") + 1]
    return _decode_lines(raw)


def _stamp(stream: BinaryIO, offset: int) -> str:
    # Inodes may survive truncate/rewrite. Witness the consumed prefix and
    # the boundary closest to the cursor to catch a log that already regrew.
    digest = hashlib.blake2s(digest_size=12)
    count = min(256, offset)
    stream.seek(0)
    digest.update(stream.read(count))
    stream.seek(offset - count)
    digest.update(stream.read(count))
    return digest.hexdigest()


def _scope_tag(scope: str, path: Path | None, identity: str) -> str:
    key = scope.encode() + b"\0" + (os.fsencode(path) if path is not None else b"")
    return hashlib.blake2s(key + b"\0" + identity.encode(), digest_size=12).hexdigest()


def _window(
    stream: BinaryIO, start: int, size: int, *, final: bool
) -> tuple[list[str], int]:
    stream.seek(start)
    raw = stream.read(size - start)
    if start:
        stream.seek(start - 1)
        if stream.read(1) != b"\n":
            cut = raw.find(b"\n") + 1
            if not cut:
                return [], start
            raw = raw[cut:]
            start += cut
    if not final:
        raw = raw[: raw.rfind(b"\n") + 1]
    return _decode_lines(raw), start + len(raw)


def log_page(
    path: Path | None, scope: str, cursor: str | None = None, *, final: bool = False
) -> dict[str, Any]:
    if cursor is not None and (
        not isinstance(cursor, str)
        or len(cursor) > 66
        or CURSOR.fullmatch(cursor) is None
    ):
        raise ValueError("Invalid log cursor.")

    if path is not None:
        try:
            with path.open("rb", buffering=0) as stream:
                return _log_page(stream, path, scope, cursor, final=final)
        except (FileNotFoundError, NotADirectoryError):
            pass
    tag = _scope_tag(scope, path, "missing")
    empty = hashlib.blake2s(digest_size=12).hexdigest()
    next_cursor = f"{tag}.0.{empty}"
    return {"log": [], "logCursor": next_cursor, "logReset": cursor != next_cursor}


def _log_page(
    stream: BinaryIO, path: Path, scope: str, cursor: str | None, *, final: bool
) -> dict[str, Any]:
    stat = os.fstat(stream.fileno())
    size = stat.st_size
    tag = _scope_tag(scope, path, f"{stat.st_dev}:{stat.st_ino}")
    previous_tag, previous_offset, stamp = (
        cursor.split(".") if cursor is not None else ("", "0", "")
    )
    offset = int(previous_offset, 16)
    reset = (
        previous_tag != tag
        or offset > size
        or size - offset > TAIL_BYTES
        or _stamp(stream, offset) != stamp
    )
    if reset:
        offset = max(0, size - TAIL_BYTES)
    lines, consumed = _window(stream, offset, size, final=final)

    # A same-inode rewrite can also land between cursor validation and the
    # read. Rebuild a bounded tail rather than joining two log generations.
    if not reset and _stamp(stream, offset) != stamp:
        reset = True
        size = os.fstat(stream.fileno()).st_size
        lines, consumed = _window(stream, max(0, size - TAIL_BYTES), size, final=final)

    next_cursor = f"{tag}.{consumed:x}.{_stamp(stream, consumed)}"
    return {
        "log": lines[-LOG_LINES:],
        "logCursor": next_cursor,
        "logReset": reset or len(lines) > LOG_LINES,
    }


def _telemetry(line: str, iteration: int) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if match := SPEED.search(line):
        speed = float(match[1])
        if math.isfinite(speed):
            report["iterationsPerSecond"] = speed
        if match[2] is not None and 0 <= int(match[2]) <= 100:
            report["dataWaitPercent"] = int(match[2])
    if match := DATA_WAIT.search(line):
        wait = float(match[1])
        if 0 <= wait <= 100:
            report["dataWaitPercent"] = wait
    if match := WALL_SPEED.search(line):
        rate = float(match[1])
        if math.isfinite(rate):
            report["wallUpdatesPerSecond"] = rate
    if match := ETA.search(line):
        report["eta"] = match[1]
    if match := VRAM.search(line):
        allocated, reserved = float(match[1]), float(match[2])
        if math.isfinite(allocated) and math.isfinite(reserved):
            report["vram"] = {"peakAllocatedGB": allocated, "peakReservedGB": reserved}
    elif match := NEO_VRAM.search(line):
        allocated, capacity = float(match[1]), float(match[2])
        if math.isfinite(allocated) and math.isfinite(capacity):
            report["vram"] = {"peakAllocatedGB": allocated, "deviceTotalGB": capacity}
    if report:
        report["telemetryIteration"] = iteration
    return report


def progress(lines: list[str], *, metrics: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scalars: dict[str, float] = {}
    telemetry: dict[str, Any] = {}
    dataset = None
    for line in lines:
        if match := ITERATION.search(line) or NEO_UPDATE.match(line):
            iteration = int((match[1] or match[2]).replace(",", ""))
            result["iteration"] = iteration
            if match[1]:
                telemetry = _telemetry(line, iteration)
            elif not match[0].startswith("for "):
                telemetry = {}
            if metrics and match[1]:
                for name, raw in SCALAR.findall(line):
                    value = float(raw)
                    if math.isfinite(value):
                        scalars[name] = value
        elif match := NEO_START.match(line) or NEO_END.match(line):
            result["iteration"] = int(match[1].replace(",", ""))
        if not metrics or not line.strip():
            continue
        if match := VALIDATION.match(line):
            dataset = match[1].strip()
        elif dataset and (match := METRIC.match(line)):
            value = float(match[2])
            if math.isfinite(value):
                scalars[f"{dataset}/{match[1].strip()}"] = value
        else:
            dataset = None
    if scalars:
        result["metrics"] = scalars
    return {**result, **telemetry}


@lru_cache(maxsize=96)
def _report(path: Path, version: tuple[int, int, int, int, int]) -> dict[str, Any]:
    # File versions contain only metadata. Keep the small parsed summary and
    # avoid rereading the engine log when a poll sees the same version.
    return progress(tail(path, complete=True), metrics=True)


def report(path: Path | None, *, metrics: bool = False) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    fields = _report(
        path,
        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    )
    return {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in fields.items()
        if metrics or key != "metrics"
    }


def artifact_path(experiment: Path, key: str) -> Path:
    if not isinstance(key, str) or "\x00" in key:
        raise FileNotFoundError("Artifact was not found.")
    relative = Path(key)
    allowed = ARTIFACT_FOLDERS.get(relative.parent.as_posix(), ())
    if (
        key != relative.as_posix()
        or relative.name.startswith(".")
        or relative.suffix not in allowed
    ):
        raise FileNotFoundError("Artifact was not found.")
    target = experiment / relative
    if not regular_file(target):
        raise FileNotFoundError("Artifact was not found.")
    return target


def artifacts(experiment: Path, run_id: str) -> list[dict[str, str]]:
    found = [
        path
        for folder, suffixes in ARTIFACT_FOLDERS.items()
        for path in files(experiment / folder)
        if path.suffix in suffixes
        and not path.name.startswith(".")
        and regular_file(path)
    ]
    indexed = []
    for path in sorted(found, key=mtime_ns, reverse=True)[:100]:
        key = path.relative_to(experiment).as_posix()
        query = urlencode({"id": run_id, "path": key})
        indexed.append(
            {"name": path.name, "path": key, "url": f"/api/run/artifact?{query}"}
        )
    return indexed

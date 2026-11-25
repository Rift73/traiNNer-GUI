"""Index TensorBoard scalar history from individual writers or a complete run."""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .native import executable

_MAX_RECORD = 64 * 1024**2
_MAX_RECORDS = 65536


@lru_cache(maxsize=1)
def _reader():
    library = ctypes.CDLL(str(executable("tfrecord", "records.so", source="records.cpp")))
    fn = library.index_records
    index = ctypes.POINTER(ctypes.c_size_t)
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, index, ctypes.c_size_t, index,
                   ctypes.POINTER(ctypes.c_uint64)]
    fn.restype = ctypes.c_int
    return fn


def sources(experiment: Path) -> list[Path]:
    folders = [experiment]
    if experiment.parent.name == "experiments":
        folders.append(experiment.parent.parent / "tb_logger" / experiment.name)
    files = []
    for folder in folders:
        if not folder.is_dir() or folder.resolve() != folder:
            continue
        files.extend(p for p in folder.glob("events.out.tfevents.*") if p.is_file() and not p.is_symlink())
    return sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)


def _scalar(value):
    kind = value.WhichOneof("value")
    if kind == "simple_value":
        return value.simple_value
    if kind != "tensor":
        return None
    tensor = value.tensor
    if value.metadata.plugin_data.plugin_name not in ("", "scalars"):
        return None
    if tensor.tensor_shape.unknown_rank or tensor.tensor_shape.dim:
        return None
    from tensorboard.compat.proto import types_pb2 as dt
    if tensor.dtype not in (dt.DT_FLOAT, dt.DT_HALF, dt.DT_BFLOAT16, dt.DT_DOUBLE, dt.DT_INT32,
                            dt.DT_INT64, dt.DT_INT8, dt.DT_INT16, dt.DT_UINT8, dt.DT_UINT16,
                            dt.DT_UINT32, dt.DT_UINT64, dt.DT_BOOL):
        return None
    from tensorboard.util.tensor_util import make_ndarray
    return float(make_ndarray(tensor).item())


def consume(db, stream, start: int, end: int, budget: int, *, first=0, last=2**63 - 1) -> tuple[int, bool]:
    from google.protobuf.message import DecodeError
    from tensorboard.compat.proto import event_pb2

    stream.seek(start)
    raw = stream.read(min(budget, end - start))
    offsets = (ctypes.c_size_t * (min(len(raw) // 16, _MAX_RECORDS) + 1))()
    count, needed = ctypes.c_size_t(), ctypes.c_uint64()
    while True:
        code = _reader()(raw, len(raw), offsets, len(offsets),
                         ctypes.byref(count), ctypes.byref(needed))
        if code:
            what = {1: "length checksum", 2: "data checksum", 3: "record length"}.get(code, "record")
            raise ValueError(f"Invalid TensorBoard {what} at byte {start + offsets[count.value]}.")
        if count.value or len(raw) < 12 or needed.value <= len(raw) or needed.value > end - start:
            break
        if needed.value > _MAX_RECORD:
            raise ValueError(f"TensorBoard record at byte {start} exceeds this scalar viewer's 64 MiB per-record limit.")
        stream.seek(start)
        raw = stream.read(needed.value)
        # A record larger than a page is processed on its own.
        offsets = (ctypes.c_size_t * 2)()
        if len(raw) != needed.value:
            return start, False

    view = memoryview(raw)
    points = []
    def flush():
        db.executemany("INSERT INTO points(metric,step,value) VALUES(?,?,?) ON CONFLICT(metric,step) "
                       "DO UPDATE SET value=excluded.value", points)
        points.clear()

    for i in range(count.value):
        pos = offsets[i]
        try:
            event = event_pb2.Event.FromString(view[pos+12:offsets[i+1]-4])
        except DecodeError as exc:
            raise ValueError(f"Invalid TensorBoard Event at byte {start + pos}.") from exc
        if event.step < 0 or event.step > last:
            continue
        if event.HasField("session_log") and event.session_log.status == event_pb2.SessionLog.START:
            flush()
            db.execute("DELETE FROM points WHERE step>=?", (max(first, event.step),))
        if event.step < first:
            continue
        for value in event.summary.value:
            number = _scalar(value)
            if number is not None and value.tag:
                points.append((value.tag, event.step, number if math.isfinite(number) else None))
    flush()
    consumed = start + offsets[count.value]
    remaining = end - consumed
    more = bool(remaining and count.value and (
        count.value == len(offsets) - 1 or needed.value <= remaining
    ))
    return consumed, more


@dataclass(frozen=True)
class Writer:
    path: Path
    first: int = 0
    last: int = 2**63 - 1


def writer_order(path):
    # TensorBoard names writers by creation time. Copying an archive changes its mtime.
    stamp = re.match(r"events\.out\.tfevents\.(\d+)(?:\.|$)", path.name)
    natural = tuple((0, int(part)) if part.isdigit() else (1, part)
                    for part in re.split(r"(\d+)", path.name))
    return (int(stamp[1]) if stamp else path.stat().st_mtime, natural)


def folder_writers(folder):
    if not folder.is_dir():
        raise ValueError("Choose the TensorBoard folder for one run.")
    paths = (p for p in folder.glob("events.out.tfevents.*") if p.is_file() and not p.is_symlink())
    return [Writer(p) for p in sorted(paths, key=writer_order)]


def combined_history(writers, identity, state, source, metric, budget, from_step, to_step, *, validation=None):
    from .history import index_file, indexed_history, plot_bounds

    bounds = plot_bounds(budget, from_step, to_step)
    files = [str(writer.path) for writer in writers]
    reply = {"sources": files, "sourceFormat": "tensorboard", "writerCount": len(files)}
    if source:
        if source not in files:
            raise ValueError("The selected event file does not belong to this history.")
        meta, index = index_file(state, Path(source), f"TensorBoard/experiment/{Path(source).name}")
        return indexed_history({**reply, **meta}, index, metric, budget, bounds)
    if not writers:
        raise ValueError("This folder contains no TensorBoard event files. Choose the individual run folder inside tb_logger.")

    folder = state.resolve() / "history"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(b"tensorboard-run-v1\0" + os.fsencode(identity)).hexdigest()
    index = folder / (digest + ".sqlite3")
    with closing(sqlite3.connect(index, timeout=15)) as db:
        indexed, size, more = _refresh_writers(db, writers)
    label = f"Full history ({len(files)} event file{'s' if len(files) != 1 else ''})"
    reply.update(source=label, indexedBytes=indexed, sourceBytes=size, more=more)
    return indexed_history(reply, index, metric, budget, bounds, validation=validation)


def _refresh_writers(db, writers):
    from .history import BUDGET, _stamp

    db.executescript("""
        CREATE TABLE IF NOT EXISTS writers (
            ordinal INTEGER PRIMARY KEY, path TEXT, first INTEGER, last INTEGER, device INTEGER,
            inode INTEGER, size INTEGER, modified INTEGER, offset INTEGER, stamp TEXT, done INTEGER);
        CREATE TABLE IF NOT EXISTS points (
            metric TEXT NOT NULL, step INTEGER NOT NULL, value REAL, PRIMARY KEY(metric,step)
        ) WITHOUT ROWID;
    """)
    with db:
        db.execute("BEGIN IMMEDIATE")
        cached = list(db.execute("SELECT path,first,last,device,inode,size,modified,offset,stamp,done "
                                 "FROM writers ORDER BY ordinal"))
        stats = [w.path.stat() for w in writers]
        signatures = [(str(w.path), w.first, w.last, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                      for w, st in zip(writers, stats, strict=True)]
        # Appending to the final indexed writer is incremental. A replaced, reordered or
        # edited earlier writer needs replay because later records can overwrite or purge it.
        rebuild = len(cached) > len(writers)
        for i, (row, sig) in enumerate(zip(cached, signatures)):
            if row[:7] == sig:
                continue
            if i == len(cached) - 1 and row[:5] == sig[:5] and row[5] < sig[5]:
                with writers[i].path.open("rb") as stream:
                    if _stamp(stream, row[7]) == row[8]:
                        continue
            rebuild = True
            break
        if rebuild:
            db.execute("DELETE FROM points")
            db.execute("DELETE FROM writers")
            cached = []

        consumed = indexed = 0
        more = False
        for i, (writer, sig) in enumerate(zip(writers, signatures, strict=True)):
            old = cached[i] if i < len(cached) else None
            offset = old[7] if old else 0
            indexed += offset
            if old and old[:7] == sig and old[9]:
                continue
            if consumed >= BUDGET:
                more = True
                continue
            with writer.path.open("rb", buffering=0) as stream:
                end, pending = consume(db, stream, offset, sig[5], BUDGET,
                                       first=writer.first, last=writer.last)
                stamp = _stamp(stream, end)
            db.execute("INSERT OR REPLACE INTO writers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (i, *sig, end, stamp, int(not pending)))
            consumed += end - offset
            indexed += end - offset
            if pending:
                more = True
                break
        return indexed, sum(st.st_size for st in stats), more

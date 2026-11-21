"""Index scalar summaries from one TensorBoard writer, without loading TensorFlow."""
from __future__ import annotations

import ctypes
import math
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


def consume(db, stream, start: int, end: int, budget: int) -> tuple[int, bool]:
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
        if event.step < 0:
            continue
        if event.HasField("session_log") and event.session_log.status == event_pb2.SessionLog.START:
            flush()
            db.execute("DELETE FROM points WHERE step>=?", (event.step,))
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

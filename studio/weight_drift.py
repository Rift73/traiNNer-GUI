"""Stream comparable checkpoint values through LLVM without importing PyTorch."""

from __future__ import annotations

import ctypes
import heapq
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path

from safetensors import SafetensorError, safe_open

from .native import executable

# Numeric summaries intentionally exclude discrete counters and packed types.
_TYPES = {"F16": (0, 2), "BF16": (1, 2), "F32": (2, 4)}
_CHUNK = 262_144


class _Measures(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint64) for k in ("finite", "changed", "left_bad", "right_bad")] + [
        (k, ctypes.c_double) for k in ("absolute", "squared", "left_sq", "right_sq", "cross", "maximum")
    ]

    def add(self, other):
        for key, _ in self._fields_[:-1]:
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.maximum = max(self.maximum, other.maximum)

    def report(self):
        return {
            "finitePairs": self.finite, "changedFinite": self.changed,
            "leftNonfinite": self.left_bad, "rightNonfinite": self.right_bad,
            "meanAbsoluteDelta": self.absolute / self.finite if self.finite else None,
            "rmsDelta": math.sqrt(self.squared / self.finite) if self.finite else None,
            "maximumAbsoluteDelta": self.maximum if self.finite else None,
            "relativeL2": math.sqrt(self.squared / self.left_sq) if self.left_sq else None,
            "cosine": max(-1.0, min(1.0, self.cross / (math.sqrt(self.left_sq) * math.sqrt(self.right_sq))))
            if self.left_sq and self.right_sq else None,
        }


def _stat(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _header(stream):
    # Validate dtype, shape, offsets and exact file size using the installed
    # safetensors reader on the same open inode, then recover its byte ranges.
    try:
        with safe_open(f"/proc/self/fd/{stream.fileno()}", framework="numpy", device="cpu") as weights:
            count = len(weights.keys())
    except SafetensorError as exc:
        raise ValueError(f"Invalid safetensors checkpoint: {exc}") from exc
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("Checkpoint changed during inspection.")
    size = struct.unpack("<Q", raw)[0]
    if size > 100_000_000:
        raise ValueError("Checkpoint header exceeds the safetensors size limit.")
    header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    if len(header) != count:
        raise ValueError("Checkpoint header changed during inspection.")
    return 8 + size, header


def measure_drift(left: Path, right: Path, report: Path, *, cancel=None, progress=None):
    """Write all tensor outcomes to JSONL and return a bounded numerical summary."""
    dll = ctypes.CDLL(str(executable("weights", "measure.so", source="measure.c")))
    compare = dll.measure_values
    compare.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                        ctypes.c_size_t, ctypes.POINTER(_Measures)]
    compare.restype = ctypes.c_int
    counts = Counter()
    total = _Measures()
    biggest = []
    total_bytes = 0
    with left.open("rb") as lhs, right.open("rb") as rhs, report.open("x", encoding="utf-8") as rows:
        versions = [_stat(os.fstat(stream.fileno())) for stream in (lhs, rhs)]
        left_start, a = _header(lhs)
        right_start, b = _header(rhs)
        names = sorted(a.keys() | b.keys())
        for index, name in enumerate(names):
            if cancel:
                cancel()
            first, second = a.get(name), b.get(name)
            row = {"name": name}
            if first is None:
                status = "rightOnly"
            elif second is None:
                status = "leftOnly"
            elif first["shape"] != second["shape"]:
                status = "differentShape"
            elif first["dtype"] not in _TYPES or second["dtype"] not in _TYPES:
                status = "nonTrainingFloatOrOtherType"
            else:
                status = "measured"
                lt, ls = _TYPES[first["dtype"]]
                rt, rs = _TYPES[second["dtype"]]
                elements = math.prod(first["shape"])
                lhs.seek(left_start + first["data_offsets"][0])
                rhs.seek(right_start + second["data_offsets"][0])
                tensor = _Measures()
                for start in range(0, elements, _CHUNK):
                    if cancel:
                        cancel()
                    n = min(_CHUNK, elements - start)
                    x, y = lhs.read(n * ls), rhs.read(n * rs)
                    if len(x) != n * ls or len(y) != n * rs:
                        raise ValueError("A checkpoint changed during comparison.")
                    chunk = _Measures()
                    if compare(x, lt, y, rt, n, ctypes.byref(chunk)):
                        raise RuntimeError("The native checkpoint reader rejected the tensor type.")
                    tensor.add(chunk)
                    total_bytes += len(x) + len(y)
                total.add(tensor)
                row.update(tensor.report())
                if tensor.changed:
                    item = (row["rmsDelta"], name, row)
                    if len(biggest) < 24:
                        heapq.heappush(biggest, item)
                    elif item[:2] > biggest[0][:2]:
                        heapq.heapreplace(biggest, item)
            counts[status] += 1
            for side, value in (("left", first), ("right", second)):
                if value is not None:
                    row[side] = {"shape": value["shape"], "dtype": value["dtype"]}
            row["status"] = status
            rows.write(json.dumps(row, allow_nan=False, ensure_ascii=True) + "\n")
            if progress:
                progress({"stage":"comparing", "completedTensors":index+1, "totalTensors":len(names),
                          "payloadBytesRead":total_bytes})
        for path, stream, version in zip((left, right), (lhs, rhs), versions, strict=True):
            if _stat(os.fstat(stream.fileno())) != version or _stat(path.stat()) != version:
                raise ValueError(f"{path.name} changed during comparison. Select completed checkpoints.")
    return {"left":str(left), "right":str(right), "tensors":dict(counts), "numerical":total.report(),
            "largestRmsChanges":[entry[-1] for entry in sorted(biggest, reverse=True)],
            "payloadBytesRead":total_bytes}

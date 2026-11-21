"""Bounded parallel writes of a serialized Linux file to a WSL Windows drive."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

_CHUNK = 1 << 20
_WORKERS = 8


def copy_checkpoint(source: int, destination: int) -> None:
    size = os.fstat(source).st_size
    workers = min(_WORKERS, (size + _CHUNK - 1) // _CHUNK)

    def copy_stripe(index: int) -> None:
        first = index * _CHUNK
        buffer = memoryview(bytearray(min(_CHUNK, size - first)))
        for start in range(first, size, workers * _CHUNK):
            end = min(start + _CHUNK, size)
            offset = start
            while offset < end:
                count = os.preadv(source, [buffer[:end - offset]], offset)
                if not count:
                    raise OSError("Checkpoint staging file ended during transfer.")
                block = buffer[:count]
                while block:
                    written = os.pwrite(destination, block, offset)
                    if not written:
                        raise OSError("Checkpoint write made no progress.")
                    offset += written
                    block = block[written:]

    if workers == 0:
        return
    if workers == 1:
        copy_stripe(0)
        return
    # Independent offsets let DrvFS overlap requests while the source stays on Linux.
    # Only one chunk per worker is in flight. All workers finish before the caller fsyncs.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(copy_stripe, range(workers)):
            pass

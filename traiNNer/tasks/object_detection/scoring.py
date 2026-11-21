"""Streaming AP50 with native and Torch CPU per-image matchers."""

import ctypes
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor

from traiNNer.tasks.object_detection.metrics import ReferenceScorer, metric_totals
from traiNNer.utils.types import DetectionTarget


_LIBRARY = Path(__file__).resolve().parents[3] / "native/studio/detection/build/clang/detection.so"


class _NativeScorer:
    def __init__(self, lib: ctypes.CDLL, names: Sequence[str]) -> None:
        self.lib = lib
        self.names = names
        ptr, size = ctypes.c_void_p, ctypes.c_size_t
        lib.detection_create.argtypes = [size]
        lib.detection_create.restype = ptr
        lib.detection_free.argtypes = [ptr]
        lib.detection_free.restype = None
        lib.detection_add.argtypes = [ptr, ctypes.c_int, ptr, ptr, ptr, size, ptr, ptr, ptr, size]
        lib.detection_add.restype = ctypes.c_int
        lib.detection_finish.argtypes = [ptr, size, ptr, ptr, ptr, ptr]
        lib.detection_finish.restype = ctypes.c_int
        self.state = lib.detection_create(len(names))
        if not self.state:
            raise MemoryError("Could not allocate the detection evaluator.")

    def add(self, prediction: dict[str, Tensor], target: DetectionTarget) -> None:
        # checked_detections has validated predictions. The dataset owns float32
        # target geometry. Float64 detector outputs retain their full precision.
        boxes = prediction["boxes"]
        wide = boxes.dtype == torch.float64
        boxes = boxes.to(dtype=torch.float64 if wide else torch.float32).contiguous()
        scores = prediction["scores"].double().contiguous()
        labels = prediction["labels"].contiguous()
        truth = target["boxes"].contiguous()
        classes = target["labels"].contiguous()
        ignore = target["ignore"].contiguous()
        if (labels.dtype != torch.int64 or classes.dtype != torch.int64
                or truth.dtype != torch.float32 or ignore.dtype != torch.bool
                or labels.shape != (len(boxes),) or scores.shape != labels.shape
                or boxes.shape != (len(labels), 4) or truth.shape != (len(classes), 4)
                or classes.ndim != 1 or ignore.shape != classes.shape
                or any(t.device.type != "cpu" for t in (boxes, scores, labels, truth, classes, ignore))):
            raise ValueError("Invalid CPU detection metric inputs.")
        status = self.lib.detection_add(
            self.state, wide, boxes.data_ptr(), scores.data_ptr(), labels.data_ptr(), len(labels),
            truth.data_ptr(), classes.data_ptr(), ignore.data_ptr(), len(classes),
        )
        if status == 1:
            raise ValueError("Invalid detection metric class or confidence.")
        if status:
            raise MemoryError("Could not accumulate detection metrics.")

    def result(self) -> dict[str, float]:
        n = len(self.names)
        ap = (ctypes.c_double * n)()
        counts = [(ctypes.c_uint64 * n)() for _ in range(3)]
        if self.lib.detection_finish(self.state, n, ap, *counts):
            raise RuntimeError("Could not finish detection metrics.")
        total, tp, fp = counts
        self.support = total
        return metric_totals(self.names, ap, total, tp, fp)

    def groups(self, category: int) -> Iterator[tuple[float, int, int]]:
        """Read the ranked matches sorted by detection_finish, a page at a time."""
        read = self.lib.detection_score_groups
        size = ctypes.c_size_t
        ptr = ctypes.c_void_p
        read.argtypes = [ptr, size, ctypes.POINTER(size), size, ptr, ptr, ptr, ctypes.POINTER(size)]
        read.restype = ctypes.c_int
        capacity = 64
        scores = (ctypes.c_double * capacity)()
        tp = (ctypes.c_uint64 * capacity)()
        fp = (ctypes.c_uint64 * capacity)()
        offset, written = size(), size()
        while True:
            if read(self.state, category, ctypes.byref(offset), capacity, scores, tp, fp, ctypes.byref(written)):
                raise RuntimeError("Could not read ranked detection groups.")
            for i in range(written.value):
                yield scores[i], tp[i], fp[i]
            if written.value < capacity:
                return


@contextmanager
def scorer(names: Sequence[str], *, operating_points: bool = False) -> Iterator[_NativeScorer | ReferenceScorer]:
    lib = ctypes.CDLL(str(_LIBRARY)) if _LIBRARY.is_file() else None
    # Older local builds support AP50 but cannot export their ranked matches.
    if lib is None or (operating_points and not hasattr(lib, "detection_score_groups")):
        yield ReferenceScorer(names)
        return
    native = _NativeScorer(lib, names)
    try:
        yield native
    finally:
        native.lib.detection_free(native.state)

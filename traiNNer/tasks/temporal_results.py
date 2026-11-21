"""Reference-relative changes between adjacent, numbered holdout frames."""

import ctypes
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile

from msgspec.json import encode
from torch import Tensor
import torch

_NUMBERED = re.compile(r"(.*?)([0-9]+)$")


def _address(path: str):
    file = Path(path)
    match = _NUMBERED.fullmatch(file.stem)
    return (file.parent, match[1], int(match[2])) if match else None


def _next(left, right):
    return left is not None and right is not None and left[:2] == right[:2] and left[2] + 1 == right[2]


class TemporalResults:
    def __init__(self, directory: Path, library: Path):
        self.directory = directory
        self.outputs: dict[str, Path] = {}
        self.summaries: dict[str, dict] = {}
        self.reduce = ctypes.CDLL(str(library)).trainner_temporal_change
        pointer = ctypes.c_void_p
        self.reduce.argtypes = [pointer, pointer, pointer, pointer, pointer, pointer, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_uint64)]
        self.reduce.restype = ctypes.c_int

    def update_history(self, history: Tensor, prediction: Tensor, target: Tensor, compare: bool) -> int:
        pred = prediction.detach().float().resolve_neg()
        gt = target.detach().float().resolve_neg()
        triple = ctypes.c_int64 * 3
        total = ctypes.c_uint64()
        rc = self.reduce(pred.data_ptr(), triple(*pred.stride()[1:]), gt.data_ptr(), triple(*gt.stride()[1:]),
                         triple(*gt.shape[1:]), history.data_ptr(), compare, ctypes.byref(total))
        if rc == 2:
            raise ValueError("Temporal error cannot be measured on nonfinite predictions or references.")
        if rc:
            raise ValueError("Temporal report encountered an invalid image layout.")
        return total.value

    @contextmanager
    def dataset(self, name: str):
        if name in self.outputs:
            raise ValueError(f"Temporal results already exist for split {name!r}.")
        self.directory.mkdir(parents=True, exist_ok=True)
        output = self.directory / f"split-{len(self.outputs) + 1}.jsonl"
        previous = None
        history = None
        frames = pairs = total_error = elements = 0
        exclusions = Counter()
        largest = None
        temporary = NamedTemporaryFile(dir=self.directory, suffix=".partial", delete=False)

        def append(batch, prediction: Tensor, target: Tensor | None):
            nonlocal previous, history, frames, pairs, total_error, elements, largest
            if target is None or target.shape != prediction.shape or target.ndim != 4 or target.size(0) != 1:
                raise ValueError("Temporal evaluation needs one aligned prediction and ground-truth centre frame per sample.")
            context = tuple(frame[0] for frame in batch.get("context_paths", ()))
            source, reference = batch["lq_path"][0], batch["gt_path"][0]
            if not context or context[len(context) // 2] != source:
                raise ValueError("The reported centre frame must match the evaluated sequence window.")
            source_key, target_key = _address(source), _address(reference)
            frame_keys = [_address(frame) for frame in context]
            dense = all(_next(a, b) for a, b in zip(frame_keys, frame_keys[1:]))
            reason = None
            if previous is not None:
                _, _, old_source, old_target, old_context, old_dense = previous
                if not _next(old_source, source_key) or not _next(old_target, target_key):
                    reason = "nonconsecutiveCentres"
                elif not dense or not old_dense or old_context[1:] != context[:-1]:
                    reason = "discontinuousWindow"
                elif history is None or history.shape != prediction.shape:
                    reason = "sizeChange"
            if prediction.device.type != "cpu" or target.device.type != "cpu":
                raise ValueError("Temporal reporting requires a CPU holdout.")
            if history is None or history.shape != prediction.shape:
                history = torch.empty(prediction.shape, dtype=torch.int16, device="cpu")
            compare = previous is not None and reason is None
            error_sum = self.update_history(history, prediction, target, compare)
            if compare:
                count = prediction.numel()
                mean = error_sum / (255 * count)
                row = {"previous": previous[0], "current": source, "previousTarget": previous[1],
                       "currentTarget": reference, "meanAbsoluteDeltaError": mean, "elements": count}
                temporary.write(encode(row) + b"\n")
                total_error += error_sum
                elements += count
                pairs += 1
                if largest is None or mean > largest["meanAbsoluteDeltaError"]:
                    largest = row
            elif reason is not None:
                exclusions[reason] += 1
            previous = (source, reference, source_key, target_key, context, dense)
            frames += 1

        try:
            with temporary:
                yield append
            Path(temporary.name).replace(output)
            self.outputs[name] = output
            self.summaries[name] = {
                "frames": frames, "scoredTransitions": pairs, "excludedTransitions": sum(exclusions.values()),
                "exclusions": dict(exclusions),
                "meanAbsoluteDeltaError": total_error / (255 * elements) if elements else None,
                "largest": largest,
            }
        finally:
            Path(temporary.name).unlink(missing_ok=True)


_active: ContextVar[TemporalResults | None] = ContextVar("temporal_holdout_report", default=None)


def recording() -> bool:
    return _active.get() is not None


@contextmanager
def record(directory: Path, library: Path) -> Iterator[TemporalResults]:
    result = TemporalResults(directory, library)
    token = _active.set(result)
    try:
        yield result
    finally:
        _active.reset(token)


@contextmanager
def dataset_sink(name: str):
    result = _active.get()
    if result is None:
        yield None
    else:
        with result.dataset(name) as append:
            yield append

"""Stream detections in COCO bbox results format during an explicit holdout job."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile

from msgspec.json import encode
from torch import Tensor


_ACTIVE: ContextVar["CocoResults | None"] = ContextVar("coco_result_export", default=None)


@contextmanager
def record(directory: Path) -> Iterator["CocoResults"]:
    results = CocoResults(directory)
    token = _ACTIVE.set(results)
    try:
        yield results
    finally:
        _ACTIVE.reset(token)


def dataset_sink(name: str, category_ids: Sequence[int]):
    results = _ACTIVE.get()
    return results.dataset(name, category_ids) if results is not None else nullcontext(None)


class CocoResults:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.outputs: dict[str, Path] = {}

    @contextmanager
    def dataset(self, name: str, category_ids: Sequence[int]) -> Iterator[Callable]:
        if name in self.outputs:
            raise ValueError(f"A detection export already exists for dataset {name!r}.")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Dataset names are arbitrary display strings. Each output uses a safe
        # ordinal and the job result publishes the corresponding dataset name.
        output = self.directory / f"split-{len(self.outputs) + 1}.json"
        temp = NamedTemporaryFile(dir=self.directory, suffix=".partial", delete=False)
        first = True

        def append(image_id: int, prediction: dict[str, Tensor]) -> None:
            nonlocal first
            records = []
            for (x1, y1, x2, y2), label, score in zip(
                prediction["boxes"].tolist(), prediction["labels"].tolist(),
                prediction["scores"].tolist(), strict=True,
            ):
                records.append({"image_id": image_id, "category_id": category_ids[label - 1],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            if records:
                if not first:
                    temp.write(b",")
                temp.write(encode(records)[1:-1])
                first = False

        try:
            with temp:
                temp.write(b"[")
                yield append
                temp.write(b"]\n")
            Path(temp.name).replace(output)
            self.outputs[name] = output
        finally:
            Path(temp.name).unlink(missing_ok=True)

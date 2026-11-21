"""Optional per-image decisions from the same pass as holdout accuracy."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile

from msgspec.json import encode
from torch import Tensor


class ClassificationResults:
    def __init__(self, directory: Path):
        self.directory = directory
        self.outputs: dict[str, Path] = {}

    @contextmanager
    def dataset(self, name: str, names: Sequence[str]):
        if name in self.outputs:
            raise ValueError(f"Classification results already exist for dataset {name!r}.")
        self.directory.mkdir(parents=True, exist_ok=True)
        output = self.directory / f"split-{len(self.outputs) + 1}.jsonl"
        temp = NamedTemporaryFile(dir=self.directory, suffix=".partial", delete=False)

        def append(paths: Sequence[str], targets: Tensor, logits: Tensor, ranked: Tensor):
            probabilities = logits.float().softmax(dim=1).gather(1, ranked)
            for path, target, choices, scores in zip(
                paths, targets.tolist(), ranked.tolist(), probabilities.tolist(), strict=True
            ):
                record = {
                    "input": str(path), "targetId": target, "target": names[target],
                    "correct": target == choices[0],
                    "top": [{"id": index, "label": names[index], "probability": probability}
                            for index, probability in zip(choices, scores, strict=True)],
                }
                temp.write(encode(record) + b"\n")

        try:
            with temp:
                yield append
            Path(temp.name).replace(output)
            self.outputs[name] = output
        finally:
            Path(temp.name).unlink(missing_ok=True)


_exports: ContextVar[ClassificationResults | None] = ContextVar("classification_exports", default=None)


@contextmanager
def record(directory: Path) -> Iterator[ClassificationResults]:
    results = ClassificationResults(directory)
    token = _exports.set(results)
    try:
        yield results
    finally:
        _exports.reset(token)


@contextmanager
def dataset_sink(name: str, class_names: Sequence[str]):
    results = _exports.get()
    if results is None:
        yield None
    else:
        with results.dataset(name, class_names) as sink:
            yield sink

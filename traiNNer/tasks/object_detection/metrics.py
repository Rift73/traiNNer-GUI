"""CPU reference for the trainer's score-ranked, single-IoU detector metrics."""

from collections.abc import Iterator, Sequence
from itertools import groupby
from operator import itemgetter

import numpy as np
from torch import Tensor
from torchvision.ops import box_iou

from traiNNer.utils.types import DetectionTarget

IOU_THRESHOLD = 0.5


def metric_totals(names: Sequence[str], aps: Sequence[float], support: Sequence[int],
                  tp: Sequence[int], fp: Sequence[int]) -> dict[str, float]:
    results = {f"AP50/{name}": aps[i] for i, name in enumerate(names) if support[i]}
    count = sum(support)
    if count:
        results["AP50"] = sum(results.values()) / len(results)
        results["recall50"] = sum(tp) / count
    results["false_positives50"] = float(sum(fp))
    results["targets"] = float(count)
    return results


class ReferenceScorer:
    """Match once per image. Only confidence and hit/miss survive to the next image."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = names
        self.support = [0] * len(names)
        self.ranked: list[list[tuple[float, bool]]] = [[] for _ in names]

    def add(self, prediction: dict[str, Tensor], target: DetectionTarget) -> None:
        for i, ranked in enumerate(self.ranked):
            same = target["labels"] == i + 1
            regular = target["boxes"][same & ~target["ignore"]]
            self.support[i] += len(regular)
            selected = prediction["labels"] == i + 1
            scores = prediction["scores"][selected]
            if not len(scores):
                continue
            boxes = prediction["boxes"][selected]
            ignored = target["boxes"][same & target["ignore"]]
            # A prediction can consume one ordinary box. Ignored boxes may
            # suppress any number of predictions with no remaining ordinary match.
            matched = np.zeros(len(regular), dtype=bool)
            order = scores.argsort(descending=True, stable=True)
            # Bound the overlap matrices to one image and a block of predictions.
            rows = max(1, 65536 // max(1, len(regular), len(ignored)))
            for indices in order.split(rows):
                batch = boxes[indices]
                overlaps = box_iou(batch, regular).numpy()
                skip = (box_iou(batch, ignored) >= IOU_THRESHOLD).any(dim=1).tolist()
                for score, ious, suppress in zip(scores[indices].tolist(), overlaps, skip, strict=True):
                    available = (ious >= IOU_THRESHOLD) & ~matched
                    if available.any():
                        matched[np.where(available, ious, -1).argmax()] = True
                        ranked.append((score, True))
                    elif not suppress:
                        ranked.append((score, False))

    def result(self) -> dict[str, float]:
        aps, tp, fp = [], [], []
        for ranked, support in zip(self.ranked, self.support, strict=True):
            ranked.sort(key=itemgetter(0), reverse=True)
            hits = np.fromiter((matched for _, matched in ranked), dtype=np.int64, count=len(ranked))
            count = int(hits.sum())
            tp.append(count)
            fp.append(len(hits) - count)
            ap = 0.0
            if support and len(hits):
                precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
                envelope = np.maximum.accumulate(precision[::-1])[::-1]
                ap = float(np.dot(envelope, hits) / support)
            aps.append(ap)
        return metric_totals(self.names, aps, self.support, tp, fp)

    def groups(self, category: int) -> Iterator[tuple[float, int, int]]:
        """Yield descending (score, new TP, new FP) groups after result()."""
        for score, tied in groupby(self.ranked[category], key=itemgetter(0)):
            count = tp = 0
            for _, matched in tied:
                count += 1
                tp += matched
            yield score, tp, count - tp


def detection_metrics(predictions: Sequence[dict[str, Tensor]], targets: Sequence[DetectionTarget],
                      class_names: Sequence[str]) -> dict[str, float]:
    """Macro AP50 over classes with targets, micro recall and FP count."""
    scorer = ReferenceScorer(class_names)
    for prediction, target in zip(predictions, targets, strict=True):
        scorer.add(prediction, target)
    return scorer.result()

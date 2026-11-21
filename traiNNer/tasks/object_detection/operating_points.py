"""Opt-in grouped confidence curves for a CPU detection holdout."""

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from heapq import merge
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from msgspec.json import encode

from .metrics import IOU_THRESHOLD

if TYPE_CHECKING:
    from .metrics import ReferenceScorer
    from .scoring import _NativeScorer


_ACTIVE: ContextVar["OperatingPoints | None"] = ContextVar("detection_operating_points", default=None)
active_report = _ACTIVE.get


@contextmanager
def record(directory: Path) -> Iterator["OperatingPoints"]:
    report = OperatingPoints(directory)
    token = _ACTIVE.set(report)
    try:
        yield report
    finally:
        _ACTIVE.reset(token)


def _summary(support: int, **identity) -> dict:
    return {**identity, "support": support, "tp": 0, "fp": 0, "fn": support,
            "scoredDetections": 0, "pointCount": 0, "bestF1": None}


def _advance(group: dict, threshold: float, hits: int, misses: int) -> dict:
    tp = group["tp"] + hits
    fp = group["fp"] + misses
    support = group["support"]
    detections = tp + fp
    point = {"threshold": threshold, "support": support, "tp": tp, "fp": fp, "fn": support - tp,
             "precision": tp / detections, "recall": tp / support if support else None,
             "f1": 2 * tp / (support + detections) if support else None}
    group.update(tp=tp, fp=fp, fn=support - tp, scoredDetections=detections,
                 pointCount=group["pointCount"] + 1)
    best = group["bestF1"]
    # Exact rational comparison retains the highest confidence on an F1 tie.
    if support and (best is None or
                    tp * (support + best["tp"] + best["fp"]) > best["tp"] * (support + detections)):
        group["bestF1"] = point
    return point


def _category_groups(scorer: "ReferenceScorer | _NativeScorer", i: int) -> Iterator[tuple[float, int, int, int]]:
    for score, tp, fp in scorer.groups(i):
        yield score, i, tp, fp


class OperatingPoints:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.outputs: dict[str, Path] = {}
        self.summaries: dict[str, dict] = {}

    def write(self, dataset: str, scorer: "ReferenceScorer | _NativeScorer",
              category_ids: Sequence[int], geometry: dict) -> None:
        """Consume grouped matches after AP50 has sorted each category."""
        if dataset in self.outputs:
            raise ValueError(f"An operating-point report already exists for {dataset!r}.")
        categories = [
            _summary(n, scope="category", classId=i, categoryId=category, name=name)
            for i, (n, category, name) in enumerate(zip(scorer.support, category_ids, scorer.names, strict=True), 1)
        ]
        pooled = _summary(sum(scorer.support), scope="global")
        streams = [_category_groups(scorer, i) for i in range(len(categories))]
        ranked = merge(*streams, key=itemgetter(0), reverse=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        output = self.directory / f"split-{len(self.outputs) + 1}.jsonl"
        with NamedTemporaryFile(dir=self.directory, suffix=".partial") as temp:
            for threshold, tied in groupby(ranked, key=itemgetter(0)):
                hits = misses = 0
                for _, i, tp, fp in tied:
                    group = categories[i]
                    point = _advance(group, threshold, tp, fp)
                    temp.write(encode({"scope": "category", "classId": group["classId"],
                                       "categoryId": group["categoryId"], "name": group["name"], **point}) + b"\n")
                    hits += tp
                    misses += fp
                temp.write(encode({"scope": "global", **_advance(pooled, threshold, hits, misses)}) + b"\n")
            temp.flush()
            os.link(temp.name, output)

        head = geometry.get("faster_rcnn", {})
        self.outputs[dataset] = output
        self.summaries[dataset] = {
            "iouThreshold": IOU_THRESHOLD, "scoreComparison": ">=",
            "detectorLimits": {"scoreThresholdExclusive": head.get("score_threshold_exclusive"),
                               "nmsIou": head.get("nms_iou"),
                               "maxDetectionsPerImage": head.get("detections_per_image")},
            "global": pooled, "categories": categories,
        }

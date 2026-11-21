"""Advance a metric-driven LR schedule only after the selected validation dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def validation_score(value, *, device, distributed: bool, rank: int) -> float:
    score = float(value) if value is not None else math.nan
    if distributed:
        # The restoration evaluator runs on rank zero. All ranks receive
        # one identical decision when the selected dataset has finished.
        packet = torch.tensor(score if rank == 0 else math.nan, dtype=torch.float64, device=device)
        torch.distributed.broadcast(packet, src=0)
        score = float(packet.item())
    return score


@dataclass(slots=True)
class MetricSchedule:
    scheduler: ReduceLROnPlateau
    optimizer_index: int
    dataset: str
    metric: str
    updated: bool = False

    def observe(self, results, *, device, distributed: bool, rank: int, logger) -> None:
        score = validation_score(results.get(self.metric) if self.updated else None,
                                 device=device, distributed=distributed, rank=rank)
        if math.isfinite(score):
            self.scheduler.step(score)
        elif self.updated and (not distributed or rank == 0):
            logger.warning("Skipped LR observation for %s/%s: the validation score is absent or nonfinite.",
                           self.dataset,self.metric)
        self.updated = False

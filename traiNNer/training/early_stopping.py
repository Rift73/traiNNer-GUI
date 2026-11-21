"""Patience on finite observations of one scheduled validation metric."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainner_config.options import ReduxOptions


@dataclass(slots=True)
class EarlyStopping:
    policy: dict
    observations: int = 0
    bad_observations: int = 0
    best: float | None = None
    best_step: int | None = None
    last: float | None = None
    last_step: int | None = None
    reference: float | None = None

    @classmethod
    def from_options(cls, opt: ReduxOptions) -> EarlyStopping:
        assert opt.train and opt.train.early_stopping and opt.val
        early = opt.train.early_stopping
        return cls({
            "monitor": {"dataset": early.monitor.dataset, "metric": early.monitor.metric},
            "mode": early.mode, "min_delta": early.min_delta, "patience": early.patience,
            "val_freq": opt.val.val_freq, "metricDefinition": deepcopy((opt.val.metrics or {}).get(early.monitor.metric)),
            "task": opt.task, "taskOptions": deepcopy(opt.task_options),
        })

    @property
    def stopped(self) -> bool:
        return self.bad_observations >= self.policy["patience"]

    def observe(self, step: int, score: float) -> None:
        if not math.isfinite(score) or self.last_step == step or self.stopped:
            return
        if self.last_step is not None and step < self.last_step:
            raise ValueError("Early-stopping validation steps must increase.")
        sign = 1 if self.policy["mode"] == "max" else -1
        self.observations += 1
        self.last, self.last_step = score, step
        if self.best is None or sign * (score - self.best) > 0:
            self.best, self.best_step = score, step
        if self.reference is None or sign * (score - self.reference) > self.policy["min_delta"]:
            self.reference = score
            self.bad_observations = 0
        else:
            self.bad_observations += 1

    def state_dict(self) -> dict:
        return {
            "version": 1, "policy": self.policy, "observations": self.observations,
            "badObservations": self.bad_observations, "best": self.best, "bestStep": self.best_step,
            "last": self.last, "lastStep": self.last_step, "reference": self.reference,
        }

    def restore(self, state: dict, checkpoint_step: int) -> None:
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported early-stopping checkpoint state.")
        if state.get("policy") != self.policy:
            raise ValueError("Cannot continue early stopping with a changed monitor, task, cadence or patience policy. "
                             "Restore the checkpoint policy or start a weights-only run with fresh history.")
        count, bad = state.get("observations"), state.get("badObservations")
        best_step, last_step = state.get("bestStep"), state.get("lastStep")
        scores = [state.get(k) for k in ("best", "last", "reference")]
        valid = type(count) is int and type(bad) is int and 0 <= count <= checkpoint_step
        if valid and count:
            valid = (type(best_step) is int and type(last_step) is int
                     and 0 < best_step <= last_step <= checkpoint_step
                     and count <= last_step and 0 <= bad <= min(count - 1, self.policy["patience"])
                     and (bad < self.policy["patience"] or last_step == checkpoint_step)
                     and all(type(v) in (float, int) and math.isfinite(v) for v in scores))
        elif valid:
            valid = bad == 0 and best_step is None and last_step is None and scores == [None, None, None]
        if not valid:
            raise ValueError("Invalid early-stopping history for this checkpoint iteration.")
        self.observations, self.bad_observations = count, bad
        self.best_step, self.last_step = best_step, last_step
        self.best, self.last, self.reference = scores

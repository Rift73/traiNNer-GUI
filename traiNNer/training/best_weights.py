"""Publish the weights that produced a winning score while they are still selected."""

from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors.torch import save_file
from trainner_config.metrics import best_weight_direction, best_weight_filename
from trainner_config.tasks import task_name

from traiNNer.models.checkpoint_payload import state_dict_for_export
from traiNNer.training.checkpoint_association import network_role
from traiNNer.training.validation_summary import attempt_id
from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.misc import is_json_compatible


class BestWeights:
    def __init__(self, opt) -> None:
        self.direction = best_weight_direction(opt)
        self.metric = opt.val.save_best
        self.attempt = attempt_id()
        self.task = task_name(opt)
        self.directory = Path(opt.path.models)
        self.winners: dict[str, float] = {}
        self.observed: tuple[str, int, float] | None = None

    def check_observed(self, model, dataset: str, step: int) -> None:
        score = model.metric_results.get(self.metric)
        if score is not None and math.isfinite(score) and self.observed != (dataset, step, float(score)):
            raise RuntimeError(
                f"Validation reported {self.metric!r} for {dataset!r} at step {step} without "
                "observing that score. Call best_weight_tracker.observe inside the inference context."
            )

    def observe(self, model, network, dataset: str, step: int) -> None:
        if model.opt.rank not in (None, 0):
            return
        score = model.metric_results.get(self.metric)
        if score is None or not math.isfinite(score):
            return
        score = float(score)
        self.observed = (dataset, step, score)
        previous = self.winners.get(dataset)
        if previous is not None and (score <= previous if self.direction == "higher" else score >= previous):
            return
        from traiNNer.models.base_model import _write_checkpoint_atomically

        bare = model.get_bare_model(network)
        role = network_role(model, bare)
        if role is None:
            raise ValueError("The validation network could not be associated with one portable checkpoint role.")
        path = self.directory / best_weight_filename(self.attempt, dataset, self.metric)
        receipt = {"version": 1, "task": self.task, "attempt": self.attempt, "dataset": dataset,
                   "metric": self.metric, "direction": self.direction, "value": float(score), "step": step, "role": role}
        header = {"trainner_best_validation": json.dumps(receipt, allow_nan=False)}
        if hasattr(bare, "hyperparameters"):
            header["metadata"] = json.dumps({k: v for k, v in bare.hyperparameters.items() if is_json_compatible(v)})
        weights = state_dict_for_export(bare.state_dict(), for_safetensors=True)
        _write_checkpoint_atomically(str(path), lambda destination: save_file(weights, destination, metadata=header))
        self.winners[dataset] = score
        get_root_logger().info("Best validation weights at step %s for %s (%s=%g): %s",
                               step, dataset, self.metric, score, path)

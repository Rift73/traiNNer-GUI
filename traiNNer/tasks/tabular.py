"""Supervised regression on numeric CSV records, with named checkpoint columns."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated

import torch
from msgspec import Meta
from torch import Tensor, nn

from trainner_config.tabular_task import RegressionOptions
from trainner_config.tasks import resolve_task
from trainner_data.numeric_records import read_values, selected_columns
from traiNNer.archs import build_network
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.models.base_model import BaseModel
from traiNNer.models.checkpoint_payload import load_matching_network_checkpoint
from traiNNer.tasks.supervised_parameters import check_grad_norm, select_trainable
from traiNNer.tasks.supervised_precision import check_bfloat16, forward_logits
from traiNNer.tasks.supervised_weights import (
    evaluation_weights,
    initialize_ema,
    save_supervised,
)
from traiNNer.training.supervised_validation import SupervisedValidationEvidence
from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY


def numeric_rows(path: Path, columns: list[str]) -> Tensor:
    """Read only selected fields into one contiguous FP32 table."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        positions = selected_columns(file, path, columns)
        values = read_values(file, path, positions)
    if not len(values):
        raise ValueError(f"{path} needs at least one numeric record.")
    return torch.from_numpy(values)


@DATASET_REGISTRY.register()
class NumericCsvDataset(BaseDataset):
    supports_batched_evaluation = True

    @property
    def label(self) -> str:
        return "records"

    def __init__(self, opt, settings: RegressionOptions):
        super().__init__(opt)
        self.source = str(Path(opt.meta_info).expanduser().absolute())
        self.values = numeric_rows(
            Path(self.source), [*settings.features, *settings.targets]
        )
        self.width = len(settings.features)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, index):
        row = self.values[index]
        return {
            "features": row[: self.width],
            "targets": row[self.width :],
            "row": index,
        }

    def __getitems__(self, indices: list[int]):
        rows = self.values[indices]
        features = rows[:, : self.width].unbind()
        targets = rows[:, self.width :].unbind()
        return [
            {"features": x, "targets": y, "row": index}
            for x, y, index in zip(features, targets, indices, strict=True)
        ]


@ARCH_REGISTRY.register()
def tabular_mlp(
    in_features: Annotated[int, Meta(ge=1)],
    out_features: Annotated[int, Meta(ge=1)],
    hidden: list[int] | None = None,
    scale: int = 1,
) -> nn.Module:
    """A dense regressor. An empty hidden list selects a linear model."""
    layers = []
    previous = in_features
    widths = [64, 32] if hidden is None else hidden
    if (
        scale != 1
        or min(in_features, out_features) < 1
        or any(type(w) is not int or w < 1 for w in widths)
    ):
        raise ValueError("Use positive layer widths and scale 1 for tabular_mlp.")
    for width in widths:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, out_features))
    return nn.Sequential(*layers)


def column_metadata(settings: RegressionOptions) -> dict:
    return {"features": list(settings.features), "targets": list(settings.targets)}


def _network(opt, settings):
    network = build_network(opt.network_g)
    network.hyperparameters = {
        **getattr(network, "hyperparameters", {}),
        **column_metadata(settings),
        "task": "tabular_regression",
    }
    return network


def _load(network, weights, key, settings):
    load_matching_network_checkpoint(
        network,
        Path(weights),
        key,
        expected_task="tabular_regression",
        expected_metadata=column_metadata(settings),
    )


class TabularModel(BaseModel):
    def __init__(self, opt, settings: RegressionOptions):
        super().__init__(opt)
        self.columns = settings
        self.use_amp = opt.use_amp
        if self.use_amp:
            check_bfloat16(self.device)
            self.amp_dtype = torch.bfloat16
        self.features = self.targets = None
        self._loss_sum = None
        self._observations = 0
        self.validation_evidence = None
        self.with_metrics = bool(
            opt.val and opt.val.val_enabled and opt.val.metrics_enabled
        )
        network = _network(opt, settings)
        if opt.path.pretrain_network_g:
            if self._initial_transfer_g is not None:
                self.load_initial_g(network)
            else:
                _load(
                    network, opt.path.pretrain_network_g, opt.path.param_key_g, settings
                )
        elif opt.path.resume_state:
            raise ValueError(
                "Resuming regression requires the numbered network weights."
            )
        if opt.is_train:
            select_trainable(network, opt.train)
        self.net_g = self.model_to_device(network, compile=opt.use_compile)
        self.net_g.train(bool(opt.is_train))
        if opt.is_train:
            self.accum_iters = opt.datasets["train"].accum_iter
            self.scaler_g = torch.amp.GradScaler(self.device.type, enabled=False)
            self.optimizer_g = self.get_optimizer(
                [p for p in network.parameters() if p.requires_grad], opt.train.optim_g
            )
            self.optimizers.append(self.optimizer_g)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                callable(getattr(self.optimizer_g, "eval", None))
            )
            self.setup_schedulers()
            initialize_ema(self, expected_metadata=column_metadata(settings))

    def feed_data(self, data):
        self.features = data["features"].to(self.device, non_blocking=True)
        self.targets = data["targets"].to(self.device, non_blocking=True)

    def _estimate(self, network):
        values = forward_logits(network, self.features, mixed=self.use_amp)
        if (
            not isinstance(values, Tensor)
            or values.shape != self.targets.shape
            or not values.is_floating_point()
        ):
            raise ValueError(
                "Regression output must have shape [batch, number of target columns]."
            )
        return values.float()

    def optimize_parameters(self, current_iter, current_accum_iter, apply_gradient):
        self.begin_training_attempt()
        loss = (self._estimate(self.net_g) - self.targets).square().sum()
        loss.backward()
        self._loss_sum = (
            loss.detach() if self._loss_sum is None else self._loss_sum + loss.detach()
        )
        self._observations += self.targets.numel()
        if not apply_gradient:
            return
        try:
            if not torch.isfinite(self._loss_sum).item():
                raise FloatingPointError("Nonfinite numeric regression loss.")
            grads = [p.grad for p in self.net_g.parameters() if p.grad is not None]
            torch._foreach_div_(grads, self._observations)
            check_grad_norm(
                self.net_g, self.opt.train.grad_clip_max_norm if self.opt.train.grad_clip else None
            )
            self.step_optimizer(self.scaler_g, self.optimizer_g)
            if self.net_g_ema is not None and not self.optimizers_skipped[0]:
                self.update_ema(self.net_g_ema)
            self.add_log_contributions(self._observations, {"l_mse": self._loss_sum})
        finally:
            self.optimizer_g.zero_grad(set_to_none=True)
            self._loss_sum = None
            self._observations = 0

    def validate_validation_datasets(self, val_loaders):
        if self.with_metrics and (
            not val_loaders
            or any(not isinstance(x.dataset, NumericCsvDataset) for x in val_loaders)
        ):
            raise ValueError(
                "Numeric regression metrics require at least one NumericCsvDataset validation split."
            )

    @torch.inference_mode()
    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, multi_val_datasets
    ):
        if not self.with_metrics:
            return
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        count = 0
        dataset = dataloader.dataset.opt.name
        evidence = None
        if self.opt.val.record_samples:
            evidence = SupervisedValidationEvidence(
                dataloader.dataset.opt,
                {
                    "task": "tabular_regression",
                    **column_metadata(self.columns),
                    "inference": {
                        "device": self.device.type,
                        "precision": "bfloat16_autocast" if self.use_amp else "float32",
                        "batchSize": dataloader.batch_size,
                    },
                    "metrics": {
                        "mae": {
                            "reduction": "mean_absolute_error_over_all_target_values",
                            "better": "lower",
                        },
                        "rmse": {
                            "reduction": "sqrt_mean_squared_error_over_all_target_values",
                            "better": "lower",
                        },
                    },
                },
            )
        with evaluation_weights(self, current_iter) as network:
            for batch in dataloader:
                if evidence is not None:
                    for index in batch["row"].tolist():
                        row = (
                            dataloader.dataset.values[index]
                            .numpy()
                            .astype("<f4", copy=False)
                        )
                        evidence.add_record(
                            dataloader.dataset.source,
                            index,
                            hashlib.sha256(row).hexdigest(),
                        )
                self.feed_data(batch)
                difference = self._estimate(network).double() - self.targets
                totals[0] += difference.abs().sum()
                totals[1] += difference.square().sum()
                count += difference.numel()
            if not count:
                raise ValueError(
                    f"Validation dataset {dataset!r} contains no target values."
                )
            mae, mse = (totals / count).tolist()
            if not (math.isfinite(mae) and math.isfinite(mse)):
                raise ValueError(
                    "Numeric regression produced nonfinite validation estimates."
                )
            self.metric_results = {"mae": mae, "rmse": math.sqrt(mse)}
            if self.best_weight_tracker is not None:
                self.best_weight_tracker.observe(self, network, dataset, current_iter)
        self.validation_evidence = evidence
        get_root_logger().info(
            "Validation %s at iteration %d | mae %.7g, rmse %.7g",
            dataset,
            current_iter,
            mae,
            math.sqrt(mse),
        )
        if tb_logger is not None:
            for key, score in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset}/{key}", score, current_iter)

    dist_validation = nondist_validation

    def save(self, epoch, current_iter):
        save_supervised(self, epoch, current_iter)

    def get_current_visuals(self):
        return {}


def _deployment(opt, checkpoint):
    resolved = resolve_task(opt)
    assert resolved is not None
    settings = resolved[2]
    network = _network(opt, settings)
    _load(network, checkpoint, opt.path.param_key_g, settings)
    return network.eval(), settings


@torch.inference_mode()
def predict_records(opt, checkpoint, source, destination):
    from traiNNer.tasks.predict import _write_new_file

    if not source.is_file():
        raise ValueError("Select a CSV file with the trained feature columns.")
    network, settings = _deployment(opt, checkpoint)
    metadata = {"task": "tabular_regression", **column_metadata(settings)}
    count = 0

    def write(file):
        nonlocal count
        file.write((json.dumps(metadata)[:-1] + ', "predictions": [').encode())
        with source.open(encoding="utf-8-sig", newline="") as stream:
            positions = selected_columns(stream, source, settings.features)
            while len(values := read_values(stream, source, positions, 1024)):
                predictions = network(torch.from_numpy(values))
                if not isinstance(predictions, Tensor) or predictions.shape != (
                    len(values),
                    len(settings.targets),
                ):
                    raise ValueError(
                        "The network output shape disagrees with the declared target columns."
                    )
                encoded = json.dumps(
                    predictions.float().tolist(), allow_nan=False, separators=(",", ":")
                )
                file.write((("," if count else "") + encoded[1:-1]).encode())
                count += len(values)
        if not count:
            raise ValueError(f"{source} needs at least one numeric record.")
        file.write(f'],"count":{count}}}'.encode())

    _write_new_file(destination, write)
    return {**metadata, "count": count, "outputs": [str(destination)]}


def export_records(opt, checkpoint, destination, *, opset):
    from traiNNer.tasks.onnx_export import write_portable_onnx

    network, settings = _deployment(opt, checkpoint)
    data = torch.zeros((1, len(settings.features)))
    with torch.inference_mode():
        if network(data).shape != (1, len(settings.targets)):
            raise ValueError(
                "The network output shape disagrees with the declared target columns."
            )
    return write_portable_onnx(
        network,
        data,
        destination,
        {
            "task": "tabular_regression",
            "architecture": opt.network_g["type"],
            **column_metadata(settings),
            "input_name": "records",
            "output_name": "predictions",
            "input_shape": ["batch", len(settings.features)],
            "output_shape": ["batch", len(settings.targets)],
            "input_dtype": "float32",
            "output_dtype": "float32",
            "opset": opset,
        },
    )

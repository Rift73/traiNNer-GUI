"""Scalar-label cross-entropy training and classification accuracy."""

import math
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import torch
from msgspec import Struct
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from trainner_config.image_classification_task import ClassificationOptions

from traiNNer.archs import build_network
from traiNNer.models.base_model import BaseModel
from traiNNer.tasks.supervised_parameters import check_grad_norm, select_trainable
from traiNNer.tasks.supervised_weights import evaluation_weights, initialize_ema, save_supervised
from traiNNer.tasks.supervised_precision import check_bfloat16, forward_logits
from traiNNer.tasks.image_classification.dataset import (
    ClassFolderDataset,
    class_folders,
)
from traiNNer.training.supervised_validation import SupervisedValidationEvidence
from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


class ImageClassificationModel(BaseModel):
    def __init__(self, opt: ReduxOptions, settings: Struct | None) -> None:
        super().__init__(opt)
        self.use_amp = opt.use_amp
        if self.use_amp:
            check_bfloat16(self.device)
            self.amp_dtype = torch.bfloat16
        self.validation_evidence: SupervisedValidationEvidence | None = None
        assert isinstance(settings, ClassificationOptions)
        self.class_names = tuple(settings.class_names)
        self._class_ids = {name: index for index, name in enumerate(self.class_names)}
        self.image_size = settings.image_size
        self.accuracy_ranks = settings.accuracy_ranks()
        self.confusion_matrices: dict[str, Tensor] = {}
        self.image: Tensor | None = None
        self.labels: Tensor | None = None
        self._window_loss: Tensor | None = None
        self._window_samples = 0
        self.with_metrics = bool(
            opt.val and opt.val.val_enabled and opt.val.metrics_enabled
        )

        train_dataset = opt.datasets.get("train")
        if opt.is_train:
            assert train_dataset is not None
            names = {folder.name for folder in class_folders(train_dataset)}
            self._check_labels(train_dataset.name, names, require_all=True)

        assert opt.network_g is not None
        network = build_network({**opt.network_g, "scale": opt.scale})
        if opt.path.pretrain_network_g:
            self.load_initial_g(
                network,
                class_names=self.class_names,
                expected_task="image_classification",
            )
        elif opt.path.resume_state:
            raise ValueError(
                "Resuming image classification requires the matching generator checkpoint."
            )

        network.hyperparameters = {
            **getattr(network, "hyperparameters", {}),
            "task": "image_classification",
            "class_names": list(self.class_names),
            "image_size": self.image_size,
        }
        if opt.is_train and opt.train is not None:
            select_trainable(network, opt.train)
        self.net_g = self.model_to_device(network, compile=opt.use_compile)
        self.net_g.train(bool(opt.is_train))

        if opt.is_train:
            assert opt.train is not None and opt.train.optim_g is not None
            assert train_dataset is not None
            self.accum_iters = train_dataset.accum_iter
            self.grad_clip = opt.train.grad_clip
            self.grad_clip_max_norm = opt.train.grad_clip_max_norm
            self.scaler_g = GradScaler(self.device.type, enabled=False)
            self.optimizer_g: Optimizer = self.get_optimizer(
                [
                    parameter
                    for parameter in self.net_g.parameters()
                    if parameter.requires_grad
                ],
                opt.train.optim_g,
            )
            self.optimizers.append(self.optimizer_g)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in opt.train.optim_g["type"].upper()
            )
            self.setup_schedulers()
            initialize_ema(self)

    def _check_labels(
        self, dataset_name: str, names: Collection[str], *, require_all: bool = False
    ) -> None:
        unknown = sorted(set(names) - self._class_ids.keys())
        if unknown:
            raise ValueError(
                f"Dataset {dataset_name!r} uses unknown class folders: {unknown!r}."
            )
        missing = sorted(self._class_ids.keys() - set(names))
        if missing and require_all:
            raise ValueError(
                f"Training dataset {dataset_name!r} is missing class folders: {missing!r}."
            )

    def feed_data(self, data: DataFeed) -> None:
        image = data.get("lq")
        names = data.get("class_name")
        if isinstance(names, str):
            names = [names]
        if names is None or not names:
            raise ValueError(
                "Image classification requires a class label on every image."
            )
        if image is None or image.shape != (
            len(names),
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Image classification expects aligned BCHW RGB input and scalar labels."
            )
        try:
            ids = [self._class_ids[name] for name in names]
        except KeyError as exc:
            raise ValueError(f"Unknown classification label {exc.args[0]!r}.") from exc
        self.image = image.to(self.device, non_blocking=True)
        self.labels = torch.tensor(ids, dtype=torch.long, device=self.device)

    def _logits(self, network: torch.nn.Module | None = None) -> Tensor:
        assert self.image is not None
        logits = forward_logits(self.net_g if network is None else network, self.image, mixed=self.use_amp)
        expected = (self.image.shape[0], len(self.class_names))
        if (
            not isinstance(logits, Tensor)
            or logits.shape != expected
            or not logits.is_floating_point()
        ):
            raise ValueError(
                f"Image classification requires one logits tensor of shape {expected}."
            )
        return logits.float() if self.use_amp else logits

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        self.begin_training_attempt()
        assert self.labels is not None and self.scaler_g is not None
        loss = cross_entropy(self._logits(), self.labels, reduction="sum")
        loss.backward()
        value = loss.detach()
        self._window_loss = (
            value if self._window_loss is None else self._window_loss + value
        )
        self._window_samples += self.labels.numel()
        if not apply_gradient:
            return

        try:
            if not math.isfinite(self._window_loss.item()):
                raise FloatingPointError("Nonfinite image classification loss.")
            for group in self.optimizer_g.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self._window_samples)
            check_grad_norm(self.net_g, self.grad_clip_max_norm if self.grad_clip else None)
            self.step_optimizer(self.scaler_g, self.optimizer_g)
            if self.net_g_ema is not None and not self.optimizers_skipped[0]:
                self.update_ema(self.net_g_ema)
            self.add_log_contributions(
                self._window_samples, {"l_classification_ce": self._window_loss}
            )
        finally:
            self.optimizer_g.zero_grad(set_to_none=True)
            self._window_samples = 0
            self._window_loss = None

    def validate_validation_datasets(self, val_loaders: Sequence[DataLoader]) -> None:
        if not self.with_metrics:
            return
        if not val_loaders:
            raise ValueError(
                "Classification metrics require a ClassFolderDataset validation loader."
            )
        for loader in val_loaders:
            dataset = self._checked_dataset(loader)
            missing = sorted(self._class_ids.keys() - dataset.class_names)
            if missing:
                get_root_logger().warning(
                    "Validation dataset %s has no examples for %s. Class IDs remain fixed.",
                    dataset.opt.name,
                    ", ".join(missing),
                )

    def _checked_dataset(self, loader: DataLoader) -> ClassFolderDataset:
        dataset = loader.dataset
        if not isinstance(dataset, ClassFolderDataset):
            raise ValueError(
                "Classification metrics require ClassFolderDataset validation data."
            )
        self._check_labels(dataset.opt.name, dataset.class_names)
        return dataset

    @torch.inference_mode()
    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if not self.with_metrics:
            return

        dataset = self._checked_dataset(dataloader)
        evidence = None
        if self.opt.val and self.opt.val.record_samples:
            evidence = SupervisedValidationEvidence(
                dataset.opt,
                {
                    "task": "image_classification",
                    "inference": {"device": self.device.type, "precision": "bfloat16_autocast" if self.use_amp else "float32",
                                  "batchSize": dataloader.batch_size},
                    "labels": [
                        {"id": index, "name": name}
                        for index, name in enumerate(self.class_names)
                    ],
                    "ignore": None,
                    "geometry": {
                        "input": "whole_rgb_float32_0_1",
                        "resize": "pillow_bilinear",
                        "size": [self.image_size, self.image_size],
                    },
                    "metrics": {
                        f"top{k}": {
                            "reduction": "images_with_target_in_top_k / labeled_images",
                            "k": k,
                            "score_ties": "ascending_class_id",
                            "better": "higher",
                        }
                        for k in self.accuracy_ranks
                    },
                },
            )
        classes = len(self.class_names)
        confusion = torch.zeros(
            (classes, classes), dtype=torch.int64, device=self.device
        )
        correct = torch.zeros(
            len(self.accuracy_ranks), dtype=torch.int64, device=self.device
        )
        from .results import dataset_sink
        with evaluation_weights(self, current_iter) as inference:
            with dataset_sink(dataset.opt.name, self.class_names) as sink:
                for data in dataloader:
                    self.feed_data(data)
                    if evidence is not None:
                        for path, name in zip(
                            data["lq_path"], data["class_name"], strict=True
                        ):
                            evidence.add(
                                path,
                                (self.image_size, self.image_size),
                                target_identity=self._class_ids[name],
                            )
                    assert self.labels is not None
                    logits = self._logits(inference)
                    if not torch.isfinite(logits).all():
                        raise ValueError(
                            "Image classification returned nonfinite validation logits."
                        )
                    if self.accuracy_ranks[-1] == 1:
                        ranked = logits.argmax(dim=1, keepdim=True)
                    else:
                        ranked = logits.argsort(dim=1, descending=True, stable=True)[
                            :, : self.accuracy_ranks[-1]
                        ]
                    if sink is not None:
                        sink(data["lq_path"], self.labels, logits, ranked)
                    matches = ranked == self.labels[:, None]
                    for index, k in enumerate(self.accuracy_ranks):
                        correct[index] += matches[:, :k].any(dim=1).sum()
                    pairs = self.labels * classes + ranked[:, 0]
                    confusion += torch.bincount(pairs, minlength=classes**2).view_as(
                        confusion
                    )

            # Rows are target IDs, columns are predicted IDs, both in class_names order.
            confusion = confusion.cpu()
            self.confusion_matrices[dataset.opt.name] = confusion
            count = confusion.sum().item()
            if not count:
                raise ValueError(
                    f"Validation dataset {dataset.opt.name!r} contains no labeled images."
                )
            self.metric_results = {
                f"top{k}": hits / count
                for k, hits in zip(self.accuracy_ranks, correct.tolist(), strict=True)
            }
            if self.best_weight_tracker is not None:
                self.best_weight_tracker.observe(self, inference, dataset.opt.name, current_iter)
        self.validation_evidence = evidence
        get_root_logger().info(
            "Validation %s at iteration %d | %s",
            dataset.opt.name,
            current_iter,
            ", ".join(
                f"{name} {value:.4f}" for name, value in self.metric_results.items()
            ),
        )
        if tb_logger is not None:
            for name, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset.opt.name}/{name}", value, current_iter
                )

    def get_current_visuals(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in (("image", self.image), ("labels", self.labels))
            if value is not None
        }

    def save(self, epoch: int, current_iter: int) -> None:
        save_supervised(self, epoch, current_iter)

"""Train an image-to-logits network with masked cross-entropy and class IoU."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from msgspec import Struct
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from trainner_config.semantic_segmentation_task import SegmentationOptions

from traiNNer.archs import build_network
from traiNNer.models.base_model import BaseModel
from traiNNer.tasks.supervised_parameters import check_grad_norm, select_trainable
from traiNNer.tasks.supervised_weights import evaluation_weights, initialize_ema, save_supervised
from traiNNer.tasks.supervised_precision import check_bfloat16, forward_logits
from traiNNer.tasks.semantic_segmentation.dataset import ImageMaskDataset
from traiNNer.training.supervised_validation import SupervisedValidationEvidence
from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


class SemanticSegmentationModel(BaseModel):
    def __init__(self, opt: ReduxOptions, settings: Struct | None) -> None:
        super().__init__(opt)
        self.use_amp = opt.use_amp
        if self.use_amp:
            check_bfloat16(self.device)
            self.amp_dtype = torch.bfloat16
        self.validation_evidence: SupervisedValidationEvidence | None = None
        assert isinstance(settings, SegmentationOptions)
        self.class_names = tuple(settings.class_names)
        self.ignore_label = settings.ignore_label
        self.confusion_matrices: dict[str, Tensor] = {}
        self.image: Tensor | None = None
        self.mask: Tensor | None = None
        self.prediction: Tensor | None = None
        self._window_loss: Tensor | None = None
        self._batch_pixels = 0
        self._window_pixels = 0
        self._empty_windows = 0

        assert opt.network_g is not None
        network = build_network({**opt.network_g, "scale": opt.scale})
        if opt.path.pretrain_network_g:
            self.load_initial_g(
                network,
                class_names=self.class_names,
                expected_task="semantic_segmentation",
                allow_legacy_class_order=True,
                strict=opt.path.strict_load_g or bool(opt.path.resume_state),
            )
        elif opt.path.resume_state:
            raise ValueError(
                "Resuming semantic segmentation requires the matching generator checkpoint."
            )

        network.hyperparameters = {
            **getattr(network, "hyperparameters", {}),
            "task": "semantic_segmentation",
            "class_names": list(self.class_names),
        }
        if opt.is_train and opt.train is not None:
            select_trainable(network, opt.train)
        self.net_g = self.model_to_device(network, compile=opt.use_compile)
        self.net_g.train(bool(opt.is_train))
        self.with_metrics = bool(
            opt.val and opt.val.val_enabled and opt.val.metrics_enabled
        )

        if opt.is_train:
            assert opt.train is not None and opt.train.optim_g is not None
            self.accum_iters = opt.datasets["train"].accum_iter
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

    def feed_data(self, data: DataFeed) -> None:
        image, mask = data["lq"], data["gt"]
        if (
            image.ndim != 4
            or image.shape[1] != 3
            or mask.ndim != 3
            or mask.dtype != torch.long
            or mask.shape != (image.shape[0], *image.shape[-2:])
        ):
            raise ValueError(
                "Semantic segmentation expects RGB BCHW images and aligned int64 BHW masks."
            )
        annotated = mask.ne(self.ignore_label)
        invalid = annotated & ((mask < 0) | (mask >= len(self.class_names)))
        # Count before the device transfer and keep one scalar read for a device-side mask.
        self._batch_pixels = int(torch.where(invalid.any(), -1, annotated.sum()).item())
        if self._batch_pixels < 0:
            label = mask[invalid][0].item()
            raise ValueError(
                f"Mask {data.get('gt_path', '')} contains class ID {label}. "
                f"Use IDs 0 to {len(self.class_names) - 1} and ignore ID {self.ignore_label}."
            )
        self.image = image.to(self.device, non_blocking=True)
        self.mask = mask.to(self.device, non_blocking=True)
        self.prediction = None

    def _logits(self, network: torch.nn.Module | None = None) -> Tensor:
        assert self.image is not None and self.mask is not None
        output = forward_logits(self.net_g if network is None else network, self.image, mixed=self.use_amp)
        logits = output.get("out") if isinstance(output, dict) else output
        expected = (self.mask.shape[0], len(self.class_names), *self.mask.shape[-2:])
        if not isinstance(logits, Tensor) or logits.shape != expected:
            raise ValueError(
                f"Semantic segmentation network must return logits of shape {expected}, "
                "as a tensor or under the 'out' key."
            )
        return logits.float() if self.use_amp else logits

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        self.begin_training_attempt()
        assert self.mask is not None and self.scaler_g is not None
        if self._batch_pixels:
            loss_sum = cross_entropy(
                self._logits(), self.mask, ignore_index=self.ignore_label, reduction="sum"
            )
            loss_sum.backward()
            loss = loss_sum.detach()
            self._window_loss = loss if self._window_loss is None else self._window_loss + loss
            self._window_pixels += self._batch_pixels
        if not apply_gradient:
            return

        try:
            valid_pixels = self._window_pixels
            self.optimizers_skipped[0] = valid_pixels == 0
            if not valid_pixels:
                self.reject_training_step("no_annotated_pixels")
                self._empty_windows += 1
                # A labeled sample can move from the start of one shuffled pass to
                # the end of the next. The nominal pass size rounds down in the loader.
                empty_limit = max(20, 2 * (self.opt.switch_iter_per_epoch + 1))
                if self._empty_windows >= empty_limit:
                    raise ValueError(
                        f"No annotated pixels in {self._empty_windows} consecutive training windows "
                        f"(ignore label {self.ignore_label}). Check the masks and training crop size."
                    )
                return

            self._empty_windows = 0
            assert self._window_loss is not None
            if not math.isfinite(self._window_loss.item()):
                raise FloatingPointError("Nonfinite semantic segmentation loss.")
            for group in self.optimizer_g.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(valid_pixels)
            check_grad_norm(self.net_g, self.grad_clip_max_norm if self.grad_clip else None)
            self.step_optimizer(self.scaler_g, self.optimizer_g)
            if self.net_g_ema is not None and not self.optimizers_skipped[0]:
                self.update_ema(self.net_g_ema)
            self.add_log_contributions(
                valid_pixels, {"l_segmentation_ce": self._window_loss}
            )
        finally:
            self.optimizer_g.zero_grad(set_to_none=True)
            self._window_loss = None
            self._window_pixels = 0

    def validate_validation_datasets(self, val_loaders: Sequence[DataLoader]) -> None:
        if self.with_metrics and (
            not val_loaders
            or not all(
                isinstance(loader.dataset, ImageMaskDataset) for loader in val_loaders
            )
        ):
            raise ValueError(
                "Semantic segmentation metrics require ImageMaskDataset validation data."
            )

    @torch.inference_mode()
    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if not self.with_metrics and not save_img:
            return

        dataset = dataloader.dataset
        assert isinstance(dataset, ImageMaskDataset)
        masks = None
        if save_img:
            from .masks import ValidationMasks

            root = self.opt.path.experiments_root if self.opt.is_train else self.opt.path.results_root
            assert root is not None and self.opt.path.visualization is not None
            masks = ValidationMasks(Path(root), Path(self.opt.path.visualization), dataset.opt,
                                    current_iter, len(self.class_names))
        evidence = None
        if self.with_metrics and self.opt.val and self.opt.val.record_samples:
            evidence = SupervisedValidationEvidence(
                dataset.opt,
                {
                    "task": "semantic_segmentation",
                    "inference": {"device": self.device.type, "precision": "bfloat16_autocast" if self.use_amp else "float32"},
                    "labels": [
                        {"id": index, "name": name}
                        for index, name in enumerate(self.class_names)
                    ],
                    "ignore": {"label_id": self.ignore_label, "exclude_pixels": True},
                    "geometry": {"input": "whole_rgb_float32_0_1", "resize": None},
                    "metrics": {
                        **{
                            f"IoU/{name}": {
                                "reduction": "pooled_pixel_intersection / pooled_pixel_union",
                                "class_id": index,
                                "better": "higher",
                            }
                            for index, name in enumerate(self.class_names)
                        },
                        "mIoU": {
                            "reduction": "mean_class_iou_over_nonempty_pooled_unions",
                            "better": "higher",
                        },
                    },
                },
            )
        num_classes = len(self.class_names)
        confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=self.device
        )
        with evaluation_weights(self, current_iter) as inference:
            for data in dataloader:
                self.feed_data(data)
                assert self.mask is not None
                if evidence is not None:
                    size = (self.mask.shape[-1], self.mask.shape[-2])
                    for path, mask_path in zip(
                        data["lq_path"], data["gt_path"], strict=True
                    ):
                        evidence.add(path, size, target_path=mask_path)
                self.prediction = self._logits(inference).argmax(dim=1)
                if masks is not None:
                    masks.add(self.prediction, data["lq_path"], data["gt_path"])
                if not self.with_metrics:
                    continue
                valid = self.mask.ne(self.ignore_label)
                pairs = self.mask[valid] * num_classes + self.prediction[valid]
                confusion += torch.bincount(pairs, minlength=num_classes**2).reshape_as(
                    confusion
                )

            if masks is not None:
                masks.publish()
            if not self.with_metrics:
                return
            # Rows are ground-truth IDs, columns are predicted IDs.
            confusion = confusion.cpu()
            self.confusion_matrices[dataset.opt.name] = confusion
            intersection = confusion.diag()
            union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
            self.metric_results = {
                f"IoU/{name}": intersection[index].item() / union[index].item()
                for index, name in enumerate(self.class_names)
                if union[index].item() > 0
            }
            self.validation_evidence = evidence
            logger = get_root_logger()
            if not self.metric_results:
                logger.warning(
                    "Validation dataset %s has no annotated pixels.", dataset.opt.name
                )
                return
            self.metric_results["mIoU"] = sum(self.metric_results.values()) / len(
                self.metric_results
            )
            if self.best_weight_tracker is not None:
                self.best_weight_tracker.observe(self, inference, dataset.opt.name, current_iter)
        logger.info(
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
            for name, value in (
                ("image", self.image),
                ("mask", self.mask),
                ("prediction", self.prediction),
            )
            if value is not None
        }

    def save(self, epoch: int, current_iter: int) -> None:
        save_supervised(self, epoch, current_iter)

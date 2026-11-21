"""Single-device trainer for detectors with variable image and target sizes."""

import json
import math
from collections.abc import Sequence
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import torch
from msgspec import Struct
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from trainner_config.object_detection_task import DetectionOptions

from traiNNer.archs import build_network
from traiNNer.models.base_model import BaseModel
from traiNNer.tasks.supervised_parameters import check_grad_norm, select_trainable
from traiNNer.tasks.supervised_weights import evaluation_weights, initialize_ema, save_supervised
from traiNNer.tasks.object_detection.dataset import CocoDetectionDataset
from traiNNer.tasks.object_detection.metrics import IOU_THRESHOLD
from traiNNer.tasks.object_detection.operating_points import active_report
from traiNNer.tasks.object_detection.results import dataset_sink
from traiNNer.tasks.object_detection.scoring import scorer
from traiNNer.training.supervised_validation import SupervisedValidationEvidence
from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed, DetectionTarget

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


def checked_detections(
    outputs: object, *, batch_size: int, num_classes: int
) -> list[dict[str, Tensor]]:
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise ValueError(
            "An evaluation detector must return one detection dictionary per image."
        )
    predictions = []
    for output in outputs:
        if not isinstance(output, dict) or not all(
            isinstance(output.get(key), Tensor) for key in ("boxes", "scores", "labels")
        ):
            raise ValueError("A detection must provide boxes, scores, and labels.")
        boxes, scores, labels = (
            output[key].detach().cpu() for key in ("boxes", "scores", "labels")
        )
        if (
            labels.ndim != 1
            or labels.dtype != torch.long
            or boxes.shape != (len(labels), 4)
            or scores.shape != labels.shape
            or not boxes.is_floating_point()
            or not scores.is_floating_point()
            or bool((boxes[:, 2:] <= boxes[:, :2]).any())
            or not torch.isfinite(boxes).all()
            or not torch.isfinite(scores).all()
            or bool(((labels < 1) | (labels > num_classes)).any())
        ):
            raise ValueError(
                "Detector returned invalid boxes, scores, or foreground labels."
            )
        predictions.append({"boxes": boxes, "scores": scores, "labels": labels})
    return predictions


class ObjectDetectionModel(BaseModel):
    def __init__(self, opt: ReduxOptions, settings: Struct | None) -> None:
        super().__init__(opt)
        self.validation_evidence: SupervisedValidationEvidence | None = None
        assert isinstance(settings, DetectionOptions)
        self.class_names = tuple(category.name for category in settings.categories)
        self.category_ids = tuple(category.id for category in settings.categories)
        self.images: list[Tensor] = []
        self.targets: list[DetectionTarget] = []
        self._window_losses: dict[str, Tensor] = {}
        self._window_samples = 0
        self.with_metrics = bool(
            opt.val and opt.val.val_enabled and opt.val.metrics_enabled
        )

        assert opt.network_g is not None
        network = build_network({**opt.network_g, "scale": opt.scale})
        if opt.path.pretrain_network_g:
            self.load_initial_g(
                network,
                class_names=self.class_names,
                category_ids=self.category_ids,
                expected_task="object_detection",
            )
        elif opt.path.resume_state:
            raise ValueError(
                "Resuming object detection requires the matching generator checkpoint."
            )

        network.hyperparameters = {
            **getattr(network, "hyperparameters", {}),
            "task": "object_detection",
            "class_names": list(self.class_names),
            "category_ids": list(self.category_ids),
        }
        if opt.is_train and opt.train is not None:
            select_trainable(network, opt.train)
        self.net_g = self.model_to_device(network)
        self.net_g.train(bool(opt.is_train))

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
        images = data.get("detection_images")
        targets = data.get("detection_targets")
        if not images or targets is None or len(images) != len(targets):
            raise ValueError(
                "Object detection needs one target per image, including empty targets."
            )
        self.images = [image.to(self.device, non_blocking=True) for image in images]
        self.targets = targets

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        self.begin_training_attempt()
        assert self.scaler_g is not None
        training_targets = []
        for target in self.targets:
            keep = ~target["ignore"]
            training_targets.append(
                {
                    "boxes": target["boxes"][keep].to(self.device, non_blocking=True),
                    "labels": target["labels"][keep].to(self.device, non_blocking=True),
                }
            )

        losses = self.net_g(self.images, training_targets)
        if (
            not isinstance(losses, dict)
            or not losses
            or not all(
                isinstance(value, Tensor) and value.ndim == 0
                for value in losses.values()
            )
        ):
            raise ValueError(
                "A training detector must return a dictionary of scalar losses."
            )
        total = torch.stack(list(losses.values())).sum()
        count = len(self.images)
        (total * count).backward()
        self._window_samples += count
        for name, value in {"total": total, **losses}.items():
            key = f"l_detection_{name}"
            contribution = value.detach() * count
            previous = self._window_losses.get(key)
            self._window_losses[key] = (
                contribution if previous is None else previous + contribution
            )

        if not apply_gradient:
            return

        try:
            if not math.isfinite(self._window_losses["l_detection_total"].item()):
                raise FloatingPointError("Nonfinite object detection loss.")
            for group in self.optimizer_g.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self._window_samples)
            check_grad_norm(self.net_g, self.grad_clip_max_norm if self.grad_clip else None)
            self.step_optimizer(self.scaler_g, self.optimizer_g)
            if self.net_g_ema is not None and not self.optimizers_skipped[0]:
                self.update_ema(self.net_g_ema)
            self.add_log_contributions(self._window_samples, self._window_losses)
        finally:
            self.optimizer_g.zero_grad(set_to_none=True)
            self._window_losses = {}
            self._window_samples = 0

    def validate_validation_datasets(self, val_loaders: Sequence[DataLoader]) -> None:
        if self.with_metrics and (
            not val_loaders
            or not all(
                isinstance(loader.dataset, CocoDetectionDataset)
                for loader in val_loaders
            )
        ):
            raise ValueError(
                "Detection metrics require CocoDetectionDataset validation data."
            )

    def _validation_geometry(self, inference: torch.nn.Module | None = None) -> dict[str, object]:
        from torchvision.models.detection import FasterRCNN

        network = self.get_bare_model(self.net_g if inference is None else inference)
        geometry: dict[str, object] = {
            "input": "whole_rgb_float32_0_1",
            "boxes": "xyxy_original_pixels",
            "target_box_clipping": "image_bounds",
        }
        if isinstance(network, FasterRCNN):
            transform, head = network.transform, network.roi_heads
            geometry["faster_rcnn"] = {
                "min_size": transform.min_size[-1],
                "max_size": transform.max_size,
                "fixed_size": transform.fixed_size,
                "size_divisible": transform.size_divisible,
                "mean": transform.image_mean,
                "std": transform.image_std,
                "score_threshold_exclusive": head.score_thresh,
                "nms_iou": head.nms_thresh,
                "detections_per_image": head.detections_per_img,
            }
        else:
            geometry["detector_options"] = self.opt.network_g
        return geometry

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
        self.validate_validation_datasets([dataloader])
        dataset = dataloader.dataset
        assert isinstance(dataset, CocoDetectionDataset)
        evidence = None
        if self.opt.val and self.opt.val.record_samples:
            evidence = SupervisedValidationEvidence(
                dataset.opt,
                {
                    "task": "object_detection",
                    "inference": {"device": self.device.type, "precision": "float32"},
                    "labels": [
                        {"id": index, "name": name, "category_id": category}
                        for index, (name, category) in enumerate(
                            zip(self.class_names, self.category_ids, strict=True), 1
                        )
                    ],
                    "ignore": {
                        "target_flags": ["iscrowd", "ignore"],
                        "exclude_unmatched_predictions_overlapping_ignored_same_class": True,
                    },
                    "geometry": self._validation_geometry(),
                    "matching": {
                        "iou_threshold": IOU_THRESHOLD,
                        "match": "one_to_one_highest_iou_within_image_and_class",
                        "ranking": "descending_score_stable_in_evaluation_order",
                    },
                    "metrics": {
                        **{
                            f"AP50/{name}": {
                                "reduction": "pooled_interpolated_precision_recall_integral",
                                "class_id": index,
                                "better": "higher",
                            }
                            for index, name in enumerate(self.class_names, 1)
                        },
                        "AP50": {
                            "reduction": "mean_class_ap_over_classes_with_targets",
                            "better": "higher",
                        },
                        "recall50": {
                            "reduction": "pooled_true_positives / pooled_targets",
                            "better": "higher",
                        },
                        "false_positives50": {
                            "reduction": "pooled_false_positive_count",
                            "better": "lower",
                        },
                        "targets": {"reduction": "pooled_nonignored_target_count"},
                    },
                },
            )
        report = active_report()
        if report is not None and (self.opt.is_train or self.device.type != "cpu"):
            raise ValueError("Detection operating points require a CPU holdout evaluation.")
        with evaluation_weights(self, current_iter) as inference:
            with (scorer(self.class_names, operating_points=report is not None) as metrics,
                  dataset_sink(dataset.opt.name, self.category_ids) as export):
                for data in dataloader:
                    self.feed_data(data)
                    predictions = checked_detections(
                        inference(self.images), batch_size=len(self.images), num_classes=len(self.class_names)
                    )
                    for i, (prediction, target) in enumerate(zip(predictions, self.targets, strict=True)):
                        metrics.add(prediction, target)
                        if export is not None:
                            export(data["detection_image_ids"][i], prediction)
                        if evidence is not None:
                            decoded = json.dumps(
                                [target[key].tolist() for key in ("boxes", "labels", "ignore")],
                                separators=(",", ":"), allow_nan=False,
                            ).encode()
                            evidence.add(
                                data["detection_paths"][i],
                                (self.images[i].shape[-1], self.images[i].shape[-2]),
                                target_path=dataset.opt.annotation_file,
                                target_identity="sha256:" + sha256(decoded).hexdigest(),
                            )
                self.metric_results = metrics.result()
                if report is not None:
                    report.write(dataset.opt.name, metrics, self.category_ids, self._validation_geometry(inference))
                if self.best_weight_tracker is not None:
                    self.best_weight_tracker.observe(self, inference, dataset.opt.name, current_iter)

        self.validation_evidence = evidence
        logger = get_root_logger()
        if not self.metric_results["targets"]:
            logger.warning(
                "Validation dataset %s has no evaluable boxes. AP50 and recall50 are undefined.",
                dataset.opt.name,
            )
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

    def get_current_visuals(self) -> dict[str, Any]:
        return {
            "images": [image.detach().cpu() for image in self.images],
            "targets": self.targets,
        }

    def save(self, epoch: int, current_iter: int) -> None:
        save_supervised(self, epoch, current_iter)

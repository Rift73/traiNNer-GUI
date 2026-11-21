"""Eager CPU detection with named COCO labels and original-image boxes."""

from pathlib import Path
from stat import S_ISREG
from typing import Any

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.ops import clip_boxes_to_image
from trainner_config.object_detection_task import DetectionOptions

from traiNNer.tasks.object_detection.image import (
    detection_image_tensor,
    read_detection_image,
)
from traiNNer.tasks.object_detection.model import checked_detections


def checkpoint_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Version a checkpoint using the managed ledger's field order."""
    info = path.stat()
    if not S_ISREG(info.st_mode):
        raise ValueError(f"Checkpoint {path} must be a regular file.")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class DetectionPredictor:
    output_kind = "detections"

    def __init__(
        self,
        network: nn.Module,
        architecture: str,
        settings: DetectionOptions,
        checkpoint: Path,
        signature: tuple[int, int, int, int, int],
        *,
        score_threshold: float,
        top_limit: int,
    ) -> None:
        if not isinstance(network, FasterRCNN):
            raise ValueError(
                "Headless detection requires an eager torchvision Faster R-CNN network."
            )
        self.network = network.eval()
        self.architecture = architecture
        self.categories = tuple(settings.categories)
        self.checkpoint = checkpoint
        self.signature = signature
        self.score_threshold = score_threshold
        self.top_limit = top_limit
        self.network.roi_heads.score_thresh = score_threshold
        self.network.roi_heads.detections_per_img = top_limit

    def metadata(self) -> dict[str, Any]:
        return {
            "task": "object_detection",
            "architecture": self.architecture,
            "output_kind": self.output_kind,
            "backend": "pytorch-eager-cpu",
            "class_names": [category.name for category in self.categories],
            "category_ids": [category.id for category in self.categories],
            "checkpoint": str(self.checkpoint),
            "checkpoint_signature": list(self.signature),
            "box_format": "xyxy",
            "coordinate_space": "original_image_pixels",
            "score_threshold": self.score_threshold,
            "top_limit": self.top_limit,
        }

    @torch.inference_mode()
    def predict(self, path: Path, *, file_name: str) -> dict[str, Any]:
        image = read_detection_image(path)
        width, height = image.size
        try:
            (prediction,) = checked_detections(
                self.network([detection_image_tensor(image)]),
                batch_size=1,
                num_classes=len(self.categories),
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Detection failed for {path}. {exc}") from exc

        scores = prediction["scores"]
        boxes = clip_boxes_to_image(prediction["boxes"], (height, width))
        kept = (scores > self.score_threshold).nonzero(as_tuple=True)[0]
        order = scores[kept].argsort(descending=True, stable=True)[: self.top_limit]
        kept = kept[order]
        boxes = boxes[kept]
        if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError(f"Detector returned a box with no area in {path}.")

        detections = []
        for box, label, score in zip(
            boxes.tolist(),
            prediction["labels"][kept].tolist(),
            scores[kept].tolist(),
            strict=True,
        ):
            category = self.categories[label - 1]
            detections.append(
                {
                    "box_xyxy": box,
                    "category_id": category.id,
                    "class_id": label,
                    "class_name": category.name,
                    "score": score,
                }
            )
        return {
            "input": str(path),
            "file_name": file_name,
            "width": width,
            "height": height,
            "detections": detections,
        }

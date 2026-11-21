"""Whole-image object detection provider."""

from typing import Annotated

from msgspec import Meta
from torch import nn

from traiNNer.tasks.object_detection.model import ObjectDetectionModel
from traiNNer.utils.registry import ARCH_REGISTRY

__all__ = ["ObjectDetectionModel"]


@ARCH_REGISTRY.register()
def fasterrcnn_mobilenet_v3_large_320_fpn(
    num_classes: Annotated[int, Meta(ge=2)],
    scale: int = 1,
    min_size: Annotated[int, Meta(ge=1)] = 320,
    max_size: Annotated[int, Meta(ge=1)] = 640,
    box_score_thresh: Annotated[float, Meta(ge=0, le=1)] = 0.05,
) -> nn.Module:
    """Torchvision Faster R-CNN with randomly initialized model and backbone."""
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn as build_fasterrcnn,
    )

    if scale != 1 or num_classes < 2 or not 1 <= min_size <= max_size:
        raise ValueError(
            "Faster R-CNN needs scale 1, foreground plus background, and valid resize bounds."
        )
    if not 0 <= box_score_thresh <= 1:
        raise ValueError("Detection box_score_thresh must be between 0 and 1.")
    return build_fasterrcnn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
    )

"""Semantic segmentation runtime and its registered architecture."""

from typing import Annotated

from msgspec import Meta
from torch import nn

from traiNNer.tasks.semantic_segmentation.model import SemanticSegmentationModel
from traiNNer.utils.registry import ARCH_REGISTRY

__all__ = ["SemanticSegmentationModel"]


@ARCH_REGISTRY.register()
def lraspp_mobilenet_v3_large(
    num_classes: Annotated[int, Meta(ge=2)], scale: int = 1
) -> nn.Module:
    """Torchvision LR-ASPP with randomly initialized weights."""
    from torchvision.models.segmentation import (
        lraspp_mobilenet_v3_large as build_lraspp,
    )

    if scale != 1 or num_classes < 2:
        raise ValueError("LR-ASPP requires scale 1 and at least two output classes.")
    return build_lraspp(weights=None, weights_backbone=None, num_classes=num_classes)

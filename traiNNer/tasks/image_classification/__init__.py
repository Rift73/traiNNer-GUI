"""Whole-image classification provider."""

from typing import Annotated

from msgspec import Meta
from torch import nn

from traiNNer.tasks.image_classification.model import ImageClassificationModel
from traiNNer.utils.registry import ARCH_REGISTRY

__all__ = ["ImageClassificationModel"]


@ARCH_REGISTRY.register()
def mobilenet_v3_small(
    num_classes: Annotated[int, Meta(ge=2)], scale: int = 1
) -> nn.Module:
    """Torchvision MobileNetV3 Small, randomly initialized."""
    from torchvision.models import mobilenet_v3_small as build_mobilenet

    if scale != 1 or num_classes < 2:
        raise ValueError(
            "MobileNetV3 classification requires scale 1 and at least two classes."
        )
    return build_mobilenet(weights=None, num_classes=num_classes)

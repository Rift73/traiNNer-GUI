"""Pure frozen-teacher distillation calculations used by SRModel.

Model setup, feature pairing, and mutable training state remain on SRModel;
these helpers make the numerical transforms independently readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

if TYPE_CHECKING:
    from traiNNer.utils.redux_options import DistillationOptions


def distillation_distance(
    prediction: Tensor,
    target: Tensor,
    loss_type: str,
    epsilon: float,
) -> Tensor:
    """Compute the configured distance in float32 for AMP-stable supervision."""
    difference = prediction.float() - target.float()
    if loss_type == "charbonnier":
        return torch.sqrt(difference.square() + epsilon * epsilon).mean()
    if loss_type == "l1":
        return difference.abs().mean()
    if loss_type == "mse":
        return difference.square().mean()
    raise ValueError(f"unsupported distillation loss: {loss_type}")


def forward_with_distillation_features(
    network: nn.Module, lq: Tensor
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Validate the architecture's output-plus-feature-tuple forward contract."""
    result = network(lq, return_distillation_features=True)
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], Tensor)
        or not isinstance(result[1], tuple)
        or not all(isinstance(feature, Tensor) for feature in result[1])
    ):
        raise TypeError(
            "feature distillation networks must return (output, tuple[feature, ...])"
        )
    return result


def distillation_scale(
    options: DistillationOptions,
    current_iter: int,
    start_iter: int,
    warmup_iters: int,
) -> float:
    """Apply the existing inclusive warmup and optional linear decay schedule."""
    if current_iter < start_iter:
        return 0.0
    if warmup_iters <= 0:
        scale = 1.0
    else:
        elapsed = current_iter - start_iter + 1
        scale = min(1.0, elapsed / warmup_iters)

    decay_start = options.decay_start_iter
    end_iter = options.end_iter
    if decay_start is not None and end_iter is not None:
        if current_iter >= end_iter:
            return 0.0
        if current_iter >= decay_start:
            scale *= (end_iter - current_iter) / (end_iter - decay_start)
    return scale


def distillation_feature_representation(
    feature: Tensor, options: DistillationOptions
) -> Tensor:
    """Build the configured feature representation without learned parameters."""
    feature = feature.float()
    transform = options.feature_transform
    if transform == "none":
        return feature
    if feature.ndim != 4:
        raise ValueError(
            "distillation features must use BCHW layout, got "
            f"shape={tuple(feature.shape)}"
        )
    if transform == "channel_l2":
        return F.normalize(
            feature,
            p=2,
            dim=1,
            eps=options.epsilon,
        )
    if transform == "spatial_affinity":
        pool_size = options.feature_pool_size
        if pool_size > 1:
            feature = F.avg_pool2d(
                feature,
                kernel_size=pool_size,
                stride=pool_size,
                ceil_mode=True,
            )
        feature = feature.flatten(2)
        feature = F.normalize(
            feature,
            p=2,
            dim=1,
            eps=options.epsilon,
        )
        return torch.bmm(feature.transpose(1, 2), feature)
    raise ValueError(f"unsupported distillation feature transform: {transform}")

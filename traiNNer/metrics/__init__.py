import inspect
from copy import deepcopy
from typing import Any

import torch

from traiNNer.metrics.dists import calculate_dists
from traiNNer.metrics.learned import (
    LEARNED_TENSOR_METRICS,
    calculate_learned_metric_pt,
    quantize_metric_tensor,
)
from traiNNer.metrics.lpips import calculate_lpips
from traiNNer.metrics.neosr_pixels import calculate_neosr_psnr, calculate_neosr_ssim
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from traiNNer.metrics.topiq import calculate_topiq
from traiNNer.utils.registry import METRIC_REGISTRY

__all__ = [
    "LEARNED_TENSOR_METRICS",
    "calculate_learned_metric_pt",
    "quantize_metric_tensor",
    "calculate_dists",
    "calculate_lpips",
    "calculate_neosr_psnr",
    "calculate_neosr_ssim",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_topiq",
]


def requires_reference(metric_type: str) -> bool:
    function = METRIC_REGISTRY.get(metric_type)
    reference = inspect.signature(function).parameters.get("img2")
    return reference is not None and reference.default is inspect.Parameter.empty


def calculate_metric(
    data: dict[str, Any], opt: dict[str, Any], device: torch.device
) -> float:
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")
    opt.pop("better", None)  # evaluation ranking metadata, outside the metric signature
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt, device=device)
    return metric

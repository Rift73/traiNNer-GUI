"""Single-image learned validation metrics on the same 8-bit grid as image export."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from traiNNer.utils.img_util import img2batchedtensor
from traiNNer.utils.registry import METRIC_REGISTRY


LEARNED_TENSOR_METRICS = frozenset({
    "calculate_dists", "calculate_lpips", "calculate_topiq",
    "calculate_neosr_topiq", "calculate_topiq_nr", "calculate_butteraugli", "calculate_ssimulacra2",
})


def quantize_metric_tensor(image: Tensor) -> Tensor:
    """Round unit-range floating pixels (or normalize uint8) on their own device.

    Returns an independent FP32 tensor in [0, 1], preserving layout and channel
    order. Like tensor2img's uint8 cast, NaN becomes zero. Call once per image
    and share the result among metrics.
    """
    if image.dtype != torch.uint8 and not image.is_floating_point():
        raise TypeError("Expected floating-point image values or uint8 pixels.")
    grid = image.detach().to(dtype=torch.float32, copy=True)
    if image.dtype != torch.uint8:
        grid.nan_to_num_().clamp_(0, 1).mul_(255).round_()
    # A device divisor gives the CPU uint8 / 255 quotient on CUDA as well.
    return grid.div_(grid.new_full((), 255.0))


def learned_rgb(image: np.ndarray | Tensor, device: torch.device) -> Tensor:
    """Accept the existing NumPy image or one already normalized tensor image."""
    if not isinstance(image, Tensor):
        return img2batchedtensor(image, device, from_bgr=False)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] not in (1, 3, 4):
        raise ValueError("A learned tensor metric expects one NCHW gray, RGB or RGBA image.")
    if image.dtype != torch.float32:
        raise TypeError("Learned metrics expect FP32 pixels from quantize_metric_tensor.")
    rgb = image[:, :3].to(device)
    return rgb.expand(-1, 3, -1, -1) if rgb.shape[1] == 1 else rgb


def calculate_learned_metric_pt(
    output: Tensor, reference: Tensor | None, options: Mapping[str, Any]
) -> Tensor:
    """Score one quantized NCHW image; return a detached scalar on output.device.

    Pass quantize_metric_tensor results. TOPIQ NR ignores the reference. Model
    lifetime follows the caller's metric_session, just as for the NumPy API.
    """
    name = options["type"]
    if name not in LEARNED_TENSOR_METRICS:
        raise ValueError(f"No learned tensor scorer for {name!r}.")
    kwargs = {k: v for k, v in options.items() if k not in ("type", "better")}
    if name != "calculate_topiq_nr":
        if reference is None:
            raise ValueError(f"{name} requires a reference image.")
        kwargs["img2"] = reference
    score = METRIC_REGISTRY.get(name)(img=output, device=output.device, **kwargs)
    return score.detach().reshape(())

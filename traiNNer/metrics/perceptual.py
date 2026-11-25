"""Full-reference validation scores for opaque SDR sRGB image pairs."""

from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from traiNNer.losses._butteraugli.metric import Metric
from traiNNer.losses.quality_loss import _check_pair, _linear
from traiNNer.metrics.learned import learned_rgb
from traiNNer.utils.registry import METRIC_REGISTRY


@lru_cache(maxsize=8)
def _scorer(
    device: torch.device, butteraugli: bool
) -> tuple[torch.nn.Module, torch.device]:
    if device.type == "cuda" and torch.version.cuda is not None:
        try:
            if butteraugli:
                from ._butteraugli_cuda import ValidationButteraugli

                return ValidationButteraugli().to(device).eval(), device
            from ._ssimulacra2_cuda import ValidationSsimulacra2

            return ValidationSsimulacra2().to(device).eval(), device
        except ModuleNotFoundError as error:
            if error.name not in {"triton", "triton.language"}:
                raise
    if butteraugli:
        return Metric().to(device).eval(), device
    from ._ssimulacra2_cpu import CpuSsimulacra2

    return CpuSsimulacra2().eval(), torch.device("cpu")


def _rgb(image: np.ndarray | Tensor, device: torch.device, border: int) -> Tensor:
    rgb = learned_rgb(image, device)
    if rgb.shape[1] == 1:
        rgb = rgb.expand(-1, 3, -1, -1)
    if border < 0 or 2 * border >= min(rgb.shape[-2:]):
        raise ValueError("crop_border must leave at least one pixel on each side.")
    return rgb[..., border:-border, border:-border] if border else rgb


@torch.inference_mode()
def _score(
    img: np.ndarray | Tensor,
    img2: np.ndarray | Tensor,
    device: torch.device,
    crop_border: int,
    butteraugli: bool,
) -> Tensor | float:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    scorer, score_device = _scorer(device, butteraugli)
    output = _rgb(img, score_device, crop_border)
    reference = _rgb(img2, score_device, crop_border)
    _check_pair(output, reference, 1 if butteraugli else 8)
    with torch.autocast(device_type=output.device.type, enabled=False):
        distance = scorer(_linear(reference), _linear(output))
        score = distance.amax() if butteraugli else 100.0 - 100.0 * distance
    return score.to(device) if isinstance(img, Tensor) else score.item()


@METRIC_REGISTRY.register()
def calculate_butteraugli(
    img: np.ndarray | Tensor,
    img2: np.ndarray | Tensor,
    device: torch.device,
    crop_border: int = 0,
) -> Tensor | float:
    """Butteraugli maximum perceptual distance at 80 nits. Lower is better."""
    return _score(img, img2, device, crop_border, True)


@METRIC_REGISTRY.register()
def calculate_ssimulacra2(
    img: np.ndarray | Tensor,
    img2: np.ndarray | Tensor,
    device: torch.device,
    crop_border: int = 0,
) -> Tensor | float:
    """SSIMULACRA2 perceptual quality. A perfect match scores 100. Scores can be negative."""
    return _score(img, img2, device, crop_border, False)

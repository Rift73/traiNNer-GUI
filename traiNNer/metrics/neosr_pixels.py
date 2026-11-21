"""NeoSR pixel metrics on the RGB sample values written to an 8-bit PNG."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from traiNNer.utils.registry import METRIC_REGISTRY
from .psnr_ssim import _ssim_pth


def _luminance(grid: Tensor) -> Tensor:
    # NeoSR's published evaluator follows its NumPy float32 division, float64
    # BT.601 matrix and float32 output. A device divisor retains that rounding.
    divisor = grid.new_full((), 255, dtype=torch.float32)
    rgb = grid.float() / divisor
    if rgb.size(1) == 3:
        r, g, b = rgb.double().unbind(1)
        y = b * 24.966 + g * 128.553 + r * 65.481 + 16.0
        rgb = (y / divisor.double()).float().unsqueeze(1)
    return rgb * 255


def score_grid(kind: str, left: Tensor, right: Tensor, border: int, luma: bool, *, streamed_cpu: bool) -> Tensor:
    """Score one or more NCHW images in 0..255 units; retain per-image values."""
    if left.ndim != 4 or left.shape != right.shape or left.shape[1] not in (1, 3):
        raise ValueError("NeoSR pixel metrics expect matching grayscale or RGB images.")
    needed = 11 if kind == "ssim" else 1
    if type(border) is not int or border < 0 or min(left.shape[-2:]) - 2 * border < needed:
        raise ValueError(f"{kind.upper()} needs at least {needed} pixels on both axes after a nonnegative border crop.")
    if border:
        left, right = left[..., border:-border, border:-border], right[..., border:-border, border:-border]
    if luma:
        left, right = _luminance(left), _luminance(right)
    if kind == "psnr":
        mse = None
        if streamed_cpu:
            from .cpu_rows import streamed_mse
            mse = streamed_mse(left, right)
        if mse is None:
            mse = (left.double() - right.double()).square().mean(dim=(1, 2, 3))
        return 10 * torch.log10(255**2 / mse)
    if kind != "ssim":
        raise ValueError(f"Unknown NeoSR pixel metric {kind!r}.")
    if streamed_cpu:
        from .cpu_rows import streamed_ssim
        score = streamed_ssim(left, right, on_grid=True)
        if score is not None:
            return score
    return _ssim_pth(left.double(), right.double())


def _grid(image: np.ndarray, device) -> Tensor:
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("Pass a grayscale or HWC RGB validation image.")
    return torch.as_tensor(np.ascontiguousarray(image), device=device).permute(2, 0, 1).unsqueeze(0).float()


@METRIC_REGISTRY.register()
def calculate_neosr_psnr(img: np.ndarray, img2: np.ndarray, device: torch.device,
                         crop_border: int = 4, test_y_channel: bool = False) -> Tensor:
    return score_grid("psnr", _grid(img, device), _grid(img2, device), crop_border, test_y_channel,
                      streamed_cpu=False).mean()


@METRIC_REGISTRY.register()
def calculate_neosr_ssim(img: np.ndarray, img2: np.ndarray, device: torch.device,
                         crop_border: int = 4, test_y_channel: bool = False) -> Tensor:
    return score_grid("ssim", _grid(img, device), _grid(img2, device), crop_border, test_y_channel,
                      streamed_cpu=torch.device(device).type == "cpu").mean()

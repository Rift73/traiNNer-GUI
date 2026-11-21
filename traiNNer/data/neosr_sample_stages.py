"""Per-image decisions in NeoSR's still-image degradation pipeline."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor

from trainner_config.options import NeoSRSampleOptions
from traiNNer.data.degradations import resize_neosr_pt
from traiNNer.utils.img_process_util import filter2d


def sample_mask(batch: Tensor, probability: float) -> Tensor:
    return torch.rand((batch.size(0), 1, 1, 1), device=batch.device) < probability


def clean_resize(source: Tensor, size: tuple[int, int], settings: NeoSRSampleOptions) -> Tensor:
    modes = settings.clean_mode
    if settings.clean_mode_prob is None:
        mode = random.choice(modes)
    else:
        mode = random.choices(modes, settings.clean_mode_prob)[0]
    return resize_neosr_pt(source, mode, size=size)


def sharpen_samples(image: Tensor, settings: NeoSRSampleOptions) -> Tensor:
    p = settings.sharpen_prob
    if p <= 0:
        return image
    batch = image.size(0)
    sigma_min, sigma_max = settings.sharpen_sigma
    amount_min, amount_max = settings.sharpen_amount
    sigma = torch.rand((batch, 1), device=image.device, dtype=image.dtype) * (sigma_max - sigma_min) + sigma_min
    amount = torch.rand((batch, 1, 1, 1), device=image.device, dtype=image.dtype) * (amount_max - amount_min) + amount_min
    radius = math.ceil(3 * sigma_max)
    if min(image.shape[-2:]) <= radius:
        raise ValueError(f"The final LQ side must exceed {radius} pixels for this sharpening sigma.")
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    taps = torch.exp(-x.square() / (2 * sigma.square()))
    taps /= taps.sum(1, keepdim=True)
    blurred = filter2d(image.contiguous(), taps.unsqueeze(2) * taps.unsqueeze(1))
    sharp = (image + amount * (image - blurred)).clamp(0, 1)
    return sharp if p == 1 else torch.where(sample_mask(image, p), sharp, image)

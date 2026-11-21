"""Stateless high-frequency target-noise generation for RealESRGAN."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from numpy.random import Generator

    from traiNNer.utils.redux_options import ReduxOptions

BetaNoiseSampler = Callable[
    [int, int, int, Tensor, Tensor, torch.dtype, torch.device], Tensor
]


def sample_beta_noise(
    channels: int,
    height: int,
    width: int,
    shape_a: Tensor,
    shape_b: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Draw Beta samples using Torch's seeded generator in the original shape."""
    distribution = torch.distributions.Beta(
        shape_a.view(-1, 1, 1, 1).expand(-1, channels, height, width),
        shape_b.view(-1, 1, 1, 1).expand(-1, channels, height, width),
    )
    return distribution.sample().to(device=device, dtype=dtype)


def generate_hf_noise(
    img: Tensor,
    options: ReduxOptions,
    rng: Generator,
    beta_sampler: BetaNoiseSampler,
) -> Tensor:
    """Generate HF noise while preserving project/Torch RNG draw order.

    Beta sampling remains an explicit input so RealESRGANModel subclasses can
    keep overriding their existing sampler method. All samples are constructed
    in float32 before the final cast to the input image dtype.
    """
    batch_size, channels, height, width = img.shape
    sample_dtype = torch.float32
    device = img.device

    shape_a = torch.as_tensor(
        rng.uniform(*options.otf_hf_noise_beta_shape_range, size=batch_size),
        device=device,
        dtype=sample_dtype,
    )
    if options.otf_hf_noise_beta_offset_range is not None:
        shape_b = shape_a + torch.as_tensor(
            rng.uniform(*options.otf_hf_noise_beta_offset_range, size=batch_size),
            device=device,
            dtype=sample_dtype,
        )
    else:
        shape_b = torch.as_tensor(
            rng.uniform(*options.otf_hf_noise_beta_shape_range, size=batch_size),
            device=device,
            dtype=sample_dtype,
        )
    alpha = torch.as_tensor(
        rng.uniform(*options.otf_hf_noise_alpha_range, size=batch_size),
        device=device,
        dtype=sample_dtype,
    ).view(-1, 1, 1, 1)

    noise = beta_sampler(
        channels,
        height,
        width,
        shape_a,
        shape_b,
        sample_dtype,
        device,
    )

    if channels > 1 and options.otf_hf_noise_gray_prob > 0:
        gray_mask = torch.as_tensor(
            rng.uniform(size=batch_size) < options.otf_hf_noise_gray_prob,
            device=device,
            dtype=torch.bool,
        )
        gray_count = int(gray_mask.sum().item())
        if gray_count > 0:
            gray_noise = beta_sampler(
                1,
                height,
                width,
                shape_a[gray_mask],
                shape_b[gray_mask],
                sample_dtype,
                device,
            ).expand(-1, channels, -1, -1)
            noise[gray_mask] = gray_noise

    if options.otf_hf_noise_normalize:
        noise = noise - noise.mean(dim=(1, 2, 3), keepdim=True)
        noise = noise / (noise.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        noise = torch.clamp(noise, -3, 3) * alpha
    else:
        noise = (noise - 0.5) * (2 * alpha)

    return noise.to(dtype=img.dtype)

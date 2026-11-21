"""Window self-attention with a learned relative-position table on the fused kernel.

This adapts Swin-convention window attention, ``softmax(q k^T * scale + bias +
mask) v`` with ``bias = table[relative_position_index]``, to the fused dense
relative-bias attention in :mod:`traiNNer.ops.drft_ocab_attention`.

Index conventions.  A Swin table for an ``h x w`` window has ``(2h-1)(2w-1)``
rows and is read at ``(qy - ky + h - 1) * (2w - 1) + (qx - kx + w - 1)``
(query minus key).  The fused kernel reads its table at
``(ky - qy + h - 1) * (2w - 1) + (kx - qx + w - 1)`` (key minus query).  The two
row numbers always add up to ``(2h-1)(2w-1) - 1``, so the kernel receives the
table with its rows reversed.  The reversal runs inside the forward, so the
gradient reaches the original parameter.

Shifted windows.  The usual additive mask (0 within a region of the cyclically
shifted image, -100 across regions) has no kernel input.  It is reproduced
with two extra query/key channels holding one of four unit vectors at right
angles per region, scaled so that pairs in the same region gain at least 100
after ``scale`` and every other pair at least 100 less.  Values get two zero
channels and the extra output channels are removed.  This needs the head width
plus two to stay inside the kernel's power-of-two channel block.
"""

from __future__ import annotations

import math

import torch

from traiNNer.ops.drft_ocab_attention import ocab_attention

_REGION_CHANNELS = 2
_MAX_REGIONS = 4
_MASK_GAP = 100.0
# Covers the rounding of the region channels to bf16/fp16 (relative error
# below 2**-8), so that the gap after rounding is never under _MASK_GAP.
_ROUNDING_MARGIN = 1.01


def _channel_block(head_dim: int) -> int:
    return 1 << max(head_dim - 1, 0).bit_length()


def supports_region_channels(head_dim: int) -> bool:
    """True when ``head_dim + 2`` fits the kernel's channel block of ``head_dim``."""
    return _channel_block(head_dim + _REGION_CHANNELS) == _channel_block(head_dim)


def mask_to_regions(mask: torch.Tensor) -> torch.Tensor:
    """Turn an additive 0 / -100 shift mask (windows, tokens, tokens) into labels.

    Returns an int64 tensor (windows, tokens) with labels 0..3 such that a
    query may attend to a key exactly when both carry the same label.  Raises
    ``ValueError`` when the mask is not such a same-region rule or a window
    holds more than four regions.  Intended to run once per mask.
    """
    if mask.dim() != 3 or mask.shape[1] != mask.shape[2]:
        raise ValueError(
            f"expected a (windows, tokens, tokens) mask, got {tuple(mask.shape)}"
        )
    mask = mask.detach().to(device="cpu", dtype=torch.float32)
    if not bool(((mask == 0) | (mask == -_MASK_GAP)).all()):
        raise ValueError("mask entries must be 0 or -100")
    allowed = mask == 0
    windows, tokens, _ = mask.shape
    labels = torch.empty(windows, tokens, dtype=torch.long)
    for window in range(windows):
        first_allowed = allowed[window].to(torch.int32).argmax(dim=1)
        _, inverse = torch.unique(first_allowed, return_inverse=True)
        labels[window] = inverse
    if not torch.equal(labels[:, :, None] == labels[:, None, :], allowed):
        raise ValueError("mask is not a same-region rule")
    if int(labels.max()) >= _MAX_REGIONS:
        raise ValueError(
            f"a window holds {int(labels.max()) + 1} regions; "
            f"at most {_MAX_REGIONS} are supported"
        )
    return labels


def _region_channels(
    regions: torch.Tensor,
    windows: int,
    num_heads: int,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    windows_per_image, tokens = regions.shape
    if windows % windows_per_image:
        raise ValueError(
            f"{windows} windows are not a whole number of images of "
            f"{windows_per_image} windows"
        )
    magnitude = math.sqrt(_MASK_GAP / scale) * _ROUNDING_MARGIN
    # Labels 0..3 -> (1, 0), (0, 1), (-1, 0), (0, -1): equal labels give
    # magnitude**2, different labels 0 or -magnitude**2.
    first = (regions == 0).to(dtype) - (regions == 2).to(dtype)
    second = (regions == 1).to(dtype) - (regions == 3).to(dtype)
    channels = torch.stack((first, second), dim=-1) * magnitude
    # The window order matches the hand-written mask broadcast
    # attn.view(windows // windows_per_image, windows_per_image, ...).
    return (
        channels[None, :, None]
        .expand(
            windows // windows_per_image,
            windows_per_image,
            num_heads,
            tokens,
            _REGION_CHANNELS,
        )
        .reshape(windows, num_heads, tokens, _REGION_CHANNELS)
    )


def relative_bias_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    table: torch.Tensor,
    window_height: int,
    window_width: int,
    scale: float,
    regions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Swin-convention window self-attention.

    q, k, v: (windows, heads, tokens, head_dim) in bf16 or fp16 on CUDA, any
    strides; q is not pre-scaled.  table: Swin-ordered
    ((2h-1)(2w-1), heads).  regions: optional (windows_per_image, tokens)
    labels 0..3 from :func:`mask_to_regions`.  Returns
    (windows, heads, tokens, head_dim); without regions it keeps q's stride
    order, so a token-major q gives a token-major output.
    """
    kernel_table = table.flip(0)
    if regions is None:
        return ocab_attention(
            q,
            k,
            v,
            kernel_table,
            window_height,
            window_width,
            window_height,
            window_width,
            scale,
            True,
        )

    windows, num_heads, _, head_dim = q.shape
    if not supports_region_channels(head_dim):
        raise ValueError(
            f"head width {head_dim} leaves no room for "
            f"{_REGION_CHANNELS} region channels"
        )
    channels = _region_channels(regions, windows, num_heads, scale, q.dtype)
    out = ocab_attention(
        torch.cat((q, channels), dim=-1),
        torch.cat((k, channels), dim=-1),
        torch.cat((v, torch.zeros_like(channels)), dim=-1),
        kernel_table,
        window_height,
        window_width,
        window_height,
        window_width,
        scale,
        False,
    )
    return out[..., :head_dim]


__all__ = [
    "mask_to_regions",
    "relative_bias_window_attention",
    "supports_region_channels",
]

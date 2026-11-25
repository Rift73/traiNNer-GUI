"""CUDA evaluation of the 11x11 Gaussian SSIM window in double precision."""


# Triton infers device argument types; uppercase kernel parameters identify buffers and dimensions.
# ruff: noqa: ANN001, ANN202, N803

from functools import lru_cache

import cv2
import torch
import triton
import triton.language as tl


@lru_cache(maxsize=8)
def _window(device: torch.device) -> torch.Tensor:
    taps = cv2.getGaussianKernel(11, 1.5).ravel()
    return torch.from_numpy(taps).to(device=device)


@triton.jit
def _horizontal(
    A,
    B,
    Taps,
    Out,
    height,
    out_width,
    channels,
    an,
    ac,
    ay,
    ax,
    bn,
    bc,
    by,
    bx,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    plane = tl.program_id(2)
    batch = plane.to(tl.int64) // channels
    channel = plane % channels
    left = (
        batch * an
        + channel.to(tl.int64) * ac
        + row.to(tl.int64) * ay
        + x.to(tl.int64) * ax
    )
    right = (
        batch * bn
        + channel.to(tl.int64) * bc
        + row.to(tl.int64) * by
        + x.to(tl.int64) * bx
    )
    valid = x < out_width
    a = tl.full((BLOCK,), 0, tl.float64)
    b = tl.full((BLOCK,), 0, tl.float64)
    aa = tl.full((BLOCK,), 0, tl.float64)
    bb = tl.full((BLOCK,), 0, tl.float64)
    ab = tl.full((BLOCK,), 0, tl.float64)
    for k in tl.static_range(11):
        l = tl.load(A + left + k * ax, valid, 0).to(tl.float64) * 255.0
        r = tl.load(B + right + k * bx, valid, 0).to(tl.float64) * 255.0
        w = tl.load(Taps + k)
        a += l * w
        b += r * w
        aa += l * l * w
        bb += r * r * w
        ab += l * r * w
    plane_size = height.to(tl.int64) * out_width
    pos = plane * 5 * plane_size + row.to(tl.int64) * out_width + x
    tl.store(Out + pos, a, valid)
    tl.store(Out + pos + plane_size, b, valid)
    tl.store(Out + pos + 2 * plane_size, aa, valid)
    tl.store(Out + pos + 3 * plane_size, bb, valid)
    tl.store(Out + pos + 4 * plane_size, ab, valid)


@triton.jit
def _vertical(Input, Taps, Sum, height, width, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    plane = tl.program_id(2)
    valid = x < width
    plane_size = height.to(tl.int64) * width
    pos = plane * 5 * plane_size + row.to(tl.int64) * width + x
    a = tl.full((BLOCK,), 0, tl.float64)
    b = tl.full((BLOCK,), 0, tl.float64)
    aa = tl.full((BLOCK,), 0, tl.float64)
    bb = tl.full((BLOCK,), 0, tl.float64)
    ab = tl.full((BLOCK,), 0, tl.float64)
    for k in tl.static_range(11):
        ptr = Input + pos + k * width
        w = tl.load(Taps + k)
        a += tl.load(ptr, valid, 0) * w
        b += tl.load(ptr + plane_size, valid, 0) * w
        aa += tl.load(ptr + 2 * plane_size, valid, 0) * w
        bb += tl.load(ptr + 3 * plane_size, valid, 0) * w
        ab += tl.load(ptr + 4 * plane_size, valid, 0) * w
    a2, b2, cross = a * a, b * b, a * b
    cs = (2 * (ab - cross) + 58.5225) / ((aa - a2) + (bb - b2) + 58.5225)
    score = ((2 * cross + 6.5025) / (a2 + b2 + 6.5025)) * cs
    out = (plane.to(tl.int64) * (height - 10) + row) * tl.num_programs(
        0
    ) + tl.program_id(0)
    tl.store(Sum + out, tl.sum(tl.where(valid, score, 0), axis=0))


def gaussian_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Score each NCHW image using the valid separable Gaussian in float64."""
    n, c, h, w = a.shape
    oh, ow = h - 10, w - 10
    taps = _window(a.device)
    tmp = torch.empty((n * c, 5, h, ow), dtype=torch.float64, device=a.device)
    blocks = triton.cdiv(ow, 128)
    sums = torch.empty((n, c, oh, blocks), dtype=torch.float64, device=a.device)
    with torch.cuda.device(a.device):
        _horizontal[(blocks, h, n * c)](
            a, b, taps, tmp, h, ow, c, *a.stride(), *b.stride(), 128
        )
        _vertical[(blocks, oh, n * c)](tmp, taps, sums, h, ow, 128)
    return sums.sum((1, 2, 3)) / (c * oh * ow)

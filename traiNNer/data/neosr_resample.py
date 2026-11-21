"""Separable OTF filters adapted from NeoSR 3fe738f4, neosr/data/resample.py (Apache-2.0).

Weights use double-precision half-pixel coordinates and the input dtype after normalization.
Only compact, clipped filter footprints are materialized. Random resize shapes are never cached.
For the optional contiguous FP32 CPU filter, build with `make -C native/studio/resample`
before starting the process. Other layouts and differentiable inputs use Torch.
"""

import ctypes
import math
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from trainner_config.resampling import NEOSR_AA_MODES

_RADIUS = {
    "aa_box": 0.5, "aa_bilinear": 1, "aa_bicubic": 2, "aa_mitchell": 2,
    "aa_lanczos2": 2, "aa_lanczos3": 3,
}
_MAX_COEFFICIENTS = 32768
_MAX_GATHER_ELEMENTS = 262144


@lru_cache(maxsize=1)
def _native_filter():
    """Load the optional CPU operator once. Training never invokes a compiler."""
    library = Path(__file__).resolve().parents[2] / "native/studio/resample/build/clang/resample.so"
    if not library.is_file():
        return None
    function = ctypes.CDLL(str(library)).neosr_filter_rows_f32
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 7 + [ctypes.c_bool]
    function.restype = None
    return function


def _kernel(mode: str, x: Tensor) -> Tensor:
    if mode == "aa_bilinear":
        return (1 - x.abs()).clamp_min_(0)
    if mode in ("aa_bicubic", "aa_mitchell"):
        b, c = (0.0, 0.5) if mode == "aa_bicubic" else (1 / 3, 1 / 3)
        x = x.abs()
        near = (12 - 9 * b - 6 * c) * x**3 + (-18 + 12 * b + 6 * c) * x**2 + (6 - 2 * b)
        far = (
            (-b - 6 * c) * x**3 + (6 * b + 30 * c) * x**2
            + (-12 * b - 48 * c) * x + (8 * b + 24 * c)
        )
        return torch.where(x < 1, near, torch.where(x < 2, far, 0.0)) / 6
    a = _RADIUS[mode]
    return torch.where(x.abs() < a, torch.sinc(x) * torch.sinc(x / a), 0.0)


def _raw_weights(mode: str, n: int, m: int, rows: Tensor, columns: Tensor, centers: Tensor) -> Tensor:
    if mode == "aa_box":
        # The lower edge belongs to the box on downsampling. At an exact nearest
        # neighbour tie on upsampling, the lower input index wins.
        if n >= m:
            pos = (2 * columns + 1) * m
            return ((2 * rows * n <= pos) & (pos < 2 * (rows + 1) * n)).double()
        pos = (2 * rows + 1) * n
        return ((2 * columns * m < pos) & (pos <= 2 * (columns + 1) * m)).double()
    return _kernel(mode, (columns - centers) / max(n / m, 1.0))


def _tap_blocks(
    x: Tensor, mode: str, n: int, m: int, begin: int, end: int,
) -> Iterator[tuple[int, Tensor, Tensor, bool]]:
    stretch = max(1.0, n / m)
    radius = stretch * _RADIUS[mode]
    taps = min(n, math.ceil(2 * radius) + 1)
    block_rows = max(1, _MAX_COEFFICIENTS // taps)
    for row in range(begin, end, block_rows):
        rows = torch.arange(row, min(row + block_rows, end), device=x.device)[:, None]
        centers = (rows.double() + 0.5) * (n / m) - 0.5
        first = (centers - radius).floor().long().clamp_(0, n - taps)
        block_taps = min(taps, _MAX_COEFFICIENTS // len(rows))

        def weights(offset: int) -> Tensor:
            columns = first + torch.arange(offset, min(taps, offset + block_taps), device=x.device)
            return _raw_weights(mode, n, m, rows, columns, centers)

        if block_taps == taps:
            w = weights(0)
            yield row - begin, first[:, 0], (w / w.sum(1, keepdim=True)).to(x.dtype), True
        else:
            # Very large reductions can put more than a block of input samples
            # in one footprint. Normalize over the entire footprint before casting.
            total = torch.zeros_like(centers)
            for offset in range(0, taps, block_taps):
                total += weights(offset).sum(1, keepdim=True)
            for offset in range(0, taps, block_taps):
                w = (weights(offset) / total).to(x.dtype)
                yield row - begin, first[:, 0] + offset, w, offset == 0


def _axis(x: Tensor, axis: int, m: int, mode: str, begin: int, end: int) -> Tensor:
    n = x.shape[axis]
    if n == m:
        return x.narrow(axis, begin, end - begin)
    blocks = _tap_blocks(x, mode, n, m, begin, end)
    shape = list(x.shape)
    shape[axis] = end - begin
    if (x.device.type == "cpu" and x.dtype == torch.float32 and x.is_contiguous()
            and not x.is_neg() and not (x.requires_grad and torch.is_grad_enabled())):
        native = _native_filter()
        if native is not None:
            out = x.new_empty(shape)
            outer = math.prod(x.shape[:axis])
            inner = math.prod(x.shape[axis + 1:])
            for row, starts, w, first in blocks:
                native(x.data_ptr(), out.data_ptr(), starts.data_ptr(), w.data_ptr(),
                       outer, n, inner, end - begin, row, len(starts), w.size(1), first)
            return out

    # Put the sample axis first and tile the independent channel/spatial values
    # as well as the filter footprint. A gather never exceeds the workspace cap.
    moved = x.movedim(axis, 0)
    lanes = math.prod(moved.shape[1:])
    src = moved.reshape(n, lanes)
    acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    out = torch.empty((end - begin, lanes), device=x.device, dtype=acc_dtype)
    for row, starts, w, first in blocks:
        indices = starts[:, None] + torch.arange(w.size(1), device=x.device)
        coeff = w.to(acc_dtype)[:, :, None]
        lane_block = max(1, _MAX_GATHER_ELEMENTS // w.numel())
        for col in range(0, lanes, lane_block):
            part = src[:, col:col + lane_block][indices]
            value = (part * coeff).sum(dim=1)
            dst = out[row:row + len(starts), col:col + lane_block]
            if first:
                dst.copy_(value)
            else:
                dst.add_(value)
    return out.to(x.dtype).reshape(end - begin, *moved.shape[1:]).movedim(0, axis)


def resample(x: Tensor, size: tuple[int, int], mode: str) -> Tensor:
    """Resize the last two axes of a floating tensor. Preserve dtype, device and RGB range."""
    if mode not in NEOSR_AA_MODES:
        raise ValueError(f"Unknown NeoSR antialiased resize mode {mode!r}. Choose one of {NEOSR_AA_MODES}.")
    if x.ndim < 2 or x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("NeoSR AA resize requires a floating-point image with spatial axes.")
    h, w = x.shape[-2:]
    oh, ow = size
    if min(h, w, oh, ow) < 1:
        raise ValueError(f"NeoSR AA resize needs positive spatial sizes, got {(h, w)} -> {size}.")
    if (h, w) == size:
        return x

    with torch.autocast(x.device.type, enabled=False):
        # ATen implements the same PIL triangle and Keys (-0.5) cubic kernels.
        # Singleton axes use the explicit filter, including the vertical-only case.
        if (x.device.type == "cpu" and x.ndim == 4 and x.dtype in (torch.float32, torch.float64)
                and mode in ("aa_bilinear", "aa_bicubic") and min(h, w, oh, ow) > 1
                and (h - oh) * (w - ow) >= 0):
            return F.interpolate(x, size=size, mode=mode[3:], align_corners=False, antialias=True)

        y_axis, x_axis = x.ndim - 2, x.ndim - 1
        # Height first, matching the reference rounding between passes. When height expands
        # and width shrinks, a full intermediate could dwarf both input and output.
        if oh > h and ow < w:
            row_budget = max(1, (1 << 20) // max(1, x.numel() // h))
            if oh > row_budget:
                out = x.new_empty((*x.shape[:-2], oh, ow))
                for row in range(0, oh, row_budget):
                    end = min(oh, row + row_budget)
                    band = _axis(x, y_axis, oh, mode, row, end)
                    out[..., row:end, :] = _axis(band, x_axis, ow, mode, 0, ow)
                return out
        y = _axis(x, y_axis, oh, mode, 0, oh)
        return _axis(y, x_axis, ow, mode, 0, ow)

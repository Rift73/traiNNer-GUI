"""Aligned x2 DVD response and the pair geometry that keeps it aligned.

The response (traiNNer/csrc/training_cuda.cu) halves RGB with centred x2
Catmull-Rom, filters luma and the two colour differences with the symmetric
17x17 kernels of traiNNer/csrc/dvd_coefficients.hpp and limits the filtered
colour to the RGB gamut; strength mixes the plain halving (0) with the full
response (1). CUDA float32 tensors run the PTX kernels; every other tensor runs
the same math in torch, with the coefficients read from the same sources.

Pair geometry: the dataset draws its flips and rotation and crops a work
canvas from the oriented image. The response runs on the image in its
canonical orientation instead, over a window with real image context around
the patch; LQ and GT are cropped from it and then oriented like the canvas.
"""

import ctypes
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.utils.data import default_collate

from traiNNer.data.ptx_cuda import PtxModule
from traiNNer.utils.rng import RNG

# An artificial patch edge needs this much real image beyond it, in LQ pixels:
# the 17x17 filter's radius of 8 plus the halving's reach of 2.
DVD_CONTEXT_LQ = 10

_csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
_RESPONSE_PTX = PtxModule("training_cuda.ptx")
_BLOCK = (16, 16, 1)


@dataclass(frozen=True)
class Window:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class DvdGeometry:
    """A sample's canonical source size, its work canvas and the D4 it drew.

    width and height are the preprocessed source's, before the flips and the
    rotation. The canvas is the square the dataset cut from the oriented,
    reflect-padded image; its origin is in the oriented image (the padding
    subtracted), so it is negative on a padded axis.
    """

    width: int
    height: int
    canvas_left: int
    canvas_top: int
    canvas_size: int
    hflip: bool
    vflip: bool
    clockwise: bool

    def canonical_origin(self, left: int, top: int, size: int) -> tuple[int, int]:
        """The canonical origin of an oriented square of the given size."""
        if self.clockwise:
            left, top = top, self.height - left - size
        if self.hflip:
            left = self.width - left - size
        if self.vflip:
            top = self.height - top - size
        return left, top

    def window(self) -> Window:
        """The canonical source region the response reads: the canvas with real
        context beyond it, on the even source grid, inside the even image."""
        x, y = self.canonical_origin(
            self.canvas_left, self.canvas_top, self.canvas_size
        )
        left, width = _window_axis(x, self.canvas_size, self.width)
        top, height = _window_axis(y, self.canvas_size, self.height)
        return Window(left, top, width, height)

    def bounds(self, gt_size: int) -> tuple[int, int, int, int]:
        """Inclusive LQ origin ranges in the window, x then y, for a gt_size
        patch with real context at every artificial edge."""
        window = self.window()
        patch = gt_size // 2
        # Only an even image edge can use the response's own edge rule; a
        # window edge inside the image, or a trimmed odd last pixel, needs context.
        right_guard = 0 if window.left + window.width == self.width else DVD_CONTEXT_LQ
        bottom_guard = (
            0 if window.top + window.height == self.height else DVD_CONTEXT_LQ
        )
        return (
            DVD_CONTEXT_LQ if window.left else 0,
            window.width // 2 - patch - right_guard,
            DVD_CONTEXT_LQ if window.top else 0,
            window.height // 2 - patch - bottom_guard,
        )

    def has_context(self, gt_size: int) -> bool:
        lo_x, hi_x, lo_y, hi_y = self.bounds(gt_size)
        return lo_x <= hi_x and lo_y <= hi_y

    def resident(self, gt_size: int) -> bool:
        """Whether the canvas is the whole source, centred in its reflect
        padding, so the delivered canvas already holds the response's input."""
        margin = (gt_size - self.canvas_size) // 2
        return (
            self.width == self.height == gt_size
            and self.canvas_left == self.canvas_top == margin
        )


@dataclass(frozen=True)
class DvdSample:
    """A sample's DVD geometry and, unless it is resident, its canonical source
    (and target) window as CPU float32 CHW."""

    geometry: DvdGeometry
    source: Tensor | None
    target: Tensor | None


def _window_axis(origin: int, length: int, extent: int) -> tuple[int, int]:
    context = 2 * DVD_CONTEXT_LQ
    end = extent // 2 * 2
    first = max(0, min(end, (origin - context) // 2 * 2))
    stop = min(end, max(first, (origin + length + context + 1) // 2 * 2))
    return first, stop - first


def orient(image: Tensor, geometry: DvdGeometry) -> Tensor:
    """Apply the sample's flips, then its clockwise rotation, as the dataset does."""
    if geometry.hflip:
        image = image.flip(-1)
    if geometry.vflip:
        image = image.flip(-2)
    return image.rot90(-1, (-2, -1)) if geometry.clockwise else image


def unorient(image: Tensor, geometry: DvdGeometry) -> Tensor:
    """Undo orient."""
    if geometry.clockwise:
        image = image.rot90(1, (-2, -1))
    if geometry.vflip:
        image = image.flip(-2)
    return image.flip(-1) if geometry.hflip else image


def collate_dvd(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """default_collate, keeping each sample's "dvd" entry as one list item."""
    sidecars = [sample["dvd"] for sample in samples if "dvd" in sample]
    batch = default_collate(
        [{key: value for key, value in s.items() if key != "dvd"} for s in samples]
    )
    if sidecars:
        if len(sidecars) != len(samples):
            raise ValueError("Every sample of a DVD batch needs its DVD geometry")
        batch["dvd"] = sidecars
    return batch


def _table(file_name: str, name: str) -> list[float]:
    """The values of the C array `name` in a file of traiNNer/csrc."""
    text = (_csrc_dir / file_name).read_text()
    match = re.search(rf"\b{name}\[(\d+)\]\s*=\s*\{{(.*?)\}};", text, re.DOTALL)
    if match is None:
        raise ValueError(f"{name} not found in {file_name}")
    values = []
    for term in match.group(2).split(","):
        numerator, _, denominator = term.strip().partition("/")
        if numerator:
            value = float(numerator.strip().rstrip("f"))
            values.append(value / float(denominator) if denominator else value)
    if len(values) != int(match.group(1)):
        raise ValueError(f"{name} in {file_name} has {len(values)} values")
    return values


@cache
def _coefficients(device: torch.device) -> tuple[Tensor, Tensor]:
    """The halving taps (8) and the luma, chroma, chroma filters (3, 1, 17, 17)."""
    catrom = torch.tensor(_table("training_cuda.cu", "catrom"), device=device)
    y_kernel, c_kernel = (
        torch.tensor(_table("dvd_coefficients.hpp", name), device=device).view(17, 17)
        for name in ("y_kernel", "c_kernel")
    )
    return catrom, torch.stack([y_kernel, c_kernel, c_kernel]).unsqueeze(1)


def _mirror(index: Tensor, n: int) -> Tensor:
    """Mirror without repeating the edge sample, as the kernels' mirror_edge."""
    if n == 1:
        return torch.zeros_like(index)
    period = 2 * (n - 1)
    index = index.remainder(period)
    return torch.where(index < n, index, period - index)


def _response_torch(source: Tensor, strength: float) -> Tensor:
    catrom, filters = _coefficients(source.device)
    _, _, h, w = source.shape
    taps = torch.arange(-3, 5, device=source.device)
    columns = _mirror(2 * torch.arange(w // 2, device=source.device)[:, None] + taps, w)
    rows = _mirror(2 * torch.arange(h // 2, device=source.device)[:, None] + taps, h)
    half = (source[..., columns] * catrom).sum(-1)
    rgb = (half[..., rows, :] * catrom[:, None]).sum(-2).clamp(0, 1)
    luma = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    planes = torch.stack([luma, rgb[:, 0] - luma, rgb[:, 2] - luma], 1)

    oh, ow = h // 2, w // 2
    pad_rows = _mirror(torch.arange(-8, oh + 8, device=source.device), oh)
    pad_columns = _mirror(torch.arange(-8, ow + 8, device=source.device), ow)
    filtered = conv2d(planes[..., pad_rows, :][..., pad_columns], filters, groups=3)
    lum = filtered[:, 0].clamp(0, 1)
    dr, db = filtered[:, 1], filtered[:, 2]
    dg = -(0.299 * dr + 0.114 * db) / 0.587
    delta = torch.stack([dr, dg, db], 1)
    hi, lo = delta.amax(1), delta.amin(1)
    # Scale the colour difference back into the gamut around the filtered luma.
    t = torch.where(lum + hi > 1, (1 - lum) / hi, torch.ones_like(lum))
    t = torch.where(lum + lo < 0, torch.minimum(t, -lum / lo), t)
    base, cr, cb = planes.unbind(1)
    before = torch.stack(
        [base + cr, base - (0.299 * cr + 0.114 * cb) / 0.587, base + cb], 1
    )
    after = (lum[:, None] + t[:, None] * delta).clamp(0, 1)
    return (before + strength * (after - before)).clamp(0, 1)


def _grid(width: int, height: int, depth: int) -> tuple[int, int, int]:
    return ((width + 15) // 16, (height + 15) // 16, depth)


def _response_cuda(source: Tensor, strength: float) -> Tensor:
    b, _, h, w = source.shape
    rgb = source.contiguous()
    half = rgb.new_empty((b, 3, h, w // 2))
    planes = rgb.new_empty((b, 3, h // 2, w // 2))
    out = torch.empty_like(planes)
    _RESPONSE_PTX.launch(
        "dvd_horizontal",
        _grid(w // 2, h, b * 3),
        _BLOCK,
        rgb,
        half,
        ctypes.c_int(w),
        ctypes.c_int(h),
    )
    _RESPONSE_PTX.launch(
        "dvd_vertical",
        _grid(w // 2, h // 2, b),
        _BLOCK,
        half,
        planes,
        ctypes.c_int(w // 2),
        ctypes.c_int(h),
    )
    _RESPONSE_PTX.launch(
        "dvd_aligned_fir",
        _grid(w // 2, h // 2, b),
        _BLOCK,
        planes,
        out,
        ctypes.c_int(w // 2),
        ctypes.c_int(h // 2),
        ctypes.c_float(strength),
    )
    return out


@torch.no_grad()
def dvd_response(source: Tensor, strength: float = 1.0) -> Tensor:
    """The aligned x2 DVD LQ of BCHW RGB in [0, 1] with even height and width,
    as float32 without rounding.

    strength 0 gives the plain centred x2 Catmull-Rom halving, 1 the full response.
    """
    if source.ndim != 4 or source.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB, got shape {tuple(source.shape)}")
    b, _, h, w = source.shape
    if min(b, h, w) < 1 or h % 2 or w % 2:
        raise ValueError(f"The x2 DVD response needs even sizes, got {h}x{w}")
    if not 0 <= strength <= 1:
        raise ValueError(f"DVD strength must be in [0, 1], got {strength}")
    if (
        source.is_cuda
        and source.dtype == torch.float32
        and b * 3 <= 65535
        and h <= 16 * 65535
        and h * w < 2**31
    ):
        return _response_cuda(source, strength)
    return _response_torch(source.float(), strength)


def _upload(windows: list[Tensor], device: torch.device) -> Tensor:
    """Stack same-sized CPU windows in pinned memory and move them to device."""
    host = torch.empty(
        (len(windows), *windows[0].shape),
        dtype=torch.float32,
        pin_memory=device.type == "cuda",
    )
    for row, window in enumerate(windows):
        host[row].copy_(window)
    return host.to(device, non_blocking=True)


@torch.no_grad()
def dvd_pair(
    samples: Sequence[DvdSample],
    source: Tensor,
    target: Tensor | None,
    gt_size: int,
    strength: float,
    *,
    center_height: bool,
    center_width: bool,
) -> tuple[Tensor, Tensor]:
    """The LQ and GT patches of a DVD batch from its delivered work canvases.

    source and target are the delivered (B, 3, S, S) canvases. The crop is drawn
    like paired_random_crop's (top, then left, inclusive, from the project RNG;
    a centred axis draws nothing), mapped into each sample's canonical window
    and moved onto the even grid and inside the real context. A resident sample
    takes its canonical source from the centre of its source canvas and its GT
    from the centre of the target canvas; the others upload their windows. The
    response runs once per window size. Every sample must have DVD context.
    LQ is contiguous; GT has the strides of a crop of the GT canvas.
    """
    count, _, height, width = source.shape
    if len(samples) != count or height != width or gt_size % 2:
        raise ValueError(
            "DVD samples must match a square canvas batch and an even gt_size"
        )
    limit = (height - gt_size) // 2
    rng = RNG.get_rng()
    top = limit // 2 if center_height else int(rng.integers(0, limit, endpoint=True))
    left = limit // 2 if center_width else int(rng.integers(0, limit, endpoint=True))

    patch = gt_size // 2
    supervision = source if target is None else target
    lq_rows: list[Tensor | None] = [None] * count
    gt_rows: list[Tensor | None] = [None] * count
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, sample in enumerate(samples):
        key = () if sample.source is None else tuple(sample.source.shape)
        groups.setdefault(key, []).append(index)
    for key, indices in groups.items():
        geometries = [samples[index].geometry for index in indices]
        if key:
            windows = _upload([samples[i].source for i in indices], source.device)
            targets = (
                windows
                if target is None
                else _upload([samples[i].target for i in indices], source.device)
            )
        else:
            margin = [-g.canvas_left for g in geometries]
            windows = torch.stack(
                [
                    unorient(source[i, :, m : m + gt_size, m : m + gt_size], g)
                    for i, m, g in zip(indices, margin, geometries, strict=True)
                ]
            )
            targets = None
            for i, m in zip(indices, margin, strict=True):
                gt_rows[i] = supervision[i, :, m : m + gt_size, m : m + gt_size]
        low = dvd_response(windows.contiguous(), strength)
        for row, (index, geometry) in enumerate(zip(indices, geometries, strict=True)):
            x, y = geometry.canonical_origin(
                geometry.canvas_left + 2 * left, geometry.canvas_top + 2 * top, gt_size
            )
            window = geometry.window()
            lo_x, hi_x, lo_y, hi_y = geometry.bounds(gt_size)
            x = min(hi_x, max(lo_x, (x - window.left) // 2))
            y = min(hi_y, max(lo_y, (y - window.top) // 2))
            lq_rows[index] = orient(low[row, :, y : y + patch, x : x + patch], geometry)
            if targets is not None:
                gt_rows[index] = orient(
                    targets[row, :, 2 * y : 2 * y + gt_size, 2 * x : 2 * x + gt_size],
                    geometry,
                )
    # GT is a crop of a canvas-shaped buffer, so it has the strides of the
    # ordinary chain's crop of the delivered canvas (compiled losses guard on them).
    gt = torch.empty_like(supervision)[..., :gt_size, :gt_size]
    return torch.stack(lq_rows), gt.copy_(torch.stack(gt_rows))

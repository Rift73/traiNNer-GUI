"""HR crops of clip frames made from a window of the decoded YUV planes.

VideoClipDataset.hr_frame (RGB conversion, descale, resize to hr_size, clip)
is linear along each axis apart from the transfer curves, so the HR crop of a
frame is a chain of matrix products restricted to what the crop reads: the
descale matrix P (native x source, descale.descale_matrix) and the resize
weights W (target x native, resize_matrix). The descale is local, so the crop
reads a window of the source frame; the loader workers send that window as
uint8 planes and hr_clip builds the crop on the planes' device.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import Tensor

from traiNNer.data.descale import descale_matrix
from traiNNer.data.video_index import KB, KG, KR
from traiNNer.utils.redux_options import DatasetOptions

# B and C of the cubic filters of chainner_ext's resize.
_CUBIC = {
    "catrom": (0.0, 0.5),
    "mitchell": (1 / 3, 1 / 3),
    "hermite": (0.0, 0.0),
}
_NEGLIGIBLE = 1e-8  # summed |P| a native pixel may lose outside the window
_CHROMA_REACH = 2  # luma pixels past a window edge that the chroma upsampling reads


def _cubic(d: np.ndarray, b: float, c: float) -> np.ndarray:
    d = np.abs(d)
    near = (
        (12 - 9 * b - 6 * c) * d**3 + (-18 + 12 * b + 6 * c) * d**2 + (6 - 2 * b)
    ) / 6
    far = (
        (-b - 6 * c) * d**3
        + (6 * b + 30 * c) * d**2
        + (-12 * b - 48 * c) * d
        + (8 * b + 24 * c)
    ) / 6
    return np.where(d < 1, near, np.where(d < 2, far, 0.0))


@lru_cache(maxsize=16)
def resize_matrix(source: int, target: int, name: str) -> np.ndarray:
    """The target x source float64 weights of chainner_ext's resize along one
    axis with the named cubic filter: support 2 widened by the downscale ratio,
    centred on (i + 0.5) * source / target, taps cut at the image edges and the
    rest normalized to sum 1. chainner_ext also clamps its result to 0..1,
    which the weights leave to the caller. Read-only, as it is cached."""
    b, c = _CUBIC[name]
    ratio = source / target
    scale = max(ratio, 1.0)
    weights = np.zeros((target, source))
    for i in range(target):
        centre = (i + 0.5) * ratio
        lo = max(int(np.floor(centre - 2 * scale)), 0)
        hi = min(int(np.ceil(centre + 2 * scale)), source)
        taps = _cubic((np.arange(lo, hi) + 0.5 - centre) / scale, b, c)
        weights[i, lo:hi] = taps / taps.sum()
    weights.flags.writeable = False
    return weights


def _margin(descale: np.ndarray) -> int:
    """The smallest distance m in source pixels such that, for every native
    pixel, the summed |P| of the source pixels further than m from its centre
    is at most _NEGLIGIBLE."""
    native, source = descale.shape
    centre = (np.arange(native) + 0.5) * source / native
    distance = np.ceil(np.abs(np.arange(source) + 0.5 - centre[:, None]))
    bucket = distance.astype(np.int64)
    mass = np.zeros((native, source + 1))
    np.add.at(mass, (np.arange(native)[:, None], bucket), np.abs(descale))
    # beyond[:, m] = summed |P| further than m.
    beyond = mass[:, ::-1].cumsum(axis=1)[:, ::-1]
    beyond = np.append(beyond[:, 1:], np.zeros((native, 1)), axis=1)
    return int(np.argmax(beyond.max(axis=0) <= _NEGLIGIBLE))


@dataclass(frozen=True, eq=False)
class ClipAxis:
    """One axis of the HR making for a crop of length crop: the matrices, the
    native pixels each crop start reads, and the source window that holds them
    (size pixels from origins[start], even, inside the frame)."""

    descale: np.ndarray  # native x source
    resize: np.ndarray  # target x native
    first: np.ndarray  # first native pixel read, per crop start
    stop: np.ndarray  # one past the last native pixel read, per crop start
    crop: int
    margin: int
    size: int
    origins: np.ndarray


@lru_cache(maxsize=8)
def clip_axis(
    source: int,
    native: int,
    target: int,
    crop: int,
    kernel: str,
    border: str,
    name: str,
) -> ClipAxis:
    """The axis for a source of even length descaled to native with kernel and
    border, then resized to target with the named filter."""
    if source % 2:
        raise ValueError(f"Clip HR needs an even frame size, got {source}")
    if crop > target:
        raise ValueError(f"Crop {crop} is larger than the HR size {target}")
    descale = descale_matrix(native, source, kernel, border)
    resize = resize_matrix(native, target, name)
    read = resize != 0
    first = sliding_window_view(read.argmax(axis=1), crop).min(axis=1)
    last = read.shape[1] - read[:, ::-1].argmax(axis=1)
    stop = sliding_window_view(last, crop).max(axis=1)
    margin = _margin(descale)
    centre = (np.arange(native) + 0.5) * source / native
    lo = np.ceil(centre[first] - margin - 0.5).astype(np.int64) - _CHROMA_REACH
    hi = np.floor(centre[stop - 1] + margin + 0.5).astype(np.int64) + _CHROMA_REACH
    origins = lo // 2 * 2
    size = int(-(-(hi - origins).max() // 2) * 2)
    size = min(size, source)
    origins = np.clip(origins, 0, source - size)
    return ClipAxis(descale, resize, first, stop, crop, margin, size, origins)


def clip_axes(
    frame_height: int, frame_width: int, opt: DatasetOptions
) -> tuple[ClipAxis, ClipAxis]:
    """The (rows, columns) axes of the dataset options for this frame size."""
    width, height = opt.descale_size
    hr_w, hr_h = opt.hr_size
    assert opt.lq_size is not None
    assert opt.scale is not None
    crop = opt.lq_size * opt.scale
    rest = (opt.descale_kernel, opt.descale_border, opt.downscale_algorithm)
    return (
        clip_axis(frame_height, height, hr_h, crop, *rest),
        clip_axis(frame_width, width, hr_w, crop, *rest),
    )


@lru_cache(maxsize=8)
def _on_device(axis: ClipAxis, device: torch.device) -> tuple[Tensor, Tensor]:
    """The float32 descale and resize matrices of the axis on the device."""
    descale = torch.from_numpy(np.array(axis.descale, dtype=np.float32))
    resize = torch.from_numpy(np.array(axis.resize, dtype=np.float32))
    return descale.to(device), resize.to(device)


def _upsample_chroma(c: Tensor) -> Tensor:
    """video_index._upsample_chroma on the last two dimensions."""
    c = c.float()
    padded = torch.cat([c[..., :1, :], c, c[..., -1:, :]], dim=-2)
    above = 0.75 * padded[..., 1:-1, :] + 0.25 * padded[..., :-2, :]
    below = 0.75 * padded[..., 1:-1, :] + 0.25 * padded[..., 2:, :]
    rows = torch.stack([above, below], dim=-2).flatten(-3, -2)
    right = torch.cat([rows[..., 1:], rows[..., -1:]], dim=-1)
    return torch.stack([rows, 0.5 * (rows + right)], dim=-1).flatten(-2)


# Optional: _to_rgb of uint8 CUDA planes runs as one Triton kernel.
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in {"triton", "triton.language"}:
        raise
    _HAS_TRITON = False
else:
    _HAS_TRITON = True

if _HAS_TRITON:

    @triton.jit
    def _upsampled_chroma(
        plane: tl.tensor,
        iy: tl.tensor,
        ix: tl.tensor,
        height: int,
        width: int,
        stride_h: int,
        stride_w: int,
    ) -> tl.tensor:
        """_upsample_chroma of one frame's chroma plane at luma (iy, ix)."""
        cy = iy // 2
        cx = ix // 2
        # Even luma rows take 1/4 of the chroma row above, odd rows the one below.
        other = cy + tl.where(iy % 2 == 0, -1, 1)
        other = tl.minimum(tl.maximum(other, 0), height // 2 - 1)
        right = tl.minimum(cx + 1, width // 2 - 1)
        own = tl.load(plane + cy * stride_h + cx * stride_w).to(tl.float32)
        near = tl.load(plane + other * stride_h + cx * stride_w).to(tl.float32)
        rows = 0.75 * own + 0.25 * near
        own = tl.load(plane + cy * stride_h + right * stride_w).to(tl.float32)
        near = tl.load(plane + other * stride_h + right * stride_w).to(tl.float32)
        rows_right = 0.75 * own + 0.25 * near
        return tl.where(ix % 2 == 0, rows, 0.5 * (rows + rows_right))

    @triton.jit(
        do_not_specialize=[
            "height",
            "width",
            "y_stride_t",
            "y_stride_h",
            "y_stride_w",
            "u_stride_t",
            "u_stride_h",
            "u_stride_w",
            "v_stride_t",
            "v_stride_h",
            "v_stride_w",
        ]
    )
    def _to_rgb_kernel(
        y: tl.tensor,
        u: tl.tensor,
        v: tl.tensor,
        out: tl.tensor,
        height: int,
        width: int,
        y_stride_t: int,
        y_stride_h: int,
        y_stride_w: int,
        u_stride_t: int,
        u_stride_h: int,
        u_stride_w: int,
        v_stride_t: int,
        v_stride_h: int,
        v_stride_w: int,
        inv_luma: tl.constexpr,
        inv_chroma: tl.constexpr,
        red_cr: tl.constexpr,
        green_cb: tl.constexpr,
        green_cr: tl.constexpr,
        blue_cb: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        # One program: block pixels of one frame; the operations and their order
        # are those of the torch path (run without fused multiply-adds).
        frame = tl.program_id(1)
        area = height * width
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        inside = offsets < area
        position = tl.minimum(offsets, area - 1)
        iy = position // width
        ix = position % width
        luma = tl.load(y + frame * y_stride_t + iy * y_stride_h + ix * y_stride_w)
        luma = (luma.to(tl.float32) - 16) * inv_luma
        cb = _upsampled_chroma(
            u + frame * u_stride_t, iy, ix, height, width, u_stride_h, u_stride_w
        )
        cr = _upsampled_chroma(
            v + frame * v_stride_t, iy, ix, height, width, v_stride_h, v_stride_w
        )
        cb = (cb - 128) * inv_chroma
        cr = (cr - 128) * inv_chroma
        base = out + frame * 3 * area + offsets
        tl.store(base, luma + red_cr * cr, mask=inside)
        tl.store(base + area, luma - green_cb * cb - green_cr * cr, mask=inside)
        tl.store(base + 2 * area, luma + blue_cb * cb, mask=inside)


def _to_rgb_fits(y: Tensor, u: Tensor, v: Tensor) -> bool:
    """Whether _to_rgb_triton can convert these planes."""
    if not _HAS_TRITON or not y.is_cuda or y.ndim != 4:
        return False
    if not all(p.dtype == torch.uint8 and p.device == y.device for p in (y, u, v)):
        return False
    n, one, h, w = y.shape
    half = (n, 1, h // 2, w // 2)
    return (
        one == 1
        and h % 2 == 0
        and w % 2 == 0
        and u.shape == half
        and v.shape == half
        and 0 < n <= 65535
        and 3 * y.numel() < 2**31
    )


def _to_rgb_triton(y: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """_to_rgb of uint8 CUDA planes (see _to_rgb_fits) in one kernel, same values."""
    n, _, h, w = y.shape
    out = torch.empty((n, 3, h, w), dtype=torch.float32, device=y.device)
    block = 256
    # Torch divides by a scalar as a multiply by its float32 reciprocal, and
    # multiplies by a Python float rounded to float32.
    inv_luma = float(np.float32(1) / np.float32(219))
    inv_chroma = float(np.float32(1) / np.float32(224))
    _to_rgb_kernel[(triton.cdiv(h * w, block), n)](
        y,
        u,
        v,
        out,
        h,
        w,
        y.stride(0),
        y.stride(2),
        y.stride(3),
        u.stride(0),
        u.stride(2),
        u.stride(3),
        v.stride(0),
        v.stride(2),
        v.stride(3),
        inv_luma,
        inv_chroma,
        2 * (1 - KR),
        2 * (1 - KB) * KB / KG,
        2 * (1 - KR) * KR / KG,
        2 * (1 - KB),
        block,
        enable_fp_fusion=False,
    )
    return out


def _to_rgb(y: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """video_index.to_rgb on T,1,h,w luma and T,1,h/2,w/2 chroma: T,3,h,w."""
    if _to_rgb_fits(y, u, v):
        return _to_rgb_triton(y, u, v)
    luma = (y.float() - 16) / 219
    cb = (_upsample_chroma(u) - 128) / 224
    cr = (_upsample_chroma(v) - 128) / 224
    r = luma + 2 * (1 - KR) * cr
    g = luma - (2 * (1 - KB) * KB / KG) * cb - (2 * (1 - KR) * KR / KG) * cr
    b = luma + 2 * (1 - KB) * cb
    return torch.cat([r, g, b], dim=1)


def _linearize(image: Tensor, transfer: str) -> Tensor:
    """descale.linearize."""
    a = image.abs()
    if transfer == "gamma24":
        v = a**2.4
    elif transfer == "srgb":
        v = torch.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    elif transfer == "gamma22":
        v = a**2.2
    elif transfer == "bt709":
        v = torch.where(a < 0.081, a / 4.5, ((a + 0.099) / 1.099) ** (1 / 0.45))
    else:
        raise ValueError(f"Unknown transfer {transfer!r}")
    return torch.copysign(v, image)


def _encode_light(image: Tensor, transfer: str) -> Tensor:
    """descale.encode_light."""
    a = image.abs()
    if transfer == "gamma24":
        v = a ** (1 / 2.4)
    elif transfer == "srgb":
        v = torch.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)
    elif transfer == "gamma22":
        v = a ** (1 / 2.2)
    elif transfer == "bt709":
        v = torch.where(a < 0.018, 4.5 * a, 1.099 * a**0.45 - 0.099)
    else:
        raise ValueError(f"Unknown transfer {transfer!r}")
    return torch.copysign(v, image)


def hr_clip(
    y: Tensor, u: Tensor, v: Tensor, geometry: Tensor, opt: DatasetOptions
) -> Tensor:
    """The T,3,crop,crop float32 HR crops of one clip, on the planes' device.

    y is T,1,h,w and u, v T,1,h/2,w/2 uint8: the window of the frames at
    (row, column) in luma pixels. geometry holds frame height, frame width,
    crop top, crop left (in HR pixels), window row, window column, hflip, vflip
    and transpose. The steps are those of VideoClipDataset.hr_frame followed by
    the crop, flips and transpose, in float32 at full matmul precision.
    """
    frame_h, frame_w, top, left, row, col, hflip, vflip, transpose = geometry.tolist()
    rows, cols = clip_axes(frame_h, frame_w, opt)
    p_v, w_v = _on_device(rows, y.device)
    p_h, w_h = _on_device(cols, y.device)
    r0, r1 = int(rows.first[top]), int(rows.stop[top])
    c0, c1 = int(cols.first[left]), int(cols.stop[left])
    crop = rows.crop
    h, w = y.shape[-2:]
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad(), torch.autocast(y.device.type, enabled=False):
            image = _to_rgb(y, u, v)
            light = opt.descale_linear_light
            if light is not None:
                image = _linearize(image, light)
            image = image @ p_h[c0:c1, col : col + w].T
            image = p_v[r0:r1, row : row + h] @ image
            if light is not None:
                image = _encode_light(image, light)
            light = opt.hr_linear_light
            if light is not None:
                image = _linearize(image, light)
            image = w_v[top : top + crop, r0:r1] @ image
            image = image @ w_h[left : left + crop, c0:c1].T
            image = image.clamp(0.0, 1.0)
            if light is not None:
                image = _encode_light(image, light)
            image = image.clamp(0.0, 1.0)
    finally:
        torch.set_float32_matmul_precision(precision)
    if hflip:
        image = image.flip(-1)
    if vflip:
        image = image.flip(-2)
    if transpose:
        image = image.transpose(-1, -2)
    return image.contiguous()

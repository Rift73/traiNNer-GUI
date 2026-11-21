"""Triton kernels of the GPU codec approximations in codec_approx.

The kernels model the pixels of a codec round trip, not a bitstream:

- pack: 8-bit truncation of the RGB input, BT.601 luma and chroma differences,
  box-averaged chroma subsampling and limited-range quantization.
- transform: 8x8 (or 4x4) block DCT with frequency-weighted quantization; for
  clips each frame after the first predicts its blocks from the previous
  decoded frame with a per-block motion vector, codes the residual and may
  skip it.
- webp_plane: a WebP-style intra predictor (DC, top, left, gradient) per 4x4
  block with a 4x4 transform and a dead zone.
- unpack: chroma upsampling, conversion back to RGB and rounding to 8 bits.

Every launch runs on the current CUDA stream. The kernels are compiled without
floating-point contraction so that their rounding is that of separate multiply
and add operations.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

# Motion search candidates (dy, dx) in luma pixels for every 8x8 block.
MOTION_OFFSETS = (
    (0, 0),
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (0, -2),
    (0, 2),
    (-2, 0),
    (2, 0),
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2),
    (0, -4),
    (0, 4),
    (-4, 0),
    (4, 0),
)

_PIXELS_PER_PROGRAM = 256


@triton.jit(do_not_specialize=["h", "w", "ch", "cw", "lead", "sn", "sc", "sh", "sw"])
def _pack(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    uv_ptr: tl.tensor,
    h: tl.tensor,
    w: tl.tensor,
    ch: tl.tensor,
    cw: tl.tensor,
    lead: tl.tensor,
    sn: tl.tensor,
    sc: tl.tensor,
    sh: tl.tensor,
    sw: tl.tensor,
    sub_y: tl.constexpr,
    sub_x: tl.constexpr,
    block: tl.constexpr,
    center: tl.constexpr,
) -> None:
    # One program covers `block` chroma samples of one output frame. Output frame
    # t reads input frame lead - t before the lead-in ends (the mirrored
    # frames) and t - lead after it. Reads past the right and bottom edges
    # repeat the last row and column.
    k = tl.program_id(0) * block + tl.arange(0, block)
    t = tl.program_id(1)
    src = tl.where(t < lead, lead - t, t - lead)
    cy = k // cw
    cx = k % cw
    u_sum = tl.full((block,), 0, tl.float32)
    v_sum = tl.full((block,), 0, tl.float32)
    for j in tl.static_range(sub_y):
        for i in tl.static_range(sub_x):
            yy = cy * sub_y + j
            xx = cx * sub_x + i
            offset = src * sn + tl.minimum(yy, h - 1) * sh + tl.minimum(xx, w - 1) * sw
            r = tl.clamp(tl.floor(tl.load(x_ptr + offset) * 255.0), 0.0, 255.0)
            g = tl.clamp(tl.floor(tl.load(x_ptr + offset + sc) * 255.0), 0.0, 255.0)
            b = tl.clamp(tl.floor(tl.load(x_ptr + offset + 2 * sc) * 255.0), 0.0, 255.0)
            r = r * (1.0 / 255.0)
            g = g * (1.0 / 255.0)
            b = b * (1.0 / 255.0)
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            u_sum += (b - luma) * (1.0 / 1.772)
            v_sum += (r - luma) * (1.0 / 1.402)
            limited = (libdevice.rint(luma * 219.0 + 16.0) - 16.0) * (1.0 / 219.0)
            tl.store(y_ptr + (t * h + yy) * w + xx, limited, (xx < w) & (yy < h))
    base = t * 2 * ch * cw + k
    u = (libdevice.rint(u_sum * (224.0 / (sub_y * sub_x)) + 128.0) - 128.0) * (
        1.0 / 224.0
    )
    v = (libdevice.rint(v_sum * (224.0 / (sub_y * sub_x)) + 128.0) - 128.0) * (
        1.0 / 224.0
    )
    tl.store(uv_ptr + base, u + center, k < ch * cw)
    tl.store(uv_ptr + base + ch * cw, v + center, k < ch * cw)


@triton.jit
def _interp(
    plane_ptr: tl.tensor,
    yy: tl.tensor,
    xx: tl.tensor,
    h: tl.tensor,
    w: tl.tensor,
    sub_y: tl.constexpr,
    sub_x: tl.constexpr,
) -> tl.tensor:
    # Bilinear chroma upsampling with 3/4 and 1/4 weights (chroma sited
    # between the luma samples), edges repeated.
    cy = yy // sub_y
    cx = xx // sub_x
    if sub_y == 2:
        ny = tl.minimum(h - 1, tl.maximum(0, cy + tl.where(yy % 2 == 0, -1, 1)))
    else:
        ny = cy
    if sub_x == 2:
        nx = tl.minimum(w - 1, tl.maximum(0, cx + tl.where(xx % 2 == 0, -1, 1)))
    else:
        nx = cx
    a = tl.load(plane_ptr + cy * w + cx).to(tl.float32)
    b = tl.load(plane_ptr + cy * w + nx).to(tl.float32)
    c = tl.load(plane_ptr + ny * w + cx).to(tl.float32)
    d = tl.load(plane_ptr + ny * w + nx).to(tl.float32)
    if sub_x == 2:
        top = a * 0.75 + b * 0.25
        bottom = c * 0.75 + d * 0.25
    else:
        top = a
        bottom = c
    if sub_y == 2:
        return top * 0.75 + bottom * 0.25
    return top


@triton.jit(do_not_specialize=["h", "w", "ch", "cw", "lead", "bias"])
def _unpack(
    y_ptr: tl.tensor,
    uv_ptr: tl.tensor,
    out_ptr: tl.tensor,
    h: tl.tensor,
    w: tl.tensor,
    ch: tl.tensor,
    cw: tl.tensor,
    lead: tl.tensor,
    bias: tl.tensor,
    sub_y: tl.constexpr,
    sub_x: tl.constexpr,
    block: tl.constexpr,
    center: tl.constexpr,
    decoder_colors: tl.constexpr,
) -> None:
    # One program covers `block` pixels of one output frame; the first `lead`
    # planes (the lead-in) are skipped.
    k = tl.program_id(0) * block + tl.arange(0, block)
    t = tl.program_id(1)
    pos = tl.minimum(k, h * w - 1)
    yy = pos // w
    xx = pos % w
    src = t + lead
    y = tl.load(y_ptr + src * h * w + pos).to(tl.float32)
    uv = uv_ptr + src * 2 * ch * cw
    if decoder_colors:
        # Nearest chroma and the integer limited-range conversion of a
        # software decoder.
        addr = (yy // sub_y) * cw + xx // sub_x
        u = tl.load(uv + addr).to(tl.float32) - center
        v = tl.load(uv + ch * cw + addr).to(tl.float32) - center
        luma = libdevice.rint(y * 219.0)
        red = libdevice.rint(v * 224.0) * (1.402 * 219.0 / 224.0)
        blue = libdevice.rint(u * 224.0) * (1.772 * 219.0 / 224.0)
        green = tl.floor(red * (-0.299 / 0.587)) + tl.floor(blue * (-0.114 / 0.587))
        r = tl.floor((luma + tl.floor(red)) * (255.0 / 219.0)) * (1.0 / 255.0)
        g = tl.floor((luma + green) * (255.0 / 219.0)) * (1.0 / 255.0)
        b = tl.floor((luma + tl.floor(blue)) * (255.0 / 219.0)) * (1.0 / 255.0)
    else:
        u = _interp(uv, yy, xx, ch, cw, sub_y, sub_x) - center
        v = _interp(uv + ch * cw, yy, xx, ch, cw, sub_y, sub_x) - center
        r = y + 1.402 * v
        b = y + 1.772 * u
        g = (y - 0.299 * r - 0.114 * b) * (1.0 / 0.587)
    base = out_ptr + t * 3 * h * w + k
    for c in tl.static_range(3):
        p = r if c == 0 else (g if c == 1 else b)
        level = tl.clamp(libdevice.rint(p * 255.0 + bias), 0.0, 255.0)
        # True division, as the codec round trip converts its uint8 frames.
        tl.store(base + c * h * w, tl.math.div_rn(level, 255.0), k < h * w)


@triton.jit(do_not_specialize=["h", "w"])
def _motion(
    y_ptr: tl.tensor,
    field_ptr: tl.tensor,
    h: tl.tensor,
    w: tl.tensor,
    hb: tl.tensor,
    wb: tl.tensor,
    off_y: tl.constexpr,
    off_x: tl.constexpr,
) -> None:
    # Full search over the candidate offsets for one 8x8 block of frame t + 1
    # against frame t; stores (dy, dx, mean absolute error of the best match).
    col = tl.program_id(0)
    row = tl.program_id(1)
    t = tl.program_id(2)
    p = tl.arange(0, 64)
    iy = row * 8 + p // 8
    ix = col * 8 + p % 8
    inside = (iy < h) & (ix < w)
    current = tl.load(y_ptr + ((t + 1) * h + iy) * w + ix, inside, 0.0)
    best = 1.0e9
    best_y = 0
    best_x = 0
    for k in tl.static_range(len(off_x)):
        dy = off_y[k]
        dx = off_x[k]
        py = tl.maximum(0, tl.minimum(h - 1, iy + dy))
        px = tl.maximum(0, tl.minimum(w - 1, ix + dx))
        reference = tl.load(y_ptr + (t * h + py) * w + px)
        error = tl.sum(tl.where(inside, tl.abs(current - reference), 0.0), 0)
        better = error < best
        best_y = tl.where(better, dy, best_y)
        best_x = tl.where(better, dx, best_x)
        best = tl.minimum(best, error)
    entry = ((t * hb + row) * wb + col) * 3
    tl.store(field_ptr + entry, best_y)
    tl.store(field_ptr + entry + 1, best_x)
    tl.store(field_ptr + entry + 2, best / tl.sum(inside.to(tl.float32), 0))


@triton.jit(do_not_specialize=["h", "w", "t", "step", "skip"])
def _blocks(
    src_ptr: tl.tensor,
    out_ptr: tl.tensor,
    motion_ptr: tl.tensor,
    dct_ptr: tl.tensor,
    weights_ptr: tl.tensor,
    h: tl.tensor,
    w: tl.tensor,
    hb: tl.tensor,
    wb: tl.tensor,
    t: tl.tensor,
    step: tl.tensor,
    skip: tl.tensor,
    channels: tl.constexpr,
    n: tl.constexpr,
    sub_y: tl.constexpr,
    sub_x: tl.constexpr,
    predicted: tl.constexpr,
    batched: tl.constexpr,
) -> None:
    # One program codes a 16x16 tile (four 8x8 or sixteen 4x4 blocks through a
    # block-diagonal basis) of one plane. With `batched` the frame comes from
    # the grid (independent frames), otherwise it is `t`. With `predicted` the
    # blocks predict from the previous decoded frame in `out_ptr`.
    col = tl.program_id(0) * 16
    row = tl.program_id(1) * 16
    channel = tl.program_id(2) % channels
    if batched:
        t = tl.program_id(2) // channels
    k = tl.arange(0, 16)
    y = row + k[:, None]
    x = col + k[None, :]
    iy = tl.minimum(y, h - 1)
    ix = tl.minimum(x, w - 1)
    a = tl.load(src_ptr + ((t * channels + channel) * h + iy) * w + ix)
    if predicted:
        # One motion vector and one decision per coded block. The vectors are
        # in luma pixels; chroma planes divide them by their subsampling.
        by = tl.minimum(hb - 1, ((iy // n) * n * sub_y) // 8)
        bx = tl.minimum(wb - 1, ((ix // n) * n * sub_x) // 8)
        addr = (((t - 1) * hb + by) * wb + bx) * 3
        dy = libdevice.rint(tl.load(motion_ptr + addr) / sub_y).to(tl.int32)
        dx = libdevice.rint(tl.load(motion_ptr + addr + 1) / sub_x).to(tl.int32)
        cost = tl.load(motion_ptr + addr + 2)
        inter = cost < (0.019 + 0.67 * step)
        py = tl.maximum(0, tl.minimum(h - 1, iy + dy))
        px = tl.maximum(0, tl.minimum(w - 1, ix + dx))
        prior = tl.load(out_ptr + (((t - 1) * channels + channel) * h + py) * w + px)
        pred = tl.where(inter, prior, 0.0)
        delta = a - pred
        sub = tl.reshape(tl.abs(delta), (16 // n, n, 16 // n, n))
        mad = tl.sum(tl.sum(sub, 3), 1) / (n * n)
        average = tl.reshape(
            tl.broadcast_to(
                tl.expand_dims(tl.expand_dims(mad, 1), 3), (16 // n, n, 16 // n, n)
            ),
            (16, 16),
        )
        allow_skip = inter & (average < step * skip)
        steps = tl.where(inter, step * 1.15, step)
    else:
        pred = tl.full((16, 16), 0, tl.float32)
        allow_skip = tl.full((16, 16), 0, tl.int1)
        delta = a
        steps = step
    table = k[:, None] * 16 + k[None, :]
    basis = tl.load(dct_ptr + table).to(tl.float16)
    q = tl.load(weights_ptr + table) * steps
    partial = tl.dot(basis, delta.to(tl.float16), out_dtype=tl.float32)
    coeff = tl.dot(partial.to(tl.float16), tl.trans(basis), out_dtype=tl.float32)
    quant = libdevice.rint(coeff / q) * q
    partial = tl.dot(tl.trans(basis), quant.to(tl.float16), out_dtype=tl.float32)
    rec = tl.dot(partial.to(tl.float16), basis, out_dtype=tl.float32)
    value = pred + tl.where(allow_skip, 0.0, rec)
    tl.store(
        out_ptr + ((t * channels + channel) * h + y) * w + x, value, (y < h) & (x < w)
    )


@triton.jit
def _mean_tile(z: tl.tensor) -> tl.tensor:
    # Mean of every 4x4 block of a 16x16 tile, broadcast back over the block.
    p = tl.reshape(z, (4, 4, 4, 4))
    stat = tl.sum(tl.sum(p, 3), 1) / 16.0
    return tl.reshape(tl.broadcast_to(stat[:, None, :, None], (4, 4, 4, 4)), (16, 16))


@triton.jit(do_not_specialize=["h", "w", "step"])
def _intra(
    x_ptr: tl.tensor,
    out_ptr: tl.tensor,
    dct_ptr: tl.tensor,
    weights_ptr: tl.tensor,
    h: tl.tensor,
    w: tl.tensor,
    step: tl.tensor,
) -> None:
    # One program codes a 16x16 tile of one plane as sixteen 4x4 blocks. Each
    # block takes the predictor (DC, top row, left column or clamped gradient
    # from the source neighbours, 0.5 outside the image) with the lowest mean
    # absolute error, then codes the residual with a dead zone of 0.72 steps.
    k = tl.arange(0, 16)
    col = tl.program_id(0) * 16
    row = tl.program_id(1) * 16
    plane = tl.program_id(2)
    xp = col + k[None, :]
    yp = row + k[:, None]
    x = tl.minimum(xp, w - 1)
    y = tl.minimum(yp, h - 1)
    buf = x_ptr + plane * h * w
    value = tl.load(buf + y * w + x)
    top_y = tl.minimum(tl.maximum(0, (yp // 4) * 4 - 1), h - 1)
    left_x = tl.minimum(tl.maximum(0, (xp // 4) * 4 - 1), w - 1)
    above = tl.where(yp < 4, 0.5, tl.load(buf + top_y * w + x))
    left = tl.where(xp < 4, 0.5, tl.load(buf + y * w + left_x))
    corner = tl.where((xp < 4) | (yp < 4), 0.5, tl.load(buf + top_y * w + left_x))
    dc = 0.5 * (
        _mean_tile(above + tl.zeros((16, 16), tl.float32))
        + _mean_tile(left + tl.zeros((16, 16), tl.float32))
    )
    gradient = tl.clamp(above + left - corner, 0.0, 1.0)
    pred = dc
    loss = _mean_tile(tl.abs(value - dc))
    for mode in tl.static_range(3):
        if mode == 0:
            z = above
        elif mode == 1:
            z = left
        else:
            z = gradient
        error = _mean_tile(tl.abs(value - z))
        pred = tl.where(error < loss, z, pred)
        loss = tl.minimum(loss, error)
    basis = tl.load(dct_ptr + k[:, None] * 16 + k[None, :]).to(tl.float16)
    partial = tl.dot(basis, (value - pred).to(tl.float16), out_dtype=tl.float32)
    coeff = tl.dot(partial.to(tl.float16), tl.trans(basis), out_dtype=tl.float32)
    q = step * tl.load(weights_ptr + k[:, None] * 16 + k[None, :])
    code = libdevice.rint(coeff / q)
    code = tl.where(tl.abs(coeff) < 0.72 * q, 0.0, code)
    partial = tl.dot(tl.trans(basis), (code * q).to(tl.float16), out_dtype=tl.float32)
    result = tl.dot(partial.to(tl.float16), basis, out_dtype=tl.float32) + pred
    tl.store(
        out_ptr + plane * h * w + yp * w + xp,
        tl.clamp(result, 0.0, 1.0),
        (yp < h) & (xp < w),
    )


def _dct_basis(n: int, device: torch.device) -> Tensor:
    """Orthonormal DCT-II matrix of size n."""
    k = torch.arange(n, dtype=torch.float32, device=device)
    m = torch.cos((math.pi / n) * (k[:, None] * (k[None, :] + 0.5))) * math.sqrt(2 / n)
    m[0] *= math.sqrt(0.5)
    return m


def _frequency_weights(n: int, mpeg: bool, device: torch.device) -> Tensor:
    """Quantizer weight per coefficient: rises with frequency, steeper for the
    MPEG-2/MPEG-4 group, with a finer DC."""
    uv = torch.arange(n, device=device, dtype=torch.float32)
    radius = (uv[:, None] + uv[None, :]) / (2 * n - 2)
    weights = 0.75 + radius * 1.7 if mpeg else 0.85 + radius * 0.65
    weights[0, 0] = 0.55
    return weights


@lru_cache(maxsize=32)
def _transform_tables(
    n: int, mpeg: bool, device: torch.device
) -> tuple[Tensor, Tensor]:
    """The 16x16 block-diagonal basis and weights of n x n blocks."""
    basis = torch.block_diag(*[_dct_basis(n, device)] * (16 // n)).contiguous()
    weights = _frequency_weights(n, mpeg, device).repeat(16 // n, 16 // n)
    return basis, weights.contiguous()


@lru_cache(maxsize=4)
def _webp_tables(device: torch.device) -> tuple[Tensor, Tensor]:
    """The 16x16 block-diagonal 4x4 basis and weights of the WebP model."""
    basis = torch.block_diag(*[_dct_basis(4, device)] * 4).contiguous()
    r = torch.arange(4, device=device, dtype=torch.float32)
    weights = (0.7 + 0.85 * (r[:, None] + r[None, :]) / 6).repeat(4, 4)
    return basis, weights.contiguous()


def pack(
    x: Tensor, lead_in: int, sy: int, sx: int, center: float
) -> tuple[Tensor, Tensor]:
    """BCHW float RGB (any strides) → Y planes (T, 1, H, W) and UV planes
    (T, 2, ceil(H / sy), ceil(W / sx)), T = B + lead_in, frames in coding order
    (the mirrored lead-in first). Chroma is stored offset by `center`."""
    t, _, h, w = x.shape
    ch = triton.cdiv(h, sy)
    cw = triton.cdiv(w, sx)
    frames = t + lead_in
    y = x.new_empty((frames, 1, h, w), dtype=torch.float32)
    uv = x.new_empty((frames, 2, ch, cw), dtype=torch.float32)
    grid = (triton.cdiv(ch * cw, _PIXELS_PER_PROGRAM), frames)
    _pack[grid](
        x,
        y,
        uv,
        h,
        w,
        ch,
        cw,
        lead_in,
        *x.stride(),
        sy,
        sx,
        _PIXELS_PER_PROGRAM,
        center,
        enable_fp_fusion=False,
    )
    return y, uv


def unpack(
    y: Tensor,
    uv: Tensor,
    lead_in: int,
    sy: int,
    sx: int,
    center: float,
    bias: float,
    decoder_colors: bool,
) -> Tensor:
    """Inverse of pack without its lead-in frames: contiguous float32 RGB
    (T - lead_in, 3, H, W) on the 8-bit grid. `bias` is added to the levels
    before rounding."""
    frames, _, h, w = y.shape
    _, _, ch, cw = uv.shape
    t = frames - lead_in
    out = y.new_empty((t, 3, h, w))
    grid = (triton.cdiv(h * w, _PIXELS_PER_PROGRAM), t)
    _unpack[grid](
        y,
        uv,
        out,
        h,
        w,
        ch,
        cw,
        lead_in,
        bias,
        sy,
        sx,
        _PIXELS_PER_PROGRAM,
        center,
        decoder_colors,
        enable_fp_fusion=False,
    )
    return out


def motion(y: Tensor) -> Tensor:
    """Motion field (T - 1, ceil(H / 8), ceil(W / 8), 3) of luma planes
    (T, 1, H, W): per block of frame i + 1 the offset into frame i and the
    mean absolute error of the match."""
    frames, _, h, w = y.shape
    hb = triton.cdiv(h, 8)
    wb = triton.cdiv(w, 8)
    field = torch.empty(
        (max(1, frames - 1), hb, wb, 3), device=y.device, dtype=torch.float32
    )
    if frames > 1:
        _motion[(wb, hb, frames - 1)](
            y,
            field,
            h,
            w,
            hb,
            wb,
            tuple(offset[0] for offset in MOTION_OFFSETS),
            tuple(offset[1] for offset in MOTION_OFFSETS),
            num_warps=1,
        )
    return field


def transform(
    planes: Tensor,
    field: Tensor | None,
    step: float,
    n: int,
    mpeg: bool,
    sy: int,
    sx: int,
    skip: float,
) -> Tensor:
    """Code planes (T, C, H, W) with n x n blocks and quantizer `step`.

    Without a motion field every frame is coded on its own (intra). With one,
    frame 0 is intra and every later frame predicts from the previous decoded
    frame, so the frames are coded in order. (sy, sx) is the subsampling of the
    planes relative to the luma raster of the field.
    """
    planes = planes.contiguous()
    t, c, h, w = planes.shape
    out = torch.empty_like(planes)
    basis, weights = _transform_tables(n, mpeg, planes.device)
    tiles_x = triton.cdiv(w, 16)
    tiles_y = triton.cdiv(h, 16)
    if field is None:
        _blocks[(tiles_x, tiles_y, t * c)](
            planes,
            out,
            planes,
            basis,
            weights,
            h,
            w,
            1,
            1,
            0,
            float(step),
            float(skip),
            c,
            n,
            sy,
            sx,
            False,
            True,
            num_warps=4,
        )
        return out
    for i in range(t):
        _blocks[(tiles_x, tiles_y, c)](
            planes,
            out,
            field,
            basis,
            weights,
            h,
            w,
            field.shape[1],
            field.shape[2],
            i,
            float(step),
            float(skip),
            c,
            n,
            sy,
            sx,
            i > 0,
            False,
            num_warps=4,
        )
    return out


def webp_plane(planes: Tensor, step: float) -> Tensor:
    """Code planes (T, C, H, W) with the WebP-style 4x4 intra model."""
    planes = planes.contiguous()
    t, c, h, w = planes.shape
    out = torch.empty_like(planes)
    basis, weights = _webp_tables(planes.device)
    _intra[(triton.cdiv(w, 16), triton.cdiv(h, 16), t * c)](
        planes, out, basis, weights, h, w, float(step), num_warps=4
    )
    return out

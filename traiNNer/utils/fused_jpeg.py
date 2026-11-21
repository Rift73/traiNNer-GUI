"""DiffJPEG(differentiable=False) as one Triton kernel over whole 4:2:0 MCUs.

One program reads a run of 16x16 MCUs of one image (zero past the right and
bottom edges, as DiffJPEG's padding), converts them to YCbCr, averages the
chroma over 2x2 pixels, runs the separable float32 DCT of every 8x8 block,
quantizes with the image's own quality (round half to even), and inverts each
step back to clamped RGB. The result differs from DiffJPEG only by float
rounding, mostly from DiffJPEG's tensordot transforms.
"""

from functools import lru_cache

import torch
from torch import Tensor

from traiNNer.utils.diffjpeg import c_table, y_table

# Without Triton, fused_jpeg_fits is False and DiffJPEG runs instead.
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ModuleNotFoundError as error:
    if error.name not in {"triton", "triton.language", "triton.language.extra"}:
        raise
    _HAS_TRITON = False
else:
    _HAS_TRITON = True

_GROUP = 4  # MCUs per program along a row of MCUs


if _HAS_TRITON:

    @triton.jit
    def _dct_rows(
        values: tl.tensor,
        group: tl.constexpr,
        blocks: tl.constexpr,
        inverse: tl.constexpr,
    ) -> tl.tensor:
        """The orthonormal 8-point DCT (or its inverse) along the last axis."""
        values = tl.reshape(values, (group, blocks, 8, 2, 2, 2))
        even, odd = tl.split(values)
        even_lo, even_hi = tl.split(even)
        odd_lo, odd_hi = tl.split(odd)
        x0, x4 = tl.split(even_lo)
        x2, x6 = tl.split(even_hi)
        x1, x5 = tl.split(odd_lo)
        x3, x7 = tl.split(odd_hi)

        # cos(k*pi/16) / 2, with the orthonormal DC weight c4.
        c1: tl.constexpr = 0.4903926402016152
        c2: tl.constexpr = 0.4619397662556434
        c3: tl.constexpr = 0.4157348061512726
        c4: tl.constexpr = 0.3535533905932738
        c5: tl.constexpr = 0.2777851165098011
        c6: tl.constexpr = 0.1913417161825449
        c7: tl.constexpr = 0.0975451610080642

        if not inverse:
            s0, s1, s2, s3 = x0 + x7, x1 + x6, x2 + x5, x3 + x4
            d0, d1, d2, d3 = x0 - x7, x1 - x6, x2 - x5, x3 - x4
            dc0, dc1 = s0 + s3, s1 + s2
            ac0, ac1 = s0 - s3, s1 - s2
            z0 = c4 * (dc0 + dc1)
            z2 = c2 * ac0 + c6 * ac1
            z4 = c4 * (dc0 - dc1)
            z6 = c6 * ac0 - c2 * ac1
            z1 = c1 * d0 + c3 * d1 + c5 * d2 + c7 * d3
            z3 = c3 * d0 - c7 * d1 - c1 * d2 - c5 * d3
            z5 = c5 * d0 - c1 * d1 + c7 * d2 + c3 * d3
            z7 = c7 * d0 - c5 * d1 + c3 * d2 - c1 * d3
        else:
            a = c4 * (x0 + x4)
            b = c4 * (x0 - x4)
            c = c2 * x2 + c6 * x6
            d = c6 * x2 - c2 * x6
            e0, e1, e2, e3 = a + c, b + d, b - d, a - c
            o0 = c1 * x1 + c3 * x3 + c5 * x5 + c7 * x7
            o1 = c3 * x1 - c7 * x3 - c1 * x5 - c5 * x7
            o2 = c5 * x1 - c1 * x3 + c7 * x5 + c3 * x7
            o3 = c7 * x1 - c5 * x3 + c3 * x5 - c1 * x7
            z0, z1, z2, z3 = e0 + o0, e1 + o1, e2 + o2, e3 + o3
            z4, z5, z6, z7 = e3 - o3, e2 - o2, e1 - o1, e0 - o0

        even = tl.join(tl.join(z0, z4), tl.join(z2, z6))
        odd = tl.join(tl.join(z1, z5), tl.join(z3, z7))
        return tl.reshape(tl.join(even, odd), (group, blocks, 8, 8))

    @triton.jit
    def _quantized_blocks(
        values: tl.tensor,
        steps: tl.tensor,
        group: tl.constexpr,
        blocks: tl.constexpr,
    ) -> tl.tensor:
        """8x8 blocks after the DCT, quantization by steps, and the inverse DCT."""
        horizontal = _dct_rows(values, group, blocks, False)
        vertical = _dct_rows(tl.permute(horizontal, (0, 1, 3, 2)), group, blocks, False)
        frequency = tl.permute(vertical, (0, 1, 3, 2))
        rounded = libdevice.rint(tl.div_rn(frequency, steps))
        vertical = _dct_rows(
            tl.permute(rounded * steps, (0, 1, 3, 2)), group, blocks, True
        )
        return _dct_rows(tl.permute(vertical, (0, 1, 3, 2)), group, blocks, True)

    @triton.jit
    def _box_chroma(plane: tl.tensor, group: tl.constexpr) -> tl.tensor:
        """The mean of every 2x2 pixels of 16x16 chroma: one 8x8 block per MCU."""
        tiles = tl.reshape(plane, (group, 8, 2, 8, 2))
        return (tl.sum(tl.sum(tiles, 4), 2) * 0.25)[:, None, :, :]

    @triton.jit
    def _repeat_chroma(plane: tl.tensor, group: tl.constexpr) -> tl.tensor:
        """8x8 chroma blocks repeated over 2x2 pixels: 16x16 per MCU."""
        cells = tl.reshape(plane, (group, 8, 8))
        expanded = tl.broadcast_to(cells[:, :, None, :, None], (group, 8, 2, 8, 2))
        return tl.reshape(expanded, (group, 16, 16))

    @triton.jit(
        do_not_specialize=[
            "height",
            "width",
            "stride_n",
            "stride_c",
            "stride_h",
            "stride_w",
            "stride_q",
        ]
    )
    def _jpeg_kernel(
        image: tl.tensor,
        quality: tl.tensor,
        y_table_ptr: tl.tensor,
        c_table_ptr: tl.tensor,
        out: tl.tensor,
        height: int,
        width: int,
        stride_n: int,
        stride_c: int,
        stride_h: int,
        stride_w: int,
        stride_q: int,
        group: tl.constexpr,
    ) -> None:
        # Grid: (runs of group MCUs along a row of MCUs, MCU rows, images).
        batch = tl.program_id(2)
        row = 16 * tl.program_id(1) + tl.arange(0, 16)[None, :, None]
        mcu = tl.program_id(0) * group + tl.arange(0, group)
        col = 16 * mcu[:, None, None] + tl.arange(0, 16)[None, None, :]
        valid = (row < height) & (col < width)
        offset = batch * stride_n + row * stride_h + col * stride_w
        r = tl.load(image + offset, mask=valid, other=0.0) * 255
        g = tl.load(image + offset + stride_c, mask=valid, other=0.0) * 255
        b = tl.load(image + offset + 2 * stride_c, mask=valid, other=0.0) * 255

        y = 0.299 * r + 0.587 * g + 0.114 * b - 128.0
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        # Four 8x8 luma blocks per MCU, in row-major block order.
        luma = tl.permute(tl.reshape(y, (group, 2, 8, 2, 8)), (0, 1, 3, 2, 4))
        luma = tl.reshape(luma, (group, 4, 8, 8))

        # diffjpeg.quality_to_factor
        q = tl.load(quality + batch * stride_q)
        factor = tl.where(q < 50.0, tl.div_rn(5000.0, q), 200.0 - q * 2.0)
        factor = tl.maximum(tl.div_rn(factor, 100.0), 1.0e-7)
        index = 8 * tl.arange(0, 8)[:, None] + tl.arange(0, 8)[None, :]
        y_steps = tl.load(y_table_ptr + index)[None, None, :, :] * factor
        c_steps = tl.load(c_table_ptr + index)[None, None, :, :] * factor
        luma = _quantized_blocks(luma, y_steps, group, 4)
        cb = _quantized_blocks(_box_chroma(cb, group), c_steps, group, 1)
        cr = _quantized_blocks(_box_chroma(cr, group), c_steps, group, 1)

        luma = tl.permute(tl.reshape(luma, (group, 2, 2, 8, 8)), (0, 1, 3, 2, 4))
        y = tl.reshape(luma, (group, 16, 16)) + 128.0
        cb = _repeat_chroma(cb, group)
        cr = _repeat_chroma(cr, group)
        r = tl.clamp(y + 1.402 * cr, 0.0, 255.0) * (1.0 / 255.0)
        g = tl.clamp(y - 0.344136 * cb - 0.714136 * cr, 0.0, 255.0) * (1.0 / 255.0)
        b = tl.clamp(y + 1.772 * cb, 0.0, 255.0) * (1.0 / 255.0)
        area = height * width
        target = out + batch * 3 * area + row * width + col
        tl.store(target, r, mask=valid)
        tl.store(target + area, g, mask=valid)
        tl.store(target + 2 * area, b, mask=valid)


@lru_cache(maxsize=4)
def _tables(device: torch.device) -> tuple[Tensor, Tensor]:
    """Copies of DiffJPEG's luma and chroma quantization tables on the device.

    DiffJPEG's modules share the module-level tables, so moving a DiffJPEG
    between devices moves them; the copies stay where they were made.
    """
    return (
        y_table.detach().to(device, torch.float32).contiguous().clone(),
        c_table.detach().to(device, torch.float32).contiguous().clone(),
    )


def fused_jpeg_fits(image: Tensor, quality: Tensor) -> bool:
    """Whether fused_jpeg can replace DiffJPEG for this image and quality."""
    if not _HAS_TRITON or not image.is_cuda or image.dtype != torch.float32:
        return False
    if image.ndim != 4 or (image.requires_grad and torch.is_grad_enabled()):
        return False
    b, c, h, w = image.shape
    return (
        c == 3
        and 0 < b <= 65535
        and h > 0
        and w > 0
        and -(-h // 16) <= 65535
        and image.numel() < 2**31
        and quality.dtype == torch.float32
        and quality.shape == (b,)
        and quality.device == image.device
    )


def fused_jpeg(image: Tensor, quality: Tensor) -> Tensor:
    """DiffJPEG(differentiable=False)(image, quality) for one quality per image.

    image is B,3,H,W RGB in [0, 1] (see fused_jpeg_fits); the result is a
    contiguous B,3,H,W tensor.
    """
    b, _, h, w = image.shape
    y_steps, c_steps = _tables(image.device)
    out = torch.empty((b, 3, h, w), dtype=torch.float32, device=image.device)
    grid = (triton.cdiv(w, 16 * _GROUP), triton.cdiv(h, 16), b)
    _jpeg_kernel[grid](
        image,
        quality,
        y_steps,
        c_steps,
        out,
        h,
        w,
        *image.stride(),
        quality.stride(0),
        _GROUP,
        num_warps=4,
    )
    return out

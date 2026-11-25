"""CUDA adjoint for the sparse 9×9 Malta orientation bank."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _adjoint(grad, offsets, out, height: tl.constexpr, width: tl.constexpr,
             taps: tl.constexpr, block: tl.constexpr):
    pixels = tl.program_id(0) * block + tl.arange(0, block)
    batch = tl.program_id(1)
    lines = tl.arange(0, 16)
    y = pixels // width
    x = pixels % width
    total = tl.full((16, block), 0, tl.float32)
    for tap in tl.static_range(taps):
        location = tl.load(offsets + lines * taps + tap)
        sy = y[None, :] - (location // 9 - 4)[:, None]
        sx = x[None, :] - (location % 9 - 4)[:, None]
        valid = (location[:, None] >= 0) & (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
        index = ((batch * 16 + lines[:, None]) * height + sy) * width + sx
        total += tl.load(grad + index, mask=valid, other=0)
    tl.store(out + batch * height * width + pixels, tl.sum(total, 0), pixels < height * width)


class Stencil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, bank, offsets):
        ctx.save_for_backward(offsets)
        return F.conv2d(image, bank, padding=4)

    @staticmethod
    def backward(ctx, grad):
        (offsets,) = ctx.saved_tensors
        n, _, h, w = grad.shape
        out = grad.new_empty((n, 1, h, w))
        _adjoint[(triton.cdiv(h * w, 32), n)](grad.contiguous(), offsets, out, h, w, offsets.shape[1], 32)
        return out, None, None

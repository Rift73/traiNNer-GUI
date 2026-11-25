"""Valid separable VIF convolution and its fixed-filter adjoint for CUDA."""

import torch
import triton
import triton.language as tl


@triton.jit
def _filter(src, weights, out, h: tl.constexpr, w: tl.constexpr, oh: tl.constexpr,
            ow: tl.constexpr, k: tl.constexpr, stride: tl.constexpr,
            vertical: tl.constexpr, backward: tl.constexpr, block: tl.constexpr):
    pos = tl.program_id(0) * block + tl.arange(0, block)
    n = tl.program_id(1)
    y, x = pos // ow, pos % ow
    value = tl.full((block,), 0, tl.float32)
    for i in tl.static_range(k):
        if backward:
            cy, cx = (y - i, x) if vertical else (y, x - i)
            sy, sx = (cy // stride, x) if vertical else (y, cx // stride)
            aligned = cy % stride == 0 if vertical else cx % stride == 0
            valid = aligned & (sy >= 0) & (sx >= 0) & (sy < h) & (sx < w)
        else:
            sy, sx = (y * stride + i, x) if vertical else (y, x * stride + i)
            valid = pos < oh * ow
        value += tl.load(src + (n*h + sy)*w + sx, mask=valid, other=0) * tl.load(weights+i)
    tl.store(out+n*oh*ow+pos, value, mask=pos < oh*ow)


def _pass(src, weights, stride, vertical, shape, backward):
    n, c, h, w = src.shape
    oh, ow = shape
    out = src.new_empty((n, c, oh, ow))
    _filter[(triton.cdiv(oh*ow, 128), n*c)](
        src.contiguous(), weights, out, h, w, oh, ow, weights.numel(), stride, vertical, backward, 128,
    )
    return out


class GaussianBlur(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, weights, stride):
        h, w = src.shape[-2:]
        k = weights.numel()
        oh, ow = (h-k)//stride+1, (w-k)//stride+1
        ctx.save_for_backward(weights)
        ctx.input_shape = (h, w)
        ctx.stride = stride
        horizontal = _pass(src, weights, stride, False, (h, ow), False)
        return _pass(horizontal, weights, stride, True, (oh, ow), False)

    @staticmethod
    def backward(ctx, grad):
        (weights,) = ctx.saved_tensors
        h, w = ctx.input_shape
        horiz = _pass(grad, weights, ctx.stride, True, (h, grad.shape[-1]), True)
        return _pass(horiz, weights, ctx.stride, False, (h, w), True), None, None

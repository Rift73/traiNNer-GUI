"""CUDA inference operators for DUAL regional attention, spatial experts and token normalization."""


# Triton infers device argument types; uppercase kernel parameters identify buffers and dimensions.
# ruff: noqa: ANN001, ANN202, N803

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _pack_regions(
    Q,
    K,
    V,
    PQ,
    PK,
    PV,
    Count,
    H,
    W,
    C,
    NY,
    NX,
    SB,
    SC,
    SY,
    SX,
    SHIFTY: tl.constexpr,
    SHIFTX: tl.constexpr,
    BC: tl.constexpr,
):
    cc = tl.program_id(1) * BC + tl.arange(0, BC)
    batch = tl.program_id(2)
    blockrow = tl.program_id(0)
    win = blockrow // 32
    wy = win // NX
    wx = win % NX
    y = wy * 32 + blockrow % 32 - SHIFTY
    px = tl.arange(0, 32)
    x = wx * 32 + px - SHIFTX
    valid = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    inp = (
        batch.to(tl.int64) * SB
        + y.to(tl.int64) * SY
        + x[:, None].to(tl.int64) * SX
        + cc[None, :] * SC
    )
    dest = (
        ((batch.to(tl.int64) * NY * NX + win) * C + cc[None, :]) * 1024
        + (blockrow % 32) * 32
        + px[:, None]
    )
    mask = valid[:, None] & (cc[None, :] < C)
    tl.store(PQ + dest, tl.load(Q + inp, mask, 0).to(tl.float32), cc[None, :] < C)
    tl.store(PK + dest, tl.load(K + inp, mask, 0).to(tl.float32), cc[None, :] < C)
    tl.store(PV + dest, tl.load(V + inp, mask, 0), cc[None, :] < C)
    if (blockrow % 32 == 0) & (tl.program_id(1) == 0) & (batch == 0):
        yc = tl.minimum((wy + 1) * 32 - SHIFTY, H) - tl.maximum(wy * 32 - SHIFTY, 0)
        xc = tl.minimum((wx + 1) * 32 - SHIFTX, W) - tl.maximum(wx * 32 - SHIFTX, 0)
        tl.store(Count + win, yc * xc)


@triton.jit
def _weighted_region(A, plane, pix, c, b, h, w, sy: tl.constexpr, sx: tl.constexpr):
    y = pix // w + sy
    x = pix % w + sx
    ny = tl.cdiv(h + sy, 32)
    nx = tl.cdiv(w + sx, 32)
    pos = (
        ((b.to(tl.int64) * ny * nx + (y // 32) * nx + x // 32) * c + plane) * 1024
        + (y % 32) * 32
        + x % 32
    )
    item = tl.load(A + pos, pix < h * w, 0).to(tl.float32)
    wy = 1.0 - tl.abs((y % 32).to(tl.float32) - 15.5) / 16.0
    wx = 1.0 - tl.abs((x % 32).to(tl.float32) - 15.5) / 16.0
    return item * wy * wx


@triton.jit
def _blend_regions(A, B, C, D, O, H, W, CHANNELS, P: tl.constexpr):
    pix = tl.program_id(0) * P + tl.arange(0, P)
    plane = tl.program_id(1)
    batch = tl.program_id(2)
    y = _weighted_region(A, plane, pix, CHANNELS, batch, H, W, 0, 0) + _weighted_region(
        B, plane, pix, CHANNELS, batch, H, W, 0, 16
    )
    y = y + _weighted_region(C, plane, pix, CHANNELS, batch, H, W, 16, 0)
    y = y + _weighted_region(D, plane, pix, CHANNELS, batch, H, W, 16, 16)
    tl.store(O + (batch.to(tl.int64) * CHANNELS + plane) * H * W + pix, y, pix < H * W)


def mix(module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b, c, h, w = q.shape
    heads = module.heads
    dim = c // heads

    result = []
    with torch.cuda.device(q.device), torch.autocast("cuda", enabled=False):
        # Keep the original batched matrix shapes and cuBLAS accumulation order.
        for sy, sx in ((0, 0), (0, 16), (16, 0), (16, 16)):
            ny, nx = triton.cdiv(h + sy, 32), triton.cdiv(w + sx, 32)
            shape = (b, ny * nx, heads, dim, 1024)
            qp = torch.empty(shape, device=q.device, dtype=torch.float32)
            kp = torch.empty_like(qp)
            vp = torch.empty(shape, device=q.device, dtype=v.dtype)
            count = torch.empty(
                (1, ny * nx, 1, 1), device=q.device, dtype=torch.float32
            )
            _pack_regions[(ny * nx * 32, triton.cdiv(c, 32), b)](
                q, k, v, qp, kp, vp, count, h, w, c, ny, nx, *q.stride(), sy, sx, 32
            )
            a = module.channel_attention(qp, kp, count)
            result.append(a.to(v.dtype) @ vp)
            del qp, kp, vp, count, a
        out = torch.empty((b, c, h, w), device=q.device, dtype=torch.float32)
        _blend_regions[(triton.cdiv(h * w, 256), c, b)](
            *result, out, h, w, c, 256, enable_fp_fusion=False
        )
        return out


@triton.jit
def _reduce_experts(
    Parts,
    Coef,
    Out,
    C,
    H,
    W,
    ps0,
    ps1,
    ps2,
    ps3,
    cs0,
    cs1,
    cs2,
    cs3,
    CHANNEL_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.program_id(1)
    if CHANNEL_LAST:
        c = tid % C
        pos = tid // C
    else:
        pos = tid % (H * W)
        c = tid // (H * W)
    y = pos // W
    x = pos % W
    ok = tid < C * H * W
    pp = Parts + b.to(tl.int64) * ps0 + c * ps1 + y * ps2 + x * ps3
    cp = Coef + b.to(tl.int64) * cs0 + y * cs2 + x * cs3
    first = tl.load(pp, ok, 0).to(tl.float32)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(3):
        weight = tl.load(cp + k * cs1, ok, 0).to(tl.float32)
        part = tl.load(pp + (k + 1) * C * ps1, ok, 0).to(tl.float32)
        acc += weight * part
    tl.store(Out + b.to(tl.int64) * C * H * W + tid, first + acc, ok)


def reduce_experts(
    convolved: torch.Tensor, coefs: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    b, c, h, w = x.shape
    out = torch.empty_like(
        x,
        dtype=convolved.dtype,
        memory_format=torch.contiguous_format
        if x.is_contiguous()
        else torch.channels_last,
    )
    with torch.cuda.device(x.device):
        _reduce_experts[(triton.cdiv(c * h * w, 256), b)](
            convolved,
            coefs,
            out,
            c,
            h,
            w,
            *convolved.stride(),
            *coefs.stride(),
            not x.is_contiguous(),
            256,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _down(x, i: tl.constexpr):
    return tl.inline_asm_elementwise(
        "shfl.sync.down.b32 $0,$1,$2,31,0xffffffff;",
        "=f,f,r",
        [x, i],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _broadcast(x):
    return tl.inline_asm_elementwise(
        "shfl.sync.idx.b32 $0,$1,0,31,0xffffffff;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _token_norm_128(X, G, B, Y, H, W, sb, sc, sy, sx, EPS: tl.constexpr):
    lanes = tl.arange(0, 128)
    pixel = tl.program_id(0) * 4 + lanes // 32
    lane = lanes % 32
    batch = tl.program_id(1)
    x = (
        X
        + batch.to(tl.int64) * sb
        + (pixel // W).to(tl.int64) * sy
        + (pixel % W).to(tl.int64) * sx
        + 4 * lane * sc
    )
    mask = pixel < H * W
    mean = tl.load(x, mask, 0).to(tl.float32)
    var = tl.full((128,), 0.0, tl.float32)
    # Each lane summarizes four adjacent channels, then combines the same
    # half-warps as the original FP32 vectorized layer normalization.
    for i in tl.static_range(1, 4):
        val = tl.load(x + i * sc, mask, 0).to(tl.float32)
        diff = val - mean
        mean = tl.fma(diff, 1.0 / (i + 1), mean)
        var = tl.fma(diff, val - mean, var)
    for i in tl.static_range(5):
        other_m = _down(mean, 16 >> i)
        other_v = _down(var, 16 >> i)
        diff = mean - other_m
        mean = tl.fma(0.5, other_m, 0.5 * mean)
        var = tl.fma(diff * diff, 2.0 ** (1 + i), var + other_v)
    mean = _broadcast(mean)
    inv = rsqrt(_broadcast(var) / 128.0 + EPS)
    dest = Y + ((batch.to(tl.int64) * H * W + pixel) * 128 + 4 * lane)
    for i in tl.static_range(4):
        val = tl.load(x + i * sc, mask, 0).to(tl.float32)
        weight = tl.load(G + 4 * lane + i)
        bias = tl.load(B + 4 * lane + i)
        z = tl.fma(weight, inv * (val - mean), bias)
        tl.store(dest + i, z, mask)


def token_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    b, _, h, w = x.shape
    out = torch.empty_like(x, memory_format=torch.channels_last)
    with torch.cuda.device(x.device):
        _token_norm_128[(triton.cdiv(h * w, 4), b)](
            x,
            weight,
            bias,
            out,
            h,
            w,
            *x.stride(),
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out

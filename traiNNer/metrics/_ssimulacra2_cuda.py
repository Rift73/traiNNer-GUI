"""SSIMULACRA2 full-frame FP32 validation. The recursive filter order follows libjxl."""

# Triton infers kernel arguments and uses uppercase compile-time constants.
# ruff: noqa: ANN001, ANN202, N803

import torch
import triton
import triton.language as tl

from traiNNer.losses._ssimulacra2 import Ssimulacra2

from ._ssimulacra2_cpu import GAUSSIAN_COEFFICIENTS


@triton.jit
def _scalar(total, pp, p, C: tl.constexpr):
    return tl.fma(C[1], p, tl.fma(-1.0, pp, total * C[0]))


@triton.jit
def _quad(a, b, c, d, pp, p, C: tl.constexpr):
    o0 = tl.fma(C[1], p, tl.fma(-1.0, pp, a * C[0]))
    o1 = tl.fma(C[2], p, tl.fma(C[5], pp, tl.fma(C[0], b, a * C[8])))
    o2 = tl.fma(C[3], p, tl.fma(C[6], pp, tl.fma(C[0], c, tl.fma(C[8], b, a * C[9]))))
    o3 = tl.fma(
        C[4],
        p,
        tl.fma(C[7], pp, tl.fma(C[0], d, tl.fma(C[8], c, tl.fma(C[9], b, a * C[10])))),
    )
    return o0, o1, o2, o3


@triton.jit
def _scan(src, dst, lines, length, C: tl.constexpr, HOR: tl.constexpr, B: tl.constexpr):
    line = tl.program_id(0) * B + tl.arange(0, B)
    offset = line
    p0 = tl.full((B,), 0, tl.float32)
    q0 = tl.full((B,), 0, tl.float32)
    p1 = tl.full((B,), 0, tl.float32)
    q1 = tl.full((B,), 0, tl.float32)
    p2 = tl.full((B,), 0, tl.float32)
    q2 = tl.full((B,), 0, tl.float32)
    if HOR:
        prefix = tl.minimum(8, length)
    else:
        prefix = length
    for n in range(-4, prefix):
        a = tl.load(src + (n - 6) * lines + offset, (n >= 6) & (line < lines), 0)
        b = tl.load(
            src + (n + 4) * lines + offset, (n + 4 < length) & (line < lines), 0
        )
        v = a + b
        if HOR:
            t0 = _scalar(v, q0, p0, C[0])
            t1 = _scalar(v, q1, p1, C[1])
            t2 = _scalar(v, q2, p2, C[2])
        else:
            t0 = tl.fma(C[0][0], v, tl.fma(C[0][1], p0, -q0))
            t1 = tl.fma(C[1][0], v, tl.fma(C[1][1], p1, -q1))
            t2 = tl.fma(C[2][0], v, tl.fma(C[2][1], p2, -q2))
        q0, p0 = p0, t0
        q1, p1 = p1, t1
        q2, p2 = p2, t2
        tl.store(dst + n * lines + offset, t0 + (t1 + t2), (n >= 0) & (line < lines))
    if HOR:
        groups = tl.maximum(0, (length - 4 - prefix) // 4)
        for g in range(groups):
            n = prefix + g * 4
            a = tl.load(src + (n - 6) * lines + offset, line < lines, 0) + tl.load(
                src + (n + 4) * lines + offset, line < lines, 0
            )
            b = tl.load(src + (n - 5) * lines + offset, line < lines, 0) + tl.load(
                src + (n + 5) * lines + offset, line < lines, 0
            )
            c = tl.load(src + (n - 4) * lines + offset, line < lines, 0) + tl.load(
                src + (n + 6) * lines + offset, line < lines, 0
            )
            d = tl.load(src + (n - 3) * lines + offset, line < lines, 0) + tl.load(
                src + (n + 7) * lines + offset, line < lines, 0
            )
            a0, b0, q0, p0 = _quad(a, b, c, d, q0, p0, C[0])
            a1, b1, q1, p1 = _quad(a, b, c, d, q1, p1, C[1])
            a2, b2, q2, p2 = _quad(a, b, c, d, q2, p2, C[2])
            tl.store(dst + n * lines + offset, a0 + (a1 + a2), line < lines)
            tl.store(dst + (n + 1) * lines + offset, b0 + (b1 + b2), line < lines)
            tl.store(dst + (n + 2) * lines + offset, q0 + (q1 + q2), line < lines)
            tl.store(dst + (n + 3) * lines + offset, p0 + (p1 + p2), line < lines)
        for n in range(prefix + groups * 4, length):
            a = tl.load(src + (n - 6) * lines + offset, (n >= 6) & (line < lines), 0)
            b = tl.load(
                src + (n + 4) * lines + offset, (n + 4 < length) & (line < lines), 0
            )
            v = a + b
            t0 = _scalar(v, q0, p0, C[0])
            t1 = _scalar(v, q1, p1, C[1])
            t2 = _scalar(v, q2, p2, C[2])
            q0, p0 = p0, t0
            q1, p1 = p1, t1
            q2, p2 = p2, t2
            tl.store(dst + n * lines + offset, t0 + (t1 + t2), line < lines)


@triton.jit
def _cone(x):
    # libjxl CubeRootAndAdd, including the bias subtraction in the final FMA.
    exponent = x.to(tl.int32, bitcast=True) >> 23
    r = (0x54800000 - exponent * 0x002AAAAA).to(tl.float32, bitcast=True)
    for _step in tl.static_range(3):
        rr = r * r
        r = tl.fma(-x * (1.0 / 3.0), rr * rr, (4.0 / 3.0) * r)
    rr = r * r
    r = tl.fma(1.0 / 3.0, tl.fma(-x, rr * rr, r), r)
    return tl.fma(r * r, x, -0.15595420054924863)


@triton.jit
def _xyb_pair(a, b, out, pixels, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    side = tl.program_id(1)
    image = a if side == 0 else b
    r = tl.load(image + i, i < pixels, 0)
    g = tl.load(image + pixels + i, i < pixels, 0)
    blue = tl.load(image + 2 * pixels + i, i < pixels, 0)
    bias = 0.0037930732552754493
    l = _cone(tl.fma(0.3, r, tl.fma(0.622, g, tl.fma(0.078, blue, bias))))
    m = _cone(tl.fma(0.23, r, tl.fma(0.692, g, tl.fma(0.078, blue, bias))))
    s = _cone(
        tl.fma(
            0.24342268924547819,
            r,
            tl.fma(0.20476744424496821, g, tl.fma(0.5518098666, blue, bias)),
        )
    )
    y = (l + m) * 0.5
    base = out + side * 3 * pixels + i
    tl.store(base, (l - m) * 7 + 0.42, i < pixels)
    tl.store(base + pixels, y + 0.01, i < pixels)
    tl.store(base + 2 * pixels, s - y + 0.55, i < pixels)


@triton.jit
def _pack(pair, packed, h, w, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    y = tl.program_id(1) * B + tl.arange(0, B)
    c = tl.program_id(2)
    keep = (y[:, None] < h) & (x[None, :] < w)
    idx = y[:, None] * w + x[None, :]
    p = pair + c * h * w + idx
    r = tl.trans(tl.load(p, keep, 0))
    d = tl.trans(tl.load(p + 3 * h * w, keep, 0))
    base = packed + ((x[:, None] * 15 + c) * h + y[None, :])
    for k in tl.static_range(5):
        if k == 0:
            v = r
        elif k == 1:
            v = d
        elif k == 2:
            v = r * r
        elif k == 3:
            v = d * d
        else:
            v = r * d
        tl.store(base + 3 * k * h, v, tl.trans(keep))


@triton.jit
def _reduce(pair, blur, partials, h, w, B: tl.constexpr):
    c = tl.program_id(1)
    block = tl.program_id(0)
    i = block * B + tl.arange(0, B)
    y = i // w
    x = i % w
    valid = i < h * w
    base = blur + (y * 15 + c) * w + x
    m = tl.load(base, valid, 0)
    n = tl.load(base + 3 * w, valid, 0)
    a = tl.load(base + 6 * w, valid, 0)
    b = tl.load(base + 9 * w, valid, 0)
    p = tl.load(base + 12 * w, valid, 0)
    dm = m - n
    numerator = (1 - dm * dm) * (2 * (p - m * n) + 0.0009)
    denominator = (a - m * m) + (b - n * n) + 0.0009
    s = tl.maximum(1 - tl.div_rn(numerator, denominator), 0)
    r = tl.load(pair + c * h * w + i, valid, 0)
    d = tl.load(pair + (c + 3) * h * w + i, valid, 0)
    edge = (tl.abs(d - n) - tl.abs(r - m)) / (1 + tl.abs(r - m))
    blocks = tl.num_programs(0)
    for k in tl.static_range(3):
        if k == 0:
            v = s
        elif k == 1:
            v = tl.maximum(edge, 0)
        else:
            v = tl.maximum(-edge, 0)
        v = tl.where(valid, v, 0)
        sq = v * v
        dest = partials + (c * 6 + k) * blocks + block
        tl.store(dest, tl.sum(v, 0))
        tl.store(dest + 3 * blocks, tl.sum(sq * sq, 0))


class ValidationSsimulacra2(Ssimulacra2):
    def features(
        self, reference: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        h, w = reference.shape[-2:]
        pair = reference.new_empty((2, 3, h, w))
        blocks = triton.cdiv(h * w, 256)
        _xyb_pair[(blocks, 2)](
            reference.contiguous(),
            prediction.contiguous(),
            pair,
            h * w,
            256,
            enable_fp_fusion=False,
        )
        inp = reference.new_empty((w, 15, h))
        tmp = torch.empty_like(inp)
        _pack[(triton.cdiv(w, 32), triton.cdiv(h, 32), 3)](pair, inp, h, w, 32)
        _scan[(triton.cdiv(15 * h, 32),)](
            inp,
            tmp,
            15 * h,
            w,
            GAUSSIAN_COEFFICIENTS,
            True,
            32,
            num_warps=1,
            enable_fp_fusion=False,
        )
        del inp
        vertical = tmp.permute(2, 1, 0).contiguous()
        del tmp
        blur = torch.empty_like(vertical)
        _scan[(triton.cdiv(15 * w, 32),)](
            vertical,
            blur,
            15 * w,
            h,
            GAUSSIAN_COEFFICIENTS,
            False,
            32,
            num_warps=1,
            enable_fp_fusion=False,
        )
        del vertical
        partial = reference.new_empty((3, 2, 3, blocks))
        _reduce[(blocks, 3)](pair, blur, partial, h, w, 256, enable_fp_fusion=False)
        means = partial.sum(-1) / (h * w)
        return torch.stack((means[:, 0], means[:, 1].sqrt().sqrt()), 1).unsqueeze(0)

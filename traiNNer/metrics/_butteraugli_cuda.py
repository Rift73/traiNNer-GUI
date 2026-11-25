"""Full-frame forward filters for Butteraugli validation."""

# Triton infers kernel arguments and uses uppercase compile-time constants.
# ruff: noqa: ANN001, ANN202, N803

import math

import torch
import triton
import triton.language as tl

from traiNNer.losses._butteraugli.metric import (
    FULL_LINES,
    LF_LINES,
    LOW_SCALE,
    MALTA_BANDS,
    Y_TO_B,
    Metric,
)


@triton.jit
def _blur(
    src,
    kernel,
    dest,
    h,
    w,
    length: tl.constexpr,
    mirror: tl.constexpr,
    vertical: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    plane = tl.program_id(1)
    size = h * w
    coord = i // w if vertical else i % w
    limit = h if vertical else w
    step = w if vertical else 1
    total = tl.full((BLOCK,), 0, tl.float32)
    denominator = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(length):
        v = coord + k - length // 2
        keep = i < size
        if mirror:
            v = tl.where(v < 0, -v - 1, tl.minimum(v, 2 * limit - 1 - v))
        else:
            keep = keep & (v >= 0) & (v < limit)
        weight = tl.load(kernel + k)
        total += tl.load(src + plane * size + i + (v - coord) * step, keep, 0) * weight
        if not mirror:
            denominator += tl.where(keep, weight, 0)
    if not mirror:
        total /= denominator
    tl.store(dest + plane * size + i, total, i < size)


@triton.jit
def _energy(scaled, out, h, w, lines: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scaled += tl.program_id(1) * h * w
    out += tl.program_id(1) * h * w
    x = i % w
    y = i // w
    energy = tl.full((BLOCK,), 0, tl.float32)
    for line in tl.static_range(len(lines)):
        s = tl.full((BLOCK,), 0, tl.float32)
        for tap in tl.static_range(len(lines[line])):
            dx = lines[line][tap][0]
            dy = lines[line][tap][1]
            keep = (x + dx >= 0) & (x + dx < w) & (y + dy >= 0) & (y + dy < h)
            s += tl.load(scaled + i + dy * w + dx, keep, 0)
        energy += s * s
    tl.store(out + i, energy, i < h * w)


@triton.jit
def _absorb(r, g, b, plane: tl.constexpr):
    if plane == 0:
        return r * 0.29956550340058319 + (
            g * 0.63373087833825936 + (b * 0.077705617820981968 + 1.7557483643287353)
        )
    if plane == 1:
        return r * 0.22158691104574774 + (
            g * 0.69391388044116142 + (b * 0.0987313588422 + 1.7557483643287353)
        )
    return r * 0.02 + (g * 0.02 + (b * 0.20480129041026129 + 12.226454707163354))


@triton.jit
def _cone_response(r, g, b, br, bg, bb, plane: tl.constexpr, minimum: tl.constexpr):
    pre = tl.maximum(_absorb(br, bg, bb, plane), minimum)
    sensitivity = tl.maximum(
        (tl.log(pre + 9.9710635769299145) * 19.245013259874995 - 23.16046239805755)
        / pre,
        1e-4,
    )
    return tl.maximum(_absorb(r, g, b, plane) * sensitivity, minimum)


@triton.jit
def _opsin(image, blur, out, pixels, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = i < pixels
    pos = tl.program_id(1) * 3 * pixels + i
    r = tl.load(image + pos, keep, 0) * 80
    g = tl.load(image + pos + pixels, keep, 0) * 80
    b = tl.load(image + pos + 2 * pixels, keep, 0) * 80
    br = tl.load(blur + pos, keep, 0) * 80
    bg = tl.load(blur + pos + pixels, keep, 0) * 80
    bb = tl.load(blur + pos + 2 * pixels, keep, 0) * 80
    a = _cone_response(r, g, b, br, bg, bb, 0, 1.7557483643287353)
    c = _cone_response(r, g, b, br, bg, bb, 1, 1.7557483643287353)
    s = _cone_response(r, g, b, br, bg, bb, 2, 12.226454707163354)
    tl.store(out + pos, a - c, keep)
    tl.store(out + pos + pixels, a + c, keep)
    tl.store(out + pos + 2 * pixels, s, keep)


@triton.jit
def _low(
    xyb,
    blurred,
    features,
    size,
    scales: tl.constexpr,
    y_to_b: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1)
    src = n * 3 * size + i
    out = n * 10 * size + i
    keep = i < size
    y = tl.load(blurred + src + size, keep, 0)
    for ch in tl.static_range(3):
        smooth = tl.load(blurred + src + ch * size, keep, 0)
        value = smooth + y * y_to_b if ch == 2 else smooth
        tl.store(features + out + ch * size, value * scales[ch], keep)
        raw = tl.load(xyb + src + ch * size, keep, 0) - smooth
        tl.store(blurred + src + ch * size, raw, keep)


@triton.jit
def _remove(v, w):
    return tl.maximum(v - w, 0) + tl.minimum(v + w, 0)


@triton.jit
def _amplify(v, w):
    return v + tl.maximum(-w, tl.minimum(v, w))


@triton.jit
def _clamp(v, w):
    return (
        v
        + tl.maximum(v - w, 0) * (0.724216145665 - 1)
        + tl.minimum(v + w, 0) * (0.724216145665 - 1)
    )


@triton.jit
def _middle(raw, smooth, features, high, size, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1)
    src = n * 3 * size + i
    out = n * 10 * size + i
    keep = i < size
    x = tl.load(smooth + src, keep, 0)
    y = tl.load(smooth + src + size, keep, 0)
    b = tl.load(smooth + src + 2 * size, keep, 0)
    tl.store(features + out + 3 * size, _remove(x, 0.29), keep)
    tl.store(features + out + 4 * size, _amplify(y, 0.1), keep)
    tl.store(features + out + 5 * size, b, keep)
    dx = tl.load(raw + src, keep, 0) - x
    dy = tl.load(raw + src + size, keep, 0) - y
    cross = 0.653020556257 + (1 - 0.653020556257) * 46 / (dy * dy + 46)
    dest = n * 2 * size + i
    tl.store(high + dest, dx * cross, keep)
    tl.store(high + dest + size, dy, keep)


@triton.jit
def _high(raw, smooth, features, size, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1)
    src = n * 2 * size + i
    base = n * 10 * size + i
    keep = i < size
    x = tl.load(smooth + src, keep, 0)
    y = _clamp(tl.load(smooth + src + size, keep, 0), 28.4691806922)
    dx = tl.load(raw + src, keep, 0) - x
    dy = tl.load(raw + src + size, keep, 0) - y
    tl.store(features + base + 6 * size, _remove(x, 1.5), keep)
    tl.store(features + base + 7 * size, _amplify(y * 2.155, 0.132), keep)
    tl.store(features + base + 8 * size, _remove(dx, 0.04), keep)
    tl.store(
        features + base + 9 * size, _clamp(dy, 5.19175294647) * 2.69313763794, keep
    )


@triton.jit
def _erode(image, out, h, w, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = i % w
    y = i // w
    v = tl.load(image + i, i < h * w, 0)
    first = v
    second = 2 * v
    third = 2 * v
    for row in tl.static_range(3):
        for col in tl.static_range(3):
            if row != 1 or col != 1:
                dx = (col - 1) * 3
                dy = (row - 1) * 3
                a = tl.load(
                    image + i + dy * w + dx,
                    (x + dx >= 0) & (x + dx < w) & (y + dy >= 0) & (y + dy < h),
                    65504,
                )
                b = tl.maximum(a, first)
                first = tl.minimum(a, first)
                c = tl.maximum(b, second)
                second = tl.minimum(b, second)
                third = tl.minimum(c, third)
    tl.store(out + i, 0.45 * first + 0.30 * second + 0.25 * third, i < h * w)


@triton.jit
def _scaled(
    a,
    b,
    out,
    size,
    planes: tl.constexpr,
    norms: tl.constexpr,
    scales: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.program_id(1)
    # Six different nonlinear normalizations share the same traversal.
    for j in tl.static_range(6):
        if idx == j:
            x = tl.load(a + planes[j] * size + i, i < size, 0)
            y = tl.load(b + planes[j] * size + i, i < size, 0)
            denom = 1 + (tl.abs(x) + tl.abs(y)) * (0.5 / norms[j])
            lo = tl.minimum(x * 0.55, x * 1.05)
            hi = tl.maximum(x * 0.55, x * 1.05)
            impact = tl.maximum(lo - y, 0) - tl.maximum(y - hi, 0)
            value = (scales[j] / denom) * ((x - y) + 0.812403840463596 * impact)
            tl.store(out + idx * size + i, value, i < size)


@triton.jit
def _activity(a, b, out, size, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1)
    src = a if n == 0 else b
    x = (
        tl.load(src + 8 * size + i, i < size, 0)
        + tl.load(src + 6 * size + i, i < size, 0)
    ) * 2.5
    y = (
        tl.load(src + 9 * size + i, i < size, 0) * 0.4
        + tl.load(src + 7 * size + i, i < size, 0) * 0.4
    )
    bias = 6.19424080439 * 12.61050594197
    v = tl.sqrt(tl.sqrt(x * x + y * y) * 6.19424080439 + bias) - tl.sqrt(bias)
    tl.store(out + n * size + i, v, i < size)


@triton.jit
def _combine(a, b, energy, activity, mask, out, size, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < size
    ac0 = (
        tl.load(energy + size + i, valid, 0)
        + tl.load(energy + 3 * size + i, valid, 0)
        + tl.load(energy + 5 * size + i, valid, 0)
    )
    ac1 = (
        tl.load(energy + i, valid, 0)
        + tl.load(energy + 2 * size + i, valid, 0)
        + tl.load(energy + 4 * size + i, valid, 0)
    )
    for ch in tl.static_range(2):
        x = tl.load(a + (6 + ch) * size + i, valid, 0)
        y = tl.load(b + (6 + ch) * size + i, valid, 0)
        delta = x - y
        lo = tl.minimum(x * 0.4, x)
        hi = tl.maximum(x * 0.4, x)
        sec = tl.maximum(lo - y, 0) + tl.maximum(y - hi, 0)
        if ch == 0:
            ac0 += delta * delta * (400 * 0.8) + sec * sec * (400 * 0.8)
            dm = tl.load(a + 3 * size + i, valid, 0) - tl.load(
                b + 3 * size + i, valid, 0
            )
            ac0 += dm * dm * 2150
        else:
            ac1 += delta * delta * (1.50815703118 * 0.8) + sec * sec * (
                1.50815703118 * 0.8
            )
            dm = tl.load(a + 4 * size + i, valid, 0) - tl.load(
                b + 4 * size + i, valid, 0
            )
            ac1 += dm * dm * 10.6195433239
    diff = tl.load(activity + i, valid, 0) - tl.load(activity + size + i, valid, 0)
    ac1 += diff * diff * 10
    blue = tl.load(a + 5 * size + i, valid, 0) - tl.load(b + 5 * size + i, valid, 0)
    ac2 = blue * blue * 16.2176043152
    dc = tl.full((BLOCK,), 0, tl.float32)
    for ch in tl.static_range(3):
        d = tl.load(a + ch * size + i, valid, 0) - tl.load(b + ch * size + i, valid, 0)
        dc += d * d * (29.2353797994, 0.844626970982, 0.703646627719)[ch]
    m = tl.load(mask + i, valid, 0)
    gs = 1 / (17.83 * 0.79079917404)
    ma = (1 + 2.5485944793 / (m * 0.451936922203 + 0.829591754942)) * gs
    md = (1 + 0.505054525019 / (m * 3.87449418804 + 0.20025578522)) * gs
    tl.store(
        out + i,
        tl.sqrt(tl.maximum(dc * (md * md) + (ac0 + ac1 + ac2) * (ma * ma), 0)),
        valid,
    )


class ValidationButteraugli(Metric):
    def blur(self, x: torch.Tensor, index: int) -> torch.Tensor:
        kernel = getattr(self, f"kernel_{index}")
        n, c, h, w = x.shape
        grid = (triton.cdiv(h * w, 256), n * c)
        tmp = x.new_empty(x.shape)
        out = x.new_empty(x.shape)
        _blur[grid](
            x.contiguous(), kernel, tmp, h, w, kernel.numel(), index == 0, False, 256
        )
        _blur[grid](tmp, kernel, out, h, w, kernel.numel(), index == 0, True, 256)
        return out

    def frequencies(self, xyb: torch.Tensor) -> torch.Tensor:
        """Fill the final feature planes as each scale becomes available."""
        n, _, h, w = xyb.shape
        grid = (triton.cdiv(h * w, 256), n)
        raw = self.blur(xyb, 1)
        features = xyb.new_empty((n, 10, h, w))
        _low[grid](
            xyb, raw, features, h * w, LOW_SCALE, Y_TO_B, 256, enable_fp_fusion=False
        )
        medium = self.blur(raw, 2)
        high = xyb.new_empty((n, 2, h, w))
        _middle[grid](raw, medium, features, high, h * w, 256, enable_fp_fusion=False)
        del medium, raw
        smooth = self.blur(high, 3)
        _high[grid](high, smooth, features, h * w, 256, enable_fp_fusion=False)
        return features

    def opsin(self, image: torch.Tensor) -> torch.Tensor:
        image = image.contiguous()
        blur = self.blur(image, 0)
        out = torch.empty_like(image)
        n, _, h, w = image.shape
        _opsin[(triton.cdiv(h * w, 256), n)](
            image, blur, out, h * w, 256, enable_fp_fusion=False
        )
        return out

    def erosion(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        out = image.new_empty(image.shape)
        _erode[(triton.cdiv(h * w, 256),)](
            image.contiguous(), out, h, w, 256, enable_fp_fusion=False
        )
        return out

    def difference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.contiguous()
        b = b.contiguous()
        h, w = a.shape[-2:]
        nb = triton.cdiv(h * w, 256)
        scaled = a.new_empty((6, 1, h, w))
        energy = torch.empty_like(scaled)
        planes = tuple(v[0] for v in MALTA_BANDS)
        norms = tuple(v[3] for v in MALTA_BANDS)
        scales = tuple(
            (0.611612573796 if low else 0.39905817637) * math.sqrt(0.5 * weight) / 8.5
            for _, _, weight, _, low in MALTA_BANDS
        )
        _scaled[(nb, 6)](
            a, b, scaled, h * w, planes, norms, scales, 256, enable_fp_fusion=False
        )
        for idx, count, lines in ((0, 2, FULL_LINES), (2, 4, LF_LINES)):
            _energy[(nb, count)](
                scaled[idx:], energy[idx:], h, w, tuple(tuple(v) for v in lines), 256
            )
        del scaled
        raw = a.new_empty((2, 1, h, w))
        _activity[(nb, 2)](a, b, raw, h * w, 256, enable_fp_fusion=False)
        activity = self.blur(raw, 4)
        del raw
        mask = self.erosion(activity[:1])
        out = a.new_empty((1, 1, h, w))
        _combine[(nb,)](
            a, b, energy, activity, mask, out, h * w, 256, enable_fp_fusion=False
        )
        return out

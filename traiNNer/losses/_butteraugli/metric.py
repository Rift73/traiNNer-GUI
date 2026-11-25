"""Differentiable batched PyTorch adaptation of JPEG XL Butteraugli.

Math and constants: JPEG XL Project Authors (BSD-3-Clause), pinned
third_party/libjxl/lib/jxl/butteraugli/butteraugli.cc. The stencil transcription
also follows GJXL's readable scalar implementation (Apache-2.0, Yunho Cho).
Default parameters: intensity_target=80, hf_asymmetry=1, xmul=1.
"""

# Adapted for batched PyTorch autograd and CUDA by the traiNNer-GUI project.
import math

import torch
from torch import nn
from torch.nn import functional as F

LOW_SCALE = (33.832837186260, 14.458268100570, 49.87984651440)
Y_TO_B = -0.362267051518
SIGMAS = (1.2, 7.15593339443, 3.22489901262, 1.56416327805, 2.7)

# (dx,dy) offsets for the 16 oriented Malta sums. Repeated full-frequency
# orientations are intentional: they occur in the reference implementation.
LF_LINES = [
    [(-4, 0), (-2, 0), (0, 0), (2, 0), (4, 0)],
    [(0, -4), (0, -2), (0, 0), (0, 2), (0, 4)],
    [(-3, -3), (-2, -2), (0, 0), (2, 2), (3, 3)],
    [(3, -3), (2, -2), (0, 0), (-2, 2), (-3, 3)],
    [(1, -4), (1, -2), (0, 0), (-1, 2), (-1, 4)],
    [(-1, -4), (-1, -2), (0, 0), (1, 2), (1, 4)],
    [(-4, -1), (-2, -1), (0, 0), (2, 1), (4, 1)],
    [(-4, 1), (-2, 1), (0, 0), (2, -1), (4, -1)],
    [(-2, -3), (-1, -2), (0, 0), (1, 2), (2, 3)],
    [(2, -3), (1, -2), (0, 0), (-1, 2), (-2, 3)],
    [(-3, -2), (-2, -1), (0, 0), (2, 1), (3, 2)],
    [(3, -2), (2, -1), (0, 0), (-2, 1), (-3, 2)],
    [(-4, 2), (-2, 1), (0, 0), (2, -1), (4, -2)],
    [(-4, -2), (-2, -1), (0, 0), (2, 1), (4, 2)],
    [(-2, -4), (-1, -2), (0, 0), (1, 2), (2, 4)],
    [(2, -4), (1, -2), (0, 0), (-1, 2), (-2, 4)],
]
FULL_LINES = [
    [(x, 0) for x in range(-4, 5)],
    [(0, y) for y in range(-4, 5)],
    [(x, x) for x in range(-3, 4)],
    [(-x, x) for x in range(-3, 4)],
    [(1, -4), (1, -3), (1, -2), (0, -1), (0, 0), (0, 1), (-1, 2), (-1, 3), (-1, 4)],
    [(-1, -4), (-1, -3), (-1, -2), (0, -1), (0, 0), (0, 1), (1, 2), (1, 3), (1, 4)],
    [(-4, -1), (-3, -1), (-2, -1), (-1, 0), (0, 0), (1, 0), (2, 1), (3, 1), (4, 1)],
    [(-4, 1), (-3, 1), (-2, 1), (-1, 0), (0, 0), (1, 0), (2, -1), (3, -1), (4, -1)],
    [(-2, -3), (-1, -2), (-1, -1), (0, 0), (1, 1), (1, 2), (2, 3)],
    [(2, -3), (1, -2), (1, -1), (0, 0), (-1, 1), (-1, 2), (-2, 3)],
    [(-3, -2), (-2, -1), (-1, -1), (0, 0), (1, 1), (2, 1), (3, 2)],
    [(3, -2), (2, -1), (1, -1), (0, 0), (-1, 1), (-2, 1), (-3, 2)],
]
FULL_LINES += [FULL_LINES[i] for i in (7, 6, 5, 4)]


MALTA_BANDS = (
    (9, 1, 1.10039032555, 71.7800275169, False),
    (8, 0, 173.5, 5, False),
    (7, 1, 18.7237414387, 4498534.45232, True),
    (6, 0, 6923.99476109, 8051.15833247, True),
    (4, 1, 37.0819870399, 130262059.556, True),
    (3, 0, 8246.75321353, 1009002.70582, True),
)


def stencil_bank(lines):
    weights = torch.zeros(len(lines), 1, 9, 9)
    for c, line in enumerate(lines):
        for dx, dy in line:
            weights[c, 0, dy + 4, dx + 4] += 1
    return weights


def safe_sqrt(x):
    # Select a finite zero subgradient when both images agree exactly.
    return torch.where(x > 0, x.clamp_min(1e-30).sqrt(), 0.0)


class Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lf_bank", stencil_bank(LF_LINES))
        self.register_buffer("full_bank", stencil_bank(FULL_LINES))
        for name, lines in (("lf_offsets", LF_LINES), ("full_offsets", FULL_LINES)):
            offsets = torch.full((16, max(map(len, lines))), -1, dtype=torch.int32)
            for i, line in enumerate(lines):
                offsets[i, :len(line)] = torch.tensor([(dy + 4) * 9 + dx + 4 for dx, dy in line])
            self.register_buffer(name, offsets)
        for index, sigma in enumerate(SIGMAS):
            radius = max(1, int(2.25 * sigma))
            positions = torch.arange(-radius, radius + 1, dtype=torch.float64)
            kernel = torch.exp(-positions.square() / (2 * sigma**2)).float()
            self.register_buffer(f"kernel_{index}", kernel / kernel.sum())

    def blur(self, x, index):
        kernel = getattr(self, f"kernel_{index}")
        radius = kernel.numel() // 2
        n, c, h, w = x.shape
        planes = x.reshape(n * c, 1, h, w)
        horizontal = kernel.view(1, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1)
        if index == 0:
            # libjxl's 5-tap path mirrors with the edge pixel included.
            planes = torch.cat((planes[..., :radius].flip(-1), planes, planes[..., -radius:].flip(-1)), -1)
            planes = F.conv2d(planes, horizontal)
            planes = torch.cat((planes[:, :, :radius].flip(-2), planes, planes[:, :, -radius:].flip(-2)), -2)
            return F.conv2d(planes, vertical).reshape(n, c, h, w)
        # Longer kernels truncate and renormalize at each image edge.
        sx = F.conv2d(x.new_ones(1, 1, 1, w), horizontal, padding=(0, radius))
        sy = F.conv2d(x.new_ones(1, 1, h, 1), vertical, padding=(radius, 0))
        planes = F.conv2d(planes, horizontal, padding=(0, radius)) / sx
        return (F.conv2d(planes, vertical, padding=(radius, 0)) / sy).reshape(n, c, h, w)

    def opsin(self, rgb):
        def absorb(image, clamp):
            r, g, b = [image[:, c : c + 1] * 80 for c in range(3)]
            values = [
                r * 0.29956550340058319
                + (
                    g * 0.63373087833825936
                    + (b * 0.077705617820981968 + 1.7557483643287353)
                ),
                r * 0.22158691104574774
                + (
                    g * 0.69391388044116142 + (b * 0.0987313588422 + 1.7557483643287353)
                ),
                r * 0.02 + (g * 0.02 + (b * 0.20480129041026129 + 12.226454707163354)),
            ]
            return [
                v.clamp_min(m) if clamp else v
                for v, m in zip(
                    values, (1.7557483643287353, 1.7557483643287353, 12.226454707163354)
                )
            ]

        pre = absorb(self.blur(rgb, 0), True)
        sensitivity = [
            (
                (
                    (v.clamp_min(0) + 9.9710635769299145).log() * 19.245013259874995
                    - 23.16046239805755
                )
                / v.clamp_min(1e-4)
            ).clamp_min(1e-4)
            for v in pre
        ]
        cur = [
            (v * s).clamp_min(m)
            for v, s, m in zip(
                absorb(rgb, False),
                sensitivity,
                (1.7557483643287353, 1.7557483643287353, 12.226454707163354),
            )
        ]
        return torch.cat([cur[0] - cur[1], cur[0] + cur[1], cur[2]], dim=1)

    @staticmethod
    def remove(x, width):
        return (x - width).clamp_min(0) + (x + width).clamp_max(0)

    @staticmethod
    def amplify(x, width):
        return x + x.clamp_max(width).clamp_min(-width)

    @staticmethod
    def clamp(x, maximum):
        return (
            x
            + (x - maximum).clamp_min(0) * (0.724216145665 - 1)
            + (x + maximum).clamp_max(0) * (0.724216145665 - 1)
        )

    def frequencies(self, xyb):
        blur = self.blur(xyb, 1)
        low = [blur[:, i : i + 1] for i in range(3)]
        low[2] = low[2] + low[1] * Y_TO_B
        low = [v * s for v, s in zip(low, LOW_SCALE)]
        raw = xyb - blur
        mf = self.blur(raw, 2)
        hf = [raw[:, i : i + 1] - mf[:, i : i + 1] for i in range(2)]
        medium = [
            self.remove(mf[:, 0 : 1], 0.29),
            self.amplify(mf[:, 1 : 2], 0.1),
            mf[:, 2 : 3],
        ]
        hf[0] = hf[0] * (
            0.653020556257 + (1 - 0.653020556257) * 46 / (hf[1].square() + 46)
        )
        filtered = self.blur(torch.cat(hf, dim=1), 3)
        hx, hy = filtered[:, 0 : 1], self.clamp(filtered[:, 1 : 2], 28.4691806922)
        ux = self.remove(hf[0] - hx, 0.04)
        uy = self.clamp(hf[1] - hy, 5.19175294647) * 2.69313763794
        high = [self.remove(hx, 1.5), self.amplify(hy * 2.155, 0.132)]
        return torch.cat([*low, *medium, *high, ux, uy], dim=1)

    def malta(self, a, b, weight, norm, low):
        multiplier = 0.611612573796 if low else 0.39905817637
        # Dividing by norm first keeps large reference constants out of FP16.
        # This algebra is mathematically equivalent; reference comparisons gate it.
        denom = 1 + (abs(a) + abs(b)) * (0.5 / norm)
        s0 = multiplier * math.sqrt(0.5 * weight) / 8.5 / denom
        s1 = multiplier * math.sqrt(0.33 * weight) / 8.5 / denom
        # Signed excess relative to the permitted [0.55*a,1.05*a] interval.
        lo = (a * 0.55).clamp_max(a * 1.05)
        hi = (a * 0.55).clamp_min(a * 1.05)
        impact = (lo - b).clamp_min(0) - (b - hi).clamp_min(0)
        scaled = s0 * (a - b) + s1 * impact
        return self.stencil_energy(scaled, low)

    def stencil_energy(self, scaled, low):
        bank = self.lf_bank if low else self.full_bank
        if scaled.is_cuda and torch.version.cuda is not None:
            from .stencil_cuda import Stencil
            sums = Stencil.apply(scaled, bank, self.lf_offsets if low else self.full_offsets)
        else:
            sums = F.conv2d(scaled, bank, padding=4)
        return sums.square().sum(1, keepdim=True)

    def activity(self, p):
        x = (p[:, 8 : 9] + p[:, 6 : 7]) * 2.5
        y = p[:, 9 : 10] * 0.4 + p[:, 7 : 8] * 0.4
        activity = safe_sqrt(x.square() + y.square())
        bias = 6.19424080439 * 12.61050594197
        pre = (abs(activity) * 6.19424080439 + bias).sqrt() - math.sqrt(bias)
        return self.blur(pre, 4)

    def erosion(self, x):
        m0, m1, m2 = x, x * 2, x * 2
        h, w = x.shape[-2:]
        padded = F.pad(x, (3, 3, 3, 3), value=65504)
        for dy, dx in (
            (0, -3),
            (-3, -3),
            (3, -3),
            (0, 3),
            (-3, 3),
            (3, 3),
            (-3, 0),
            (3, 0),
        ):
            v = padded[..., 3+dy:3+dy+h, 3+dx:3+dx+w]
            v, m0 = v.clamp_min(m0), v.clamp_max(m0)
            v, m1 = v.clamp_min(m1), v.clamp_max(m1)
            m2 = v.clamp_max(m2)
        return m0 * 0.45 + m1 * 0.3 + m2 * 0.25

    def difference(self, a, b):
        ac = [a[:, 0 : 1] * 0 for _ in range(3)]
        for plane, channel, weight, norm, low in MALTA_BANDS:
            ac[channel] = ac[channel] + self.malta(
                a[:, plane : plane + 1], b[:, plane : plane + 1], weight, norm, low
            )
        for c, weight in enumerate((400, 1.50815703118)):
            x, y = a[:, 6 + c : 7 + c], b[:, 6 + c : 7 + c]
            lo = (x * 0.4).clamp_max(x)
            hi = (x * 0.4).clamp_min(x)
            secondary = (lo - y).clamp_min(0) + (y - hi).clamp_min(0)
            ac[c] = (
                ac[c]
                + (x - y).square() * (weight * 0.8)
                + secondary.square() * (weight * 0.8)
            )
        for c, weight in enumerate((2150, 10.6195433239, 16.2176043152)):
            ac[c] = ac[c] + (a[:, 3 + c : 4 + c] - b[:, 3 + c : 4 + c]).square() * weight
        dc = [
            (a[:, c : c + 1] - b[:, c : c + 1]).square() * weight
            for c, weight in enumerate((29.2353797994, 0.844626970982, 0.703646627719))
        ]
        activity0, activity1 = self.activity(a), self.activity(b)
        mask = self.erosion(activity0)
        ac[1] = ac[1] + (activity0 - activity1).square() * 10
        global_scale = 1 / (17.83 * 0.79079917404)
        mask_ac = (
            (1 + 2.5485944793 / (mask * 0.451936922203 + 0.829591754942)) * global_scale
        ).square()
        mask_dc = (
            (1 + 0.505054525019 / (mask * 3.87449418804 + 0.20025578522)) * global_scale
        ).square()
        return safe_sqrt(sum(dc) * mask_dc + sum(ac) * mask_ac)

    def single_scale(self, reference, prediction):
        # Keep reference activations off the backward tape.
        ref = self.frequencies(self.opsin(reference))
        pred = self.frequencies(self.opsin(prediction))
        return self.difference(ref, pred)

    def forward(self, reference, prediction):
        h, w = reference.shape[-2:]
        if h < 8 or w < 8:
            dh, dw = max(0, 8 - h), max(0, 8 - w)
            pads = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
            main = self.single_scale(F.pad(reference, pads, mode="replicate"),
                                     F.pad(prediction, pads, mode="replicate"))
            return main[..., dh // 2:dh // 2+h, dw // 2:dw // 2+w]
        main = self.single_scale(reference, prediction)
        if h >= 15 and w >= 15:
            def down(x):
                return F.avg_pool2d(x, 2, stride=2, ceil_mode=True, count_include_pad=False)
            sub = self.single_scale(down(reference), down(prediction))
            main = main * 0.85 + F.interpolate(sub, scale_factor=2, mode="nearest")[..., :h, :w] * 0.5
        return main

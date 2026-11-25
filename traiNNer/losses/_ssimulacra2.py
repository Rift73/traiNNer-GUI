"""JPEG XL's 2023 SSIMULACRA2 scoring model. See _butteraugli/LICENSE.libjxl.

Reference: libjxl/tools/ssimulacra2.cc and tools/gauss_blur.cc.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ._vmaf.utils import fast_gaussian_blur


# Reference weights in channel, scale, norm, error-map order.
WEIGHTS = (
    0.0, 0.0007376606707406586, 0.0, 0.0, 0.0007793481682867309, 0.0,
    0.0, 0.0004371155730107379, 0.0, 1.1041726426657346, 0.00066284834129271, 0.00015231632783718752,
    0.0, 0.0016406437456599754, 0.0, 1.8422455520539298, 11.441172603757666, 0.0,
    0.0007989109436015163, 0.000176816438078653, 0.0, 1.8787594979546387, 10.94906990605142, 0.0,
    0.0007289346991508072, 0.9677937080626833, 0.0, 0.00014003424285435884, 0.9981766977854967, 0.00031949755934435053,
    0.0004550992113792063, 0.0, 0.0, 0.0013648766163243398, 0.0, 0.0,
    0.0, 0.0, 0.0, 7.466890328078848, 0.0, 17.445833984131262,
    0.0006235601634041466, 0.0, 0.0, 6.683678146179332, 0.00037724407979611296, 1.027889937768264,
    225.20515300849274, 0.0, 0.0, 19.213238186143016, 0.0011401524586618361, 0.001237755635509985,
    176.39317598450694, 0.0, 0.0, 24.43300999870476, 0.28520802612117757, 0.0004485436923833408,
    0.0, 0.0, 0.0, 34.77906344483772, 44.835625328877896, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0008680556573291698, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0005313191874358747, 0.0, 0.00016533814161379112, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0004179171803251336, 0.0017290828234722833, 0.0,
    0.0020827005846636437, 0.0, 0.0, 8.826982764996862, 23.19243343998926, 0.0,
    95.1080498811086, 0.9863978034400682, 0.9834382792465353, 0.0012286405048278493, 171.2667255897307, 0.9807858872435379,
    0.0, 0.0, 0.0, 0.0005130064588990679, 0.0, 0.00010854057858411537,
)


def _xyb(linear):
    r, g, b = linear.unbind(1)
    bias = 0.0037930732552754493
    l = (0.300 * r + 0.622 * g + 0.078 * b + bias).pow(1 / 3)
    m = (0.230 * r + 0.692 * g + 0.078 * b + bias).pow(1 / 3)
    s = (0.2434226892 * r + 0.2047674442 * g + 0.5518098666 * b + bias).pow(1 / 3)
    y = (l + m) * 0.5
    return torch.stack(((l - m) * 7 + 0.42, y - bias ** (1 / 3) + 0.01, s - y + 0.55), dim=1)


class Ssimulacra2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.tensor(WEIGHTS, dtype=torch.float32))
        # Exact finite impulse response of libjxl's three-cosine Gaussian (sigma 1.5).
        half = [0.00941436781, 0.03601111466, 0.10933537278, 0.21292859201]
        self.register_buffer("kernel", torch.tensor([*half, 0.26462110549, *reversed(half)]))

    def features(self, reference, prediction):
        ref = _xyb(reference)
        dist = _xyb(prediction)
        delta = dist - ref
        moments = torch.cat((ref, delta, ref.square(), ref * delta, delta.square()))
        n, c, h, w = moments.shape
        planes = F.pad(moments.reshape(n * c, 1, h, w), (4, 4, 4, 4))
        blurred = fast_gaussian_blur(planes, self.kernel).reshape(n, c, h, w)
        mu, dmu, second, cross, dsecond = blurred.chunk(5)
        # Expand 1 - SSIM in the residual domain to preserve near-perfect differences.
        var = second - mu.square()
        cov = cross - mu * dmu
        dvar = dsecond - dmu.square()
        structural = 2 * (var + cov) + 0.0009
        dissimilarity = ((dvar + dmu.square() * structural) / (structural + dvar)).relu()
        ref_edge = (ref - mu).abs()
        edge_difference = ((dist - (mu + dmu)).abs() - ref_edge) / (1 + ref_edge)
        maps = torch.stack((dissimilarity, edge_difference.relu(), (-edge_difference).relu()), dim=2)
        average = maps.mean(dim=(-1, -2))
        fourth = torch.linalg.vector_norm(maps, 4, dim=(-1, -2)) / (h * w) ** 0.25
        return torch.stack((average, fourth), dim=2)

    def forward(self, reference, prediction):
        scales = []
        for level in range(6):
            if min(reference.shape[-2:]) < 8:
                break
            if level:
                reference = F.avg_pool2d(reference, 2, ceil_mode=True, count_include_pad=False)
                prediction = F.avg_pool2d(prediction, 2, ceil_mode=True, count_include_pad=False)
            scales.append(self.features(reference, prediction))
        features = torch.stack(scales, dim=2).flatten(1)
        # The reference advances its weight cursor only for evaluated scales.
        error = (features * self.weights[:features.shape[1]]).sum(1) * 0.9562382616834844
        mapped = error * (2.326765642916932 + error * (-0.020884521182843837 + error * 0.00006248496625763138))
        return torch.where(mapped > 0, 0.1 * mapped.clamp_min(1e-30).pow(0.6276336467831387), 0).mean()

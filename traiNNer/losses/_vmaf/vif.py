"""VIF metric

Based on:
    http://live.ece.utexas.edu/research/Quality/vifvec_release.zip
    https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/vif.c

References:
    https://ieeexplore.ieee.org/abstract/document/1576816/

Todo:
    check that padding modes are the same as C code for nonstandard resolutions

"""

import torch
import torch.nn.functional as F
from .utils import gaussian_kernel_1d, fast_gaussian_blur


class VIF(torch.nn.Module):

    def __init__(self, NEG=False):
        super().__init__()

        if NEG:
            self.vif_enhn_gain_limit = 1.
        else:
            self.vif_enhn_gain_limit = 100.

        self.sigma_nsq = 2
        self.sigma_max_inv = 4. / (255.*255.)

        self.scales = (0, 1, 2, 3)
        self.blur_windows = torch.nn.ParameterList()
        for scale in self.scales:
            N = pow(2, 4 - scale) + 1
            win = gaussian_kernel_1d(kernel_size=N, sigma=N/5)
            self.blur_windows.append(torch.nn.Parameter(win, requires_grad=False))  # using parameters instead of buffers because there is no BufferList

    def forward(self, ref, dist):
        return self.vif_features(ref, dist)  # all VIF features are used for VMAF score regresion

    def vif_features(self, ref, dist):
        num, den = self.vif_num_den(ref, dist)
        features = num/den
        return features               # [batch_size, num_scales]

    def vif_score(self, ref, dist):
        num, den = self.vif_num_den(ref, dist)
        score = torch.sum(num, dim=-1)/torch.sum(den, dim=-1)
        return score                  # [batch_size]


    def vif_num_den(self, ref, dist):

        assert len(ref.shape) == 4 and len(dist.shape) == 4, f'Expected tensors in [b,c,h,w] format, got {ref.shape} and {dist.shape}'

        num = []
        den = []

        ref = ref - 128
        dist = dist - 128

        w = ref.shape[-1]
        h = ref.shape[-2]

        for scale in self.scales:

            win = self.blur_windows[scale]
            kernel_size = win.shape[-1]
            pad_size = [(kernel_size-1)//2]*4

            if scale > 0:
                # pad size to ensure that width_new = floor(width/2), height_new = floor(height/2)
                pad_size_downscale = [(kernel_size-1)//2,
                                      (kernel_size-1)//2-1 if w % 2 == 1 else (kernel_size-1)//2,
                                      (kernel_size-1)//2,
                                      (kernel_size-1)//2-1 if h % 2 == 1 else (kernel_size-1)//2
                                      ]

                down = fast_gaussian_blur(F.pad(torch.cat((ref, dist)), pad_size_downscale, mode='reflect'), weight=win, stride=2)
                ref, dist = down.chunk(2, dim=0)
                w = w//2
                h = h//2

            moments = torch.cat((ref, dist, ref.square(), dist.square(), ref * dist))
            moments = fast_gaussian_blur(F.pad(moments, pad_size, mode="reflect"), win)
            mu1, mu2, second1, second2, cross = moments.chunk(5, dim=0)
            sigma1_sq = second1 - mu1.square()
            sigma2_sq = second2 - mu2.square()
            sigma12 = cross - mu1 * mu2

            sigma1_sq = F.relu(sigma1_sq)
            sigma2_sq = F.relu(sigma2_sq)

            g = torch.divide(sigma12, (sigma1_sq+1e-10))

            sv_sq = sigma2_sq - torch.multiply(g, sigma12)

            g = g.masked_fill(sigma1_sq < 1e-10, 0)
            sv_sq = torch.where(sigma1_sq < 1e-10, sigma2_sq, sv_sq)
            sigma1_sq = sigma1_sq.masked_fill(sigma1_sq < 1e-10, 0)

            g = g.masked_fill(sigma2_sq < 1e-10, 0)
            sv_sq = sv_sq.masked_fill(sigma2_sq < 1e-10, 0)

            sv_sq = torch.where(g < 0, sigma2_sq, sv_sq)
            g = F.relu(g)
            sv_sq = sv_sq.masked_fill(sv_sq <= 1e-10, 1e-10)

            g = torch.clamp(g, max=self.vif_enhn_gain_limit)

            num_ar = torch.log2(1 + ((g**2) * sigma1_sq)/(sv_sq+self.sigma_nsq))
            den_ar = torch.log2(1 + sigma1_sq/self.sigma_nsq)

            num_ar = num_ar.masked_fill(sigma12 < 0, 0)
            num_ar = torch.where(sigma1_sq < self.sigma_nsq, 1 - sigma2_sq * self.sigma_max_inv, num_ar)
            den_ar = den_ar.masked_fill(sigma1_sq < self.sigma_nsq, 1)

            num.append(torch.sum(num_ar, dim=(-1, -2, -3)))
            den.append(torch.sum(den_ar, dim=(-1, -2, -3)))

        num = torch.stack(num, dim=-1)  # [batch_size, num_scales]
        den = torch.stack(den, dim=-1)

        return num, den

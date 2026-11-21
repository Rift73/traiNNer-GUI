"""Image objectives adapted from NeoSR 3fe738f4 (Apache-2.0)."""

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.losses.loss_util import restoring_clamp
from traiNNer.utils.color_util import rgb_to_linear_rgb
from traiNNer.utils.registry import LOSS_REGISTRY


def _clipped_charbonnier(pred: Tensor, target: Tensor) -> Tensor:
    # Both NeoSR criteria use CHC with lambda=0 and elementwise clip bounds [0, 1].
    dtype = torch.promote_types(pred.dtype, torch.float32)
    distance = charbonnier_loss(pred.to(dtype), target.to(dtype), reduction="none")
    return distance.clamp(0, 1).mean()


@LOSS_REGISTRY.register()
class NeoSRLDLLoss(nn.Module):
    """Compare images weighted by the local and global variance of their residual."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        criterion: Literal["l1", "l2", "huber", "chc"] = "l1",
        ksize: int = 7,
    ) -> None:
        super().__init__()
        if type(ksize) is not int or ksize < 3 or ksize % 2 == 0:
            raise ValueError("NeoSR LDL requires an odd ksize of at least 3.")
        criteria = {
            "l1": F.l1_loss,
            "l2": F.mse_loss,
            "huber": F.huber_loss,
            "chc": _clipped_charbonnier,
        }
        if criterion not in criteria:
            raise ValueError(f"Unknown NeoSR LDL criterion {criterion!r}.")
        self.criterion = criteria[criterion]
        self.ksize = ksize
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        k = self.ksize
        pad = k // 2
        if pred.ndim != 4 or pred.shape != target.shape or min(pred.shape[-2:]) <= pad:
            raise ValueError(
                f"NeoSR LDL requires matching NCHW images with H and W > {pad}."
            )
        residual = (target - pred).abs().sum(1, keepdim=True)
        global_weight = (
            residual.float()
            .var(dim=(-1, -2, -3), keepdim=True)
            .clamp_min(1e-12)
            .pow(0.2)
        )
        # Double-precision moments retain small local variance around a large mean.
        padded = F.pad(residual.double(), (pad,) * 4, mode="reflect")
        mean = F.avg_pool2d(padded, k, stride=1)
        second_moment = F.avg_pool2d(padded.square(), k, stride=1)
        n = k * k
        variance = ((second_moment - mean.square()) * (n / (n - 1))).clamp_min(0)
        weight = global_weight * variance.to(residual.dtype)
        return self.criterion(weight * pred, weight * target)


def _oklab_chroma(rgb: Tensor) -> Tensor:
    r, g, b = rgb_to_linear_rgb(rgb).unbind(1)
    long = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    medium = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    short = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    long = long.sign() * long.abs().pow(1 / 3)
    medium = medium.sign() * medium.abs().pow(1 / 3)
    short = short.sign() * short.abs().pow(1 / 3)
    a = 1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short
    b = 0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short
    return torch.stack((a, b), dim=1)


@LOSS_REGISTRY.register()
class NeoSRConsistencyLoss(nn.Module):
    """Compare CIE lightness and Oklab chroma on bounded sRGB image pairs in FP32."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        criterion: Literal["chc", "l1"] = "chc",
        blur: bool = True,
        cosim: bool = True,
        saturation: float = 1.0,
        brightness: float = 1.0,
    ) -> None:
        super().__init__()
        if criterion not in ("chc", "l1"):
            raise ValueError(f"Unknown NeoSR consistency criterion {criterion!r}.")
        self.criterion = _clipped_charbonnier if criterion == "chc" else F.l1_loss
        self.loss_weight = loss_weight
        self.blur = blur
        self.cosim = cosim
        self.saturation = saturation
        self.brightness = brightness
        self.register_buffer(
            "luma", torch.tensor([0.2126, 0.7152, 0.0722]), persistent=False
        )
        taps = torch.exp(-0.5 * (torch.linspace(-10, 10, 21) / 3).square())
        self.register_buffer("taps", taps / taps.sum(), persistent=False)

    def _lightness(self, rgb: Tensor) -> Tensor:
        if self.blur:
            padded = F.pad(rgb, (10,) * 4, mode="reflect")
            taps = self.taps.to(rgb.dtype)
            row = taps.view(1, 1, 1, 21).expand(3, 1, 1, 21)
            col = taps.view(1, 1, 21, 1).expand(3, 1, 21, 1)
            rgb = F.conv2d(F.conv2d(padded, row, groups=3), col, groups=3).clamp(0, 1)
        y = rgb_to_linear_rgb(rgb).permute(0, 2, 3, 1) @ self.luma.to(rgb.dtype)
        knee = 216 / 24389
        lightness = torch.where(
            y <= knee, y * (24389 / 27), y.clamp_min(knee).pow(1 / 3) * 116 - 16
        )
        return (lightness / 100).clamp(0, 1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim != 4 or pred.shape[1] != 3 or pred.shape != target.shape:
            raise ValueError("NeoSR consistency requires matching NCHW RGB images.")
        if self.blur and min(pred.shape[-2:]) <= 10:
            raise ValueError(
                "NeoSR consistency blur requires H and W > 10 for reflect padding."
            )
        with torch.autocast(pred.device.type, enabled=False):
            pred = restoring_clamp(pred.float(), 1 / 255, 1)
            target = target.float().clamp(1 / 255, 1)
            pred_l = self._lightness(pred)
            target_l = self._lightness(target) * self.brightness
            pred_ab = (_oklab_chroma(pred) + 0.5).clamp(0, 1)
            target_ab = (_oklab_chroma(target) * self.saturation + 0.5).clamp(0, 1)
            loss = self.criterion(pred_l, target_l) + self.criterion(pred_ab, target_ab)
            if self.cosim:
                # Upstream compares along height for BHW lightness and along channels for two-channel chroma.
                ab_cos = 1 - F.cosine_similarity(pred_ab, target_ab, dim=1).mean()
                l_cos = 1 - F.cosine_similarity(pred_l, target_l, dim=1).mean()
                cosine = 0.5 * ab_cos + 0.5 * l_cos
                loss = loss + torch.where(cosine < 1e-3, cosine, 0.0)
            return loss

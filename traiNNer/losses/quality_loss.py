"""Differentiable full-reference objectives for aligned SDR RGB restoration."""

from typing import Literal

import torch
from torch import Tensor, nn

from traiNNer.losses._butteraugli.metric import Metric
from traiNNer.losses._ssimulacra2 import Ssimulacra2
from traiNNer.losses._vmaf.adm import ADM
from traiNNer.losses._vmaf.svm_predict import SVMPredict
from traiNNer.losses._vmaf.vif import VIF
from traiNNer.utils.registry import LOSS_REGISTRY


def _check_pair(pred: Tensor, gt: Tensor, minimum: int) -> None:
    if pred.ndim != 4 or pred.shape != gt.shape or pred.shape[1] != 3:
        raise ValueError("Expected aligned N×3×H×W RGB prediction and target.")
    if min(pred.shape[-2:]) < minimum:
        raise ValueError(f"This loss needs at least {minimum} pixels on each side.")


def _linear(rgb: Tensor) -> Tensor:
    rgb = rgb.float().clamp(0, 1)
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def _luma(rgb: Tensor) -> Tensor:
    r, g, b = rgb.float().unbind(1)
    return (16 + 219 * (0.2126 * r + 0.7152 * g + 0.0722 * b)).unsqueeze(1)


def _fidelity_loss(score: Tensor, ref: Tensor, dist: Tensor) -> Tensor:
    # At an exact match several VMAF feature clamps meet at a kink. Choose zero
    # as the subgradient so the metric cannot push an already matching crop away.
    exact = (dist == ref).flatten(1).all(1)
    score = score.reshape(-1)
    return 1 - torch.where(exact, score.detach(), score).mean()


@LOSS_REGISTRY.register()
class ButteraugliLoss(nn.Module):
    """JPEG XL Butteraugli on sRGB images at 80 nits. Use max for its reference score or p3 for spatially distributed gradients."""

    def __init__(self, loss_weight: float = 1.0, pooling: Literal["p3", "max"] = "p3") -> None:
        super().__init__()
        if pooling not in {"p3", "max"}:
            raise ValueError("pooling must be p3 or max.")
        self.loss_weight = loss_weight
        self.pooling = pooling
        self.metric = Metric()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt, 1)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            dist = self.metric(_linear(gt), _linear(pred)).flatten(1)
            if self.pooling == "max":
                return dist.amax(1).mean()
            return torch.linalg.vector_norm(dist, 3, dim=1).mean() / dist.shape[1] ** (1 / 3)


@LOSS_REGISTRY.register()
class VMAFLoss(nn.Module):
    """Frame-independent VMAF v0.6.1 loss on BT.709 luma. NEG limits enhancement gain. Pair with an RGB fidelity loss."""

    def __init__(self, loss_weight: float = 1.0, neg: bool = True) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.vif = VIF(NEG=neg)
        self.adm = ADM(NEG=neg)
        self.svm = SVMPredict()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt, 32)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            ref, dist = _luma(gt), _luma(pred)
            detail = self.adm(ref, dist)
            features = torch.cat((detail, torch.zeros_like(detail), self.vif(ref, dist)), dim=1)
            return _fidelity_loss(self.svm(features) / 100, ref, dist)


@LOSS_REGISTRY.register()
class VIFLoss(nn.Module):
    """Four-scale visual information fidelity on BT.709 luma with enhancement gain capped."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.vif = VIF(NEG=True)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt, 32)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            ref, dist = _luma(gt), _luma(pred)
            return _fidelity_loss(self.vif.vif_score(ref, dist), ref, dist)


@LOSS_REGISTRY.register()
class ADMLoss(nn.Module):
    """Four-scale detail preservation on BT.709 luma, with contrast masking and enhancement gain capped."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.adm = ADM(NEG=True)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt, 32)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            ref, dist = _luma(gt), _luma(pred)
            return _fidelity_loss(self.adm(ref, dist), ref, dist)


@LOSS_REGISTRY.register()
class ChromaWaveletLoss(nn.Module):
    """Match two scales of horizontal, vertical and diagonal BT.709 chroma detail on aligned sRGB crops."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt, 4)
        r, g, b = (pred.float() - gt.float()).unbind(1)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        residual = torch.stack(((b - y) / 1.8556, (r - y) / 1.5748), dim=1)
        bands = []
        for _ in range(2):
            h, w = residual.shape[-2:]
            if h % 2 or w % 2:
                residual = nn.functional.pad(residual, (0, w % 2, 0, h % 2), mode="replicate")
            a, b = residual[..., ::2, ::2], residual[..., ::2, 1::2]
            c, d = residual[..., 1::2, ::2], residual[..., 1::2, 1::2]
            top, bottom = a + b, c + d
            dx, dy = a - b, c - d
            bands.append(((top - bottom).abs() + (dx + dy).abs() + (dx - dy).abs()).mean())
            residual = (top + bottom) * 0.25
        return (bands[0] + bands[1]) / 24


@LOSS_REGISTRY.register()
class SSIMULACRA2Loss(nn.Module):
    """Full-reference SSIMULACRA2 distance on aligned SDR sRGB. Zero denotes an exact match."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.metric = Ssimulacra2()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        _check_pair(pred, gt, 8)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return self.metric(_linear(gt), _linear(pred))

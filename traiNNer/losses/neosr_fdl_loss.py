"""Fourier distribution loss compatible with the NeoSR compact recipes.

Ni et al., Misalignment-Robust Frequency Distribution Loss for Image Transformation
(CVPR 2024). Feature taps and phase weighting follow NeoSR-dev 3fe738f4.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from math import isfinite
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as recompute

from trainner_config.feature_assets import DINO_REPOSITORY
from traiNNer.utils.registry import LOSS_REGISTRY


class _RecomputedBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, tokens: Tensor) -> Tensor:
        if torch.is_grad_enabled() and tokens.requires_grad:
            return recompute(self.block, tokens, use_reentrant=False)
        return self.block(tokens)


def _frozen_position_encoding(vit, tokens, width, height):
    # The frozen backbone sees a small set of crop geometries in training.
    key = (tokens.shape[1], width, height, tokens.dtype, tokens.device)
    cache = vit._fdl_positions
    if key not in cache:
        if len(cache) >= 8:
            del cache[next(iter(cache))]
        cache[key] = type(vit).interpolate_pos_encoding(vit, tokens, width, height).detach()
    return cache[key]


class _Features(nn.Module):
    def __init__(self, kind: str, vgg_weights, dino_layers, dino_weights, dino_norm: bool,
                 checkpoint: bool):
        super().__init__()
        from torchvision import models

        self.kind = kind
        self.normalize = kind != "dinov2" or dino_norm
        if self.normalize:
            self.register_buffer("mean", torch.tensor((.485, .456, .406)).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor((.229, .224, .225)).view(1, 3, 1, 1))
        if kind == "vgg":
            net = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
            cuts = (0, 4, 9, 18, 27, 36)
            self.stages = nn.ModuleList(net[a:b] for a, b in zip(cuts[:-1], cuts[1:], strict=True))
            self.widths = (64, 128, 256, 512, 512)
            weights = (.5, .5, 1., 1., 1.) if vgg_weights is None else vgg_weights
        elif kind == "resnet":
            net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.stages = nn.ModuleList((nn.Sequential(net.conv1, net.bn1, net.relu),
                                         nn.Sequential(net.maxpool, net.layer1), net.layer2, net.layer3))
            self.widths = (64, 256, 512, 1024)
            weights = [1.] * len(self.widths)
        elif kind == "effnet":
            net = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT).features
            self.stages = nn.ModuleList((net[:2], *net[2:6]))
            self.widths = (32, 48, 80, 160, 224)
            weights = [1.] * len(self.widths)
        elif kind == "dinov2":
            self.layers = tuple(range(8) if dino_layers is None else dino_layers)
            if not self.layers or any(type(i) is not int or not 0 <= i < 12 for i in self.layers):
                raise ValueError("Choose DINOv2 ViT-B/14 block indices between 0 and 11.")
            if tuple(sorted(set(self.layers))) != self.layers:
                raise ValueError("DINOv2 block indices must be unique and in ascending order.")
            self.widths = (768,) * len(self.layers)
            weights = (1., .5, .5, 1., .5, .5, 1., .1) if dino_weights is None else dino_weights
            if len(weights) != len(self.layers):
                raise ValueError("Supply a weight for each selected DINOv2 layer.")
            # Hub's branch-tip check rejects a historical SHA after main moves.
            # Both repository and full commit are fixed in the application.
            self.vit = torch.hub.load(DINO_REPOSITORY, "dinov2_vitb14", trust_repo=True,
                                      skip_validation=True, verbose=False)
            self.vit.to(memory_format=torch.channels_last)
            if not self.vit.chunked_blocks:
                del self.vit.blocks[self.layers[-1] + 1:]
            if checkpoint:
                for index, block in enumerate(self.vit.blocks):
                    self.vit.blocks[index] = _RecomputedBlock(block)
            self.vit._fdl_positions = {}
            self.vit.interpolate_pos_encoding = MethodType(_frozen_position_encoding, self.vit)
        else:
            raise ValueError("FDL supports dinov2, vgg, resnet and effnet.")
        self.weights = tuple(weights)
        if len(self.weights) != len(self.widths) or not all(isfinite(w) for w in self.weights):
            raise ValueError("Supply one finite weight for each FDL feature map.")
        if kind == "dinov2":
            self.register_buffer("layer_weights", torch.tensor(self.weights, dtype=torch.float32).view(-1, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.kind == "dinov2":
            height, width = x.shape[-2:]
            py, px = -height % 14, -width % 14
            if px or py:
                x = F.pad(x, (px // 2, px - px // 2, py // 2, py - py // 2), mode="reflect")
            features = self.vit.get_intermediate_layers(x, n=self.layers, reshape=True,
                                                        return_class_token=False)
            return [feat * weight for feat, weight in zip(features, self.layer_weights, strict=True)]
        else:
            features = []
            for stage in self.stages:
                x = stage(x)
                features.append(x)
        return [feature * weight if weight != 1.0 else feature
                for feature, weight in zip(features, self.weights, strict=True)]


@LOSS_REGISTRY.register()
class NeoSRFDLLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, patch_size: int = 4, stride: int = 1,
                 num_proj: int = 24, model: str = "dinov2", vgg_weights: Sequence[float] | None = None,
                 dino_layers: Sequence[int] | None = None, dino_weights: Sequence[float] | None = None,
                 dino_norm: bool = False, phase_weight: float = 1., seed: int = 0,
                 fp32_backbone: bool = True, checkpoint: bool = False):
        super().__init__()
        if min(patch_size, stride, num_proj) < 1:
            raise ValueError("FDL projection size, stride and count must be positive.")
        model = model.lower()
        if checkpoint and model != "dinov2":
            raise ValueError("Activation checkpointing is supported by the DINOv2 backbone.")
        self.loss_weight = loss_weight
        self.stride = stride
        self.phase_weight = phase_weight
        self.fp32_features = fp32_backbone or model != "dinov2"
        self.multiplier = 1. if model == "dinov2" else .01
        self.features = _Features(model, vgg_weights, dino_layers, dino_weights, dino_norm, checkpoint)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for index, width in enumerate(self.features.widths):
            filters = torch.randn(num_proj, width, patch_size, patch_size, device="cpu", generator=generator)
            filters = filters / filters.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            self.register_buffer(f"projection_{index}", filters, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self

    def _distance(self, a: Tensor, b: Tensor, kernel: Tensor) -> Tensor:
        if min(a.shape[-2:]) < kernel.shape[-1]:
            raise ValueError("The FDL projection exceeds a feature map. Increase the training crop or reduce patch_size.")
        pa = F.conv2d(a, kernel, stride=self.stride).flatten(2).sort(dim=-1).values
        pb = F.conv2d(b, kernel, stride=self.stride).flatten(2).sort(dim=-1).values
        return (pa - pb).abs().mean((1, 2))

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        dev = prediction.device.type
        force_fp32 = self.fp32_features or min(prediction.shape[-2:]) < 256
        with torch.autocast(dev, enabled=False) if force_fp32 else nullcontext():
            restored = self.features(prediction.float() if force_fp32 else prediction)
            with torch.no_grad():
                reference = self.features(target.float() if force_fp32 else target)
        with torch.autocast(dev, enabled=False):
            score = []
            for index, (x, y) in enumerate(zip(restored, reference, strict=True)):
                kernel = getattr(self, f"projection_{index}")
                a = torch.fft.fftn(x.float(), dim=(-2, -1))
                b = torch.fft.fftn(y.float(), dim=(-2, -1))
                score.append(self._distance(a.abs(), b.abs(), kernel)
                             + self.phase_weight * self._distance(a.angle(), b.angle(), kernel))
            return sum(score).mean() * self.multiplier

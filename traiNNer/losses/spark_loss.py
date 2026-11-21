"""Frozen feature loss from umzi2/SparK's ``weight_33_ep`` encoder.

Despite its filename, this checkpoint is an Inception-style ConvNeXt variant
with BatchNorm, not timm's stock ConvNeXt. The dense encoder below preserves
its parameter names and all-active evaluation mathematics. Sparse masking and
the reconstruction decoder belong to pretraining, not this restoration loss.

Adapted from umzi2/SparK, commit 73fd06a43785599552f005c62073f648966d78d8:
pretrain/models/convnext.py and pretrain/encoder.py. See SPARK_LICENSE.txt.
"""

import hashlib
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import LOSS_REGISTRY

SPARK_URL = (
    "https://github.com/umzi2/SparK/releases/download/weight_33_ep/"
    "convnext_tiny_1kpretrained_timm_style.pth"
)
SPARK_SHA256 = "37b35923783e98afeb7c0b4b4603cd28fe385331e89f6336c4569a30405051e0"
SPARK_FILENAME = f"spark_33ep-{SPARK_SHA256}.pth"
STAGE_NAMES = ("stage1", "stage2", "stage3", "stage4")


class _InceptionDWConv(nn.Module):
    """Identity, square, horizontal and vertical depthwise branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch = channels // 8
        self.split_indexes = (channels - 3 * branch, branch, branch, branch)
        self.dwconv_hw = nn.Conv2d(branch, branch, 3, padding=1, groups=branch)
        self.dwconv_w = nn.Conv2d(
            branch, branch, (1, 11), padding=(0, 5), groups=branch
        )
        self.dwconv_h = nn.Conv2d(
            branch, branch, (11, 1), padding=(5, 0), groups=branch
        )

    def forward(self, x: Tensor) -> Tensor:
        identity, square, horizontal, vertical = x.split(self.split_indexes, dim=1)
        return torch.cat(
            (
                identity,
                self.dwconv_hw(square),
                self.dwconv_w(horizontal),
                self.dwconv_h(vertical),
            ),
            dim=1,
        )


class _ConvMlp(nn.Module):
    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels * ratio, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(channels * ratio, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SparkBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int) -> None:
        super().__init__()
        self.token_mixer = _InceptionDWConv(channels)
        # SparseBatchNorm1d on all selected pixels is dense BatchNorm2d in eval.
        self.norm = nn.BatchNorm2d(channels, eps=1e-5)
        self.mlp = _ConvMlp(channels, mlp_ratio)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.mlp(self.norm(self.token_mixer(x)))
        return x + self.gamma[None, :, None, None] * mixed


class _SparkStage(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, depth: int, mlp_ratio: int
    ) -> None:
        super().__init__()
        self.downsample = (
            nn.Identity()
            if in_channels == channels
            else nn.Sequential(
                nn.BatchNorm2d(in_channels, eps=1e-5),
                nn.Conv2d(in_channels, channels, 2, stride=2),
            )
        )
        self.blocks = nn.Sequential(
            *[_SparkBlock(channels, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.downsample(x))


class SparkFeatureExtractor(nn.Module):
    """The release's four stages: widths 96/192/384/768, depths 3/3/9/3."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4), nn.BatchNorm2d(96, eps=1e-5)
        )
        self.stages = nn.ModuleList(
            [
                _SparkStage(96, 96, 3, 4),
                _SparkStage(96, 192, 3, 4),
                _SparkStage(192, 384, 9, 4),
                _SparkStage(384, 768, 3, 3),
            ]
        )

    def forward(self, x: Tensor, last_stage: int = 3) -> list[Tensor]:
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if index == last_stage:
                break
        return features


def _load_spark_weights(weights_path: str | None) -> dict[str, Tensor]:
    """Download once, verify the pinned release, then deserialize tensors only."""
    if weights_path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / SPARK_FILENAME
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                SPARK_URL, str(path), hash_prefix=SPARK_SHA256
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != SPARK_SHA256:
            raise ValueError(f"SparK release checksum mismatch: {path}")
    else:
        path = Path(weights_path).expanduser()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state.items()
        )
    ):
        raise ValueError("SparK weights must be a direct tensor state dictionary")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("SparK weights contain non-finite values")
    return state


@LOSS_REGISTRY.register()
class SparKLoss(nn.Module):
    """Sum of weighted feature distances from a frozen SparK-pretrained encoder.

    Inputs are matching BCHW RGB images in the [0, 1] convention. They receive
    ImageNet normalization without clipping, resizing, or random masking. The
    feature network and distance reduction run in FP32 even inside AMP; gradients
    flow through the prediction only. BatchNorm always uses its frozen statistics.

    ``layer_weights`` selects stage1..stage4 (strides 4, 8, 16, 32); omitted stages
    have zero weight. By default all four contribute equally. This aggregation is
    a restoration-loss choice, not an objective supplied by the SparK release.
    The trainer applies ``loss_weight`` externally, so forward does not multiply
    by it. ``weights_path`` optionally supplies an exact-compatible local encoder.
    """

    def __init__(
        self,
        loss_weight: float,
        layer_weights: dict[str, float] | None = None,
        criterion: Literal["l1", "mse"] = "l1",
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(loss_weight):
            raise ValueError("loss_weight must be finite")
        if criterion not in ("l1", "mse"):
            raise ValueError("SparK criterion must be 'l1' or 'mse'")
        weights = (
            dict.fromkeys(STAGE_NAMES, 1.0)
            if layer_weights is None
            else dict(layer_weights)
        )
        if set(weights) - set(STAGE_NAMES):
            raise ValueError(
                f"SparK layer_weights keys must be drawn from {STAGE_NAMES}"
            )
        if (
            not weights
            or any(not math.isfinite(value) or value < 0 for value in weights.values())
            or not any(value > 0 for value in weights.values())
        ):
            raise ValueError(
                "SparK layer weights must be finite, nonnegative and not all zero"
            )
        self.layer_weights = tuple(float(weights.get(name, 0)) for name in STAGE_NAMES)
        self.last_stage = max(
            i for i, weight in enumerate(self.layer_weights) if weight > 0
        )
        self.minimum_size = 4 * 2**self.last_stage
        self.loss_weight = loss_weight
        self.criterion = criterion
        self.extractor = SparkFeatureExtractor()
        self.extractor.load_state_dict(_load_spark_weights(weights_path), strict=True)
        self.extractor.requires_grad_(False).eval()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "SparKLoss":
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape != gt.shape or x.shape[1] != 3:
            raise ValueError(
                "SparK loss requires matching BCHW RGB prediction and target"
            )
        if x.shape[0] == 0 or min(x.shape[-2:]) < self.minimum_size:
            raise ValueError(
                f"SparK loss requires a nonempty batch and H/W >= {self.minimum_size} "
                "for the selected feature stages"
            )
        if (
            x.device != gt.device
            or not x.is_floating_point()
            or not gt.is_floating_point()
        ):
            raise ValueError("SparK inputs must be floating tensors on the same device")
        with torch.autocast(device_type=x.device.type, enabled=False):
            prediction = self.extractor(
                (x.float() - self.mean) / self.std, self.last_stage
            )
            with torch.no_grad():
                reference = self.extractor(
                    (gt.detach().float() - self.mean) / self.std, self.last_stage
                )
            total = x.new_zeros((), dtype=torch.float32)
            for weight, pred, target in zip(
                self.layer_weights[: self.last_stage + 1],
                prediction,
                reference,
                strict=True,
            ):
                if weight > 0:
                    distance = (
                        F.l1_loss(pred, target)
                        if self.criterion == "l1"
                        else F.mse_loss(pred, target)
                    )
                    total = total + weight * distance
        return total

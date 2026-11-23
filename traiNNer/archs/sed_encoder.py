"""Frozen OpenAI CLIP RN50 visual features for SeD."""

# The RN50 trunk is adapted from https://github.com/openai/CLIP/blob/main/clip/model.py.
# MIT License
# Copyright (c) 2021 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from trainner_config.official_weights import RN50_SHA256, ensure_rn50, rn50_cache_path

SEMANTIC_CHANNELS = 1024
SEMANTIC_STRIDE = 16

_RN50_TRUNK_PREFIXES = (
    "visual.conv",
    "visual.bn",
    "visual.layer1.",
    "visual.layer2.",
    "visual.layer3.",
)


def validate_rgb_crop(image: Tensor, name: str) -> None:
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError(
            f"SeD {name} must be a floating-point B x 3 x H x W RGB tensor."
        )
    height, width = image.shape[-2:]
    if (
        min(height, width) < SEMANTIC_STRIDE
        or height % SEMANTIC_STRIDE
        or width % SEMANTIC_STRIDE
    ):
        raise ValueError(
            f"SeD {name} height and width must be positive multiples of "
            f"{SEMANTIC_STRIDE}. Got {height} x {width}."
        )


def _load_visual_trunk(encoder_weights: str | None) -> dict[str, Tensor]:
    path = (
        rn50_cache_path()
        if encoder_weights is None
        else Path(encoder_weights).expanduser()
    )
    try:
        archive = path.read_bytes()
    except FileNotFoundError as exc:
        if encoder_weights is not None:
            raise FileNotFoundError(
                f"SeD requires the official OpenAI CLIP RN50.pt archive. Missing {path}. "
                "Set network_d.encoder_weights to its local path."
            ) from exc
        path = ensure_rn50()
        archive = path.read_bytes()
    # Verify the exact bytes that TorchScript will read.
    digest = sha256(archive).hexdigest()
    if digest != RN50_SHA256:
        raise ValueError(
            f"SeD expected the official OpenAI CLIP RN50.pt archive at {path}. "
            f"SHA-256 was {digest}, expected {RN50_SHA256}."
        )

    clip_state = torch.jit.load(BytesIO(archive), map_location="cpu").state_dict()
    return {
        name.removeprefix("visual."): value
        for name, value in clip_state.items()
        if name.startswith(_RN50_TRUNK_PREFIXES)
    }


class _Bottleneck(nn.Module):
    def __init__(self, in_channels: int, width: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = 4 * width
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Sequential | None = None
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, image: Tensor) -> Tensor:
        features = F.relu(self.bn1(self.conv1(image)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.bn3(self.conv3(self.pool(features)))
        if self.downsample is not None:
            image = self.downsample(self.pool(image))
        return F.relu(features + image, inplace=True)


def _stage(in_channels: int, width: int, blocks: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        _Bottleneck(in_channels, width, stride),
        *(_Bottleneck(width * 4, width) for _ in range(blocks - 1)),
    )


class SeDConditioner(nn.Module):
    """Encode the full HR crop to a detached B x 1024 x (H/16) x (W/16) tensor."""

    def __init__(self, encoder_weights: str | None = None) -> None:
        super().__init__()
        pretrained = _load_visual_trunk(encoder_weights)

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2)
        self.layer1 = _stage(64, 64, 3)
        self.layer2 = _stage(256, 128, 4, stride=2)
        self.layer3 = _stage(512, 256, 6, stride=2)
        self.float()
        self.load_state_dict(pretrained, strict=True)
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> Self:
        # Keep CLIP's pretrained batch-norm statistics fixed.
        return super().train(False)

    @torch.no_grad()
    def forward(self, hr: Tensor) -> Tensor:
        validate_rgb_crop(hr, "reference")
        with torch.autocast(device_type=hr.device.type, enabled=False):
            features = (hr.to(dtype=self.conv1.weight.dtype) - self.mean) / self.std
            features = F.relu(self.bn1(self.conv1(features)), inplace=True)
            features = F.relu(self.bn2(self.conv2(features)), inplace=True)
            features = F.relu(self.bn3(self.conv3(features)), inplace=True)
            features = self.layer1(self.pool(features))
            features = self.layer2(features)
            return self.layer3(features)

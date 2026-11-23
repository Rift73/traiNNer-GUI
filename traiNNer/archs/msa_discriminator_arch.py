"""MSA-ESRGAN discriminator from Chen et al., Fig. 2 and Eqs. 1-4.

https://doi.org/10.1038/s41598-024-78813-5
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


class _ChannelSpatialAttentionFusion(nn.Module):
    """Fuse an encoder skip with the decoder feature at its spatial scale."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.global_norm = nn.BatchNorm2d(channels)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, encoder: Tensor, decoder: Tensor) -> Tensor:
        combined = encoder + decoder
        global_context = self.global_context(combined)
        norm = self.global_norm
        if norm.training and global_context.shape[0] == 1:
            # Global pooling leaves one value per channel. Use the running
            # statistics when a singleton batch cannot estimate a variance.
            global_context = F.batch_norm(
                global_context,
                norm.running_mean,
                norm.running_var,
                norm.weight,
                norm.bias,
                eps=norm.eps,
            )
        else:
            global_context = norm(global_context)

        channel_weights = (self.local_context(combined) + global_context).sigmoid()
        attended = combined * channel_weights
        pooled = torch.cat(
            (attended.mean(dim=1, keepdim=True), attended.amax(dim=1, keepdim=True)),
            dim=1,
        )
        spatial_weights = self.spatial(pooled).sigmoid()
        return encoder * spatial_weights + decoder * (1.0 - spatial_weights)


def _spectral_conv_relu(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False)
        ),
        nn.ReLU(inplace=True),
    )


def _upsample_to(feature: Tensor, skip: Tensor) -> Tensor:
    return F.relu(
        F.interpolate(
            feature, size=skip.shape[-2:], mode="bilinear", align_corners=False
        ),
        inplace=True,
    )


class MSADiscriminator(nn.Module):
    """Attention U-Net with quarter-, half-, and full-resolution CS-AFM skips.

    Accepts BCHW images with H and W of at least 8. The three fusions feed
    the full-resolution B x 1 x H x W score map shown in Fig. 2.
    """

    def __init__(
        self, num_in_ch: int = 3, num_feat: int = 64, reduction: int = 4
    ) -> None:
        super().__init__()
        if min(num_in_ch, num_feat, reduction) < 1:
            raise ValueError("num_in_ch, num_feat, and reduction must be positive.")

        self.stem = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down_half = _spectral_conv_relu(num_feat, 2 * num_feat, 4, 2)
        self.down_quarter = _spectral_conv_relu(2 * num_feat, 4 * num_feat, 4, 2)
        self.down_eighth = _spectral_conv_relu(4 * num_feat, 8 * num_feat, 4, 2)

        self.up_quarter = _spectral_conv_relu(8 * num_feat, 4 * num_feat)
        self.up_half = _spectral_conv_relu(4 * num_feat, 2 * num_feat)
        self.up_full = _spectral_conv_relu(2 * num_feat, num_feat)

        self.fuse_quarter = _ChannelSpatialAttentionFusion(4 * num_feat, reduction)
        self.fuse_half = _ChannelSpatialAttentionFusion(2 * num_feat, reduction)
        self.fuse_full = _ChannelSpatialAttentionFusion(num_feat, reduction)

        self.head = nn.Sequential(
            _spectral_conv_relu(num_feat, num_feat),
            _spectral_conv_relu(num_feat, num_feat),
            # GANLoss consumes signed logits from the final convolution.
            spectral_norm(nn.Conv2d(num_feat, 1, 3, padding=1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or min(x.shape[-2:]) < 8:
            raise ValueError("MSA discriminator expects BCHW images with H and W >= 8.")

        full = self.stem(x)
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        eighth = self.down_eighth(quarter)

        decoded = self.fuse_quarter(
            quarter, self.up_quarter(_upsample_to(eighth, quarter))
        )
        decoded = self.fuse_half(half, self.up_half(_upsample_to(decoded, half)))
        decoded = self.fuse_full(full, self.up_full(_upsample_to(decoded, full)))
        return self.head(decoded)


@ARCH_REGISTRY.register()
def msa_discriminator(
    num_in_ch: int = 3, num_feat: int = 64, reduction: int = 4
) -> MSADiscriminator:
    """Build the MSA-ESRGAN multi-scale attention U-Net discriminator."""
    return MSADiscriminator(num_in_ch, num_feat, reduction)

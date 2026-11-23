"""Pixel-wise U+SeD discriminator from Li et al. (CVPR 2024).

The semantic fusion blocks follow the paper and its supplement.
https://openaccess.thecvf.com/content/CVPR2024/html/Li_SeD_Semantic-Aware_Discriminator_for_Image_Super-Resolution_CVPR_2024_paper.html
"""

from torch import Tensor, cat, nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils import spectral_norm

from traiNNer.archs.discriminator_arch import UNetDiscriminatorSN
from traiNNer.archs.sed_encoder import (
    SEMANTIC_CHANNELS,
    SEMANTIC_STRIDE,
    SeDConditioner,
    validate_rgb_crop,
)
from traiNNer.utils.registry import ARCH_REGISTRY


class _Attention(nn.Module):
    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        channels = heads * head_dim
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.out = nn.Linear(channels, channels)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        return tokens.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        query = self._split_heads(self.query(queries))
        key = self._split_heads(self.key(context))
        value = self._split_heads(self.value(context))
        mixed = F.scaled_dot_product_attention(query, key, value)
        return self.out(mixed.transpose(1, 2).flatten(-2))


class _SemanticFusion(nn.Module):
    def __init__(
        self, image_channels: int, image_stride: int, heads: int, head_dim: int
    ) -> None:
        super().__init__()
        hidden_dim = heads * head_dim
        self.shuffle_factor = SEMANTIC_STRIDE // image_stride
        self.shuffled_channels = SEMANTIC_CHANNELS // self.shuffle_factor**2

        self.semantic_norm = nn.GroupNorm(32, SEMANTIC_CHANNELS, eps=1e-6)
        self.semantic_projection = nn.Conv2d(SEMANTIC_CHANNELS, hidden_dim, 1)
        self.image_projection = nn.Conv2d(image_channels, hidden_dim, 1)
        self.self_attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = _Attention(heads, head_dim)
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = _Attention(heads, head_dim)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, 8 * hidden_dim)
        self.feedforward_projection = nn.Linear(4 * hidden_dim, hidden_dim)
        self.output_projection = nn.Conv2d(hidden_dim, SEMANTIC_CHANNELS, 1)
        nn.init.zeros_(self.output_projection.weight)
        assert self.output_projection.bias is not None
        nn.init.zeros_(self.output_projection.bias)

        self.spatial_conv = nn.Conv2d(
            self.shuffled_channels, self.shuffled_channels, 3, padding=1
        )
        self.merge = spectral_norm(
            nn.Conv2d(
                image_channels + self.shuffled_channels, image_channels, 3, padding=1
            )
        )

    def forward(self, image: Tensor, semantic: Tensor) -> Tensor:
        batch, _, height, width = semantic.shape
        semantic = self.semantic_projection(self.semantic_norm(semantic))
        semantic_tokens = semantic.flatten(2).transpose(1, 2).contiguous()
        image_tokens = (
            self.image_projection(image).flatten(2).transpose(1, 2).contiguous()
        )

        queries = self.self_attention_norm(semantic_tokens)
        queries = self.cross_attention_norm(self.self_attention(queries, queries))
        features = self.feedforward_norm(self.cross_attention(queries, image_tokens))
        values, gate = self.gate(features).chunk(2, dim=-1)
        features = self.feedforward_projection(values * F.gelu(gate))
        features = features.transpose(1, 2).reshape(batch, -1, height, width)
        features = self.output_projection(features)

        # SeD packs subpixel channels with the horizontal offset before the
        # vertical offset. Preserve that ordering when expanding the grid.
        scale = self.shuffle_factor
        features = features.reshape(
            batch, self.shuffled_channels, scale, scale, height, width
        )
        features = features.permute(0, 1, 4, 3, 5, 2).reshape(
            batch, self.shuffled_channels, height * scale, width * scale
        )
        return self.merge(cat((image, self.spatial_conv(features)), dim=1))


@ARCH_REGISTRY.register()
class SeD(nn.Module):
    """Score RGB crops with fixed CLIP RN50 conditioning from their paired HR.

    Build one conditioner per training device, then reuse its output for all
    critic calls on the batch. Both crop dimensions must be multiples of 16.
    The critic owns only trainable U+SeD weights.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        skip_connection: bool = True,
        nheads: int = 1,
        dhead: int = 64,
        encoder_weights: str | None = None,
    ) -> None:
        if num_in_ch != 3:
            raise ValueError("SeD expects three RGB input channels.")
        if min(num_feat, nheads, dhead) < 1:
            raise ValueError("SeD num_feat, nheads and dhead must be positive.")
        super().__init__()
        self.unet = UNetDiscriminatorSN(num_in_ch, num_feat, skip_connection)
        self.encoder_weights = encoder_weights
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.fuse_eighth = _SemanticFusion(num_feat * 8, 8, nheads, dhead)
        self.fuse_quarter = _SemanticFusion(num_feat * 4, 4, nheads, dhead)
        self.fuse_half = _SemanticFusion(num_feat * 2, 2, nheads, dhead)

    def build_conditioner(self) -> SeDConditioner:
        """Load the pretrained encoder as a separate, frozen module."""
        return SeDConditioner(self.encoder_weights)

    def forward(self, rgb: Tensor, condition: Tensor) -> Tensor:
        validate_rgb_crop(rgb, "candidate")
        batch, _, height, width = rgb.shape
        expected = (
            batch,
            SEMANTIC_CHANNELS,
            height // SEMANTIC_STRIDE,
            width // SEMANTIC_STRIDE,
        )
        if condition.shape != expected or not condition.is_floating_point():
            raise ValueError(
                f"SeD expected a floating-point conditioning tensor of shape {expected}, "
                f"got {tuple(condition.shape)} ({condition.dtype})."
            )
        if condition.device != rgb.device:
            raise ValueError(
                "SeD conditioning and candidate must be on the same device."
            )
        unet = self.unet
        condition = condition.detach().to(dtype=unet.conv0.weight.dtype)

        full = self.activation(unet.conv0(rgb))
        half = self.activation(unet.conv1(full))
        quarter = self.activation(unet.conv2(half))
        features = self.activation(self.fuse_eighth(unet.conv3(quarter), condition))

        features = F.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=False
        )
        features = self.activation(unet.conv4(features))
        if unet.skip_connection:
            features = features + quarter
        features = self.fuse_quarter(features, condition)

        features = F.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=False
        )
        features = self.activation(unet.conv5(features))
        if unet.skip_connection:
            features = features + half
        features = self.fuse_half(features, condition)

        features = F.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=False
        )
        features = self.activation(unet.conv6(features))
        if unet.skip_connection:
            features = features + full
        features = self.activation(unet.conv7(features))
        features = self.activation(unet.conv8(features))
        return unet.conv9(features)

"""DUnet dense scoring with DUAL local neighborhood attention at stride eight."""

from torch import Tensor

from traiNNer.archs.dual_arch import TokenNorm, _SpatialAttention
from traiNNer.archs.dunet_arch import DUnet
from traiNNer.ops.input_only_conv import InputOnlyConv2d
from traiNNer.utils.registry import ARCH_REGISTRY


class DUALDiscriminator(DUnet):
    """Map nonempty BCHW images to unnormalized Bx1xHxW logits."""

    def __init__(
        self,
        num_in_ch: int,
        num_feat: int,
        *,
        context_heads: int,
        context_window: int,
    ) -> None:
        if type(num_in_ch) is not int or num_in_ch < 1:
            raise ValueError("num_in_ch must be a positive integer")
        if type(num_feat) is not int or num_feat < 2 or num_feat % 2:
            raise ValueError("num_feat must be a positive even integer for DySample")
        if type(context_heads) is not int or context_heads < 1:
            raise ValueError("context_heads must be a positive integer")
        if type(context_window) is not int or context_window < 2:
            raise ValueError("context_window must be an integer of at least two")

        super().__init__(num_in_ch=num_in_ch, num_feat=num_feat)
        bottleneck = 8 * num_feat
        channels = 32 * context_heads  # DUAL uses 32 channels per head.
        self.context_in = InputOnlyConv2d(bottleneck, channels, 1)
        self.context_norm = TokenNorm(channels, eps=1e-5)
        self.attention = _SpatialAttention(
            channels, context_heads, context_window, pos_bias="rpb"
        )
        # Keep DUAL's own CUDA dispatch and PyTorch SDPA fallback.
        self.attention.local_na = True
        self.context_out = InputOnlyConv2d(channels, bottleneck, 1)

    def forward(self, x: Tensor) -> Tensor:
        if (
            x.ndim != 4
            or x.shape[0] < 1
            or x.shape[1] != self.in_to_dim.in_channels
            or min(x.shape[-2:]) < 1
        ):
            raise ValueError("expected nonempty BCHW input with matching channels")

        x0 = self.in_to_dim(x)
        x1 = self.e_x1(x0)
        x2 = self.e_x2(x1)
        x3 = self.e_x3(x2)
        context = self.context_in(x3)
        x3 = x3 + self.context_out(self.attention(self.context_norm(context)))

        # Stride-2 downsampling rounds up, so trim the upsampled map to its skip.
        x = self.up1(x3)[..., : x2.shape[-2], : x2.shape[-1]] + x2
        x = self.up2(x)[..., : x1.shape[-2], : x1.shape[-1]] + x1
        x = self.up3(x)[..., : x0.shape[-2], : x0.shape[-1]] + x0
        return self.end_conv(x)


@ARCH_REGISTRY.register()
def dual_discriminator(
    num_in_ch: int = 3,
    num_feat: int = 64,
    *,
    context_heads: int = 2,
    context_window: int = 16,
) -> DUALDiscriminator:
    """Build a dense discriminator with 32 channels per attention head."""
    return DUALDiscriminator(
        num_in_ch,
        num_feat,
        context_heads=context_heads,
        context_window=context_window,
    )

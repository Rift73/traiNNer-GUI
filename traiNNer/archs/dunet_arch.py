from typing import Any

from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.archs.arch_util import DySampleV1 as DySample
from traiNNer.ops.input_only_conv import InputOnlyConv2d
from traiNNer.utils.registry import ARCH_REGISTRY


class Down(nn.Sequential):
    def __init__(self, dim: int) -> None:
        super().__init__(
            spectral_norm(InputOnlyConv2d(dim, dim * 2, 3, 2, 1)), nn.Mish(True)
        )


class Up(nn.Sequential):
    def __init__(self, dim: int) -> None:
        upsampler = DySample(dim, dim, 2, 4, False)
        upsampler.followed_by_conv = True
        # Keep DySample's initialization, parameters and sampling mathematics.
        # Its offset/scope convolutions also need only input gradients in G's
        # frozen-discriminator phase; D's trainable phase stays ordinary Conv2d.
        upsampler.offset = InputOnlyConv2d.from_conv(upsampler.offset)
        upsampler.scope = InputOnlyConv2d.from_conv(upsampler.scope)
        super().__init__(
            upsampler,
            spectral_norm(InputOnlyConv2d(dim, dim // 2, 3, 1, 1)),
        )


@ARCH_REGISTRY.register()
class DUnet(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64) -> None:
        super().__init__()
        self.in_to_dim = InputOnlyConv2d(num_in_ch, num_feat, 3, 1, 1)
        # encode x
        self.e_x1 = Down(num_feat)
        self.e_x2 = Down(num_feat * 2)
        self.e_x3 = Down(num_feat * 4)
        # up
        self.up1 = Up(num_feat * 8)
        self.up2 = Up(num_feat * 4)
        self.up3 = Up(num_feat * 2)
        # end conv
        self.end_conv = nn.Sequential(
            spectral_norm(InputOnlyConv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.Mish(True),
            spectral_norm(InputOnlyConv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.Mish(True),
            InputOnlyConv2d(num_feat, 1, 3, 1, 1),
        )

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        if "_lqd_version" in state_dict and not hasattr(self, "_lqd_version"):
            raise ValueError(
                "Cannot load LQ-D weights into RGB DUnet, even with strict_load_d: false. "
                "Switch with a new run and an RGB DUnet checkpoint (or a fresh D)."
            )
        if "_mcd_version" in state_dict and not hasattr(self, "_mcd_version"):
            raise ValueError(
                "Cannot load MC-D weights into RGB DUnet, even with strict_load_d: false. "
                "Switch with a new run and an RGB DUnet checkpoint (or a fresh D)."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.in_to_dim(x)
        x1 = self.e_x1(x0)
        x2 = self.e_x2(x1)
        x3 = self.e_x3(x2)
        x = self.up1(x3) + x2
        x = self.up2(x) + x1
        x = self.up3(x) + x0
        return self.end_conv(x)

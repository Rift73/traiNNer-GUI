"""Observed-input-only discriminator conditioning; no HR statistics."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class LQCondition(nn.Module):
    """Resize the final guarded/augmented RGB LQ once for all three D passes.

    Only the desired spatial size is supplied alongside LQ, never HR pixels or
    generator output. Match MC-D's existing degraded-RGB preprocessing exactly.
    """

    @torch.no_grad()
    def forward(self, lq_rgb: Tensor, output_size: tuple[int, int]) -> Tensor:
        if (
            lq_rgb.ndim != 4
            or lq_rgb.shape[1] != 3
            or lq_rgb.shape[0] == 0
            or min(lq_rgb.shape[-2:]) < 1
        ):
            raise ValueError("LQ-D requires a nonempty NCHW RGB degraded input")
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError("LQ-D condition output size must contain positive H/W")
        with torch.autocast(device_type=lq_rgb.device.type, enabled=False):
            return F.interpolate(
                lq_rgb.detach().float(),
                size=output_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0, 1)


def lq_discriminator_input(image: Tensor, condition: Tensor | None) -> Tensor:
    """Keep the candidate image differentiable and the shared condition detached."""
    if (
        condition is None
        or image.ndim != 4
        or image.shape[1] != 3
        or condition.ndim != 4
        or condition.shape[1] != 3
        or condition.shape[0] != image.shape[0]
        or condition.shape[-2:] != image.shape[-2:]
        or condition.device != image.device
    ):
        raise ValueError(
            "LQ-D requires matching RGB image/condition batch, size and device"
        )
    return torch.cat((image, condition.detach().to(dtype=image.dtype)), dim=1)

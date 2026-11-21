"""Fixed target-medium statistics and optional degraded-input MC-D condition."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class MediumCondition(nn.Module):
    """Six detached FP32 channels: four band energies and two orientations.

    Compute from the final RGB loss target once per microbatch, after augmentation.
    Reuse the result for real and generated images alike.
    Optionally append three detached, resized degraded-RGB channels. Kernels are
    nonpersistent; the discriminator checkpoint records its conditioning version.
    """

    def __init__(self) -> None:
        super().__init__()
        for name, sigma in (("g1", 1.0), ("g2", 2.0), ("g4", 4.0), ("g15", 1.5)):
            radius = math.ceil(3 * sigma)
            positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-positions.square() / (2 * sigma * sigma))
            self.register_buffer(name, kernel / kernel.sum(), persistent=False)
        sobel = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        )
        self.register_buffer(
            "sobel", torch.stack((sobel, sobel.T)).unsqueeze(1), persistent=False
        )

    @staticmethod
    def _blur(image: Tensor, kernel: Tensor) -> Tensor:
        channels = image.shape[1]
        radius = kernel.numel() // 2
        image = F.conv2d(
            F.pad(image, (radius, radius, 0, 0), mode="reflect"),
            kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1),
            groups=channels,
        )
        return F.conv2d(
            F.pad(image, (0, 0, radius, radius), mode="reflect"),
            kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1),
            groups=channels,
        )

    @torch.no_grad()
    def forward(self, target: Tensor, lq_rgb: Tensor | None = None) -> Tensor:
        if target.ndim != 4 or target.shape[1] != 3:
            raise ValueError("MC-D requires an NCHW RGB loss target")
        if min(target.shape[-2:]) < 16 or target.shape[0] == 0:
            raise ValueError("MC-D requires nonempty targets at least 16x16")
        if lq_rgb is not None and (
            lq_rgb.ndim != 4
            or lq_rgb.shape[:2] != target.shape[:2]
            or min(lq_rgb.shape[-2:]) < 1
            or lq_rgb.device != target.device
        ):
            raise ValueError("MC-D degraded RGB input must match target batch/device")
        # Explicitly disable the surrounding training autocast for statistics.
        with torch.autocast(device_type=target.device.type, enabled=False):
            rgb = target.detach().float()
            y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            chroma = torch.cat((rgb[:, 2:3] - y, rgb[:, 0:1] - y), dim=1)
            y1 = self._blur(y, self.g1)
            y2 = self._blur(y, self.g2)
            y4 = self._blur(y, self.g4)
            bands = torch.cat((y - y1, y1 - y2, y2 - y4), dim=1)
            chroma_energy = (
                (chroma - self._blur(chroma, self.g1)).square().sum(dim=1, keepdim=True)
            )
            energy = F.avg_pool2d(torch.cat((bands.square(), chroma_energy), dim=1), 4)
            energy = torch.log1p(energy / (0.5 / 255) ** 2) / math.log(4097)

            gradients = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), self.sobel)
            gx, gy = gradients[:, 0:1], gradients[:, 1:2]
            structure = F.avg_pool2d(
                self._blur(
                    torch.cat((gx.square(), gx * gy, gy.square()), dim=1), self.g15
                ),
                4,
            )
            j11, j12, j22 = structure.split(1, dim=1)
            difference = j11 - j22
            radius = torch.sqrt(difference.square() + 4 * j12.square())
            coherence = radius / (j11 + j22 + (1 / 255) ** 2)
            orientation = torch.cat((difference, 2 * j12), dim=1)
            orientation = coherence * orientation / (radius + 1e-12)
            medium = F.interpolate(
                torch.cat((energy, orientation), dim=1),
                size=target.shape[-2:],
                mode="nearest",
            )
            if lq_rgb is None:
                return medium
            # Compute once per microbatch, outside the compiled D. Reuse
            # exactly the same condition for G-adversarial, real-D and fake-D.
            degraded = F.interpolate(
                lq_rgb.detach().float(),
                size=target.shape[-2:],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0, 1)
            return torch.cat((medium, degraded), dim=1)


def medium_discriminator_input(image: Tensor, condition: Tensor | None) -> Tensor:
    """The disabled path is an identity, retaining the existing discriminator."""
    if condition is None:
        return image
    if (
        image.ndim != 4
        or image.shape[1] != 3
        or condition.ndim != 4
        or condition.shape[1] not in (6, 9)
        or condition.shape[0] != image.shape[0]
        or condition.shape[-2:] != image.shape[-2:]
        or condition.device != image.device
    ):
        raise ValueError("MC-D image and condition batch/spatial dimensions must match")
    return torch.cat((image, condition.detach().to(dtype=image.dtype)), dim=1)

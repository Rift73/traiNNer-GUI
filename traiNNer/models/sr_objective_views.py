"""Training-only prediction clamp and discriminator input transform for image SR."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.losses.loss_util import restoring_clamp


def neosr_lq_reference(lq: Tensor, output: Tensor) -> Tensor:
    with torch.no_grad(), torch.autocast(lq.device.type, enabled=False):
        image = F.interpolate(lq.float(), size=output.shape[-2:], mode="bicubic", antialias=True)
        return image.clamp_(1 / 255, 1)


def clamp_loss_output(image: Tensor, mode: str) -> Tensor:
    if mode == "restore":
        return restoring_clamp(image)
    if mode == "hard":
        return image.clamp(1 / 255, 1)
    if mode == "none":
        return image
    raise ValueError(f"Unknown loss_output_clamp mode {mode!r}.")


# Symlet 19 reconstruction low-pass taps from PyWavelets' sym19_double table.
# https://github.com/PyWavelets/pywt/blob/v1.8.0/pywt/_extensions/c/wavelets_coeffs.template.h
_SYM19 = (
    0.0000017509367995348687,
    0.0000020623170632395688,
    -0.000028151138661550245,
    -0.000016821387029373716,
    0.0002762187768573407,
    0.00012930767650701415,
    -0.0017049602611649971,
    -0.0006179223277983108,
    0.008262236955528255,
    0.004319351874894969,
    -0.02770989693131125,
    -0.016908234861345205,
    0.08407267627924504,
    0.09363084341589714,
    -0.11624173010739675,
    -0.17659686625203097,
    0.2582661692372836,
    0.7195555257163943,
    0.578144945338605,
    0.10902582508127781,
    -0.06752505804029409,
    0.008954591173043624,
    0.0070155738571741596,
    -0.046635983534938946,
    -0.02265199337824595,
    0.01579743929567463,
    0.007968438320613306,
    -0.005122205002583014,
    -0.0011607032572062486,
    0.0021214250281823303,
    0.00015915804768084938,
    -0.000635764515004334,
    -0.00004612039600210587,
    0.0001155392333357879,
    0.000008873312173729286,
    -0.000011880518269823984,
    -6.463651303345963e-7,
    5.487732768215838e-7,
)


class WaveletDiscriminatorView(nn.Module):
    """NeoSR's periodic level-one sym19 detail bands of BT.601 luma."""

    def __init__(self, start_iter: int) -> None:
        super().__init__()
        self.start_iter = start_iter
        # conv2d correlates. Reverse the decomposition filters once. Normalize
        # only the low pass to unit DC before casting the bank to FP32.
        lo = torch.tensor(_SYM19, dtype=torch.float64)
        hi = lo.flip(0)
        hi[1::2].neg_()
        bank = torch.stack((lo / lo.sum(), hi)).float()
        self.register_buffer("row", bank.reshape(2, 1, 1, 38), persistent=False)
        self.register_buffer(
            "col", bank.reshape(2, 1, 38, 1).repeat(2, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "luma",
            torch.tensor([65.481, 128.553, 24.966], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "gain",
            torch.tensor([0.035, 0.025, 0.02], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 19:
            raise ValueError(
                "The sym19 GAN view requires NCHW RGB images at least 19 pixels high and wide."
            )
        with torch.autocast(rgb.device.type, enabled=False):
            y = 16.0 + (rgb.float() * self.luma).sum(1, keepdim=True)
            rows = F.conv2d(F.pad(y, (18, 19, 0, 0), mode="circular"), self.row)
            bands = F.conv2d(
                F.pad(rows, (0, 0, 18, 19), mode="circular"), self.col, groups=2
            )
            return bands[:, 1:] * self.gain

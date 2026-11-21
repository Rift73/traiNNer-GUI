"""VOSR 2.0 as a teacher: as a loss (vosrloss) or as the HR target (train.vosr_target, see realesrgan_model.py).

The teacher (upstream VOSR 2.0 as one 512 px window on TensorRT FP8 engines, see vosr2_trt.py) sees the RGB LR the
way upstream's inference script does: rounded to 8-bit, enlarged with Pillow BICUBIC, then colour-corrected with
upstream's wavelet_color_fix. Outputs smaller than 512 px are reflect-padded to 512 and cropped back. As a loss, the
model's loss loop scales the returned distance against the GAN term with balance_vosr. Design: docs/VOSR_loss.md.
"""

from functools import cache
from typing import Any

import torch
from torch import Tensor, nn

from traiNNer.losses.basic_loss import CharbonnierLoss, L1Loss
from traiNNer.losses.perceptual_fp16_loss import (
    VGG19_CONV_CRITERION,
    VGG19_RELU_CRITERION,
)
from traiNNer.losses.perceptual_loss import PerceptualLoss
from traiNNer.utils.registry import LOSS_REGISTRY

PRECISION_BITS = 22  # Pillow's 8-bit resampling: 32 - 8 - 2
WINDOW = 512
CRITERIA = {"charbonnier", "l1", "perceptual"}
PIXEL_CRITERIA = {"charbonnier": CharbonnierLoss, "l1": L1Loss}
PERCEPTUAL_CRITERIA = VGG19_CONV_CRITERION | VGG19_RELU_CRITERION


def _bicubic(x: float) -> float:
    """Pillow's bicubic_filter (a = -0.5, support 2)."""
    a, x = -0.5, abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


@cache
def _coefficients(
    in_size: int, out_size: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for one axis: first tap and fixed-point weights per output."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    starts, weights = [], []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(
            int(center - support + 0.5), 0
        )  # int() truncates toward zero, as the C cast
        xmax = min(int(center + support + 0.5), in_size)
        taps = [
            _bicubic((x + xmin - center + 0.5) / filterscale)
            for x in range(xmax - xmin)
        ]
        total = sum(taps)
        if total != 0.0:
            taps = [tap / total for tap in taps]
        starts.append(xmin)
        weights.append(
            tuple(
                int(tap * (1 << PRECISION_BITS) + (-0.5 if tap < 0 else 0.5))
                for tap in taps
            )
        )
    return tuple(starts), tuple(weights)


@cache
def _taps(in_size: int, out_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Gather indices and int64 weights (zero-padded to the longest tap list) for one axis, on `device`."""
    starts, weights = _coefficients(in_size, out_size)
    count = max(len(w) for w in weights)
    index = [
        [start + min(t, len(w) - 1) for t in range(count)]
        for start, w in zip(starts, weights, strict=True)
    ]
    weight = [list(w) + [0] * (count - len(w)) for w in weights]
    return torch.tensor(index, device=device), torch.tensor(
        weight, dtype=torch.int64, device=device
    )


def _resample_last_axis(image: Tensor, out_size: int) -> Tensor:
    """One 8-bit Pillow pass along the last axis: uint8 in, uint8 out (round, clip)."""
    index, weight = _taps(image.shape[-1], out_size, image.device)
    total = (image.to(torch.int64)[..., index] * weight).sum(-1) + (
        1 << (PRECISION_BITS - 1)
    )
    return (total >> PRECISION_BITS).clamp_(0, 255).to(torch.uint8)


def pillow_bicubic_up(image: Tensor, scale: int) -> Tensor:
    """(N, C, H, W) uint8 -> (N, C, H*scale, W*scale) uint8, bit-exact to Pillow's Image.resize(BICUBIC) on 8-bit
    images: horizontal pass first, its rounded 8-bit result then resampled vertically."""
    height, width = image.shape[-2:]
    wide = _resample_last_axis(image, width * scale)
    return _resample_last_axis(wide.transpose(-1, -2), height * scale).transpose(-1, -2)


def vosr_balance_state(
    distance: Tensor,
    gan_term: Tensor,
    state: tuple[Tensor, Tensor] | None,
    decay: float = 0.99,
) -> tuple[Tensor, Tensor]:
    """The running means (GAN term, raw VOSR distance) after one more step, detached and float32; the first step
    starts them at its own values."""
    gan = gan_term.detach().float()
    raw = distance.detach().float()
    if state is None:
        return gan, raw
    return decay * state[0] + (1 - decay) * gan, decay * state[1] + (1 - decay) * raw


def balance_vosr(
    distance: Tensor,
    gan_term: Tensor | None,
    state: tuple[Tensor, Tensor] | None,
    loss_weight: float,
    decay: float = 0.99,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]] | None:
    """Scale the raw VOSR distance so its running mean is `loss_weight` times the GAN term's running mean.

    Returns (weighted loss, scale, new state), or None when there is no GAN term this step. The scale is detached:
    gradients flow through `distance` only. Everything stays on the device (no host sync).
    """
    if gan_term is None:
        return None
    mean_gan, mean_raw = vosr_balance_state(distance, gan_term, state, decay)
    scale = loss_weight * mean_gan / mean_raw
    return scale * distance, scale, (mean_gan, mean_raw)


class VosrTeacher:
    """VOSR 2.0's restoration of an RGB LR batch, with fresh noise per sample.

    Args:
        engine_dir (str): folder with the 512 px FP8 stage engines (<stage>.plan).
        plugin (str): the vosr2_qkv TensorRT plugin library.
        lanes (int): windows run side by side.
    """

    def __init__(self, engine_dir: str, plugin: str, lanes: int = 2) -> None:
        # Imported here: every *_loss.py module loads at start-up, TensorRT only when VOSR is configured
        from traiNNer.losses import vosr2_trt

        self.window = vosr2_trt.Vosr2Window(engine_dir, plugin, lanes=lanes)
        self.pad_to_window = vosr2_trt.pad_to_window
        self.wavelet_colour = vosr2_trt.wavelet_colour
        self.generator = torch.Generator(device=self.window.device).manual_seed(
            torch.initial_seed()
        )

    @torch.compiler.disable
    def restore(self, lq_rgb: Tensor, scale: int) -> Tensor:
        """(N, 3, h, w) RGB in 0..1 -> VOSR's (N, 3, h*scale, w*scale) restoration in 0..1, float32, contiguous.

        The result is an inference tensor; clone it outside inference mode before autograd saves it.
        """
        height, width = lq_rgb.shape[-2] * scale, lq_rgb.shape[-1] * scale
        if height > WINDOW or width > WINDOW:
            raise ValueError(
                f"VOSR teacher: the output is at most {WINDOW}x{WINDOW} px, got {height}x{width}"
            )
        with torch.inference_mode(), torch.autocast(lq_rgb.device.type, enabled=False):
            lq = (lq_rgb.float().clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
            reference = pillow_bicubic_up(lq, scale).float() / 255 * 2 - 1
            restored = self.window(self.pad_to_window(reference), self.generator)
            restored = restored[..., :height, :width]
            return (self.wavelet_colour(restored, reference) * 0.5 + 0.5).contiguous()


@LOSS_REGISTRY.register()
class VOSRLoss(nn.Module):
    """Distance to VOSR 2.0's restoration of the same LR, normalized to the GAN loss's raw magnitude.

    Args:
        loss_weight (float): compares directly with ganloss's loss_weight: equal weights give equal pull, 0.1
            against a GAN weight of 0.02 gives 5x the GAN's pull. Must be positive.
        engine_dir (str): folder with the 512 px FP8 stage engines (<stage>.plan).
        plugin (str): the vosr2_qkv TensorRT plugin library.
        criterion (str): 'charbonnier', 'l1' (pixels) or 'perceptual' (VGG19 features, as perceptualloss).
        perceptual_options (dict): with criterion 'perceptual', perceptualloss's options other than loss_weight
            (criterion, stride_fd, w_lambda, layer_weights, ...). Its criterion 'charbonnier' (default) or 'l1'
            compares features position by position; 'fd' and 'pd' compare their distributions, which suits VOSR's
            grain (different on every noise draw); 'fd+l1' and 'pd+l1' combine both.
        lanes (int): windows run side by side.
        still_frames (int): clip training only (image training ignores it): up to this many other frames of each
            clip's key shot also train against the key frame's restoration where the picture stays still. 0 trains
            the key frame only.
    """

    def __init__(
        self,
        loss_weight: float,
        engine_dir: str,
        plugin: str,
        criterion: str = "charbonnier",
        perceptual_options: dict[str, Any] | None = None,
        lanes: int = 2,
        still_frames: int = 2,
    ) -> None:
        super().__init__()
        if not loss_weight > 0:
            raise ValueError(
                "vosrloss: loss_weight is VOSR's pull relative to the GAN's and must be positive"
            )
        if still_frames < 0:
            raise ValueError("vosrloss: still_frames must be 0 or more")
        if criterion not in CRITERIA:
            raise ValueError(
                f"vosrloss: unsupported criterion {criterion!r}, use one of {sorted(CRITERIA)}"
            )
        if perceptual_options is not None and criterion != "perceptual":
            raise ValueError(
                "vosrloss: perceptual_options applies only to criterion: perceptual"
            )
        options = dict(perceptual_options or {})
        if "loss_weight" in options:
            raise ValueError(
                "vosrloss: perceptual_options cannot set loss_weight; vosrloss's own loss_weight applies"
            )
        if options.get("criterion", "charbonnier") not in PERCEPTUAL_CRITERIA:
            raise ValueError(
                f"vosrloss: unsupported perceptual_options criterion {options['criterion']!r}, "
                f"use one of {sorted(PERCEPTUAL_CRITERIA)}"
            )
        self.loss_weight = loss_weight
        self.still_frames = still_frames
        self.criterion = (
            PerceptualLoss(loss_weight=1.0, **options)
            if criterion == "perceptual"
            else PIXEL_CRITERIA[criterion](loss_weight=1.0)
        )
        self.teacher = VosrTeacher(engine_dir, plugin, lanes)

    def forward(self, output: Tensor, lq_rgb: Tensor) -> Tensor:
        scale = output.shape[-1] // lq_rgb.shape[-1]
        if output.shape[-2:] != (lq_rgb.shape[-2] * scale, lq_rgb.shape[-1] * scale):
            raise ValueError(
                f"vosrloss: output {tuple(output.shape[-2:])} is not an integer upscale of the LR {tuple(lq_rgb.shape[-2:])}"
            )
        # clone: a normal tensor, safe for autograd to save
        target = self.teacher.restore(lq_rgb, scale).clone()
        return self.distance(output, target)

    def distance(self, output: Tensor, target: Tensor) -> Tensor:
        """The criterion's distance from the output to a given target, without the teacher."""
        distance = self.criterion(output, target)
        if isinstance(distance, dict):  # perceptual: one term per VGG layer
            return torch.stack(tuple(distance.values())).sum()
        return distance

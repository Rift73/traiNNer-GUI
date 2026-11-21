"""Read one still-SR print update from its existing backward and discriminator scores."""

import math

import torch
from torch import Tensor

from traiNNer.losses.gan_loss import GANLoss


class GanHealthSample:
    """Detached scalar measurements for the boundary microbatch."""

    def __init__(self) -> None:
        self.values: dict[str, Tensor] = {}
        self.gradient_multiplier = 1.0

    def backward(
        self, scaled_loss: Tensor, image: Tensor, gan_view: Tensor, *, compare_other: bool
    ) -> None:
        if not image.requires_grad:
            scaled_loss.backward()
            return

        adversarial: Tensor | None = None

        def measure_gan(grad: Tensor) -> None:
            nonlocal adversarial
            g = grad.detach().float()
            self.values["grad_sr_gan"] = torch.linalg.vector_norm(g)
            if compare_other:
                # Autograd can recycle the incoming gradient while summing branches.
                adversarial = g.clone() if grad.dtype == torch.float32 else g

        def measure_total(grad: Tensor) -> None:
            nonlocal adversarial
            if adversarial is not None:
                self.values["grad_sr_other"] = torch.linalg.vector_norm(
                    grad.detach().float() - adversarial
                )
                adversarial = None

        # gan_view feeds only the adversarial branch. Its hook necessarily fires
        # before the hook on the shared image, including through the wavelet view.
        with gan_view.register_hook(measure_gan), image.register_hook(measure_total):
            scaled_loss.backward()

    @torch.no_grad()
    def discriminator(self, loss: GANLoss, real: Tensor, fake: Tensor) -> None:
        # Positive signed distance means the patch is on its intended side.
        threshold, inverted = 0.0, False
        if loss.gan_type in {"vanilla", "lsgan"}:
            hi, lo = loss.real_label_val, loss.fake_label_val
            if hi == lo or not (math.isfinite(hi) and math.isfinite(lo)):
                return
            inverted = hi < lo
            if loss.gan_type == "lsgan":
                threshold = (lo + hi) * 0.5
            elif not (0 <= min(lo, hi) < 0.5 < max(lo, hi) <= 1):
                return
        elif loss.gan_type not in {"hinge", "wgan_softplus"}:
            return  # A Wasserstein critic has no absolute decision boundary.

        margins = torch.cat(
            ((real.float() - threshold).reshape(-1), (threshold - fake.float()).reshape(-1))
        )
        correct = margins < 0 if inverted else margins > 0
        self.values["out_d_acc"] = correct.float().mean()
        self.values["out_d_abs"] = margins.abs().mean()

    def logs(self) -> dict[str, float]:
        if not self.values:
            return {}
        # One host read at the existing print boundary. Use the scale of this
        # backward, even if the scaler grew on the accepted optimizer step.
        scalars = torch.stack(list(self.values.values())).tolist()
        logs = dict(zip(self.values, scalars, strict=True))
        for name in ("grad_sr_gan", "grad_sr_other"):
            if name in logs:
                logs[name] *= self.gradient_multiplier
        other = logs.get("grad_sr_other", 0.0)
        if other > 0 and math.isfinite(other) and math.isfinite(logs["grad_sr_gan"]):
            logs["grad_sr_ratio"] = logs["grad_sr_gan"] / other
        return logs

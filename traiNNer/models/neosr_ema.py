"""Image-training EMA from NeoSR 3fe738f4, using the trainer's EMA checkpoint interface."""

import torch
from ema_pytorch import EMA
from torch import nn


class NeoSREMA(EMA):
    """Count applied G updates in the existing step buffer. Step one is a full copy."""

    def __init__(self, model: nn.Module, decay: float, power: float | None) -> None:
        super().__init__(model, beta=decay, include_online_model=False)
        self.power = power
        self.ema_model.eval().requires_grad_(False)
        self._pairs = None

    @torch.no_grad()
    def update(self) -> None:
        # step lives on CPU, including on resume. It is authoritative even for a
        # state saved before the first G step (e.g. discriminator-only training).
        previous_steps = self.step.item()
        if previous_steps == 0:
            self.ema_model.load_state_dict(self.model.state_dict())
            self.initted.fill_(True)
        if self._pairs is None or previous_steps == 0:
            self._pairs = (
                list(self.ema_model.parameters()),
                list(self.model.parameters()),
                list(self.ema_model.buffers()),
                list(self.model.buffers()),
            )
        if previous_steps:
            dst_params, src_params, dst_buffers, src_buffers = self._pairs
            decay = self.beta
            if self.power is not None:
                decay = min(decay, 1.0 - (1.0 + previous_steps) ** -self.power)
            torch._foreach_lerp_(dst_params, src_params, 1.0 - decay)
            if dst_buffers:
                torch._foreach_copy_(dst_buffers, src_buffers)
        self.step.add_(1)

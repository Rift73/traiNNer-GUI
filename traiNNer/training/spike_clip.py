"""Per-optimizer spike clipping as in NeoSR 3fe738f4, phase.Phase.step."""

from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import get_total_norm
from torch.optim import Optimizer


class SpikeClip:
    """Update the threshold and logging window only after the optimizer applied a step."""

    def __init__(self, optimizer: Optimizer) -> None:
        device = next(
            p.device for group in optimizer.param_groups for p in group["params"]
        )
        self.norm_ema: Tensor | None = None
        # Sum and observation count for the raw norm and the clipped indicator.
        self.window = torch.zeros(4, dtype=torch.float32, device=device)
        optimizer.register_state_dict_post_hook(self._save)
        optimizer.register_load_state_dict_pre_hook(self._restore)

    def _save(self, optimizer: Optimizer, state: dict[str, Any]) -> None:
        state["neosr_spike_norm_ema"] = self.norm_ema

    def _restore(self, optimizer: Optimizer, state: dict[str, Any]) -> None:
        ema = state.get("neosr_spike_norm_ema")
        if ema is not None and (
            not isinstance(ema, Tensor)
            or ema.ndim != 0
            or not ema.is_floating_point()
            or not torch.isfinite(ema).item()
            or ema.item() < 0
        ):
            raise ValueError("Invalid saved NeoSR spike gradient-norm EMA.")
        self.norm_ema = None if ema is None else ema.detach().to(self.window.device)
        self.window.zero_()

    @torch.no_grad()
    def prepare(self, optimizer: Optimizer) -> tuple[Tensor, Tensor | None]:
        # The scaler has unscaled exactly these parameter groups. Other model
        # parameters may have gradients but are outside this optimizer's update.
        grads = [
            p.grad
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        norm = (
            get_total_norm(grads, foreach=True) if grads else self.window.new_zeros(())
        )
        if self.norm_ema is None:
            return norm, None
        threshold = torch.where(self.norm_ema > 0, self.norm_ema * 3.0, torch.inf)
        if grads:
            torch._foreach_mul_(grads, (threshold / (norm + 1e-6)).clamp_(max=1.0))
        return norm, threshold

    @torch.no_grad()
    def commit(self, norm: Tensor, threshold: Tensor | None) -> None:
        capped = norm if threshold is None else torch.minimum(norm, threshold)
        previous = self.norm_ema
        self.norm_ema = (
            capped.clone()
            if previous is None
            else torch.where(previous > 0, previous.lerp(capped, 1.0 - 0.98), capped)
        )
        self.window[0].add_(norm.float())
        self.window[1].add_(1)
        if threshold is not None:
            self.window[2].add_((norm > threshold).float())
            self.window[3].add_(1)

    def logs(self, role: str, distributed: bool) -> dict[str, float]:
        window = self.window
        if distributed:
            window = window.clone()
            torch.distributed.all_reduce(window)
        norm_sum, norm_count, clipped_sum, clipped_count = window.tolist()
        values = {}
        if norm_count:
            values[f"grad_norm_{role}"] = norm_sum / norm_count
        if clipped_count:
            values[f"clipped_{role}"] = clipped_sum / clipped_count
        return values

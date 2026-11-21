"""Temporary optimizer views and exact training weights for schedule-free replay."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def native_schedule_free(optimizer: Optimizer) -> bool:
    from traiNNer.optimizers.adanschedulefree_optim import AdanScheduleFree
    from traiNNer.optimizers.adamwschedulefree_optim import AdamWScheduleFree
    from traiNNer.optimizers.neosr_adan_sf_optim import NeoSRAdanScheduleFree
    return type(optimizer) in (AdanScheduleFree, AdamWScheduleFree, NeoSRAdanScheduleFree)


@contextmanager
def preserve_module_modes(network: nn.Module) -> Iterator[None]:
    """Restore training hooks and independently frozen children after evaluation."""
    modes = [(module, module.training) for module in network.modules()]
    try:
        yield
    finally:
        # Parents precede children. A parent's train() can change descendant
        # flags, so check each saved child after restoring its parent.
        for module, training in modes:
            if module.training != training:
                module.train(training)


@contextmanager
def evaluation_view(optimizer: Optimizer | None) -> Iterator[None]:
    if optimizer is None or not callable(getattr(optimizer, "eval", None)):
        yield
        return
    if not native_schedule_free(optimizer):
        optimizer.eval()
        try:
            yield
        finally:
            optimizer.train()
        return

    # The inverse lerp can differ from the original training value by ULPs.
    # Snapshot just the parameters this optimizer will actually interpolate.
    modes = [group["train_mode"] for group in optimizer.param_groups]
    params = [p for group in optimizer.param_groups if group["train_mode"] and group.get("schedule_free", True)
              for p in group["params"] if "z" in optimizer.state.get(p, {})]
    originals = [p.detach().clone(memory_format=torch.preserve_format) for p in params]
    try:
        optimizer.eval()
        yield
    finally:
        with torch.no_grad():
            if params:
                torch._foreach_copy_(params, originals)
        for group, mode in zip(optimizer.param_groups, modes, strict=True):
            group["train_mode"] = mode


def checkpoint_state(optimizer: Optimizer, *, exact: bool) -> tuple[dict, dict[int, Tensor]]:
    """Pair the portable eval-weight file with its optimizer state, without a live swap."""
    state = optimizer.state_dict()
    originals: dict[int, Tensor] = {}
    if callable(getattr(optimizer, "eval", None)) and native_schedule_free(optimizer):
        for live, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
            if not live.get("schedule_free", True):
                continue
            if exact and live["train_mode"]:
                for p, key in zip(live["params"], saved["params"], strict=True):
                    if "z" in optimizer.state.get(p, {}):
                        # Serialization is synchronous at an accepted update boundary.
                        originals[key] = p.detach()
            saved["train_mode"] = False
    return state, originals


def checked_replay_weights(optimizer: Optimizer, state: dict, originals: dict[int, Tensor]) -> list[tuple[Tensor, Tensor]]:
    if not isinstance(originals, dict):
        raise ValueError("Invalid schedule-free replay weights.")
    if not originals:
        return []
    if not native_schedule_free(optimizer):
        raise ValueError("Schedule-free replay weights require their original optimizer.")
    parameters = {key: p for group, saved in zip(optimizer.param_groups, state["param_groups"], strict=True)
                  for p, key in zip(group["params"], saved["params"], strict=True)}
    pairs = []
    for key, value in originals.items():
        p = parameters.get(key)
        if (p is None or not isinstance(value, Tensor) or value.shape != p.shape or value.dtype != p.dtype
                or "z" not in state["state"].get(key, {})):
            raise ValueError("Schedule-free replay tensor does not match its optimizer parameter.")
        pairs.append((p, value))
    return pairs

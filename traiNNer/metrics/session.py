"""Reuse pretrained metric networks for the lifetime of a validation pass."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch


_models: ContextVar[tuple[torch.device, dict] | None] = ContextVar("evaluation_metric_models", default=None)


def _metric_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


@contextmanager
def metric_session(device: torch.device | str, *, cache_accelerator: bool = False):
    device = torch.device(device)
    if device.type != "cpu" and not cache_accelerator:
        yield
        return
    device = _metric_device(device)
    active = _models.get()
    if active is not None and active[0] == device:
        yield
        return
    models = {}
    token = _models.set((device, models))
    try:
        yield
    finally:
        _models.reset(token)
        models.clear()


def cpu_metric_session():
    return metric_session("cpu")


def pretrained_metric(key: tuple, device: torch.device, factory):
    active = _models.get()
    if active is None or active[0] != _metric_device(device):
        return factory().to(device).eval()
    models = active[1]
    if key not in models:
        models[key] = factory().to(device).eval()
    return models[key]

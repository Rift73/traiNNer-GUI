"""Immutable recipe choices recorded with a full training state."""
from __future__ import annotations

from copy import deepcopy
from sys import float_info
from typing import TYPE_CHECKING

from .tasks import task_name

if TYPE_CHECKING:
    from .options import ReduxOptions


def recipe_learning_rate(value: object) -> float | None:
    """Read a finite, nonnegative numeric LR from a recipe or API request."""
    if type(value) is not int and type(value) is not float:
        return None
    return float(value) if 0 <= value <= float_info.max else None


def resume_recipe(opt: ReduxOptions) -> dict:
    """Retain initial options before model factories can consume their keyword dictionaries."""
    networks = {role: deepcopy(config) for role in ("g", "ae", "d")
                if (config := getattr(opt, "network_" + role)) is not None}
    optimizers = {role: deepcopy(config) for role in networks
                  if opt.train and (config := getattr(opt.train, "optim_" + role)) is not None}
    recipe = {"version": 1, "task": task_name(opt), "scale": opt.scale,
              "networks": networks, "optimizers": optimizers}
    if opt.train and opt.train.ema_mode == "neosr":
        recipe["neosr_ema"] = {"decay": opt.train.ema_decay, "power": opt.train.ema_power}
    if opt.train and (qat := opt.train.qat):
        calib, norm, obs = qat.schedule(opt.train.total_iter)
        recipe["qat"] = {"type": qat.type, "calibration_iters": calib, "freeze_norm_iter": norm,
                         "freeze_observers_iter": obs, "quantize_residual": (
                             qat.quantize_residual if qat.quantize_residual is not None
                             else networks.get("g", {}).get("quantize_residual")
                         )}
    return recipe

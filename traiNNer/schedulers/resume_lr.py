"""Rebase a saved learning-rate curve without restarting its optimizer or clocks."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from torch.optim import lr_scheduler
from traiNNer.utils import get_root_logger

if TYPE_CHECKING:
    from traiNNer.models.base_model import BaseModel

_GROUP_RATES = ("lr", "initial_lr", "max_lr", "min_lr")
_SCHEDULE_RATES = (
    "base_lrs",
    "_last_lr",
    "max_lrs",
    "min_lrs",
    "default_min_lr",
    "eta_min",
    "peak_lr",
)


def _rates(
    saved: dict, keys: tuple[str, ...], factor: float | None, fresh: dict
) -> dict:
    result = saved.copy()
    for key in keys:
        value = saved.get(key)
        if value is None:
            continue
        if factor is None:
            if key in fresh:
                result[key] = fresh[key]
        elif isinstance(value, list):
            result[key] = [rate * factor for rate in value]
        else:
            result[key] = value * factor
    return result


def _knee_defaults(options: dict, declared: dict, old: float, new: float) -> dict:
    # Per-role KneeLR records an inferred peak in the composite receipt.
    # Adjust only inferred peaks; keep the check on every explicit child setting.
    if "schedulers" in options and "schedulers" in declared:
        if len(options["schedulers"]) == len(declared["schedulers"]):
            return {
                **options,
                "schedulers": [
                    _knee_defaults(child, spec, old, new)
                    for child, spec in zip(
                        options["schedulers"], declared["schedulers"], strict=True
                    )
                ],
            }
    if (
        str(options.get("type", "")).lower() == "kneelr"
        and "peak_lr" not in declared
        and options.get("peak_lr") == old
    ):
        return {**options, "peak_lr": new}
    return options


def _schedule_rates(
    saved: dict,
    factor: float | None,
    fresh: dict,
    declared: dict,
    old: float,
    new: float,
) -> dict:
    keys = _SCHEDULE_RATES
    if (
        "min_lrs" in saved and "num_bad_epochs" in saved
    ):  # Plateau's eps is in LR units.
        keys += ("eps",)
    result = _rates(saved, keys, factor, fresh)
    if "_schedulers" in saved:
        receipt = saved.get("recipe")
        if receipt and receipt != fresh.get("recipe"):
            result["recipe"] = {
                **receipt,
                "options": _knee_defaults(receipt["options"], declared, old, new),
            }
        children = declared.get("schedulers", [])
        if len(children) == len(saved["_schedulers"]) == len(fresh["_schedulers"]):
            result["_schedulers"] = [
                _schedule_rates(child, factor, template, spec, old, new)
                for child, template, spec in zip(
                    saved["_schedulers"], fresh["_schedulers"], children, strict=True
                )
            ]
    return result


def _clock(saved: dict) -> int:
    if "last_epoch" in saved:
        return saved["last_epoch"]
    # ChainedScheduler advances every child once per update.
    return _clock(saved["_schedulers"][0])


def _position_fresh_curve(schedule: lr_scheduler.LRScheduler, saved: dict) -> None:
    """A zero base has no scale factor. Use the new recipe at the saved phase."""
    if isinstance(schedule, lr_scheduler.ReduceLROnPlateau):
        # Past metric reductions at zero cannot identify a nonzero effective LR.
        # Adopt the new rate and let the saved patience and best score continue.
        return
    updates = _clock(saved)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The epoch parameter in .*")
        warnings.filterwarnings(
            "ignore", message="Detected call of .*lr_scheduler.step.*"
        )
        if isinstance(
            schedule, (lr_scheduler.SequentialLR, lr_scheduler.ChainedScheduler)
        ):
            # Composites lack random access. Only the zero-to-positive edit needs replay.
            for _ in range(updates):
                schedule.step()
        else:
            schedule.step(updates)


def rebase_resume_rates(
    model: BaseModel, checkpoint: dict
) -> tuple[list[dict], list[dict]]:
    """Copy only changed checkpoint metadata. Tensor moments stay shared and untouched."""
    optimizers = list(checkpoint["optimizers"])
    schedules = list(checkpoint["schedulers"])
    recorded = checkpoint.get("resume_recipe", {}).get("optimizers", {})
    train = model.opt.train
    assert train is not None
    previous = train.resume_previous_lrs or {}
    if any(
        k not in ("g", "d", "ae") or not math.isfinite(v) or v < 0
        for k, v in previous.items()
    ):
        raise ValueError(
            "train.resume_previous_lrs needs finite nonnegative rates keyed by g, d or ae."
        )
    for role in ("g", "d", "ae"):
        optimizer = getattr(model, "optimizer_" + role, None)
        if (
            optimizer is None
            or optimizer not in model.optimizers
            or optimizer.defaults.get("lr") is None
        ):
            continue
        index = model.optimizers.index(optimizer)
        state = optimizers[index]
        groups = state["param_groups"]
        new = float(optimizer.defaults["lr"])
        attached = next((s for s in model.schedulers if s.optimizer is optimizer), None)
        owns_bounds = isinstance(
            attached,
            (
                lr_scheduler.OneCycleLR,
                lr_scheduler.CyclicLR,
                lr_scheduler.SequentialLR,
                lr_scheduler.ChainedScheduler,
            ),
        )
        inferred = None
        if groups and not owns_bounds:
            inferred = groups[0].get("initial_lr")
            if inferred is None and attached is None:
                inferred = groups[0]["lr"]
        old = state.get("recipe_lr", previous.get(role))
        if old is None:
            declared = recorded.get(role, {}).get("lr")
            if declared == new:  # Unchanged legacy recipe keeps its exact checkpoint.
                continue
            if declared is None and "lr" not in (
                getattr(train, "optim_" + role, None) or {}
            ):
                continue
            old = inferred if inferred is not None else declared
            if old is None:
                get_root_logger().warning(
                    "Legacy state has no optimizer %s base LR. Keeping the saved curve. "
                    "To rebase it, set train.resume_previous_lrs.%s to the old recipe LR.",
                    role,
                    role,
                )
                continue
        old = float(old)
        if old == new:
            continue
        factor = new / old if old else None
        for si, schedule in enumerate(model.schedulers):
            if schedule.optimizer is not optimizer or train.restart_scheduler:
                continue
            saved = schedules[si]
            if factor is None:
                _position_fresh_curve(schedule, saved)
            fresh = schedule.state_dict()
            declared = (train.schedulers or {}).get(role) or train.scheduler or {}
            schedules[si] = _schedule_rates(saved, factor, fresh, declared, old, new)
        optimizers[index] = {
            **state,
            "param_groups": [
                _rates(group, _GROUP_RATES, factor, template)
                for group, template in zip(groups, optimizer.param_groups, strict=True)
            ],
        }
        get_root_logger().info(
            "Resuming optimizer %s with base LR %g -> %g. Next group LR(s): %s",
            role,
            old,
            new,
            [g["lr"] for g in optimizers[index]["param_groups"]],
        )
    return optimizers, schedules

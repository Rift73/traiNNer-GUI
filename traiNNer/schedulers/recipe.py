"""Construct recipe-selected PyTorch schedulers on the owning optimizer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pytorch_optimizer import get_wsd_schedule
from torch.optim import Optimizer, lr_scheduler

from trainner_config.schedules import COMPOSITES, SchedulerConfigError, scheduler_name

from .cosineannealingrestartlr_scheduler import CosineAnnealingRestartLR
from .cosineannealingwarmuplr_scheduler import CosineAnnealingWarmupLR
from .kneelr_scheduler import KneeLR

if TYPE_CHECKING:
    from trainner_config.options import TrainOptions
    from traiNNer.utils.types import TrainingState


class _CompositeRecipe(lr_scheduler.LRScheduler):
    # Native composite state stores children by position, without their class
    # names. Some leaves also rebuild callables from configuration on restore.
    recipe: dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("recipe") != self.recipe:
            raise ValueError("Cannot resume this composite LR schedule. Restore its original optimizer role, "
                             "ordered child types, options and milestones, or start a new schedule from weights.")
        children = state_dict.get("_schedulers")
        if not isinstance(children, list) or len(children) != len(self.recipe["options"]["schedulers"]):
            raise ValueError("The checkpoint is missing state for one or more composite LR scheduler children.")
        super().load_state_dict(state_dict)


class _SequentialLR(_CompositeRecipe, lr_scheduler.SequentialLR):
    pass


class _ChainedScheduler(_CompositeRecipe, lr_scheduler.ChainedScheduler):
    pass


_CUSTOM = {
    "CosineAnnealingRestartLR": CosineAnnealingRestartLR,
    "CosineAnnealingWarmupLR": CosineAnnealingWarmupLR,
    "KneeLR": KneeLR,
    "WarmupStableDecayLR": get_wsd_schedule,
}


def scheduler_constructor(name: str):
    """Resolve the installed constructor for a canonical scheduler name."""
    return _CUSTOM[name] if name in _CUSTOM else getattr(lr_scheduler, name)


def build_scheduler(
    optimizer: Optimizer, spec: dict, train: TrainOptions, *, target: str | int, path: str,
) -> tuple[lr_scheduler.LRScheduler, dict]:
    """Construct a validated spec and return serializable constructor options for the receipt."""
    # Earlier children can change param_groups during construction. Each Knee
    # leaf must inherit the original target LR, as a standalone scheduler does.
    initial_lr = optimizer.param_groups[0]["lr"]

    def build(node: dict, location: str) -> tuple[lr_scheduler.LRScheduler, dict]:
        name = scheduler_name(node["type"], location + ".type")
        opts = {key: value for key, value in node.items() if key not in ("type", "monitor")}
        if name in COMPOSITES:
            schedules = []
            child_specs = []
            for i, child in enumerate(opts["schedulers"]):
                child_location = f"{location}.schedulers[{i}]"
                schedule, child_opts = build(child, child_location)
                schedules.append(schedule)
                child_specs.append({"type": scheduler_name(child["type"], child_location + ".type"), **child_opts})
            opts["schedulers"] = child_specs
            if name == "SequentialLR":
                composite = _SequentialLR(optimizer, schedulers=schedules, milestones=opts["milestones"],
                                          last_epoch=opts.get("last_epoch", -1))
            else:
                composite = _ChainedScheduler(schedules, optimizer=optimizer)
            composite.recipe = {"target": target, "type": name, "options": opts}
            return composite, opts

        constructor = scheduler_constructor(name)
        kwargs = opts
        if name == "KneeLR":
            if train.schedulers is not None:
                opts.setdefault("peak_lr", initial_lr)
            shared_peak = train.optim_g["lr"] if train.optim_g else 0.001
            kwargs = {"total_steps": train.total_iter, "peak_lr": shared_peak, **opts}
        try:
            return constructor(optimizer, **kwargs), opts
        except (TypeError, ValueError, KeyError, AssertionError, ArithmeticError) as error:
            raise SchedulerConfigError(location, f"{location} ({name}): {error}") from error

    return build(spec, path)


def restart_schedulers(
    schedules: list[lr_scheduler.LRScheduler],
    specs: list[dict],
    configured_groups: list[list[dict]],
    state: TrainingState,
) -> None:
    """Position fresh recipe schedules after loading the checkpoint optimizer groups."""
    # D can skip updates. Use the previous update clock for the same target
    # when recorded. A new target or a legacy state has only the trainer clock.
    previous = {
        spec["target"]: (spec, saved)
        for spec, saved in zip(state.get("scheduler_specs", []), state["schedulers"])
        if spec.get("monitor") is None and "last_epoch" in saved
    }
    for schedule, spec, initial in zip(schedules, specs, configured_groups, strict=True):
        groups = schedule.optimizer.param_groups
        if isinstance(schedule, lr_scheduler.ReduceLROnPlateau):
            # Its clock counts validation observations. A restart has seen none.
            schedule._last_lr = [group["lr"] for group in groups]
            continue

        old_spec, old_state = previous.get(spec["target"], ({}, {}))
        completed = old_state.get("last_epoch", state["iter"])
        if (hasattr(schedule, "_get_closed_form_lr") and spec == old_spec
                and schedule.base_lrs == old_state.get("base_lrs")):
            # These schedules step relative to the current optimizer LR. Trainer
            # warmup can overwrite a milestone, and re-evaluating a closed form
            # would also lose the rounding of repeated updates. Keep the actual LR.
            schedule.last_epoch = completed
            schedule._step_count = completed + 1
            schedule._last_lr = old_state.get("_last_lr", [g["lr"] for g in groups])
            continue
        for group, declared in zip(groups, initial, strict=True):
            # Optimizer.load_state_dict replaces these recipe-owned values.
            # OneCycleLR in particular reads its endpoints from the groups.
            for key in ("initial_lr", "max_lr", "min_lr", "base_momentum", "max_momentum"):
                if key in declared:
                    group[key] = declared[key]

        # An explicit epoch looks up the LR (and cycle phase) directly. It
        # does not apply every historical milestone to the restored LR again.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The epoch parameter in .*")
            warnings.filterwarnings("ignore", message="Detected call of .*lr_scheduler.step.*before .*optimizer.step.*")
            schedule.step(completed)
        if type(schedule).step is lr_scheduler.LRScheduler.step:
            schedule._step_count = completed + 1  # includes construction's initial step

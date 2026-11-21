"""Validate the YAML-to-scheduler boundary without constructing Torch optimizers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import ReduxOptions


class SchedulerConfigError(ValueError):
    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path


# Child constructor arguments that can be expressed as data. The interactive
# preflight uses these signatures without importing Torch or creating an optimizer.
_CHILD_OPTIONS = {
    "StepLR": (("step_size",), ("gamma",)),
    "MultiStepLR": (("milestones",), ("gamma",)),
    "ConstantLR": ((), ("factor", "total_iters")),
    "LinearLR": ((), ("start_factor", "end_factor", "total_iters")),
    "ExponentialLR": (("gamma",), ()),
    "PolynomialLR": ((), ("total_iters", "power")),
    "CosineAnnealingLR": (("T_max",), ("eta_min",)),
    "CosineAnnealingWarmRestarts": (("T_0",), ("T_mult", "eta_min")),
    "CyclicLR": (("base_lr", "max_lr"), ("step_size_up", "step_size_down", "mode", "gamma", "scale_fn",
                                       "scale_mode", "cycle_momentum", "base_momentum", "max_momentum")),
    "CosineAnnealingWarmupLR": (("total_iters",), ("warmup_iters", "eta_min_ratio")),
    "CosineAnnealingRestartLR": (("periods",), ("restart_weights", "eta_min")),
    "KneeLR": ((), ("total_steps", "peak_lr", "explore_ratio")),
    "WarmupStableDecayLR": (("num_warmup_steps", "num_stable_steps", "num_decay_steps"),
                            ("min_lr_ratio", "num_cycles", "cooldown_type")),
}
COMPOSITES = ("SequentialLR", "ChainedScheduler")
_NAMES = {name.casefold(): name for name in (*_CHILD_OPTIONS, *COMPOSITES, "OneCycleLR", "ReduceLROnPlateau")}


def scheduler_name(name: object, path: str) -> str:
    if not isinstance(name, str) or not name:
        raise SchedulerConfigError(path, "An LR scheduler type is required.")
    if name.casefold() in {"lambdalr", "multiplicativelr"}:
        raise SchedulerConfigError(path, f"{name} requires a Python callable. Choose a named recipe scheduler.")
    if name.casefold() not in _NAMES:
        raise SchedulerConfigError(path, f"Unknown recipe LR scheduler {name!r}.")
    return _NAMES[name.casefold()]


def _allowed_parents(name: str) -> tuple[str, ...]:
    if name in ("ReduceLROnPlateau", "OneCycleLR"):
        return ()
    # SequentialLR activates a later child using its scalar LR update and
    # rewinds leaf clocks at initialization. Nested composites lack that API.
    if name in COMPOSITES:
        return ("ChainedScheduler",)
    return COMPOSITES


def scheduler_catalog() -> list[dict]:
    """Recipe schedulers, all usable at the root. Parent IDs describe child placement."""
    kinds = {"SequentialLR": "sequential", "ChainedScheduler": "chained", "ReduceLROnPlateau": "metric"}
    return [{"id": name, "kind": kinds.get(name, "step"), "allowedParents": list(_allowed_parents(name))}
            for name in sorted(_NAMES.values())]


def _validate_spec(opt: ReduxOptions, spec: dict, path: str, parent: str | None = None) -> None:
    if not isinstance(spec, dict):
        raise SchedulerConfigError(path, "Each scheduler needs a mapping with its own type and options.")
    name = scheduler_name(spec.get("type"), path + ".type")
    if parent and parent not in _allowed_parents(name):
        if name == "ReduceLROnPlateau":
            message = "ReduceLROnPlateau needs a validation metric. Place it directly on an optimizer."
        elif name == "OneCycleLR":
            message = "OneCycleLR controls the complete optimizer cycle. Place it directly on an optimizer."
        else:
            message = f"{parent} needs individual step schedulers for its phases. {name} cannot be a phase."
        raise SchedulerConfigError(path + ".type", message)
    for field in ("optimizer", "role", "target"):
        if field in spec:
            raise SchedulerConfigError(path + "." + field, "Select the optimizer at train.schedulers.g, .d or .ae.")
    if name == "CyclicLR" and spec.get("scale_fn") is not None:
        raise SchedulerConfigError(path + ".scale_fn", "scale_fn requires a Python callable. Choose a built-in CyclicLR mode.")
    monitor = spec.get("monitor")
    if name == "ReduceLROnPlateau":
        if not isinstance(monitor, dict) or set(monitor) != {"dataset", "metric"}:
            raise SchedulerConfigError(path + ".monitor", "Select a validation dataset name and metric for ReduceLROnPlateau.")
        mode = spec.get("mode")
        if mode not in ("min", "max"):
            raise SchedulerConfigError(path + ".mode", "Specify min or max for this metric explicitly.")
        _validate_monitor(opt, path, monitor["dataset"], monitor["metric"], mode)
    elif monitor is not None:
        raise SchedulerConfigError(path + ".monitor", "Only ReduceLROnPlateau observes a validation metric.")
    if (parent or name in COMPOSITES) and spec.get("last_epoch", -1) != -1:
        raise SchedulerConfigError(path + ".last_epoch", "Composite child clocks start together. Use a full state to resume.")
    if name not in COMPOSITES:
        if "schedulers" in spec:
            raise SchedulerConfigError(path + ".schedulers", "Only SequentialLR and ChainedScheduler accept children.")
        if parent:
            required, optional = _CHILD_OPTIONS[name]
            for key in required:
                if key not in spec:
                    raise SchedulerConfigError(f"{path}.{key}", f"{name} needs {key}.")
            for key in spec.keys() - {"type", "last_epoch", *required, *optional}:
                raise SchedulerConfigError(f"{path}.{key}", f"{name} does not accept {key!r}.")
            if name in ("StepLR", "CosineAnnealingLR"):
                key = "step_size" if name == "StepLR" else "T_max"
                if type(spec[key]) is not int or spec[key] < 1:
                    raise SchedulerConfigError(f"{path}.{key}", "Use a positive number of optimizer updates.")
        return
    if opt.train and opt.train.restart_scheduler and opt.path.resume_state:
        raise SchedulerConfigError(path, "Clear restart_scheduler to restore the saved composite and its child clocks, "
                                   "or start from weights for a new schedule.")
    allowed = {"type", "schedulers"}
    if name == "SequentialLR":
        allowed.update(("milestones", "last_epoch"))
    for field in spec.keys() - allowed:
        raise SchedulerConfigError(f"{path}.{field}", f"{name} does not accept {field!r} here.")
    children = spec.get("schedulers")
    if not isinstance(children, list) or not children:
        raise SchedulerConfigError(path + ".schedulers", f"{name} needs a nonempty ordered list of scheduler mappings.")
    if name == "SequentialLR":
        milestones = spec.get("milestones")
        if (not isinstance(milestones, list) or len(milestones) != len(children) - 1
                or any(type(step) is not int or step <= 0 for step in milestones)
                or any(a >= b for a, b in zip(milestones, milestones[1:]))):
            raise SchedulerConfigError(path + ".milestones", "Provide one strictly increasing positive update per phase transition.")
    for index, child in enumerate(children):
        _validate_spec(opt, child, f"{path}.schedulers[{index}]", name)


def validate_schedulers(opt: ReduxOptions) -> None:
    train = opt.train
    if not train:
        return
    if train.schedulers is not None and train.scheduler is not None:
        raise SchedulerConfigError("train.schedulers","Choose either the shared schedule or per-optimizer schedules.")
    specs = []
    if train.scheduler is not None:
        specs.append(("train.scheduler",train.scheduler))
    elif train.schedulers is not None:
        for role, schedule in train.schedulers.items():
            path = "train.schedulers." + role
            if role not in ("g","d","ae"):
                raise SchedulerConfigError(path,"Choose the optimizer g, d or ae.")
            if getattr(train,"optim_"+role) is None:
                raise SchedulerConfigError(path,f"The recipe has no optim_{role} configuration.")
            specs.append((path,schedule))
    for path, spec in specs:
        _validate_spec(opt, spec, path)

    early = train.early_stopping
    if early is not None:
        if not math.isfinite(early.min_delta):
            raise SchedulerConfigError("train.early_stopping.min_delta", "Use a finite absolute improvement.")
        _validate_monitor(opt, "train.early_stopping", early.monitor.dataset, early.monitor.metric, early.mode)


def _validate_monitor(opt: ReduxOptions, path: str, dataset: str, metric: str, mode: str) -> None:
    from .metrics import configured_snapshot_metrics, validation_snapshot_metrics
    from .tasks import require_task_action, task_name

    if not all(isinstance(x,str) and x.strip() for x in (dataset,metric)):
        raise SchedulerConfigError(path+".monitor","The monitored dataset and metric need names.")
    val = opt.val
    if not val or not val.val_enabled or not val.metrics_enabled or not val.val_freq or val.val_freq<1:
        raise SchedulerConfigError(path+".monitor","Enable periodic validation and metrics for this monitor.")
    matches = [ds for phase,ds in opt.datasets.items() if phase.split("_",1)[0]=="val" and ds.name==dataset]
    if len(matches)!=1:
        raise SchedulerConfigError(path+".monitor.dataset","Use the name of exactly one configured validation dataset.")

    configured = val.metrics or {}
    definition = configured.get(metric, {})
    direction = definition.get("better") if isinstance(definition, dict) else None
    try:
        declaration = require_task_action(opt, "validate").snapshot_metrics
        # A provider can generate scores from typed settings even with an empty val.metrics.
        declared = (
            validation_snapshot_metrics(opt)
            if declaration is not None and declaration is not configured_snapshot_metrics
            else None
        )
    except ValueError as exc:
        raise SchedulerConfigError(path + ".monitor.metric", str(exc)) from exc

    if declared is not None:
        if metric in declared:
            direction = declared[metric]
        elif metric not in configured and not (opt.task == "object_detection" and metric == "targets"):
            # Detection also publishes its target count as an unranked monitor.
            raise SchedulerConfigError(path + ".monitor.metric", f"{metric!r} is not a validation metric for this recipe.")
    elif (declaration is configured_snapshot_metrics or configured) and metric not in configured:
        raise SchedulerConfigError(path + ".monitor.metric", f"Define {metric!r} in val.metrics.")

    if declaration is configured_snapshot_metrics and (
        not isinstance(definition, dict) or not isinstance(definition.get("type"), str)
        or not definition["type"].strip()
    ):
        raise SchedulerConfigError("val.metrics." + metric, "A monitored validation metric needs a metric type.")
    if direction is not None:
        if direction not in ("higher", "lower"):
            raise SchedulerConfigError("val.metrics." + metric + ".better", "Use higher or lower for the metric direction.")
        expected = "min" if direction == "lower" else "max"
        if mode != expected:
            raise SchedulerConfigError(path + ".mode", f"{metric!r} declares {direction} is better. Set mode to {expected!r}.")
    # Single-input restoration can measure a no-reference score such as
    # TOPIQ-NR. A reference metric on this specific split could never
    # drive its scheduler, even when another validation split is paired.
    if (matches[0].type.casefold() in ("singleimagedataset", "singlevideodataset")
            and task_name(opt) in ("image_restoration", "recurrent_restoration")
            and isinstance(definition, dict) and str(definition.get("type", "")).casefold() in {
                "calculate_psnr", "calculate_ssim", "calculate_lpips", "calculate_dists",
                "calculate_topiq", "calculate_neosr_psnr", "calculate_neosr_ssim", "calculate_neosr_topiq",
            }):
        raise SchedulerConfigError(path+".monitor.metric",
                                   f"{metric!r} requires a reference image on dataset {dataset!r}.")

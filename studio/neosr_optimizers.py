"""Translate NeoSR optimizer clocks and LR schedules to trainer recipe fields."""

from __future__ import annotations


def _optimizer(spec):
    kind = str(spec.get("type", "")).lower()
    args = {key: value for key, value in spec.items() if key != "type"}
    schedule_free = kind in ("adamw_sf", "adan_sf") and args.get("schedule_free", True)
    warmup = 0
    if kind in ("adam", "adamw", "nadam", "adan"):
        warmup = args.pop("warmup_steps", 0)
        if type(warmup) is not int or warmup < 0:
            raise ValueError("warmup_steps must be a nonnegative integer.")
    shared = {"lr", "betas", "eps", "weight_decay"}
    if kind in ("adam", "adamw", "nadam"):
        extra = {"momentum_decay", "decoupled_weight_decay"} if kind == "nadam" else {"amsgrad", "fused"}
        if args.keys() <= shared | extra | {"foreach", "capturable", "maximize", "differentiable"}:
            return {"type": {"adam": "Adam", "adamw": "AdamW", "nadam": "NAdam"}[kind], **args}, warmup, False
    elif kind == "adan":
        if (args.keys() <= shared | {"max_grad_norm", "no_prox", "foreach"}
                and args.pop("no_prox", True) is True and args.pop("foreach", True) is True):
            return {"type": "Adan", "lr": 5e-4, "weight_decay": 0.02, **args}, warmup, False
    elif kind in ("adan_sf", "adamw_sf"):
        allowed = shared | {"warmup_steps", "r", "weight_lr_power", "schedule_free"}
        if kind == "adan_sf" and args.keys() <= allowed | {"max_grad_norm"}:
            return {"type": "NeoSRAdanScheduleFree", **args}, 0, schedule_free
        if (kind == "adamw_sf" and args.keys() <= allowed | {"foreach"}
                and args.pop("foreach", True) is True and schedule_free is True):
            betas = args.pop("betas", [0.9, 0.99])
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError("adamw_sf needs two beta coefficients.")
            # At beta2=0 the existing NeoSR kernel uses precisely the AdamW-SF
            # direction and bias-corrected interpolation weight.
            return {"type": "NeoSRAdanScheduleFree", "lr": 8e-4, "weight_decay": 0.0,
                    **args, "betas": [betas[0], 0.0, betas[1]]}, 0, True
    raise ValueError("Review this optimizer and its execution options before selecting a native implementation.")


def _scheduler(spec, roles):
    if not isinstance(spec, dict):
        raise ValueError("Use a scheduler table.")
    kind = str(spec.get("type", "")).lower()
    names = {"multisteplr": ("MultiStepLR", {"milestones", "gamma"}),
             "cosineannealing": ("CosineAnnealingLR", {"T_max", "eta_min"})}
    if kind not in names or spec.keys() - {"type", "nets"} - names[kind][1]:
        raise ValueError("Review the scheduler type and its constructor options.")
    targets = spec.get("nets")
    if targets is None:
        targets = [role for role, (_, sf) in roles.items() if not sf]
    elif (not isinstance(targets, list) or not targets or not all(isinstance(role, str) for role in targets)
            or len(set(targets)) != len(targets) or set(targets) - roles.keys()):
        raise ValueError("Scheduler nets must name distinct, configured optimizers.")
    return {"type": names[kind][0], **{k: v for k, v in spec.items() if k not in ("type", "nets")}}, targets


def import_optimizers(train, training, unresolved):
    roles = {}
    for role in ("g", "d"):
        key = "optim_" + role
        if key not in train:
            continue
        source = train.pop(key)
        if not isinstance(source, dict):
            raise ValueError(f"train.{key} must be a TOML table.")
        try:
            training[key], warmup, sf = _optimizer(source)
        except ValueError as error:
            unresolved("train." + key, source, str(error))
        else:
            roles[role] = (warmup, sf)
    shared, targets = None, []
    spec = train.pop("scheduler", None)
    if spec is not None:
        try:
            shared, targets = _scheduler(spec, roles)
        except ValueError as error:
            unresolved("train.scheduler", spec, str(error))
    schedules = {}
    for role, (warmup, _) in roles.items():
        stages = []
        if warmup:
            stages.append({"type": "LinearLR", "start_factor": 1 / warmup, "total_iters": warmup})
        if role in targets:
            stages.append(shared)
        if stages:
            schedules[role] = stages[0] if len(stages) == 1 else {"type": "ChainedScheduler", "schedulers": stages}
    if schedules:
        training["schedulers"] = schedules

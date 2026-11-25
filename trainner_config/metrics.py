"""Direction used for ranking configured validation metrics."""

import hashlib
import json
import re
from collections.abc import Mapping


def best_weight_filename(attempt: str, dataset: str, metric: str) -> str:
    identity = json.dumps([attempt, dataset, metric], ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{dataset}-{metric}").strip("-")[:54]
    return f"best-{label}-{digest}.safetensors"


def supervised_metric_direction(task: str, metric: str, names: list[str] | tuple[str, ...]) -> str | None:
    if task == "image_classification" and metric.startswith("top"):
        rank = metric.removeprefix("top")
        if len(rank) <= len(str(len(names))) and rank.isascii() and rank.isdecimal() and 1 <= int(rank) <= len(names):
            return "higher"
    if task == "semantic_segmentation" and (
        metric in ("mIoU", "pixel_accuracy") or metric.startswith("IoU/") and metric[4:] in names
    ):
        return "higher"
    if task == "object_detection":
        if metric == "false_positives50":
            return "lower"
        if metric in ("AP50", "recall50") or metric.startswith("AP50/") and metric[5:] in names:
            return "higher"
    return None


def metric_direction(options: dict) -> str:
    metric_type = options["type"]
    default = "lower" if metric_type.lower() in ("calculate_lpips", "calculate_dists", "calculate_butteraugli") else "higher"
    value = options.get("better", default)
    if value not in ("lower", "higher"):
        raise ValueError(f"Metric {metric_type!r} needs better set to 'lower' or 'higher'.")
    return value


def configured_snapshot_metrics(opt, _settings) -> dict[str, str]:
    """For validators that compute the named metrics in val.metrics."""
    metrics = opt.val.metrics if opt.val is not None else None
    return {
        name: metric_direction(options)
        for name, options in (metrics or {}).items()
        if isinstance(options, dict) and isinstance(options.get("type"), str)
    }


def validation_snapshot_metrics(opt) -> dict[str, str] | None:
    """Recipe-specific directions, or None when this evaluator has no snapshot hook."""
    from trainner_config.tasks import TRAINING_TASKS, resolve_task, task_actions, task_name

    validate = task_actions(opt).get("validate")
    if validate is None or validate.snapshot_metrics is None:
        return None
    task = task_name(opt)
    selected = resolve_task(opt) if TRAINING_TASKS[task].options_type is not None else None
    directions = validate.snapshot_metrics(opt, selected[2] if selected else None)
    if not isinstance(directions, Mapping) or any(
        not isinstance(name, str) or not name.strip() or direction not in ("higher", "lower")
        for name, direction in directions.items()
    ):
        raise ValueError(f"Task {task!r} snapshot_metrics must map metric names to 'higher' or 'lower'.")
    return dict(directions)


def best_weight_direction(opt) -> str:
    from trainner_config.tasks import task_name

    val = opt.val
    assert val is not None and val.save_best is not None
    metric = val.save_best
    if not val.val_enabled or not val.metrics_enabled:
        raise ValueError("Enable validation and metrics to save best weights.")
    datasets = [ds.name for k, ds in opt.datasets.items() if k.split("_")[0] == "val"]
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("Saving best weights requires validation datasets with distinct names.")
    directions = validation_snapshot_metrics(opt)
    if directions is None:
        raise ValueError(f"Task {task_name(opt)!r} has no validation snapshot support for this architecture. Omit val.save_best.")
    if metric not in directions:
        raise ValueError(f"val.save_best {metric!r} must name a snapshot metric produced by this recipe.")
    return directions[metric]

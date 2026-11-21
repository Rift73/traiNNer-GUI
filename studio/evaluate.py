"""Run the trainer's own holdout evaluation against one explicit checkpoint."""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import yaml

from .preflight import launch_text
from .recipes import task_from_config
from .yaml_edit import Dumper, read_recipe


def run_evaluation(source: Path, weights: Path, output: Path, *, split: str | None,
                   save_images: bool, progress, save_detections: bool = False,
                   save_classifications: bool = False, save_temporal_errors: bool = False,
                   save_detection_operating_points: bool = False, device: str = "cpu") -> tuple[dict, list[Path]]:
    import msgspec
    from trainner_config.options import ReduxOptions
    from trainner_config.tasks import TRAINING_TASKS, require_task_action
    from traiNNer.tasks.dispatch import evaluate_holdout
    from traiNNer.utils.config import Config

    if device not in ("cpu", "cuda"):
        raise ValueError("Holdout device must be cpu or cuda.")
    accelerator = {}
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA was selected but is unavailable in the evaluation worker.")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        accelerator = {"deviceIndex": torch.cuda.current_device(), "deviceName": torch.cuda.get_device_name(),
                       "torchVersion": str(torch.__version__)}
    original = source.read_text(encoding="utf-8")
    config = read_recipe(launch_text(original, read_recipe(original), Path.cwd()))
    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("The recipe needs at least one validation or test split.")
    selected = {name: dataset for name, dataset in datasets.items()
                if name.partition("_")[0] in {"val", "test"} and (split is None or name == split)}
    if not selected:
        raise ValueError("Select a validation or test split defined in the recipe.")
    names = set()
    for key, dataset in selected.items():
        if not isinstance(dataset, dict) or not isinstance(dataset.get("name"), str):
            raise ValueError(f"Dataset {key!r} needs a name and data settings.")
        if dataset["name"] in names:
            raise ValueError("Give each selected split a distinct dataset name.")
        names.add(dataset["name"])
        if save_temporal_errors and str(dataset.get("type", "")).lower() != "pairedvideodataset":
            raise ValueError(f"Temporal error needs a PairedVideoDataset with numbered frames on both sides. Select a paired split instead of {key!r}.")
        dataset["prefetch_mode"] = None
        if device == "cpu":
            dataset["pin_memory"] = False
    val = config.get("val")
    if not isinstance(val, dict):
        raise ValueError("The recipe needs validation settings.")
    task = task_from_config(config)
    declared = require_task_action(msgspec.convert(config, type=ReduxOptions, strict=True), "holdout")
    provider_scores = declared.output == "provider_scores"
    if not provider_scores and not val.get("metrics") and not (save_images or save_temporal_errors):
        raise ValueError("Select validation metrics or request a holdout output.")
    for field, enabled in (("saveImages", save_images), ("saveClassifications", save_classifications),
                           ("saveDetections", save_detections), ("saveTemporalErrors", save_temporal_errors),
                           ("saveDetectionOperatingPoints", save_detection_operating_points)):
        if enabled and field not in declared.optional_exports:
            raise ValueError(f"{task} does not support {field} during holdout evaluation.")
    # Use the trained inference network. Discriminator, training losses/teacher,
    # training data and optimizer setup have no role in holdout evaluation.
    options = config.get("path") or {}
    autoencoder = TRAINING_TASKS[task].network_field == "network_ae"
    paths = {"pretrain_network_ae" if autoencoder else "pretrain_network_g": os.fspath(weights),
             "strict_load_g": True}
    if options.get("param_key_g") is not None and not autoencoder:
        paths["param_key_g"] = options["param_key_g"]
    for key in ("network_d", "network_g_teacher", "train"):
        config.pop(key, None)
    config.update(name="evaluation", num_gpu=int(device == "cuda"), high_order_degradation=False,
                  use_compile=False, use_amp=False, checkpoint_rng=False, datasets=selected, path=paths)
    if device == "cuda":
        config["fast_matmul"] = False
    val.update(val_enabled=True, metrics_enabled=bool(val.get("metrics") or provider_scores),
               record_samples=True, save_img=save_images, save_img_freq=None,
               save_img_iters=None, pbar=False)
    effective = output / "evaluation.yml"
    effective.write_text(yaml.dump(config, Dumper=Dumper, sort_keys=False, allow_unicode=True), encoding="utf-8")
    options, _ = Config.load_config_from_file(str(output), is_train=False, argv=["-opt", str(effective)])
    progress({"stage": "evaluating", "completed": 0, "total": len(selected)})
    from traiNNer.metrics.session import metric_session
    if save_temporal_errors:
        from traiNNer.tasks.temporal_results import record
        from .native import executable
        result_export = record(output / "temporal-errors", executable("metrics", "metrics.so", source="metrics.c"))
    elif save_detections:
        from traiNNer.tasks.object_detection.results import record
        result_export = record(output / "detections")
    elif save_classifications:
        from traiNNer.tasks.image_classification.results import record
        result_export = record(output / "classifications")
    else:
        result_export = nullcontext(None)
    operating_points = nullcontext(None)
    if save_detection_operating_points:
        from traiNNer.tasks.object_detection.operating_points import record as record_operating_points
        operating_points = record_operating_points(output / "operating-points")
    with metric_session(device, cache_accelerator=options.val.cache_metric_models_on_device), result_export as exported, operating_points as points:
        observations = evaluate_holdout(options, on_dataset=progress)
    artifacts = [effective, *sorted(p for p in (output / "results" / "evaluation").rglob("*")
                                    if p.is_file() and p.suffix in (".sqlite3", ".png"))]
    exports = {}
    if exported is not None:
        artifacts.extend(exported.outputs.values())
        exports = {name: path.relative_to(output).as_posix() for name, path in exported.outputs.items()}
    temporal = {"temporalResults": exports, "temporalSummary": exported.summaries} if save_temporal_errors else {}
    curves = {}
    if points is not None:
        artifacts.extend(points.outputs.values())
        curves = {"detectionOperatingPoints": {name: path.relative_to(output).as_posix() for name, path in points.outputs.items()},
                  "detectionOperatingSummary": points.summaries}
    tiles = {}
    if options.val.tile_size > 0 and task == "image_restoration":
        tiles = {"tiles": {"mode": options.val.tile_mode, "size": options.val.tile_size,
                           "overlap": options.val.tile_overlap,
                           "stitchDevice": "cpu" if options.val.tile_mode == "context" else device}}
    formats = {row.get("outputPixelFormat") for row in observations}
    observed_format = {"outputPixelFormat": formats.pop()} if len(formats) == 1 and None not in formats else {}
    return {"task": task, **observed_format, **tiles, **temporal, **curves, "cocoResults": exports if save_detections else {},
            "classificationResults": exports if save_classifications else {},
            "weights": str(weights), "device": device, "precision": "float32", **accelerator,
            "datasets": observations, "splits": list(selected)}, artifacts

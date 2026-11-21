"""Import a NeoSR training TOML as an editable local recipe."""

from __future__ import annotations

import datetime
import hashlib
import math
import tomllib
from typing import Any

import yaml

from .yaml_edit import MAX_RECIPE_BYTES, Dumper
from .neosr_degradations import import_otf


def _check_values(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("TOML exceeds the supported nesting depth.")
    if isinstance(value, dict):
        for child in value.values():
            _check_values(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_values(child, depth + 1)
    elif isinstance(value, (datetime.time, datetime.date)) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Training recipes use finite numeric values and strings for dates.")


def import_toml(text: str, library, *, source_root: str | None = None, archived: bool = False) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_RECIPE_BYTES:
        raise ValueError("Provide a NeoSR TOML recipe of at most 2 MiB.")
    try:
        source = tomllib.loads(text.lstrip("\ufeff"))
    except RecursionError as exc:
        raise ValueError("TOML exceeds the supported nesting depth.") from exc
    _check_values(source)
    defs = library.schema()["$defs"]
    mapped: dict[str, Any] = {"num_gpu": 1, "path": {}}
    review: dict[str, Any] = {}
    notes: list[str] = []

    def unresolved(path, value, reason):
        review[path] = {"value": value, "reason": reason}

    def table(parent, key, prefix=""):
        group = parent.pop(key, {})
        if not isinstance(group, dict):
            raise ValueError(f"{prefix}{key} must be a TOML table.")
        return group

    def move(source, destination, fields):
        for origin, target in fields.items():
            if origin in source:
                destination[target] = source.pop(origin)

    def leftovers(source, prefix):
        for key, value in source.items():
            unresolved(prefix + key, value, "Choose the corresponding trainer setting or explicitly discard this option.")

    move(source, mapped, {"name": "name", "scale": "scale", "manual_seed": "manual_seed",
                          "use_amp": "use_amp", "bfloat16": "amp_bf16", "fast_matmul": "fast_matmul",
                          "compile": "use_compile", "deterministic": "deterministic"})
    if type(mapped.get("use_amp", False)) is not bool:
        raise ValueError("use_amp must be true or false.")
    mapped["use_channels_last"] = mapped.get("use_amp", False)
    mode = source.pop("model_type", "image")
    if mode in ("image", "otf"):
        mapped["task"] = "image_restoration"
        mapped["high_order_degradation"] = mode == "otf"
        if mode == "otf":
            notes.append("The native trainer uses its own on-the-fly degradation sampler and defaults.")
    else:
        unresolved("model_type", mode, "Select the native task that implements this model family.")

    for slot in ("network_g", "network_d"):
        if slot not in source:
            continue
        network = table(source, slot)
        kind = str(network.get("type", "")).lower()
        if kind not in defs["ReduxOptions"]["properties"]["network_g"]["discriminator"]["mapping"]:
            unresolved(slot, network, "This named architecture is absent from the native catalog.")
            continue
        network["type"] = kind
        if "upscale" in network:
            scale = network.pop("upscale")
            if scale != mapped.get("scale"):
                unresolved(slot + ".upscale", scale, "Resolve the conflict with the global training scale.")
        mapped[slot] = network
        if kind != "compact":
            notes.append(f"Check the constructor and checkpoint compatibility of the native {kind} implementation.")

    paths = table(source, "path")
    move(paths, mapped["path"], {k: k for k in ("pretrain_network_g", "pretrain_network_d",
                                                "strict_load_g", "strict_load_d", "param_key_g", "param_key_d")})
    if "resume_state" in paths:
        unresolved("path.resume_state", paths.pop("resume_state"),
                   "NeoSR optimizer states have a different format. Select compatible weights for a fresh optimizer.")
    leftovers(paths, "path.")

    datasets = table(source, "datasets")
    mapped["datasets"] = {}
    training = mapped["train"] = {}
    for phase in datasets:
        original = datasets[phase]
        if not isinstance(original, dict):
            raise ValueError(f"datasets.{phase} must be a table.")
        ds = dict(original)
        prefix = f"datasets.{phase}."
        kind = ds.pop("type", "")
        names = {"paired": "pairedimagedataset", "otf": "realesrgandataset", "single": "singleimagedataset"}
        if not isinstance(kind, str) or kind not in names:
            unresolved(f"datasets.{phase}", original, "Select a supported dataset and confirm its sample contract.")
            continue
        target = {"type": names[kind], "name": ds.pop("name", phase)}
        for key in ("dataroot_gt", "dataroot_lq"):
            if key in ds:
                path = ds.pop(key)
                if not isinstance(path, str) and not (isinstance(path, list) and all(isinstance(v, str) for v in path)):
                    raise ValueError(f"{prefix}{key} must be a path or list of paths.")
                target[key] = [path] if isinstance(path, str) else path
        folder_paths = (target.get("dataroot_lq", []) or []) + (target.get("dataroot_gt", []) or [])
        if folder_paths and all(isinstance(path, str) and path.rstrip("/\\").lower().endswith(".lmdb") for path in folder_paths):
            target["io_backend"] = {"type": "lmdb"}
        if "meta_info" in ds and "meta_info_file" in ds:
            legacy = ds.pop("meta_info_file")
            if ds["meta_info"] != legacy:
                notes.append(f"{prefix}meta_info takes precedence over the legacy meta_info_file path.")
        move(ds, target, {"patch_size": "lq_size", "batch_size": "batch_size_per_gpu",
                          "accumulate": "accum_iter", "num_worker_per_gpu": "num_worker_per_gpu",
                          "use_hflip": "use_hflip", "use_rot": "use_rot", "pin_memory": "pin_memory",
                          "persistent_workers": "persistent_workers", "meta_info": "meta_info",
                          "meta_info_file": "meta_info", "color": "color", "dataset_enlarge_ratio": "dataset_enlarge_ratio"})
        if target.get("color") == "y":
            target["color"] = "luma601"
        elif target.get("color") == "rgb":
            target.pop("color")
        if phase == "train":
            if mode == "image" and kind == "paired":
                from trainner_config.neosr_batchaug import NEOSR_BATCH_FIELDS, map_paired_batchaug

                settings = {key: ds.pop(key) for key in NEOSR_BATCH_FIELDS if key in ds}
                try:
                    training.update(map_paired_batchaug(settings))
                except ValueError as exc:
                    for key, value in settings.items():
                        unresolved(prefix + key, value, str(exc))
            target.setdefault("num_worker_per_gpu", 4)
            target.setdefault("dataset_enlarge_ratio", 1)
            if target.get("num_worker_per_gpu") == 0:
                target.update(persistent_workers=False, prefetch_factor=None)
            if mode == "otf" and kind == "otf":
                group = table(source, "degradations")
                import_otf(group, ds, target, mapped, notes, unresolved)
                leftovers(group, "degradations.")
            elif (mode == "otf") != (kind == "otf"):
                unresolved(prefix + "type", kind, "Choose matching model and dataset types for on-the-fly or paired training.")
        mapped["datasets"][phase] = target
        leftovers(ds, prefix)

    log = table(source, "logger")
    train = table(source, "train")
    if "total_iter" in log:
        training["total_iter"] = log.pop("total_iter")
    logger = mapped["logger"] = {"print_freq": 100, "use_tb_logger": False}
    move(log, logger, {k:k for k in ("save_checkpoint_freq", "print_freq", "use_tb_logger")})
    leftovers(log, "logger.")
    move(train, training, {"ema": "ema_decay", "ema_power": "ema_power"})
    decay, power = training.get("ema_decay", 0), training.get("ema_power")
    if type(decay) not in (float, int) or not 0 <= decay < 1:
        unresolved("train.ema", training.pop("ema_decay"), "NeoSR EMA requires decay in [0, 1).")
    elif decay > 0:
        training["ema_mode"] = "neosr"
    if power is not None and (type(power) not in (float, int) or power <= 0):
        unresolved("train.ema_power", training.pop("ema_power"), "NeoSR EMA power must be positive.")
    clip = train.pop("grad_clip", True)
    if isinstance(clip, bool):
        training.update(grad_clip=clip, grad_clip_max_norm=1.0)
    elif clip == "spike":
        training["grad_clip"] = "spike"
    elif type(clip) in (int, float) and clip > 0:
        training.update(grad_clip=bool(clip), grad_clip_max_norm=float(clip))
    else:
        unresolved("train.grad_clip", clip, "Select a supported gradient clipping policy.")

    from .neosr_objectives import import_objectives
    from .neosr_optimizers import import_optimizers
    import_optimizers(train, training, unresolved)
    import_objectives(train, training, unresolved)
    if "enable_qat" in train:
        enabled = train.pop("enable_qat")
        if type(enabled) is not bool:
            unresolved("train.enable_qat", enabled, "Choose a boolean quantization-aware training setting.")
        elif enabled:
            from trainner_config.tasks import AETHER_ARCHITECTURES

            arch = mapped.get("network_g", {}).get("type")
            if mode != "image" or arch not in AETHER_ARCHITECTURES:
                unresolved("train.enable_qat", enabled, "INT8 QAT requires an Aether image restoration generator.")
            else:
                qat = training["qat"] = {}
                move(train, qat, {"qat_type": "type", "quantize_residual": "quantize_residual",
                                  "calibration_iters": "calibration_iters", "qat_freeze_norm_iter": "freeze_norm_iter",
                                  "qat_freeze_observers_iter": "freeze_observers_iter"})
                if "fused_init" in train:
                    fused = train.pop("fused_init")
                    if fused is not True:
                        unresolved("train.fused_init", fused, "Aether QAT requires fused reparameterized convolutions.")
    leftovers(train, "train.")

    if "val" in source:
        val = table(source, "val")
        target = mapped["val"] = {"val_enabled": bool(val.get("val_freq")), "save_img": val.pop("save_img", True),
                                  "training_precision": "float32"}
        move(val, target, {"val_freq":"val_freq"})
        tile = val.pop("tile", 0)
        halo = val.pop("tile_overlap", 16)
        if type(tile) is not int or type(halo) is not int:
            unresolved("val.tile", {"tile": tile, "tile_overlap": halo}, "Use integer input pixels for tile size and context.")
        elif tile > 0:
            if 0 <= 2 * halo < tile:
                target.update(tile_size=tile, tile_overlap=halo, tile_mode="context")
            else:
                unresolved("val.tile", {"tile": tile, "tile_overlap": halo}, "Choose a positive tile larger than twice its nonnegative context.")
        metrics = table(val, "metrics", "val.")
        native_metrics = {}
        metric_parameters = {
            "calculate_psnr": {"crop_border", "test_y_channel"},
            "calculate_ssim": {"crop_border", "test_y_channel"},
            "calculate_lpips": {"net"},
            "calculate_dists": set(),
            "calculate_topiq": {"dataset"},
        }
        for name, spec in metrics.items():
            kind = spec.get("type") if isinstance(spec, dict) else None
            if (isinstance(kind, str) and kind in metric_parameters
                    and set(spec) <= {"type", "better"} | metric_parameters[kind]):
                if kind == "calculate_topiq" and spec.get("dataset", "kadid") not in ("kadid", "pipal"):
                    unresolved("val.metrics."+name, spec, "NeoSR TOPIQ offers kadid and pipal weights.")
                elif kind in ("calculate_psnr", "calculate_ssim"):
                    crop = spec.get("crop_border", 4)
                    if type(crop) is not int or crop < 0 or type(spec.get("test_y_channel", False)) is not bool:
                        unresolved("val.metrics."+name, spec, "Use a nonnegative border crop and a boolean Y-channel setting.")
                    else:
                        native_metrics[name] = {**spec, "type": kind.replace("calculate_", "calculate_neosr_"), "crop_border": crop}
                else:
                    native_metrics[name] = {**spec, "type": "calculate_neosr_topiq"} if kind == "calculate_topiq" else spec
            else:
                unresolved("val.metrics."+name, spec, "Review the metric and its model or preprocessing.")
        target.update(metrics_enabled=bool(native_metrics),metrics=native_metrics or None)
        if "save_best" in val:
            name = val.pop("save_best")
            if isinstance(name, str) and name in native_metrics:
                target["save_best"] = name
                notes.append("Best validation weights are retained separately for each dataset and training attempt. A resumed attempt starts its own selection.")
            else:
                unresolved("val.save_best", name, "Select a mapped validation metric to keep its best weights.")
        leftovers(val, "val.")
    leftovers(source, "")

    from .preflight import input_paths
    from .reproduction_paths import source_path

    base = source_path(source_root) if source_root is not None else None
    relative_paths = False
    for segments, value in input_paths(mapped):
        if not isinstance(value, str):
            continue
        path_key = ".".join(map(str, segments))
        if base is not None:
            try:
                qualified = str(source_path(value, base))
            except ValueError as exc:
                unresolved(path_key, value, str(exc))
                continue
            owner = mapped
            for segment in segments[:-1]:
                owner = owner[segment]
            owner[segments[-1]] = qualified
        else:
            try:
                source_path(value)
            except ValueError:
                relative_paths = True
                if archived:
                    unresolved(path_key, value, "Provide the original launch directory to qualify this archived input.")
    if relative_paths:
        notes.append("Relative input paths need the original launch directory. Supply sourceRoot to qualify them.")
    if review:
        # The strict trainer options decoder rejects this section. Saving a
        # partially imported recipe can never silently train a reduced objective.
        mapped["neosr_review"] = review
    header = "# Imported NeoSR settings. Paths, weights and evaluation use the local trainer.\n"
    if review:
        header += "# Resolve neosr_review below and remove that section before training.\n"
    recipe = header + yaml.dump(mapped, Dumper=Dumper, sort_keys=False, allow_unicode=True)
    return {"yaml": recipe, "sourceSha256": hashlib.sha256(text.encode()).hexdigest(),
            "review": review, "needsReview": bool(review), "notes": notes}

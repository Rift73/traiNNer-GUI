"""Read-only training recipe checks. No model construction or GPU work."""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from stat import S_ISREG
from threading import Lock
from typing import TYPE_CHECKING, Any
from zipfile import BadZipFile

import msgspec
from trainner_data.manifests import manifest_filename
from trainner_data.zip_images import (
    ZipConfigError,
    validate_zip_archive,
    validate_zip_dataset,
    zip_archive_stat,
)

from .paths import artifact_version, local_path
from .recipes import RecipeLibrary, safe_name, task_from_config
from .yaml_edit import read_recipe, render

if TYPE_CHECKING:
    from trainner_config.options import DatasetOptions


DATA_PATHS = ("dataroot_gt", "dataroot_lq", "target_dataroot_gt", "paired_dataroot_gt",
              "meta_info", "annotation_file", "video_scene_manifest")

_zip_probe_lock = Lock()


@lru_cache(maxsize=32)
def _zip_archive_problem(
    archive: Path,
    archive_version: str,
    manifest: Path | None,
    manifest_version: str | None,
) -> tuple[str, str] | None:
    # Cache only the outcome. Central-directory records and manifest names
    # are released after checking, and the trainer validates its own snapshot.
    names = None
    problem = None
    if manifest is not None:
        try:
            with manifest.open() as stream:
                names = [manifest_filename(line) for line in stream if line.strip()]
        except UnicodeError as exc:
            problem = ("meta_info", f"Could not read ZIP manifest {manifest}: {exc}")
    if problem is None:
        try:
            validate_zip_archive(str(archive), names)
        except (BadZipFile, NotImplementedError) as exc:
            problem = ("dataroot_gt", f"Invalid ZIP archive {archive}: {exc}")
        except ValueError as exc:
            problem = ("dataroot_gt", str(exc))

    if artifact_version(archive.stat()) != archive_version or (
        manifest is not None and artifact_version(manifest.stat()) != manifest_version
    ):
        raise ValueError(f"The archive or manifest changed during ZIP preflight: {archive}")
    return problem


def _zip_dataset_issues(dataset: DatasetOptions, root: Path) -> Iterator[tuple[str, str]]:
    try:
        roots = validate_zip_dataset(dataset)
    except ZipConfigError as exc:
        yield exc.path, str(exc)
        return

    manifest = None
    manifest_version = None
    if dataset.meta_info is not None:
        try:
            manifest = local_path(dataset.meta_info, root)
            info = manifest.stat()
            if not S_ISREG(info.st_mode):
                raise ValueError(f"Expected a manifest file: {manifest}")
            manifest_version = artifact_version(info)
        except (OSError, ValueError) as exc:
            yield "meta_info", str(exc)
            return

    for index, value in enumerate(roots):
        field = f"dataroot_gt.{index}" if isinstance(dataset.dataroot_gt, list) else "dataroot_gt"
        try:
            archive = local_path(value, root)
            version = artifact_version(zip_archive_stat(str(archive)))
            # Concurrent form and launch requests can request the same scan.
            with _zip_probe_lock:
                problem = _zip_archive_problem(archive, version, manifest, manifest_version)
        except (OSError, ValueError) as exc:
            yield field, str(exc)
        else:
            if problem is not None:
                problem_field, message = problem
                yield field if problem_field == "dataroot_gt" else problem_field, message


def input_paths(config: dict):
    datasets = config.get("datasets")
    if isinstance(datasets, dict):
        for phase, ds in datasets.items():
            if not isinstance(ds, dict):
                continue
            for key in DATA_PATHS:
                val = ds.get(key)
                path = ["datasets", phase, key]
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        yield [*path, i], v
                elif val is not None:
                    yield path, val
    paths = config.get("path")
    if isinstance(paths, dict):
        for key, val in paths.items():
            if val is not None and (key.startswith("pretrain_network_") or key in ("resume_state", "neuflow_weights")):
                yield ["path", key], val


def launch_text(text: str, config: dict, root: Path) -> str:
    """The saved launch recipe uses paths understood by both train.py and the CLI."""
    edits = []
    for key, value in input_paths(config):
        if isinstance(value, str) and (value.startswith(("~", "\\\\", "//")) or
                                       (len(value) > 2 and value[1:2] == ":")):
            if key == ["path", "neuflow_weights"]:
                from trainner_config.rdual import rdual_neuflow_required
                from .run_checkpoints import import_sources, rdual_options
                opt = rdual_options(config)
                if opt is None:
                    continue
                companions = import_sources(config, root) if opt.path.resume_state else None
                if rdual_neuflow_required(opt, root, resolve_path=local_path, resume_files=companions) is False:
                    continue
            edits.append({"path": key, "value": str(local_path(value, root))})
    return render(text, edits) if edits else text


def _manifest_crop_example(dataset, root: Path):
    """Read the size of one selected LQ file. Folder scans remain a separate action."""
    from PIL import Image, UnidentifiedImageError
    from trainner_data.manifests import paired_manifest_lq_path

    lq = dataset.dataroot_lq
    if not lq or dataset.io_backend.get("type", "disk") != "disk":
        return None
    lq = local_path(lq[0] if isinstance(lq, list) else lq, root)
    if dataset.meta_info:
        manifest = local_path(dataset.meta_info, root)
    else:
        # Patch extraction records the complete inventory alongside its sibling roots.
        gt = dataset.dataroot_gt
        if (not gt or lq.name != "lq" or local_path(gt[0] if isinstance(gt, list) else gt, root) != lq.parent / "gt"
                or not (lq.parent / "dataset.json").is_file()):
            return None
        manifest = lq.parent / "meta_info.txt"
    if not manifest.is_file():
        return None
    with manifest.open("rb") as stream:
        line = stream.readline(8192)
    if not line.strip() or len(line) == 8192:
        return None
    name = manifest_filename(line.decode("utf-8"))
    path = Path(paired_manifest_lq_path(str(lq), name, dataset.filename_tmpl))
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return {"path": str(path), "width": image.width, "height": image.height}
    except (UnidentifiedImageError, Image.DecompressionBombError):
        # The trainer's libvips reader has different format and size support.
        return None


def snapshot_options(opt) -> dict[str, Any]:
    from trainner_config.metrics import validation_snapshot_metrics
    from trainner_config.tasks import task_name

    val = opt.val
    directions = (validation_snapshot_metrics(opt) or {}) if val is not None else {}
    selected = val.save_best if val else None
    datasets = [ds for key, ds in opt.datasets.items() if key.split("_")[0] == "val"]
    names = [ds.name for ds in datasets]
    eligible = []
    if val and val.val_enabled and val.metrics_enabled and selected in directions and len(set(names)) == len(names):
        metric = (val.metrics or {}).get(selected, {})
        reconstruction = task_name(opt) == "autoencoder"
        for ds in datasets:
            if (not reconstruction and ds.type.casefold() in {"singleimagedataset", "singlevideodataset"}
                    and metric.get("type", "").casefold() != "calculate_topiq_nr"):
                continue
            eligible.append(ds.name)
    return {"snapshotMetrics": directions, "bestMetric": selected, "bestValidationDatasets": eligible}


def phase_snapshot_options(base: str, edits: list) -> dict[str, Any]:
    from trainner_config.options import ReduxOptions

    try:
        opt = msgspec.convert(read_recipe(render(base, edits)), type=ReduxOptions, strict=True)
    except msgspec.ValidationError as exc:
        raise ValueError(str(exc)) from exc
    opt.is_train = True
    return snapshot_options(opt)


def preflight(text: str, library: RecipeLibrary) -> dict[str, Any]:
    issues = []
    summary = {"name":None, "task":None, "architecture":None, "scale":None}

    def issue(path, message, level="error"):
        issues.append({"level":level, "path":path, "message":str(message)})

    try:
        config = read_recipe(text)
    except ValueError as exc:
        issue("", exc)
        return {"ok":False, "summary":summary, "issues":issues}

    def section(key):
        value = config.get(key)
        return value if isinstance(value, dict) else {}

    arch = section("network_g") or section("network_ae")
    summary.update(name=config.get("name"), task=task_from_config(config),
                   architecture=arch.get("type"), scale=config.get("scale"))
    try:
        safe_name(config.get("name"))
    except ValueError as exc:
        issue("name", exc)

    # The runtime options decoder is authoritative; it permits extensions in
    # network/loss keyword dictionaries. The catalog can give more specific
    # advice where the installed implementation declares a closed signature.
    from trainner_config.options import ReduxOptions, initial_g_transfer
    from trainner_config.tasks import TRAINING_TASKS, require_task_action, task_actions, validate_task_selection
    from trainner_config.objectives import ObjectiveConfigError, resolve_objectives
    from trainner_config.schedules import SchedulerConfigError, validate_schedulers
    from trainner_config.devices import DeviceConfigError, validate_checkpoint_rng, validate_cpu_recipe
    try:
        opt = msgspec.convert(config, type=ReduxOptions, strict=True)
    except msgspec.ValidationError as exc:
        message = str(exc)
        path = message.partition(" - at `$.")[2].rstrip("`")
        issue(path, message.partition(" - at `")[0])
        return {"ok":False, "summary":summary, "issues":issues, "config":config}

    try:
        opt.is_train = True
        from .recipes import describe_actions
        from trainner_config.tasks import supports_dynamic_spatial_export
        summary["actions"] = describe_actions(task_actions(opt))
        summary.update(snapshot_options(opt))
        if "export" in summary["actions"]:
            summary["actions"]["export"]["dynamicSpatial"] = supports_dynamic_spatial_export(
                summary["task"], summary["architecture"] or ""
            )
        unavailable = {}
        for action in ("holdout", "predict", "export"):
            if action not in summary["actions"]:
                try:
                    require_task_action(opt, action)
                except ValueError as exc:
                    unavailable[action] = str(exc)
        if unavailable:
            summary["unavailableActions"] = unavailable
        require_task_action(opt, "train")
        if opt.val is not None and opt.val.val_enabled:
            require_task_action(opt, "validate")
        validate_task_selection(opt)
    except ValueError as exc:
        issue("task", exc)

    if opt.train and (opt.train.optim_include_params_g or opt.train.optim_exclude_params_g):
        from trainner_config.parameter_filters import parameter_filter
        for field in ("optim_include_params_g", "optim_exclude_params_g"):
            try:
                parameter_filter(getattr(opt.train, field), None)
            except ValueError as exc:
                issue("train." + field, exc)

    if opt.path.transfer_g is not None:
        try:
            transfer = initial_g_transfer(opt)
            summary["transferG"] = (
                {"status": "atModelInitialization", "exclude": transfer.exclude,
                 "labelPolicy": "noRemap"} if transfer is not None else {"status": "inactiveOnResume"}
            )
        except ValueError as exc:
            issue("path.transfer_g", exc)

    try:
        from trainner_config.feature_assets import feature_assets

        objectives = resolve_objectives(opt)
        metrics = opt.val.metrics if opt.val and opt.val.val_enabled and opt.val.metrics_enabled else None
        assets = feature_assets(objectives, metrics, library.root, network_d=opt.network_d)
        if assets:
            summary["featureAssets"] = assets
        for item in assets:
            if item["status"] == "present":
                continue
            if item["provider"] == "OpenAI CLIP RN50":
                if item["status"] == "missing" and item["autoDownload"]:
                    issue(item["usedBy"][0],
                          f"SeD will download the official OpenAI CLIP RN50.pt to {item['file']} on first use.",
                          "warning")
                else:
                    issue(item["usedBy"][0],
                          f"SeD needs the official OpenAI CLIP RN50.pt. The file at {item['file']} "
                          f"is {item['status']}. Set network_d.encoder_weights to a readable local copy.")
            elif item["provider"] == "checkout":
                issue(item["usedBy"][0], f"Required bundled coefficient file is {item['status']}: {item['file']}")
            else:
                issue(item["usedBy"][0],
                      f"Pretrained asset is {item['status']}: {item['file']}. Its provider may download it at first use.",
                      "warning")
    except ObjectiveConfigError as exc:
        issue(exc.path, exc)

    try:
        validate_schedulers(opt)
    except SchedulerConfigError as exc:
        issue(exc.path, exc)

    try:
        validate_cpu_recipe(opt)
        validate_checkpoint_rng(opt)
    except DeviceConfigError as exc:
        issue(exc.path, exc)

    schema = library.schema()
    defs = schema["$defs"]
    arch_map = defs["ReduxOptions"]["properties"]["network_g"]["discriminator"]["mapping"]
    provider = TRAINING_TASKS.get(summary["task"])
    for slot in ("network_g","network_d","network_ae","network_g_teacher"):
        ns = section(slot)
        if not ns:
            continue
        typ = ns.get("type")
        if not isinstance(typ, str) or not typ:
            issue(slot + ".type", "An architecture type is required.")
            continue
        ref = arch_map.get(typ.lower())
        if typ.lower() == "temporalspanv2" and type(ns.get("num_blocks")) is int:
            depth = ns["num_blocks"]
            if ns.get("num_frames", 5) != 1 and depth < 6:
                issue(slot + ".num_blocks", "TemporalSPANv2 indexes six SPAB blocks per temporal layer. Choose 6.")
            elif depth > 6:
                issue(slot + ".num_blocks", "This TemporalSPANv2 implementation uses only the first six SPAB blocks per temporal layer. Additional stored blocks do not contribute to predictions.", "warning")
        if not ref:
            # A locally added implementation can be ahead of the generated catalog.
            issue(slot + ".type", f"{typ} is absent from this catalog; its constructor could not be verified.", "warning")
            continue
        spec = defs[ref.rsplit("/",1)[-1]]
        if spec.get("additionalProperties") is False:
            for key in ns.keys() - spec.get("properties", {}).keys():
                issue(f"{slot}.{key}", f"The registered {typ} architecture does not accept {key}.")
        for key in spec.get("required", []):
            if key == "scale" and slot != "network_d" and (provider is None or provider.dataset_geometry != "records"):
                continue  # Spatial generator, teacher and AE factories receive the top-level scale.
            if key not in ns:
                issue(f"{slot}.{key}", f"The {typ} constructor needs this option.")
    if not arch:
        issue("network_g", "Specify a generator or autoencoder to train.")
    train = section("train")
    optimizer_specs = {item["id"].lower(): item for item in schema.get("x-trainner-catalog", {}).get("optimizers", ())}
    for role in ("optim_g", "optim_d", "optim_ae"):
        options = train.get(role)
        if not isinstance(options, dict):
            continue
        kind = options.get("type")
        if not isinstance(kind, str) or not kind:
            issue(f"train.{role}.type", "Specify the optimizer type.")
            continue
        spec = optimizer_specs.get(kind.lower())
        if spec is None:
            if optimizer_specs:
                issue(f"train.{role}.type", f"{kind} is absent from this catalog. Refresh after installing a local optimizer.", "warning")
            continue
        if not spec.get("signatureAvailable"):
            continue
        parameters = {p["name"]: p for p in spec["parameters"]}
        if not spec["acceptsExtraKeywords"]:
            for key in options.keys() - parameters.keys() - {"type"}:
                issue(f"train.{role}.{key}", f"The {kind} optimizer does not accept {key}.")
        for key, param in parameters.items():
            if param["required"] and key not in options:
                issue(f"train.{role}.{key}", f"The {kind} optimizer needs this option.")
    if type(train.get("total_iter")) is not int or train["total_iter"] <= 0:
        issue("train.total_iter", "Enter a positive update count.")
    if not section("logger"):
        issue("logger", "Training requires a logger section.")
    ds = section("datasets").get("train") or {}
    if not isinstance(ds,dict) or not ds:
        issue("datasets.train", "Select training data.")
    else:
        requires_patch = provider is None or provider.dataset_geometry == "aligned_patches"
        if requires_patch and (ds.get("gt_size") is None) == (ds.get("lq_size") is None):
            issue("datasets.train", "Set exactly one of gt_size and lq_size for the training crop.")
        scale = config.get("scale")
        if type(scale) is not int or scale <= 0:
            issue("scale", "Training scale must be a positive integer.")
        for key in ("gt_size", "lq_size"):
            size = ds.get(key)
            if size is not None and (type(size) is not int or size < 1):
                issue(f"datasets.train.{key}", "Training crops must have positive integer dimensions.")
        gt_size = ds.get("gt_size")
        if type(gt_size) is int and type(scale) is int and scale > 0 and gt_size % scale:
            issue("datasets.train.gt_size", "The high-quality crop size must be divisible by the training scale for aligned pairs.")
        if type(ds.get("batch_size_per_gpu")) is not int or ds["batch_size_per_gpu"] < 1:
            issue("datasets.train.batch_size_per_gpu", "Set batch_size_per_gpu to a positive integer.")
        if opt.datasets["train"].num_worker_per_gpu is None:
            issue("datasets.train.num_worker_per_gpu", "Specify a nonnegative worker count or auto.")
        if ds.get("prefetch_mode") == "cuda" and not ds.get("pin_memory", True):
            issue("datasets.train.pin_memory", "CUDA prefetch needs pinned memory.")
        kind = str(ds.get("type", "")).lower().removesuffix("_trainner")
        if kind == "pairedimagedataset" and type(scale) is int and scale > 0:
            patch = ds.get("lq_size") or (ds.get("gt_size") or 0) // scale
            if patch > 0:
                try:
                    example = _manifest_crop_example(opt.datasets["train"], library.root)
                except (OSError, ValueError, IndexError, KeyError) as exc:
                    issue("datasets.train.meta_info", str(exc))
                else:
                    if example:
                        summary["cropExample"] = example
                        if min(example["width"], example["height"]) < patch:
                            key = "lq_size" if ds.get("lq_size") else "gt_size"
                            issue("datasets.train." + key, f"The selected LQ image {example['path']} is {example['width']}×{example['height']}. It cannot supply the requested {patch}×{patch} LQ crop.")
        if (kind in ("realesrgandataset","videoclipdataset","pairedimagedataset","realesrganpaireddataset","pairedvideodataset","imagemaskdataset")
                and not ds.get("dataroot_gt") and opt.datasets["train"].io_backend.get("type") != "zip"):
            issue("datasets.train.dataroot_gt", "Select a ground-truth data root.")
        if kind in ("pairedimagedataset", "imagemaskdataset"):
            lq_roots, gt_roots = ds.get("dataroot_lq"), ds.get("dataroot_gt")
            if isinstance(lq_roots,list) and isinstance(gt_roots,list) and len(lq_roots) != len(gt_roots):
                issue("datasets.train.dataroot_lq", "Select one low-quality root for each ground-truth root.")
        if kind in ("pairedimagedataset","imagemaskdataset") and not ds.get("dataroot_lq"):
            issue("datasets.train.dataroot_lq", "Select the low-quality image data root.")
        if config.get("dvd_prob",0) > 0 and (config["scale"] != 2 or kind != "realesrgandataset"):
            issue("dvd_prob", "Aligned DVD degradation requires a 2× on-the-fly still-image dataset.")
    if opt.val is not None and opt.val.save_best is not None:
        from trainner_config.metrics import best_weight_direction

        try:
            best_weight_direction(opt)
        except ValueError as exc:
            issue("val.save_best", exc)
    if opt.val is not None and opt.val.record_samples:
        if not opt.val.val_enabled or not opt.val.metrics_enabled or not (opt.val.metrics or summary.get("snapshotMetrics")):
            issue("val.record_samples", "Enable validation with a metric to record sample evidence.", "warning")
    if opt.val and opt.val.metrics:
        from trainner_config.metrics import metric_direction
        for name, options in opt.val.metrics.items():
            if not isinstance(options, dict) or not isinstance(options.get("type"), str) or not options["type"].strip():
                issue(f"val.metrics.{name}.type", "Choose a metric implementation.")
            else:
                try:
                    metric_direction(options)
                except ValueError as exc:
                    issue(f"val.metrics.{name}.better", exc)
    validation_enabled = opt.val is not None and opt.val.val_enabled
    if validation_enabled:
        if opt.val.tile_size < 0 or (opt.val.tile_size and not 0 <= opt.val.tile_overlap < opt.val.tile_size):
            issue("val.tile_overlap", "Use a non-negative overlap smaller than the active tile size.")
        freq = opt.val.val_freq
        if freq is None or freq <= 0:
            issue("val.val_freq","Set a positive validation interval.")
        if summary["task"] == "recurrent_restoration" and opt.val.tile_size:
            issue("val.tile_size", "Recurrent validation uses a full frame sequence. Set tile size to zero.")
        names = set()
        for phase, dataset in opt.datasets.items():
            if phase.split("_",1)[0] == "val":
                if (summary["task"] == "image_restoration" and opt.output_pixel_format in ("y", "uv")
                        and opt.val.complementary_planes == "target" and dataset.type.lower() == "singleimagedataset"):
                    issue("val.complementary_planes", "This split has no ground truth for the missing planes. Select input_bicubic or supply a paired split.")
                if summary["task"] == "autoencoder" and dataset.type.lower() not in ("pairedimagedataset", "singleimagedataset"):
                    issue(f"datasets.{phase}.type", "Choose paired or single images for autoencoder reconstruction.")
                if summary["task"] == "recurrent_restoration":
                    if dataset.type.lower() not in ("pairedvideodataset", "singlevideodataset"):
                        issue(f"datasets.{phase}.type", "Select a paired or single video frame sequence.")
                    if dataset.clip_size is None or dataset.clip_size < 1:
                        issue(f"datasets.{phase}.clip_size", "Use a positive number of frames per window.")
                if dataset.name in names:
                    issue(f"datasets.{phase}.name","Give each validation dataset its own name so scores and images stay distinct.")
                names.add(dataset.name)
        if not names:
            issue("datasets", "Validation is enabled. Add a val split or disable validation.")
    zip_splits = set()
    for phase, dataset in opt.datasets.items():
        if dataset.io_backend.get("type") != "zip" or (
            phase.split("_", 1)[0] == "val" and not validation_enabled
        ):
            continue
        zip_splits.add(phase)
        for field, message in _zip_dataset_issues(dataset, library.root):
            issue(f"datasets.{phase}.{field}", message)
    rdual = str(summary["architecture"]).lower() == "rdual"
    indexed_train = isinstance(ds, dict) and str(ds.get("type", "")).lower() == "videoclipdataset"
    if indexed_train and ds.get("dataroot_gt"):
        from .video_indexing import index_inputs, inspect_inputs
        try:
            indexed = inspect_inputs(index_inputs(config, library.root))
            if indexed["pending"]:
                sample = next(item for item in indexed["files"] if item["status"] != "ready")
                issue("datasets.train.dataroot_gt",
                      f'{indexed["ready"]} of {indexed["total"]} video indexes ready. {sample["reason"]}')
        except (OSError, ValueError) as exc:
            issue("datasets.train.dataroot_gt", exc)
    for path, val in input_paths(config):
        if path[0] == "path" and (path[1] == "neuflow_weights" or
                rdual and (path[1] in ("pretrain_network_g", "pretrain_network_g_ema")
                           or opt.path.resume_state and path[1].startswith("pretrain_network_"))):
            continue  # RDUAL resolves active generator/flow assets after full-state selection below.
        if indexed_train and path[:3] == ["datasets", "train", "dataroot_gt"]:
            continue  # The shared clip inventory accepts folders and explicit video files.
        if path[0] == "datasets":
            phase, field = path[1:3]
            if phase.split("_", 1)[0] == "val" and not validation_enabled:
                continue
            if phase in zip_splits and field in ("dataroot_gt", "target_dataroot_gt", "meta_info"):
                continue
        try:
            p = local_path(val, library.root)
            label = ".".join(map(str,path))
            if not p.exists():
                issue(label, f"Path was not found: {p}")
            elif path[0] == "path" or path[2] in ("meta_info", "annotation_file", "video_scene_manifest"):
                if not p.is_file():
                    issue(label, f"Expected a file: {p}")
                elif summary.get("task") == "tabular_regression" and path[0] == "datasets" and path[2] == "meta_info":
                    from trainner_config.tabular_task import RegressionOptions
                    from trainner_data.numeric_records import selected_columns

                    columns = msgspec.convert(opt.task_options, type=RegressionOptions)
                    with p.open(encoding="utf-8-sig", newline="") as stream:
                        selected_columns(stream, p, [*columns.features, *columns.targets])
            elif not p.is_dir():
                issue(label, f"Expected a dataset directory: {p}")
        except (ValueError, OSError) as exc:
            issue(".".join(map(str,path)), exc)
    if section("path").get("resume_state") and not any(item["level"] == "error" for item in issues):
        from .resume_preflight import inspect_resume
        try:
            summary["resume"], resume_issues = inspect_resume(config, opt, library.root)
            issues.extend(resume_issues)
        except (OSError, ValueError) as exc:
            issue("path.resume_state", exc)
    if rdual:
        from trainner_config.official_weights import neuflow_cache_path
        from trainner_config.rdual import preflight_rdual
        resume_files = summary.get("resume", {}).get("files")
        opt.auto_resume = False  # Managed launches use an explicit numbered state.
        rdual_issues, inputs = preflight_rdual(opt, library.root, resolve_path=local_path, resume_files=resume_files)
        for field, message in rdual_issues:
            if not any(item["path"] == field and item["message"] == message for item in issues):
                issue(field, message)
        flow_field = "path.neuflow_weights"
        if opt.path.neuflow_weights is None and (flow := inputs.get(flow_field)) == neuflow_cache_path():
            if not any(field == flow_field for field, _ in rdual_issues):
                try:
                    flow.stat()
                except FileNotFoundError:
                    issue(flow_field, f"RDUAL will download official NeuFlow v2 weights to {flow} on first use.", "warning")
                except OSError as exc:
                    issue(flow_field, exc)
    return {"ok":all(i["level"] != "error" for i in issues), "summary":summary,
            "issues":issues, "config":config}

"""Lightweight task declarations for recipe validation and trainer dispatch.

Shipped declarations live in trainner_config modules ending in _task.
Installed extensions can expose a registration callback under the
trainner.training_tasks entry-point group. Declarations import configuration
only. Each model_factory names a runtime provider as module:callable.
The callable receives ReduxOptions and the decoded task options (or None).
A provider declaring dataset_types registers those types at import time.
Whole-image and record providers receive their decoded task settings in the dataset
factory instead of the aligned-patch sizing pass.
A dataset can declare collate_fn to retain variable-size samples.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from importlib.metadata import entry_points
from pkgutil import iter_modules
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, get_args

import msgspec

import trainner_config
from trainner_config.metrics import configured_snapshot_metrics

if TYPE_CHECKING:
    from trainner_config.options import ReduxOptions


TaskAction = Literal["train", "validate", "holdout", "predict", "export"]
TaskInput = Literal[
    "images",
    "image_pairs",
    "sequences",
    "sequence_pairs",
    "image_masks",
    "class_folders",
    "coco_boxes",
    "records",
]
_ACTIONS = get_args(TaskAction)
_INPUTS = get_args(TaskInput)


@dataclass(frozen=True, slots=True)
class TaskActionSpec:
    """Alternative input kinds and the runtime handler for a standalone action."""

    inputs: tuple[TaskInput, ...]
    handler: str | None = None
    output: str | None = None
    # Request fields implemented by the holdout handler and its output writers.
    optional_exports: tuple[str, ...] = ()
    # A validate action opts in only if its evaluator observes the selected network.
    snapshot_metrics: Callable[["ReduxOptions", msgspec.Struct | None], Mapping[str, str]] | None = None


@dataclass(frozen=True, slots=True)
class TrainingTaskSpec:
    network_field: Literal["network_g", "network_ae"]
    model_factory: str
    architecture_types: tuple[str, ...] | None = None
    excluded_architecture_types: tuple[str, ...] = ()
    training_only: bool = False
    dataset_types: tuple[str, ...] = ()
    dataset_geometry: Literal["aligned_patches", "whole_images", "records"] = "aligned_patches"
    recommended_architectures: tuple[str, ...] = ()
    options_type: type[msgspec.Struct] | None = None
    validate: Callable[["ReduxOptions", str, msgspec.Struct | None], None] | None = None
    otf_degradation: Literal["required", "unsupported"] | None = None
    actions: Mapping[TaskAction, TaskActionSpec] = field(
        default_factory=lambda: {
            "train": TaskActionSpec(()),
            "validate": TaskActionSpec(()),
        }
    )
    architecture_actions: Mapping[str, Mapping[TaskAction, TaskActionSpec]] = field(
        default_factory=dict
    )


_tasks: dict[str, TrainingTaskSpec] = {}
TRAINING_TASKS: Mapping[str, TrainingTaskSpec] = MappingProxyType(_tasks)


def task_name(opt: "ReduxOptions") -> str:
    """Identify the model family selected by named and legacy trainer recipes."""
    if opt.task:
        return opt.task
    if opt.network_ae and not opt.network_g:
        return "autoencoder"
    if (opt.network_g or {}).get("type", "").lower() in recurrent_architectures:
        return "recurrent_restoration"
    return "image_restoration"


def _named_callable(reference: object) -> bool:
    if not isinstance(reference, str):
        return False
    module, sep, callable_name = reference.partition(":")
    return bool(
        sep
        and all(part.isidentifier() for part in module.split("."))
        and callable_name.isidentifier()
    )


def _register_actions(
    task: str,
    actions: Mapping[TaskAction, TaskActionSpec],
    *,
    training_only: bool,
) -> Mapping[TaskAction, TaskActionSpec]:
    registered = {}
    allowed_outputs = {"predict": ("image", "json", "sequence"), "holdout": ("provider_scores", "reference_scores"),
                       "export": ("onnx",)}
    for name, action in actions.items():
        if name not in _ACTIONS:
            raise ValueError(f"Task {task!r} declares an unknown action {name!r}.")
        if not isinstance(action, TaskActionSpec) or any(
            kind not in _INPUTS for kind in action.inputs
        ):
            raise ValueError(f"Task {task!r} has invalid inputs for {name!r}.")
        if action.snapshot_metrics is not None and (
            name != "validate" or "train" not in actions or not callable(action.snapshot_metrics)
        ):
            raise ValueError(f"Task {task!r} requires train and a validate action with a callable snapshot_metrics.")
        if name in ("train", "validate"):
            if action.handler is not None:
                raise ValueError(f"Task {task!r} uses model hooks for {name!r}.")
        elif training_only or not (
            action.inputs and action.handler and _named_callable(action.handler)
        ):
            raise ValueError(
                f"Task {task!r} must declare input kinds and a module:callable "
                f"handler for {name!r}, and permit standalone actions."
            )
        choices = allowed_outputs.get(name, ())
        if action.output is not None and action.output not in choices:
            raise ValueError(f"Task {task!r} declares an invalid {name} output {action.output!r}.")
        output = action.output or (choices[0] if choices else None)
        exports = action.optional_exports
        if name == "holdout" and output == "reference_scores" and not exports:
            # Reference-score providers already accept val.save_img.
            exports = ("saveImages",)
        if exports and (name != "holdout" or not set(exports) <= {
            "saveImages", "saveDetections", "saveClassifications", "saveTemporalErrors", "saveDetectionOperatingPoints"
        }):
            raise ValueError(f"Task {task!r} has invalid optional outputs for {name!r}.")
        registered[name] = replace(action, output=output, optional_exports=exports)
    return MappingProxyType(registered)


def register_task(name: str, spec: TrainingTaskSpec) -> None:
    """Register a declaration before the engine imports its task providers."""
    if not name or name in _tasks:
        raise ValueError(f"Invalid or duplicate training task {name!r}.")
    if spec.dataset_geometry not in ("aligned_patches", "whole_images", "records"):
        raise ValueError(f"Invalid dataset geometry for task {name!r}.")
    if not _named_callable(spec.model_factory):
        raise ValueError(
            f"Task {name!r} needs a model_factory in module:callable form."
        )
    actions = _register_actions(name, spec.actions, training_only=spec.training_only)
    by_architecture = {
        architecture.lower(): _register_actions(
            name, overrides, training_only=spec.training_only
        )
        for architecture, overrides in spec.architecture_actions.items()
    }
    _tasks[name] = replace(
        spec, actions=actions, architecture_actions=MappingProxyType(by_architecture)
    )


def _get_task(name: str) -> TrainingTaskSpec:
    spec = TRAINING_TASKS.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown task {name!r}. Choose from {', '.join(TRAINING_TASKS)}."
        )
    return spec


def _task_architecture(opt: "ReduxOptions", task: str, spec: TrainingTaskSpec) -> str:
    field = spec.network_field
    network = opt.network_ae if field == "network_ae" else opt.network_g
    other_field = "network_g" if field == "network_ae" else "network_ae"
    if opt.task is not None and getattr(opt, other_field) is not None:
        raise ValueError(f"{task} uses {field}. Remove {other_field}.")

    architecture = network.get("type") if network is not None else None
    if not isinstance(architecture, str) or not architecture:
        raise ValueError(f"{task} requires {field}.type.")
    architecture = architecture.lower()
    if (
        spec.architecture_types is not None
        and architecture not in spec.architecture_types
    ):
        supported = ", ".join(spec.architecture_types)
        raise ValueError(
            f"{task} requires {field}.type to be one of {supported}. Got {architecture!r}."
        )
    if architecture in spec.excluded_architecture_types:
        raise ValueError(f"{task} cannot use {field}.type {architecture!r}.")
    return architecture


def still_restoration_luma(opt: "ReduxOptions") -> str | None:
    """Resolve the one-plane signal used by the builtin still-image deployment handlers."""
    train = opt.datasets.get("train")
    datasets = [train] if train is not None else list(opt.datasets.values())
    colors = {dataset.color for dataset in datasets}
    if len(colors) > 1:
        raise ValueError("Standalone prediction needs one dataset color encoding. The evaluation splits disagree.")
    color = next(iter(colors), None)
    network = opt.network_g or {}
    compact = str(network.get("type", "")).lower() in ("compact", "ultracompact", "superultracompact")
    compact_mono = compact and network.get("num_in_ch", 3) == network.get("num_out_ch", 3) == 1
    formats = (opt.input_pixel_format, opt.output_pixel_format)
    if color is not None and formats != ("rgb", "rgb"):
        raise ValueError("Dataset luma already supplies one plane. Use the default rgb pixel formats to pass it through.")
    if opt.val and opt.val.complementary_planes == "input_bicubic" and formats[1] in ("y", "uv"):
        if compact:
            channels = {"rgb": 3, "yuv444": 3, "y": 1, "uv": 2}
            if network.get("num_in_ch", 3) != channels[formats[0]] or network.get("num_out_ch", 3) != channels[formats[1]]:
                raise ValueError("Set Compact input and output channels to match the selected RGB, Y, or UV planes.")
        return None
    if formats == ("y", "y"):
        color = "y"
    if color is not None:
        if compact and not compact_mono:
            raise ValueError("Compact luma deployment requires network_g.num_in_ch and num_out_ch both set to 1.")
        if any(dataset.mean is not None and dataset.std is not None for dataset in datasets):
            raise ValueError("Standalone luma deployment does not support per-dataset mean/std normalization.")
    elif compact_mono:
        raise ValueError("Set dataset color to luma601 or y for this one-channel Compact, or select y for both pixel formats.")
    elif opt.output_pixel_format in ("y", "uv"):
        raise ValueError("This output pixel format requires reference channels for RGB restoration.")
    return color


def task_actions(opt: "ReduxOptions") -> Mapping[TaskAction, TaskActionSpec]:
    """Read recipe capabilities without importing models or validating data paths."""
    if opt.task is None and opt.task_options is not None:
        raise ValueError("task_options requires an explicit task.")
    name = task_name(opt)
    spec = _get_task(name)
    architecture = _task_architecture(opt, f"Task {name!r}", spec)
    actions = spec.architecture_actions.get(architecture, spec.actions)
    if name == "image_restoration" and architecture not in _windowed_architectures:
        try:
            still_restoration_luma(opt)
        except ValueError:
            return {key: value for key, value in actions.items() if key not in ("predict", "export")}
    if name == "image_restoration" and architecture in _span_architectures and _sequence_rgb_problem(opt) is not None:
        return _windowed_actions
    if (
        name == "recurrent_restoration"
        and "export" in actions
        and _recurrent_export_problem(opt, architecture) is not None
    ):
        return spec.actions
    return actions


def _recurrent_export_problem(opt: "ReduxOptions", architecture: str) -> str | None:
    if architecture == "rdual":
        return "RDUAL supports eager sequence prediction. Its flow-based ONNX path has no CPU parity verification."
    network = opt.network_g or {}
    if network.get("temporal_read", "block") not in ("block", "aligned"):
        return "RSR ONNX export requires temporal_read: block or aligned."
    if opt.scale not in (1, 2, 4):
        return "rsr_compact requires scale 1, 2 or 4."
    return _sequence_rgb_problem(opt)


def _sequence_rgb_problem(opt: "ReduxOptions") -> str | None:
    network = opt.network_g or {}
    if (opt.input_pixel_format != "rgb" or opt.output_pixel_format != "rgb"
            or network.get("num_in_ch", 3) != 3 or network.get("num_out_ch", 3) != 3):
        return "Temporal ONNX export requires three-channel RGB inputs and outputs."
    if opt.val and opt.val.tile_size:
        return "Temporal ONNX export uses full frames (val.tile_size: 0)."
    return None


def require_task_action(opt: "ReduxOptions", action: TaskAction) -> TaskActionSpec:
    """Reject an unsupported action before runtime imports or checkpoint I/O."""
    if action not in _ACTIONS:
        raise ValueError(f"Unknown task action {action!r}.")
    declared = task_actions(opt).get(action)
    if declared is None:
        name = task_name(opt)
        if name == "image_restoration" and action in ("predict", "export") and (
            str((opt.network_g or {}).get("type", "")).lower() not in _windowed_architectures
        ):
            still_restoration_luma(opt)  # Report the recipe-specific reason.
        if name == "image_restoration" and action == "export" and (opt.network_g or {}).get("type", "").lower() in _span_architectures:
            problem = _sequence_rgb_problem(opt)
            if problem is not None:
                raise ValueError(problem)
        if name == "recurrent_restoration" and action == "export":
            architecture = (opt.network_g or {}).get("type", "").lower()
            problem = _recurrent_export_problem(opt, architecture)
            if problem is not None:
                raise ValueError(problem)
        raise ValueError(
            f"Task {name!r} does not support {action!r} for this architecture."
        )
    return declared


AETHER_ARCHITECTURES = frozenset(("aether", "aether_small", "aether_medium", "aether_large"))
DYNAMIC_SPATIAL_ARCHITECTURES = AETHER_ARCHITECTURES | {"compact", "ultracompact", "superultracompact"}


def supports_dynamic_spatial_export(task: str, architecture: str) -> bool:
    return task == "image_restoration" and architecture.lower() in DYNAMIC_SPATIAL_ARCHITECTURES


def validate_qat(opt: "ReduxOptions") -> None:
    train = opt.train
    if not train or not train.qat:
        return
    net = opt.network_g or {}
    if task_name(opt) != "image_restoration" or str(net.get("type")).lower() not in AETHER_ARCHITECTURES:
        raise ValueError("train.qat requires an AetherNet image restoration generator.")
    if opt.use_amp or opt.use_compile or opt.dist or (isinstance(opt.num_gpu, int) and opt.num_gpu > 1):
        raise ValueError("train.qat requires single-device, FP32 eager generator training.")
    if not train.optim_g:
        raise ValueError("train.qat requires train.optim_g to advance calibration on applied G updates.")
    if train.ema_decay > 0 and train.ema_mode != "neosr":
        raise ValueError("train.qat requires ema_mode: neosr when EMA is enabled to copy all observer buffers.")
    if net.get("fused_init") and not (opt.path.pretrain_network_g or opt.path.resume_state or opt.auto_resume):
        raise ValueError("QAT from scratch requires network_g.fused_init: false for live normalization.")
    residual = train.qat.quantize_residual
    if residual is not None and net.get("quantize_residual", residual) != residual:
        raise ValueError("train.qat.quantize_residual and network_g.quantize_residual disagree.")
    train.qat.schedule(train.total_iter)


def _validate_still_luma(opt: "ReduxOptions") -> None:
    modes = {key: ds.color for key, ds in opt.datasets.items() if ds.color}
    if not modes or task_name(opt) not in ("image_restoration", "recurrent_restoration", "autoencoder"):
        return
    network = opt.network_ae or opt.network_g or {}
    architecture = str(network.get("type", "")).lower()
    for key in modes:
        ds = opt.datasets[key]
        if str(ds.type).lower() in ("pairedvideodataset", "realesrganvideodataset") or architecture in (
            *recurrent_architectures, *_windowed_architectures
        ):
            raise ValueError(f"datasets.{key}.color: sequence restoration requires RGB frames.")
        if any(values is not None and len(values) != 1 for values in (ds.mean, ds.std)):
            raise ValueError(f"datasets.{key}.color: one-channel data needs one mean and one standard deviation when normalized.")
    if opt.input_pixel_format != "rgb" or opt.output_pixel_format != "rgb":
        raise ValueError("Dataset luma already supplies one channel. The RGB-to-Y/UV model adapters require an RGB dataset.")
    if architecture in ("compact", "span"):
        if network.get("num_in_ch", 3) != 1 or network.get("num_out_ch", 3) != 1:
            raise ValueError(f"One-channel datasets require num_in_ch: 1 and num_out_ch: 1 for {architecture}.")
    if architecture in AETHER_ARCHITECTURES and network.get("in_chans", 3) != 1:
        raise ValueError(f"One-channel datasets require in_chans: 1 for {architecture}.")
    if "train" in modes and opt.is_train is not False and opt.train:
        for loss in opt.train.losses or ():
            kind = str(loss.get("type", "")).lower()
            if kind == "neosrconsistencyloss" or kind == "neosrfdlloss" and loss.get("model", "dinov2") == "dinov2":
                raise ValueError(f"{kind} requires RGB training images. Select a single-channel objective for dataset luma.")


def resolve_task(
    opt: "ReduxOptions",
) -> tuple[TrainingTaskSpec, str, msgspec.Struct | None] | None:
    """Validate a selected task and decode its options without loading the model."""
    _validate_still_luma(opt)
    if opt.val and opt.val.complementary_planes == "input_bicubic":
        if (task_name(opt) != "image_restoration" or opt.output_pixel_format not in ("y", "uv")
                or str((opt.network_g or {}).get("type", "")).lower() in _windowed_architectures):
            raise ValueError("val.complementary_planes: input_bicubic requires a still-image Y-only or UV-only generator output.")
    if opt.val and opt.val.training_precision != "inherit" and task_name(opt) != "image_restoration":
        raise ValueError("val.training_precision applies to still-image restoration training.")
    if opt.val and opt.val.tile_mode == "context" and opt.val.tile_size:
        if task_name(opt) != "image_restoration" or not 0 <= 2 * opt.val.tile_overlap < opt.val.tile_size:
            raise ValueError("Context tiles require still-image restoration and a tile_size greater than twice the nonnegative tile_overlap.")
    if opt.is_train is not False and opt.high_order_degradation:
        from trainner_config.resampling import NEOSR_AA_MODES, NEOSR_RESIZE_MODES

        modes = set((*opt.resize_mode_list, *opt.resize_mode_list2, *opt.resize_mode_list3))
        if opt.otf_resize_policy == "neosr":
            if modes.difference(NEOSR_RESIZE_MODES):
                raise ValueError("Choose a NeoSR-compatible resize mode for all three OTF stages.")
        elif modes.intersection(NEOSR_AA_MODES):
            raise ValueError("The aa_ resize kernels require otf_resize_policy: neosr.")
    if opt.is_train is not False and opt.neosr_samples is not None:
        if (not opt.high_order_degradation or opt.otf_resize_policy != "neosr" or
                str(getattr(opt.datasets.get("train"), "type", "")).lower() != "realesrgandataset"):
            raise ValueError("neosr_samples requires the NeoSR resize policy and on-the-fly still-image training.")
    if opt.is_train is not False and opt.train and opt.train.use_moa:
        from trainner_config.neosr_batchaug import validate_training_batchaug

        validate_training_batchaug(opt, task_name(opt))
    if opt.is_train is not False and opt.train and opt.train.qat:
        validate_qat(opt)
    if opt.is_train is not False and opt.train and opt.train.grad_clip == "spike":
        if task_name(opt) not in ("image_restoration", "recurrent_restoration"):
            raise ValueError("train.grad_clip 'spike' requires image or recurrent restoration training.")
    if opt.task is None:
        if opt.task_options is not None:
            raise ValueError("task_options requires an explicit task.")
        return None

    spec = _get_task(opt.task)
    task = f"task {opt.task!r}"
    architecture = _task_architecture(opt, task, spec)

    if spec.training_only and opt.is_train is False:
        raise ValueError(f"{task} only supports training.")
    if (
        spec.otf_degradation == "required"
        and opt.is_train is not False
        and not opt.high_order_degradation
    ):
        raise ValueError(f"{task} requires high_order_degradation: true.")
    if spec.otf_degradation == "unsupported" and opt.high_order_degradation:
        raise ValueError(f"{task} requires high_order_degradation: false.")

    settings = None
    if spec.options_type is not None:
        try:
            settings = msgspec.convert(opt.task_options or {}, type=spec.options_type)
        except msgspec.ValidationError as exc:
            raise ValueError(f"Invalid task_options for {task}. {exc}") from exc
    elif opt.task_options is not None:
        raise ValueError(f"{task} has no task_options.")

    if spec.validate is not None:
        spec.validate(opt, architecture, settings)
    return spec, architecture, settings


def validate_task_selection(opt: "ReduxOptions") -> str | None:
    """Return the normalized architecture of an explicit task."""
    task = resolve_task(opt)
    return task[1] if task is not None else None


recurrent_architectures = ("rdual", "rsr_compact")
# The existing center-window architectures deploy through the sequence path.
_windowed_architectures = ("tscunet", "temporalspan", "temporalspanv2", "tfdat")
_span_architectures = ("temporalspan", "temporalspanv2")
_sequence_export = TaskActionSpec(("sequences",), "traiNNer.tasks.sequence_onnx:export_sequence_onnx")
_windowed_actions: dict[TaskAction, TaskActionSpec] = {
    "train": TaskActionSpec(("sequence_pairs", "sequences")),
    "validate": TaskActionSpec(("sequence_pairs", "sequences"), snapshot_metrics=configured_snapshot_metrics),
    "holdout": TaskActionSpec(
        ("sequence_pairs", "sequences"), "traiNNer.evaluation:evaluate", "reference_scores", ("saveImages", "saveTemporalErrors")
    ),
    "predict": TaskActionSpec(("sequences",), "traiNNer.tasks.predict:_predict_frames", "sequence"),
}
_image_restoration_actions: dict[TaskAction, TaskActionSpec] = {
    "train": TaskActionSpec(("image_pairs", "images")),
    "validate": TaskActionSpec(("image_pairs", "images"), snapshot_metrics=configured_snapshot_metrics),
    "holdout": TaskActionSpec(
        ("image_pairs", "images"), "traiNNer.evaluation:evaluate", "reference_scores"
    ),
    "predict": TaskActionSpec(("images",), "traiNNer.tasks.predict:_predict_images"),
    "export": TaskActionSpec(("images",), "traiNNer.tasks.predict:_export_image_onnx"),
}
register_task(
    "image_restoration",
    TrainingTaskSpec(
        "network_g",
        "traiNNer.models:_build_image_model",
        excluded_architecture_types=recurrent_architectures,
        actions=_image_restoration_actions,
        architecture_actions={
            **dict.fromkeys(_windowed_architectures, _windowed_actions),
            **dict.fromkeys(_span_architectures, {**_windowed_actions, "export": _sequence_export}),
        },
    ),
)
register_task(
    "autoencoder",
    TrainingTaskSpec(
        "network_ae",
        "traiNNer.models:_build_autoencoder",
        ("autoencoder",),
        otf_degradation="unsupported",
        actions={
            **_image_restoration_actions,
            "train": TaskActionSpec(("image_pairs",)),
        },
    ),
)
_recurrent_actions: dict[TaskAction, TaskActionSpec] = {
    **_windowed_actions,
    "train": TaskActionSpec(("sequences",)),
}
register_task(
    "recurrent_restoration",
    TrainingTaskSpec(
        "network_g",
        "traiNNer.models:_build_recurrent_model",
        recurrent_architectures,
        otf_degradation="required",
        actions=_recurrent_actions,
        architecture_actions={
            "rsr_compact": {
                **_recurrent_actions,
                "export": _sequence_export,
            },
        },
    ),
)

for module in sorted(
    iter_modules(trainner_config.__path__), key=lambda item: item.name
):
    if module.name.endswith("_task"):
        import_module(f"{trainner_config.__name__}.{module.name}")
for entry_point in entry_points(group="trainner.training_tasks"):
    entry_point.load()()

"""Loss slots and accepted-update schedules shared by the trainer and preflight."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

import msgspec

if TYPE_CHECKING:
    from trainner_config.options import ReduxOptions, TrainOptions

_LEGACY_LOSS_FIELDS = (
    "pixel_opt",
    "mssim_opt",
    "ms_ssim_l1_opt",
    "perceptual_opt",
    "contextual_opt",
    "dists_opt",
    "hr_inversion_opt",
    "dinov2_opt",
    "topiq_opt",
    "pd_opt",
    "fd_opt",
    "ldl_opt",
    "hsluv_opt",
    "gan_opt",
    "color_opt",
    "luma_opt",
    "avg_opt",
    "bicubic_opt",
)
ModelKind = Literal["sr", "clip", "ae"]
_TASK_MODELS: dict[str, ModelKind] = {
    "image_restoration": "sr",
    "recurrent_restoration": "clip",
    "autoencoder": "ae",
}
_CLIP_ARCHITECTURES = {"rdual", "rsr_compact"}
_SPECIAL_LOSSES = {"gan", "ldl", "vosr", "multiscalegan"}
_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


class ObjectiveConfigError(ValueError):
    """An invalid objective configuration at a recipe path."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


class LossSchedule(msgspec.Struct, frozen=True):
    start_iter: Annotated[
        int, msgspec.Meta(ge=0, description="Exclusive first update boundary.")
    ] = 0
    ramp_iters: Annotated[
        int, msgspec.Meta(ge=0, description="Updates to reach full loss weight.")
    ] = 0
    end_iter: Annotated[
        Annotated[int, msgspec.Meta(ge=1)] | None,
        msgspec.Meta(description="Inclusive last update. Null keeps the loss active."),
    ] = None

    def factor(self, candidate_iter: int) -> float:
        elapsed = candidate_iter - self.start_iter
        if elapsed <= 0 or (
            self.end_iter is not None and candidate_iter > self.end_iter
        ):
            return 0.0
        if elapsed < self.ramp_iters:
            return elapsed / self.ramp_iters
        return 1.0


_SCHEDULE_FIELDS = frozenset(LossSchedule.__struct_fields__)
_METADATA_FIELDS = _SCHEDULE_FIELDS | {"name", "loss_reference"}


@dataclass(frozen=True, slots=True)
class Objective:
    options: dict[str, Any]
    schedule: LossSchedule | None
    path: str
    neosr_lq: bool = False


def _kind(loss_type: str) -> str:
    # Keep the existing trainer labels for entries without a custom name.
    return loss_type.lower().replace("loss", "")


def uses_gan_loss(losses: list[dict] | None, gan_opt: dict | None) -> bool:
    """Match the discriminator that the training objective will construct."""
    entries = losses if losses is not None else ([gan_opt] if gan_opt else [])
    return any(
        _kind(str(entry.get("type", ""))) == "gan" and entry.get("loss_weight", 0) != 0
        for entry in entries
    )


def validate_supervised_training(train: TrainOptions, objective: str) -> None:
    """Reject settings the built-in supervised providers cannot apply."""
    for field in ("losses", *_LEGACY_LOSS_FIELDS, "use_moa", "distillation", "vosr_target", "optim_d", "optim_ae"):
        if getattr(train, field):
            raise ValueError(f"Remove train.{field}. This task uses {objective} and train.optim_g.")
    if not 0 <= train.ema_decay < 1:
        raise ValueError("train.ema_decay must be in [0, 1).")
    if train.grad_clip and (not math.isfinite(train.grad_clip_max_norm) or train.grad_clip_max_norm < 0):
        raise ValueError("train.grad_clip_max_norm must be finite and nonnegative.")


def _model_kind(opt: ReduxOptions) -> ModelKind:
    if opt.task is not None and opt.task in _TASK_MODELS:
        return _TASK_MODELS[opt.task]
    network = opt.network_g or {}
    if str(network.get("type", "")).lower() in _CLIP_ARCHITECTURES:
        return "clip"
    if opt.network_ae and not opt.network_g and not opt.high_order_degradation:
        return "ae"
    return "sr"


def _parse_schedule(raw: dict[str, Any], path: str) -> LossSchedule | None:
    options = {key: raw[key] for key in _SCHEDULE_FIELDS if key in raw}
    if not options:
        return None
    try:
        schedule = msgspec.convert(options, type=LossSchedule)
    except msgspec.ValidationError as exc:
        message, _, suffix = str(exc).partition(" - at `$.")
        field = suffix.removesuffix("`")
        invalid_path = f"{path}.{field}" if field else path
        raise ObjectiveConfigError(invalid_path, message) from exc
    if schedule.end_iter is not None and schedule.end_iter <= schedule.start_iter:
        raise ObjectiveConfigError(
            f"{path}.end_iter", "end_iter must be greater than start_iter."
        )
    return schedule


def _loss_label(kind: str, raw: dict[str, Any], path: str, model: ModelKind) -> str:
    prefix = "ae" if model == "ae" else "g"
    if "name" not in raw:
        return f"l_{prefix}_{kind}"
    name = raw["name"]
    if not isinstance(name, str) or not re.fullmatch(_NAME_PATTERN, name):
        raise ObjectiveConfigError(
            f"{path}.name",
            "Use a loss name with letters, digits, underscores or hyphens.",
        )
    name = name.lower()
    if (
        kind in _SPECIAL_LOSSES
        or name in _SPECIAL_LOSSES | {"total"}
        or name.startswith("distill_")
    ):
        raise ObjectiveConfigError(
            f"{path}.name",
            "Use a unique ordinary loss name. Special loss slots are reserved.",
        )
    return f"l_{prefix}_{name}"


def _check_coverage(
    objectives: dict[str, Objective], opt: ReduxOptions, model: ModelKind
) -> None:
    train = opt.train
    assert train is not None
    if train.total_iter <= 0:
        raise ObjectiveConfigError("train.total_iter", "Set a positive update count.")
    if model == "ae" and train.optim_ae is None:
        raise ObjectiveConfigError(
            "train.optim_ae", "Scheduled AE training requires train.optim_ae."
        )
    if model == "clip" and train.optim_g is None:
        raise ObjectiveConfigError(
            "train.optim_g", "Scheduled clip training requires train.optim_g."
        )

    intervals = []
    for label, objective in objectives.items():
        if model != "ae" and train.optim_g is None and label != "l_g_gan":
            if objective.schedule is not None:
                raise ObjectiveConfigError(
                    objective.path,
                    "Scheduling this generator loss requires train.optim_g.",
                )
            continue
        # VOSR contributes only alongside GAN, which already covers that interval.
        if label == "l_g_vosr":
            continue
        schedule = objective.schedule
        first = schedule.start_iter + 1 if schedule is not None else 1
        last = (
            min(train.total_iter, schedule.end_iter)
            if schedule is not None and schedule.end_iter is not None
            else train.total_iter
        )
        if first <= last:
            intervals.append((first, last))

    next_update = 1
    for first, last in sorted(intervals):
        if first > next_update:
            break
        next_update = max(next_update, last + 1)
    if next_update <= train.total_iter:
        raise ObjectiveConfigError(
            "train.losses",
            f"No configured objective can update an optimizer at update {next_update}. "
            "Cover every planned update with an active ordinary, LDL, or GAN loss. "
            "VOSR and auxiliary objectives cannot cover a schedule gap on their own.",
        )


def validate_buffer_rollback(opt: ReduxOptions, model: ModelKind | None = None) -> None:
    """Admission at recipe decode and again after launcher/device resolution."""
    path = "train.rollback_buffers_on_reject"
    if opt.task not in (None, "image_restoration") or (model or _model_kind(opt)) != "sr":
        raise ObjectiveConfigError(path, f"{path} requires the still-image SR or OTF update loop.")
    if opt.dist or (isinstance(opt.num_gpu, int) and opt.num_gpu > 1):
        raise ObjectiveConfigError(path, f"{path} requires single-device training.")
    if opt.train is not None and opt.train.qat is not None:
        raise ObjectiveConfigError(path, f"{path} cannot be combined with Aether QAT, which owns its observer and norm rollback.")


def resolve_objectives(
    opt: ReduxOptions, *, model: ModelKind | None = None
) -> dict[str, Objective]:
    """Validate effective loss slots and return the options accepted by factories."""
    train = opt.train
    if train is None:
        return {}

    if train.rollback_buffers_on_reject:
        validate_buffer_rollback(opt, model)

    if train.ema_mode == "neosr":
        supervised = opt.task in ("image_classification", "semantic_segmentation", "object_detection", "tabular_regression")
        if not supervised and (
            (model or _model_kind(opt)) != "sr" or opt.task not in (None, "image_restoration")
        ):
            raise ObjectiveConfigError("train.ema_mode", "NeoSR EMA requires image restoration or supervised training.")
        if not 0 < train.ema_decay < 1:
            raise ObjectiveConfigError("train.ema_decay", "NeoSR EMA requires decay between 0 and 1.")
        if train.ema_power is not None and (not math.isfinite(train.ema_power) or train.ema_power <= 0):
            raise ObjectiveConfigError("train.ema_power", "NeoSR EMA power must be positive and finite.")
        if train.ema_switch_iter or train.ema_update_after_step:
            raise ObjectiveConfigError("train.ema_mode", "NeoSR EMA starts on the first G update and does not switch weights.")

    if train.loss_output_clamp != "none" or train.gan_wavelet_start_iter is not None:
        if (model or _model_kind(opt)) != "sr" or opt.task not in (
            None,
            "image_restoration",
        ):
            field = (
                "loss_output_clamp"
                if train.loss_output_clamp != "none"
                else "gan_wavelet_start_iter"
            )
            raise ObjectiveConfigError(
                f"train.{field}", "This option requires image restoration training."
            )

    legacy = [
        (f"train.{field}", loss)
        for field in _LEGACY_LOSS_FIELDS
        if (loss := getattr(train, field)) is not None
    ]
    if train.losses is not None:
        for path, raw in legacy:
            if _METADATA_FIELDS.intersection(raw):
                raise ObjectiveConfigError(
                    path,
                    "Move this loss to train.losses. That list takes precedence over legacy fields.",
                )
        entries = [
            (f"train.losses[{index}]", raw) for index, raw in enumerate(train.losses)
        ]
    else:
        entries = legacy

    scheduled = any(_SCHEDULE_FIELDS.intersection(raw) for _, raw in entries)
    if opt.task is not None and opt.task not in _TASK_MODELS:
        for path, raw in entries:
            if _METADATA_FIELDS.intersection(raw):
                raise ObjectiveConfigError(
                    path,
                    "Loss scheduling and names are supported by image_restoration, "
                    "recurrent_restoration and autoencoder.",
                )
        if model is None:
            return {}
    model = model or _model_kind(opt)
    if scheduled and opt.dist:
        raise ObjectiveConfigError(
            "train.losses",
            "Loss scheduling requires a single training process. The trainer does not "
            "coordinate rejected candidate updates across distributed ranks.",
        )

    objectives: dict[str, Objective] = {}
    for path, raw in entries:
        if model == "ae" and path == "train.gan_opt":
            if _METADATA_FIELDS.intersection(raw):
                raise ObjectiveConfigError(
                    path, "Autoencoder training has no GAN loss slot."
                )
            continue
        loss_type = raw.get("type")
        if not isinstance(loss_type, str) or not loss_type:
            raise ObjectiveConfigError(f"{path}.type", "Specify the loss type.")
        weight = raw.get("loss_weight")
        try:
            valid_weight = (
                isinstance(weight, int | float)
                and not isinstance(weight, bool)
                and math.isfinite(weight)
            )
        except OverflowError:
            valid_weight = False
        if not valid_weight:
            raise ObjectiveConfigError(
                f"{path}.loss_weight", "Specify a finite numeric loss weight."
            )

        kind = _kind(loss_type)
        reference = raw.get("loss_reference")
        if "loss_reference" in raw and (
            reference != "neosr_lq" or model != "sr" or kind in _SPECIAL_LOSSES or weight < 0
        ):
            raise ObjectiveConfigError(
                f"{path}.loss_reference",
                "neosr_lq requires an ordinary image restoration loss with a nonnegative weight.",
            )
        schedule = _parse_schedule(raw, path)
        label = _loss_label(kind, raw, path, model)
        if weight == 0:
            continue
        if model == "ae":
            if weight < 0 and scheduled:
                raise ObjectiveConfigError(
                    f"{path}.loss_weight",
                    "Autoencoder scheduling requires a positive loss weight.",
                )
            if weight < 0:
                continue
            if kind in _SPECIAL_LOSSES:
                raise ObjectiveConfigError(
                    path, f"Autoencoder training does not support {loss_type}."
                )
        if kind == "ldl" and model == "clip":
            raise ObjectiveConfigError(
                path, f"Recurrent clip training does not support {loss_type}."
            )
        if kind == "multiscalegan" and scheduled:
            raise ObjectiveConfigError(path, "Scheduled GAN training requires GANLoss.")
        if label in objectives:
            raise ObjectiveConfigError(
                path,
                f"Duplicate loss slot {label!r}. It is already defined at {objectives[label].path}.",
            )
        options = {
            key: value for key, value in raw.items() if key not in _METADATA_FIELDS
        }
        objectives[label] = Objective(options, schedule, path, reference == "neosr_lq")

    if train.gan_wavelet_start_iter is not None and "l_g_gan" not in objectives:
        raise ObjectiveConfigError(
            "train.gan_wavelet_start_iter",
            "The sym19 view requires an active GAN loss.",
        )

    if scheduled:
        if train.losses is not None and train.gan_opt:
            raise ObjectiveConfigError(
                "train.gan_opt",
                "Remove legacy gan_opt when scheduling train.losses. "
                "Discriminator checkpoint discovery gives gan_opt precedence.",
            )
        gan = objectives.get("l_g_gan")
        if gan is not None:
            if gan.options["loss_weight"] < 0:
                raise ObjectiveConfigError(
                    f"{gan.path}.loss_weight",
                    "Scheduled GAN loss requires a positive loss_weight for "
                    "discriminator checkpoint discovery.",
                )
            if opt.network_d is None:
                raise ObjectiveConfigError("network_d", "GAN loss requires network_d.")
            if train.optim_d is None:
                raise ObjectiveConfigError(
                    "train.optim_d", "GAN loss requires train.optim_d."
                )
        if (ldl := objectives.get("l_g_ldl")) is not None and train.ema_decay <= 0:
            raise ObjectiveConfigError(
                ldl.path, "LDL requires a positive train.ema_decay."
            )
        if (vosr := objectives.get("l_g_vosr")) is not None:
            if vosr.options["loss_weight"] < 0:
                raise ObjectiveConfigError(
                    f"{vosr.path}.loss_weight", "VOSR requires a positive loss weight."
                )
            if str((opt.network_g or {}).get("type", "")).lower() == "rsr_compact":
                raise ObjectiveConfigError(
                    vosr.path, "RSR clip training does not support VOSR loss."
                )
            if gan is None or gan.options.get("gan_type", "vanilla") not in (
                "vanilla",
                "lsgan",
                "wgan_softplus",
            ):
                raise ObjectiveConfigError(
                    vosr.path,
                    "VOSR needs a GANLoss using vanilla, lsgan, or wgan_softplus.",
                )
        _check_coverage(objectives, opt, model)
    return objectives


def validate_objectives(opt: ReduxOptions) -> None:
    """Validate a training recipe without importing torch or constructing losses."""
    resolve_objectives(opt)


def objective_schedules(
    objectives: dict[str, Objective],
) -> dict[str, LossSchedule] | None:
    return {
        label: objective.schedule
        for label, objective in objectives.items()
        if objective.schedule is not None
    } or None


def add_loss_schedule_schema(schema: dict[str, Any]) -> None:
    """Expose trainer-owned loss metadata on generated loss schema entries."""
    definitions = schema["$defs"]
    train_properties = definitions["TrainOptions"]["properties"]
    loss_items = train_properties["losses"]["items"]
    schedule = msgspec.json.schema(LossSchedule)["$defs"]["LossSchedule"]["properties"]
    name = {
        "type": "string",
        "pattern": _NAME_PATTERN,
        "description": "Unique custom name for an ordinary loss.",
    }
    refs = loss_items.get("anyOf", loss_items.get("oneOf", []))
    for item in refs:
        loss_type = item["$ref"].rsplit("/", 1)[1]
        properties = definitions[loss_type]["properties"]
        properties.update(schedule)
        if _kind(loss_type) not in _SPECIAL_LOSSES:
            properties["name"] = name
            properties["loss_reference"] = {
                "const": "neosr_lq",
                "description": "Image restoration only. Use the bicubic antialiased float32 LQ image at output size, clamped to [1/255, 1].",
            }
    for field in _LEGACY_LOSS_FIELDS:
        # Legacy slots construct the same registered losses and accept their options.
        train_properties[field]["anyOf"][0] = {"$ref": "#/$defs/TrainOptions/properties/losses/items"}

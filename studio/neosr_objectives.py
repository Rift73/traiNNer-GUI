"""Translate the training objectives of NeoSR's pinned compact recipes."""

import math
from collections.abc import Callable
from typing import Any

# Constructor defaults at NeoSR 3fe738f4. The native port exposes this kernel.
_MSSIM_DEFAULTS = {
    "window_size": 11,
    "in_channels": 3,
    "sigma": 1.5,
    "K1": 0.01,
    "K2": 0.03,
    "L": 1,
    "padding": 5,  # NeoSR's omitted padding is window_size // 2.
}


def _loss(cfg: dict[str, Any]) -> dict[str, Any]:
    options = dict(cfg)
    target = options.pop("target", "gt")
    if target not in ("gt", "lq"):
        raise ValueError("NeoSR loss targets are gt or lq.")
    if "perceptual_weight" in options:
        if "loss_weight" in options:
            raise ValueError("Specify one loss weight for this objective.")
        options["loss_weight"] = options.pop("perceptual_weight")
    schedule = {key: options.pop(key) for key in ("start_iter", "ramp_iters", "end_iter") if key in options}
    if any(type(v) is not int or v < 0 for v in schedule.values()) or (
        "end_iter" in schedule and schedule["end_iter"] <= schedule.get("start_iter", 0)
    ):
        raise ValueError("Loss schedules require nonnegative updates and end_iter greater than start_iter.")
    result = _loss_kernel(options)
    # NeoSR's adversarial terms always take real samples from ctx.gt.
    if target == "lq" and result["type"] != "ganloss":
        result["loss_reference"] = "neosr_lq"
    return {**result, **schedule}


def _loss_kernel(cfg: dict[str, Any]) -> dict[str, Any]:
    kind = str(cfg.get("type", "")).lower()
    weight = cfg.get("loss_weight", 0.1 if kind == "gan_loss" else 1.0)
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError(
            "Use a nonnegative numeric weight. Native SR uses negative weights to select an LQ reference."
        )
    params = {
        k: v for k, v in cfg.items() if k not in ("type", "loss_weight", "target")
    }

    if kind in ("l1loss", "mseloss"):
        if params.keys() <= {"reduction"} and params.get("reduction", "mean") in (
            "mean",
            "sum",
        ):
            return {"type": kind, "loss_weight": weight, **params}
    elif kind == "fdl_loss":
        allowed = {"model", "patch_size", "num_proj", "stride", "vgg_weights", "dino_layers",
                   "dino_weights", "dino_norm", "phase_weight", "seed", "fp32_backbone", "checkpoint"}
        if (params.keys() <= allowed and isinstance(params.get("model", "dinov2"), str)
                and params.get("model", "dinov2").lower() in {"vgg", "dinov2", "resnet", "effnet"}):
            return {"type": "neosrfdlloss", "loss_weight": weight, **params}
    elif kind == "ldl_loss":
        ksize = params.get("ksize", 7)
        if (
            params.keys() <= {"criterion", "ksize"}
            and params.get("criterion", "l1") in ("l1", "l2", "huber", "chc")
            and type(ksize) is int
            and ksize >= 3
            and ksize % 2 == 1
        ):
            return {"type": "neosrldlloss", "loss_weight": weight, **params}
        raise ValueError(
            "NeoSR LDL accepts l1, l2, huber or chc and an odd ksize of at least 3."
        )
    elif kind == "consistency_loss":
        if (
            params.keys() <= {"criterion", "blur", "cosim", "saturation", "brightness"}
            and params.get("criterion", "chc") in ("chc", "l1")
            and all(type(params.get(key, True)) is bool for key in ("blur", "cosim"))
            and all(
                type(params.get(key, 1)) in (int, float)
                for key in ("saturation", "brightness")
            )
        ):
            return {"type": "neosrconsistencyloss", "loss_weight": weight, **params}
        raise ValueError(
            "NeoSR consistency accepts chc or l1, boolean blur/cosim, and numeric saturation/brightness."
        )
    elif kind == "mssim_loss":
        if all(
            key in _MSSIM_DEFAULTS
            and type(value) in (int, float)
            and (
                key not in ("window_size", "in_channels", "padding")
                or type(value) is int
            )
            and value == _MSSIM_DEFAULTS[key]
            for key, value in params.items()
        ):
            return {"type": "neosrmssimloss", "loss_weight": weight}
        raise ValueError(
            "The registered NeoSR MS-SSIM objective implements the default RGB Gaussian and five scales. Review the supplied options."
        )
    elif kind == "gan_loss":
        aliases = {"bce": "vanilla", "mse": "lsgan", "hinge": "hinge"}
        gan = params.get("gan_type", "bce")
        if not weight:
            raise ValueError(
                "NeoSR still steps the discriminator when the generator GAN weight is zero."
            )
        if gan == "hinge" and (
            params.get("real_label_val", 1),
            params.get("fake_label_val", 0),
        ) != (1, 0):
            raise ValueError(
                "NeoSR hinge GAN requires its default real and fake labels."
            )
        if (
            isinstance(gan, str)
            and gan in aliases
            and params.keys() <= {"gan_type", "real_label_val", "fake_label_val"}
        ):
            return {
                "type": "ganloss",
                "loss_weight": weight,
                **params,
                "gan_type": aliases[gan],
            }
    raise ValueError(
        "Review the source loss type, reduction, schedule and options before choosing a native objective."
    )


def import_objectives(
    train: dict[str, Any],
    training: dict[str, Any],
    unresolved: Callable[[str, Any, str], None],
) -> None:
    """Consume source objective keys and append mappings or original review values."""
    clamp = train.pop("clamp", "restore")
    if clamp == "restore" or clamp is True:
        training["loss_output_clamp"] = "restore" if clamp == "restore" else "hard"
    elif clamp is not False:
        unresolved(
            "train.clamp",
            clamp,
            "Choose NeoSR restoring clamp ('restore'), hard clamp (true), or raw output (false).",
        )

    wavelet = train.pop("wavelet_guided", False)
    has_wavelet_start = "wavelet_init" in train
    wavelet_start = train.pop("wavelet_init", 0)
    if type(wavelet) is not bool:
        unresolved("train.wavelet_guided", wavelet, "wavelet_guided must be a boolean.")
        if has_wavelet_start:
            unresolved(
                "train.wavelet_init",
                wavelet_start,
                "Resolve the wavelet switch before applying this update.",
            )
    if wavelet is True and (type(wavelet_start) is not int or wavelet_start < 0):
        unresolved(
            "train.wavelet_init",
            wavelet_start,
            "Choose a nonnegative integer update for the sym19 GAN view.",
        )
    gan_slots = [
        key
        for key, value in train.items()
        if key.endswith("_opt")
        and isinstance(value, dict)
        and str(value.get("type", "")).lower() == "gan_loss"
    ]
    if wavelet is True and "gan_opt" not in gan_slots:
        unresolved(
            "train.wavelet_guided",
            wavelet,
            "NeoSR applies the global wavelet switch to train.gan_opt. Select the intended GAN objective.",
        )
        unresolved(
            "train.wavelet_init",
            wavelet_start,
            "Confirm the first wavelet update for the selected GAN.",
        )
    match = train.pop("match_lq_colors", False)
    if type(match) is not bool:
        unresolved(
            "train.match_lq_colors",
            match,
            "Set the default low-quality color reference using a boolean.",
        )

    losses = []
    for slot in list(train):
        if not slot.endswith("_opt"):
            continue
        cfg = train.pop(slot)
        if not isinstance(cfg, dict):
            unresolved("train." + slot, cfg, "A NeoSR objective must be a TOML table.")
        elif cfg:
            try:
                if slot not in gan_slots:
                    if match is True and slot in ("consistency_opt", "msswd_opt"):
                        cfg = {"target": "lq", **cfg}
                    losses.append(_loss(cfg))
                    continue
                if len(gan_slots) != 1:
                    raise ValueError(
                        "The native image trainer has one discriminator objective. Select the intended GAN and input view."
                    )
                # NeoSR's named GAN slot inherits the global switch. An explicit
                # GAN-local start takes precedence and can activate the view alone.
                gan_cfg = dict(cfg)
                start = gan_cfg.pop(
                    "wavelet_init",
                    wavelet_start if slot == "gan_opt" and wavelet is True else None,
                )
                if start is not None and (type(start) is not int or start < 0):
                    raise ValueError(
                        "wavelet_init must be a nonnegative integer update."
                    )
                losses.append(_loss(gan_cfg))
                if start is not None:
                    training["gan_wavelet_start_iter"] = start
            except ValueError as exc:
                unresolved("train." + slot, cfg, str(exc))
                if slot == "gan_opt" and wavelet is True:
                    unresolved(
                        "train.wavelet_guided",
                        wavelet,
                        "Resolve the GAN objective before applying its wavelet input view.",
                    )
                    unresolved(
                        "train.wavelet_init",
                        wavelet_start,
                        "Retain this update boundary for the selected GAN.",
                    )
    if losses:
        training["losses"] = losses

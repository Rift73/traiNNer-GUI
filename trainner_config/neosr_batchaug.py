"""Translate the paired NeoSR train dataset's batch augmentation fields without torch."""

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trainner_config.options import ReduxOptions

# The pinned dispatcher applies selected modes in this order, once per distinct name.
NEOSR_BATCH_ORDER = ("cutmix", "mixup", "resizemix", "cutblur")
NEOSR_BATCH_FIELDS = ("augmentation", "aug_prob", "aug_multi_prob", "aug_antialias")


def validate_training_batchaug(opt: "ReduxOptions", task: str) -> None:
    train = opt.train
    assert train is not None
    if task in ("recurrent_restoration", "image_classification", "semantic_segmentation", "object_detection"):
        raise ValueError(f"train.use_moa needs aligned image batches. Task {task!r} has another sample layout.")
    if task not in ("autoencoder", "image_restoration"):
        return  # An installed provider owns its augmentation contract.
    dataset = opt.datasets.get("train")
    size = dataset.batch_size_per_gpu if dataset is not None else None
    if train.moa_mode == "neosr":
        slots = paired_batchaug_slots(train.moa_augs, train.moa_probs, train.moa_multi_prob, train.moa_antialias)
        if size == 1 and any(weight > 0 and mode not in ("none", "cutblur") for mode, weight in slots):
            raise ValueError("NeoSR partner mixing needs batch_size_per_gpu above one, or only none and cutblur with positive weights.")
    else:
        weights = train.moa_probs
        if len(weights) != len(train.moa_augs) or not all(isfinite(w) and w >= 0 for w in weights) or not 0 < sum(weights) < float("inf"):
            raise ValueError("train.moa_probs needs one finite nonnegative weight per augmentation and a positive finite total.")
        if any(mode not in ("none", "downup", "up", *NEOSR_BATCH_ORDER) and w for mode, w in zip(train.moa_augs, weights, strict=True)):
            raise ValueError("train.moa_augs contains an unknown native augmentation with a positive weight.")
        if size == 1:
            raise ValueError("Native batch augmentation requires batch_size_per_gpu above one.")


def paired_batchaug_slots(
    modes: object, weights: object, multi: object, antialias: object
) -> list[tuple[str, float]]:
    """Validate source fields and retain slot order, duplicates and zero weights."""
    if (
        not isinstance(multi, (float, int))
        or isinstance(multi, bool)
        or not 0 <= multi <= 1
    ):
        raise ValueError("datasets.train.aug_multi_prob must be a number in [0, 1].")
    if type(antialias) is not bool:
        raise ValueError("datasets.train.aug_antialias must be a boolean.")
    if weights is not None and not isinstance(weights, list):
        raise ValueError("datasets.train.aug_prob must be a list of relative weights.")
    # The model bypasses apply_augment (including its weights check and RNG draw).
    if modes is None or modes == ["none"]:
        return []
    if not isinstance(modes, list) or not modes:
        raise ValueError(
            "datasets.train.augmentation must be a nonempty list of modes."
        )
    for name in modes:
        if name not in ("none", *NEOSR_BATCH_ORDER):
            raise ValueError(f"Unknown datasets.train.augmentation mode {name!r}.")
    if weights is None:
        weights = [1.0] * len(modes)
    if len(weights) != len(modes):
        raise ValueError("datasets.train.aug_prob needs one weight per augmentation.")
    if any(type(w) not in (float, int) or not isfinite(w) or w < 0 for w in weights):
        raise ValueError(
            "datasets.train.aug_prob weights must be finite and nonnegative."
        )
    total = sum(float(weight) for weight in weights)
    if not isfinite(total) or total <= 0:
        raise ValueError("datasets.train.aug_prob must have a finite positive sum.")
    return [(mode, float(weight)) for mode, weight in zip(modes, weights, strict=True)]


def map_paired_batchaug(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields for a fresh train section. An empty mapping leaves MoA disabled."""
    multi = dataset.get("aug_multi_prob", 0.3)
    antialias = dataset.get("aug_antialias", False)
    slots = paired_batchaug_slots(
        dataset.get("augmentation"), dataset.get("aug_prob"), multi, antialias
    )
    if not slots:
        return {}
    return {
        "use_moa": True,
        "moa_mode": "neosr",
        "moa_augs": [mode for mode, _ in slots],
        "moa_probs": [weight for _, weight in slots],
        "moa_multi_prob": float(multi),
        "moa_antialias": antialias,
    }

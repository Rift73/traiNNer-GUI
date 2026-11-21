"""Identify the ordered supervised data used for strict RNG replay."""

from __future__ import annotations

import hashlib
import os
import struct

from trainner_config.devices import PAIRED_WORKER_AUGMENTATION, replay_worker_mode
from trainner_config.tasks import resolve_task
from trainner_data.paired_identity import disk_file_identity


def uses_supervised_replay_identity(opt) -> bool:
    """Sequential and position-seeded streams both depend on the training inputs."""
    if opt.task in ("image_classification", "semantic_segmentation", "object_detection", "tabular_regression"):
        return True
    # A position-seeded supervised provider also requires a valid named task.
    return replay_worker_mode(opt) not in (None, PAIRED_WORKER_AUGMENTATION)


def supervised_replay_settings(opt) -> dict:
    ds = opt.datasets["train"]
    selected = resolve_task(opt)
    if selected is None:
        raise ValueError("Supervised replay requires the named task and its label settings.")
    settings = selected[2]
    if opt.task == "tabular_regression":
        return {"provider": ds.type.casefold(), "features": list(settings.features), "targets": list(settings.targets)}
    common = {"provider": ds.type.casefold(), "hflip": ds.use_hflip, "rotate": ds.use_rot}
    if opt.task == "image_classification":
        return {**common, "classNames": settings.class_names, "imageSize": settings.image_size}
    if opt.task == "semantic_segmentation":
        return {**common, "classNames": settings.class_names, "ignoreLabel": settings.ignore_label,
                "cropSize": ds.gt_size if ds.gt_size is not None else ds.lq_size}
    if opt.task == "object_detection":
        return {**common, "categories": [[item.id, item.name] for item in settings.categories]}
    raise ValueError("Unsupported supervised dataset for RNG replay.")


def capture_supervised_identity(dataset, opt) -> dict:
    settings = supervised_replay_settings(opt)
    if opt.task == "tabular_regression":
        values = dataset.values.numpy().astype("<f4", copy=False)
        return {"version": 1, "settings": settings, "count": len(dataset),
                "values": hashlib.sha256(values).hexdigest()}
    labels = hashlib.sha256(b"supervised-disk-targets-v1\0")

    def add(data: bytes) -> None:
        labels.update(struct.pack("<Q", len(data)))
        labels.update(data)

    cwd = os.getcwdb()

    def paths():
        if opt.task == "image_classification":
            for path, label in dataset.samples:
                add(label.encode("utf-8"))
                yield os.path.join(cwd, os.fsencode(path))
        elif opt.task == "semantic_segmentation":
            for image, mask in dataset.pairs:
                yield os.path.join(cwd, os.fsencode(image))
                yield os.path.join(cwd, os.fsencode(mask))
        else:
            for sample in dataset.samples:
                add(str(sample.image_id).encode("ascii"))
                add(struct.pack("<QQ", *sample.size))
                for field in ("boxes", "labels", "ignore"):
                    add(sample.target[field].numpy().tobytes())
                yield os.path.join(cwd, os.fsencode(sample.path))

    selection, files = disk_file_identity(paths(), b"supervised-disk")
    return {"version": 1, "count": len(dataset), "settings": settings, "selection": selection,
            "files": files, "targets": labels.hexdigest()}


def supervised_identity_error(saved: object) -> str | None:
    if saved is None:
        return (
            "This checkpoint has no supervised training-data identity. Older zero-worker "
            "image checkpoints omitted it. Set checkpoint_rng: false to resume with fresh random streams."
        )
    if (
        not isinstance(saved, dict)
        or type(saved.get("version")) is not int
        or saved["version"] != 1
        or type(saved.get("count")) is not int
        or not isinstance(saved.get("settings"), dict)
    ):
        return (
            "Incompatible supervised training-data identity. "
            "Set checkpoint_rng: false to resume with fresh random streams."
        )
    return None


def require_supervised_identity(saved, current) -> None:
    if error := supervised_identity_error(saved):
        raise ValueError(error)
    changed = [key for key in current if saved.get(key) != current[key]]
    if changed:
        raise ValueError("Supervised replay requires the original training data and settings. Changed: " + ", ".join(changed))

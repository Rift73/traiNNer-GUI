"""CPU-mode option checks shared between Studio and the real trainer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import ReduxOptions


class DeviceConfigError(ValueError):
    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path


def validate_cpu_recipe(opt: ReduxOptions):
    if opt.num_gpu != 0:
        return
    if opt.high_order_degradation:
        train_data = opt.datasets.get("train")
        if not train_data or train_data.type.lower() != "realesrgandataset":
            raise DeviceConfigError("datasets.train.type", "CPU on-the-fly training supports the still-image RealESRGANDataset.")
        if opt.train and opt.train.vosr_target:
            raise DeviceConfigError("train.vosr_target", "The VOSR training target requires CUDA and TensorRT.")
    if opt.use_amp and not opt.amp_bf16:
        raise DeviceConfigError("amp_bf16","CPU mixed precision uses BF16 autocast. Select amp_bf16 or disable AMP.")
    for phase, dataset in opt.datasets.items():
        if dataset.prefetch_mode == "cuda":
            raise DeviceConfigError(f"datasets.{phase}.prefetch_mode","Select CPU prefetch when num_gpu is zero.")


def auto_workers(world_size: int) -> int:
    """NeoSR's per-process loader budget, including affinity on Python 3.11/3.12."""
    if hasattr(os, "process_cpu_count"):
        cpus = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count()
    return max((cpus or 1) // max(1, world_size) - 1, 1)


PAIRED_WORKER_AUGMENTATION = "paired-position-v1"
WORKER_REPLAY_DATASETS = {
    "pairedimagedataset": (PAIRED_WORKER_AUGMENTATION, "traiNNer.data.paired_image_dataset.PairedImageDataset"),
    "classfolderdataset": ("class-position-v1", "traiNNer.tasks.image_classification.dataset.ClassFolderDataset"),
    "imagemaskdataset": ("mask-position-v1", "traiNNer.tasks.semantic_segmentation.dataset.ImageMaskDataset"),
    "cocodetectiondataset": ("boxes-position-v1", "traiNNer.tasks.object_detection.dataset.CocoDetectionDataset"),
}


def replay_worker_mode(opt: ReduxOptions) -> str | None:
    train = opt.datasets.get("train")
    if opt.checkpoint_rng and train and train.num_worker_per_gpu:
        spec = WORKER_REPLAY_DATASETS.get(train.type.casefold())
        if spec is not None:
            return spec[0]
    return None


def validate_checkpoint_rng(opt: ReduxOptions) -> None:
    if not opt.checkpoint_rng:
        return
    if opt.num_gpu not in (0, 1) or opt.dist:
        raise DeviceConfigError("checkpoint_rng", "Checkpoint RNG replay supports a single training process and at most one GPU.")
    if opt.high_order_degradation:
        raise DeviceConfigError("high_order_degradation", "Checkpoint RNG replay needs prepared data. The on-the-fly pipeline keeps unsaved batches and degradation state.")
    for phase, dataset in opt.datasets.items():
        if phase == "train" and dataset.num_worker_per_gpu != 0 and replay_worker_mode(opt) is None:
            raise DeviceConfigError(f"datasets.{phase}.num_worker_per_gpu", "Use zero workers for replay with this dataset. Prepared image pairs, class folders, image masks and COCO detection support position-seeded workers.")
        if dataset.prefetch_mode is not None:
            raise DeviceConfigError(f"datasets.{phase}.prefetch_mode", "Use a null prefetch mode for checkpoint RNG replay.")

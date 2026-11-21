"""Disk COCO boxes in pixel xywh, decoded to clipped pixel xyxy."""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from msgspec import Struct
from PIL import Image
from torch import Tensor
from trainner_config.object_detection_task import DetectionOptions
from trainner_data.coco import clip_box, image_path, index_coco, read_coco

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import draw_augment_vips, sample_index_rng
from traiNNer.tasks.object_detection.image import (
    detection_image_tensor,
    read_detection_image,
)
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed, DetectionTarget


@dataclass(frozen=True, slots=True)
class _Sample:
    image_id: int
    path: Path
    size: tuple[int, int]
    target: DetectionTarget


def _augment(
    image: Image.Image, boxes: Tensor, *, hflip: bool, rotation: bool, rng=random
) -> tuple[Image.Image, Tensor]:
    horizontal, vertical, transpose = draw_augment_vips(hflip, rotation, rotation, rng=rng)
    if horizontal:
        boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if vertical:
        boxes[:, [1, 3]] = image.height - boxes[:, [3, 1]]
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transpose:
        boxes = boxes[:, [1, 0, 3, 2]]
        image = image.transpose(Image.Transpose.TRANSPOSE)
    return image, boxes


@DATASET_REGISTRY.register()
class CocoDetectionDataset(BaseDataset):
    def __init__(self, opt: DatasetOptions, task_options: Struct | None = None) -> None:
        super().__init__(opt)
        if not isinstance(task_options, DetectionOptions):
            raise ValueError(
                "CocoDetectionDataset requires the object_detection task settings."
            )
        if opt.io_backend.get("type") != "disk":
            raise ValueError("CocoDetectionDataset requires the disk io_backend.")
        roots = (
            [opt.dataroot_lq] if isinstance(opt.dataroot_lq, str) else opt.dataroot_lq
        )
        if not roots or len(roots) != 1 or not opt.annotation_file:
            raise ValueError(
                "CocoDetectionDataset needs one dataroot_lq and an annotation_file."
            )
        root = Path(roots[0]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"COCO image root does not exist: {root}")
        annotation_path = Path(opt.annotation_file).expanduser()
        document = read_coco(annotation_path)
        configured = {
            category.id: category.name for category in task_options.categories
        }
        class_ids = {
            category.id: index
            for index, category in enumerate(task_options.categories, 1)
        }
        index = index_coco(document, annotation_path, configured)

        self.samples: list[_Sample] = []
        for image in index.images.values():
            records = index.annotations.get(image.id, ())
            boxes = torch.tensor(
                [clip_box(ann, image) for ann in records], dtype=torch.float32
            ).reshape(-1, 4)
            if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
                raise ValueError(
                    f"A box in image {image.id} has zero extent at float32 precision."
                )
            target: DetectionTarget = {
                "boxes": boxes,
                "labels": torch.tensor(
                    [class_ids[ann.category_id] for ann in records], dtype=torch.long
                ),
                "ignore": torch.tensor(
                    [ann.ignored for ann in records],
                    dtype=torch.bool,
                ),
            }
            self.samples.append(
                _Sample(
                    image.id,
                    image_path(root, image.file_name),
                    (image.width, image.height),
                    target,
                )
            )

    @property
    def label(self) -> str:
        return "detection images"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, str]) -> DataFeed:
        rng = random
        if isinstance(index, tuple):
            index, rng = sample_index_rng(index)
        sample = self.samples[index]
        boxes = sample.target["boxes"].clone()
        image = read_detection_image(sample.path, expected_size=sample.size)
        if self.opt.phase == "train":
            image, boxes = _augment(
                image, boxes, hflip=self.opt.use_hflip, rotation=self.opt.use_rot, rng=rng
            )
        return {
            "lq": detection_image_tensor(image),
            "lq_path": str(sample.path),
            "detection_target": {**sample.target, "boxes": boxes},
            "detection_image_id": sample.image_id,
        }

    @staticmethod
    def collate_fn(samples: list[DataFeed]) -> DataFeed:
        """Preserve each image size and each box count."""
        return {
            "detection_images": [sample["lq"] for sample in samples],
            "detection_targets": [sample["detection_target"] for sample in samples],
            "detection_paths": [sample["lq_path"] for sample in samples],
            "detection_image_ids": [sample["detection_image_id"] for sample in samples],
        }

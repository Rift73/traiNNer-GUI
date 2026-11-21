"""Aligned RGB images and integer class masks."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from trainner_data.supervised_manifest import folder_members
from trainner_data.task_folders import IMAGE_EXTENSIONS, MASK_EXTENSIONS, files_by_stem

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import draw_augment_vips, paired_random_crop, sample_index_rng
from traiNNer.tasks.sample_prep import try_pack_rgb8
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class ImageMaskDataset(BaseDataset):
    """Pair image and mask files by path relative to their roots, minus extension."""

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        if opt.io_backend.get("type") != "disk":
            raise ValueError("ImageMaskDataset requires the disk io_backend.")
        if not opt.dataroot_lq or not opt.dataroot_gt:
            raise ValueError(
                "ImageMaskDataset requires dataroot_lq images and dataroot_gt masks."
            )
        images = (
            [opt.dataroot_lq] if isinstance(opt.dataroot_lq, str) else opt.dataroot_lq
        )
        masks = (
            [opt.dataroot_gt] if isinstance(opt.dataroot_gt, str) else opt.dataroot_gt
        )
        if len(images) != len(masks):
            raise ValueError(
                "Provide one mask root per image root in ImageMaskDataset."
            )

        self.pairs: list[tuple[Path, Path]] = []
        if opt.meta_info is not None:
            image_roots = [Path(path).expanduser() for path in images]
            mask_roots = [Path(path).expanduser() for path in masks]
            for group, image, mask in folder_members(
                Path(opt.meta_info), image_roots, mask_roots
            ):
                assert mask is not None
                self.pairs.append(
                    (image_roots[group - 1] / image, mask_roots[group - 1] / mask)
                )
        else:
            for image_root, mask_root in zip(images, masks, strict=True):
                inputs = files_by_stem(Path(image_root).expanduser(), IMAGE_EXTENSIONS)
                labels = files_by_stem(Path(mask_root).expanduser(), MASK_EXTENSIONS)
                missing = inputs.keys() - labels.keys()
                extra = labels.keys() - inputs.keys()
                if missing:
                    raise ValueError(f"Missing mask for {min(missing)} in {mask_root}.")
                if extra:
                    raise ValueError(f"Missing image for {min(extra)} in {image_root}.")
                self.pairs.extend((path, labels[stem]) for stem, path in inputs.items())
        if not self.pairs:
            raise ValueError("ImageMaskDataset found no image and mask pairs.")

        self.crop_size = opt.gt_size if opt.gt_size is not None else opt.lq_size
        if opt.phase == "train" and (self.crop_size is None or self.crop_size <= 0):
            raise ValueError(
                "ImageMaskDataset training requires a positive lq_size or gt_size."
            )

    @property
    def label(self) -> str:
        return "image/mask pairs"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int | tuple[int, str]) -> DataFeed:
        rng = random
        crop_rng = None
        if isinstance(index, tuple):
            index, rng = sample_index_rng(index)
            crop_rng = np.random.default_rng(rng.getrandbits(128))
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            if image_file.size != mask_file.size:
                raise ValueError(
                    f"Image and mask sizes differ for {image_path} and {mask_path}."
                )
            image = np.asarray(image_file.convert("RGB"))
            mask = np.asarray(mask_file)
            if mask.ndim != 2 or mask.dtype.kind not in ("b", "i", "u"):
                raise ValueError(
                    f"Mask {mask_path} must store one integer class ID per pixel."
                )

        if self.opt.phase == "train":
            assert self.crop_size is not None
            mask, image = paired_random_crop(
                mask, image, self.crop_size, 1, str(mask_path),
                rng=crop_rng,
            )
            hflip, vflip, transpose = draw_augment_vips(
                self.opt.use_hflip, self.opt.use_rot, self.opt.use_rot, rng=rng
            )
            if hflip:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if vflip:
                image, mask = image[::-1], mask[::-1]
            if transpose:
                image, mask = image.swapaxes(0, 1), mask.T

        packed = try_pack_rgb8(image)
        if packed is None:
            packed = (
                torch.from_numpy(np.array(image, copy=True, order="C"))
                .permute(2, 0, 1)
                .float()
                .div_(255)
            )

        return {
            "lq": packed,
            "gt": torch.from_numpy(
                np.array(mask, dtype=np.int64, copy=True, order="C")
            ),
            "lq_path": str(image_path),
            "gt_path": str(mask_path),
        }

"""Whole images labeled by their first subdirectory."""

import random
from pathlib import Path

import numpy as np
from msgspec import Struct
from PIL import Image
from torch import Tensor
from trainner_config.image_classification_task import ClassificationOptions
from trainner_data.supervised_manifest import folder_members
from trainner_data.task_folders import class_entries, image_files

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import augment, sample_index_rng
from traiNNer.tasks.sample_prep import try_pack_rgb8
from traiNNer.utils.img_util import img2tensor
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


def _roots(opt: DatasetOptions) -> list[Path]:
    if not opt.dataroot_lq:
        raise ValueError(
            "ClassFolderDataset requires dataroot_lq with class subfolders."
        )
    roots = [opt.dataroot_lq] if isinstance(opt.dataroot_lq, str) else opt.dataroot_lq
    return [Path(value).expanduser() for value in roots]


def class_folders(opt: DatasetOptions) -> list[Path]:
    roots = _roots(opt)
    if opt.meta_info is not None:
        return list(
            dict.fromkeys(
                roots[group - 1] / image.parts[0]
                for group, image, _ in folder_members(Path(opt.meta_info), roots)
            )
        )
    folders = []
    for root in roots:
        for path, is_class in class_entries(root):
            if not is_class:
                raise ValueError(
                    f"Image {path} has no class label. Put it inside a class subfolder."
                )
            folders.append(path)
    return folders


def read_classification_image(
    path: Path, size: int, *, hflip: bool = False, rotation: bool = False, rng=random
) -> Tensor:
    """Apply the task's shared whole-image RGB resize and optional training flips."""
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame image.")
            resized = image.convert("RGB").resize(
                (size, size), resample=Image.Resampling.BILINEAR
            )
            pixels = np.array(resized)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read classification image {path}. {exc}") from exc
    if hflip or rotation:
        pixels = augment(pixels, hflip=hflip, rotation=rotation, rng=rng)
        assert isinstance(pixels, np.ndarray)
    packed = try_pack_rgb8(pixels)
    return img2tensor(np.ascontiguousarray(pixels)) if packed is None else packed


@DATASET_REGISTRY.register()
class ClassFolderDataset(BaseDataset):
    supports_batched_evaluation = True

    def __init__(self, opt: DatasetOptions, task_options: Struct | None = None) -> None:
        super().__init__(opt)
        if opt.io_backend.get("type") != "disk":
            raise ValueError("ClassFolderDataset requires the disk io_backend.")
        if isinstance(task_options, ClassificationOptions):
            self.size = task_options.image_size
        else:
            self.size = opt.lq_size if opt.lq_size is not None else opt.gt_size
        if self.size is None or self.size < 1:
            raise ValueError(
                "ClassFolderDataset requires task_options.image_size for the whole-image resize."
            )

        self.samples: list[tuple[Path, str]] = []
        names = set()
        if opt.meta_info is not None:
            roots = _roots(opt)
            for group, image, _ in folder_members(Path(opt.meta_info), roots):
                name = image.parts[0]
                names.add(name)
                self.samples.append((roots[group - 1] / image, name))
        else:
            for folder in class_folders(opt):
                images = image_files(folder)
                if not images:
                    raise ValueError(
                        f"Class folder {folder} contains no supported images."
                    )
                names.add(folder.name)
                self.samples.extend((path, folder.name) for path in images)
        if not self.samples:
            raise ValueError(
                f"Classification dataset {opt.name!r} has no labeled images."
            )
        self.class_names = frozenset(names)

    @property
    def label(self) -> str:
        return "labeled images"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, str]) -> DataFeed:
        rng = random
        if isinstance(index, tuple):
            index, rng = sample_index_rng(index)
        path, name = self.samples[index]
        training = self.opt.phase == "train"
        assert self.size is not None
        return {
            "lq": read_classification_image(
                path,
                self.size,
                hflip=training and self.opt.use_hflip,
                rotation=training and self.opt.use_rot,
                rng=rng,
            ),
            "class_name": name,
            "lq_path": str(path),
        }

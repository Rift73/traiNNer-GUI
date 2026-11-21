# https://github.com/Demetter/TSCUNet_Trainer/blob/main/Train_TSCUNet.py
import random

import torch

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import (
    augment_vips,
    augment_vips_pair,
    paired_random_crop_vips,
    single_crop_vips,
)
from trainner_data.frame_sequences import FrameWindowIndex
from traiNNer.utils.img_util import img2rgb, img2tensor, vipsimfrompath
from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class PairedVideoDataset(BaseDataset):
    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)

        assert opt.dataroot_lq is not None
        assert opt.dataroot_gt is not None
        assert opt.clip_size is not None

        assert isinstance(opt.dataroot_lq, list), (
            f"dataroot_lq must be defined for dataset {opt.name}"
        )
        assert isinstance(opt.dataroot_gt, list), (
            f"dataroot_gt must be defined for dataset {opt.name}"
        )

        if opt.io_backend.get("type", "disk") != "disk":
            raise ValueError("PairedVideoDataset reads frame folders using the disk backend.")
        self.windows = FrameWindowIndex(
            opt.dataroot_lq, opt.clip_size, opt.dataroot_gt,
            scene_manifest=opt.video_scene_manifest,
        )
        self.clip_size = opt.clip_size
        self.gt_size = opt.gt_size
        self._pack_lq = False
        if opt.phase == "train":
            from traiNNer.data.paired_video_pack import load_library

            self._pack_lq = load_library() is not None
        logger = get_root_logger()
        total_clips = len(self.windows)
        total_scenes = len(self.windows.scenes)
        logger.info(
            "Found %d valid file pairs across %d scenes.",
            total_clips,
            total_scenes,
        )

        if opt.phase == "train":
            if total_scenes < 100:
                logger.warning(
                    "Number of scene pairs is low: %d, training quality may be impacted. Please use more scene pairs for best training results.",
                    total_scenes,
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> DataFeed:
        scale = self.opt.scale
        assert scale is not None
        assert self.clip_size > 0

        lr_paths, hr_path = self.windows.window(idx)
        assert hr_path is not None
        middle_idx = self.clip_size // 2

        lr_clip = []
        lr_frames = [] if self._pack_lq else None

        force_x = None
        force_y = None
        force_hflip = None
        force_vflip = None
        force_rot90 = None
        lq_size = 0

        if self.opt.phase == "train":
            assert self.gt_size is not None
            lq_size = self.gt_size // scale

        # middle frame only
        vips_img_gt = vipsimfrompath(hr_path)

        for i in range(self.clip_size):
            lq_path = lr_paths[i]
            vips_img_lq = vipsimfrompath(lq_path)

            if self.opt.phase == "train":
                assert self.gt_size is not None

                if force_x is None:
                    force_rot90 = self.opt.use_rot and random.random() < 0.5
                    force_hflip = self.opt.use_hflip and random.random() < 0.5
                    force_vflip = self.opt.use_rot and random.random() < 0.5
                    h_lq: int = vips_img_lq.height  # pyright: ignore[reportAssignmentType]
                    w_lq: int = vips_img_lq.width  # pyright: ignore[reportAssignmentType]
                    if force_rot90:
                        h_lq, w_lq = w_lq, h_lq  # swap dimensions if rotating
                    force_y = random.randint(0, h_lq - lq_size)
                    force_x = random.randint(0, w_lq - lq_size)

                if i == middle_idx:
                    vips_img_gt_aug, vips_img_lq = augment_vips_pair(
                        (vips_img_gt, vips_img_lq),
                        self.opt.use_hflip,
                        self.opt.use_rot,
                        self.opt.use_rot,
                        force_hflip,
                        force_vflip,
                        force_rot90,
                    )
                    img_gt, img_lq = paired_random_crop_vips(
                        vips_img_gt_aug,
                        vips_img_lq,
                        self.gt_size,
                        scale,
                        lq_path,
                        hr_path,
                        force_x,
                        force_y,
                    )
                else:
                    assert force_hflip is not None
                    assert force_vflip is not None
                    assert force_rot90 is not None
                    assert force_y is not None
                    # For non-middle frames, only augment LR
                    vips_img_lq = augment_vips(
                        vips_img_lq,
                        force_hflip,
                        force_vflip,
                        force_rot90,
                        randomize=False,
                    )
                    # Crop LR only
                    img_lq = single_crop_vips(
                        vips_img_lq, lq_size, force_x, force_y, lq_path
                    )
            else:
                img_lq = img2rgb(vips_img_lq.numpy())
                if i == middle_idx:
                    img_gt = img2rgb(vips_img_gt.numpy())

            if lr_frames is None:
                lr_clip.append(img2tensor(img_lq, from_bgr=False, float32=True))
            else:
                lr_frames.append(img_lq)

        lq_tensor = None
        if lr_frames is not None:
            from traiNNer.data.paired_video_pack import try_pack_lq_clip

            lq_tensor = try_pack_lq_clip(lr_frames)
            if lq_tensor is None:
                lr_clip = [
                    img2tensor(frame, from_bgr=False, float32=True)
                    for frame in lr_frames
                ]

        # Process GT tensor
        img_gt = img2tensor(
            img_gt,  # pyright: ignore[reportPossiblyUnboundVariable]
            from_bgr=False,
            float32=True,
        )

        result: DataFeed = {
            "lq": lq_tensor if lq_tensor is not None else torch.stack(lr_clip),
            "gt": img_gt,
            "gt_path": hr_path,
            "lq_path": lr_paths[middle_idx],
        }
        if self.opt.phase != "train":
            result["context_paths"] = lr_paths
        return result

    @property
    def label(self) -> str:
        return f"{self.clip_size}-frame sequence pairs"

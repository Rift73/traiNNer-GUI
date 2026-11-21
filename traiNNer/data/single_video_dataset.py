# https://github.com/Demetter/TSCUNet_Trainer/blob/main/Train_TSCUNet.py

import torch

from traiNNer.data.base_dataset import BaseDataset
from trainner_data.frame_sequences import FrameWindowIndex
from traiNNer.utils.img_util import img2rgb, img2tensor, vipsimfrompath
from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class SingleVideoDataset(BaseDataset):
    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)

        assert opt.dataroot_lq is not None
        assert opt.clip_size is not None

        assert isinstance(opt.dataroot_lq, list), (
            f"dataroot_lq must be defined for dataset {opt.name}"
        )

        if opt.io_backend.get("type", "disk") != "disk":
            raise ValueError("SingleVideoDataset reads frame folders using the disk backend.")
        self.clip_size = opt.clip_size
        self.windows = FrameWindowIndex(opt.dataroot_lq, self.clip_size)
        get_root_logger().info(
            "Found %d frames, %d complete windows across %d eligible scenes.",
            self.windows.frame_count, len(self.windows), len(self.windows.scenes),
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> DataFeed:
        clips, _ = self.windows.window(idx)
        lr_clip = []

        for lq_path in clips:
            vips_img_lq = vipsimfrompath(lq_path)
            img_lq = img2rgb(vips_img_lq.numpy())
            img_lq = img2tensor(
                img_lq,
                from_bgr=False,
                float32=True,
            )
            lr_clip.append(img_lq)

        return {
            "lq": torch.stack(lr_clip),
            "lq_path": clips[self.clip_size // 2],
            "context_paths": clips,
        }

    @property
    def label(self) -> str:
        return f"{self.clip_size}-frame sequences"

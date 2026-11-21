"""One-channel conversion shared by the paired and single still-image loaders."""

from functools import cache

import numpy as np
from torch import Tensor
from trainner_config.options import DatasetOptions

from traiNNer.utils.color_util import dataset_luma_pt, dataset_luma_weights, rgb2ycbcr
from traiNNer.utils.img_util import img2tensor


def check_luma_normalization(opt: DatasetOptions) -> None:
    if (
        opt.mean is not None
        and opt.std is not None
        and (len(opt.mean) != 1 or len(opt.std) != 1)
    ):
        raise ValueError(
            f"Dataset color={opt.color!r} produces one channel. Provide one mean and one std."
        )


@cache
def _weights(mode: str) -> tuple[Tensor, float]:
    return dataset_luma_weights(mode, device="cpu")


def luma_tensor(rgb: np.ndarray, mode: str) -> Tensor:
    if mode == "y" and rgb.dtype in (np.uint8, np.float32):
        # Keep the original single-image conversion, including rounding uint8
        # studio codes before the tensor conversion divides by 255.
        y = rgb2ycbcr(rgb, y_only=True)
        return img2tensor(y[..., None], from_bgr=False, float32=True)
    # Integer normalization uses this image's decoded dtype independently for
    # GT and LQ. The full-range projection stays continuous even on uint8.
    tensor = img2tensor(rgb, from_bgr=False, float32=True).float()
    return dataset_luma_pt(tensor, *_weights(mode))

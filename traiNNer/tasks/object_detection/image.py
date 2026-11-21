"""Shared RGB decoding for COCO datasets and eager detection."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor

from traiNNer.tasks.sample_prep import try_pack_rgb8
from traiNNer.utils.img_util import img2tensor


def read_detection_image(
    path: Path, *, expected_size: tuple[int, int] | None = None
) -> Image.Image:
    """Decode one RGB frame in its stored pixel geometry."""
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame image.")
            if expected_size is not None and opened.size != expected_size:
                raise ValueError(
                    f"COCO declares size {expected_size}, but the file is {opened.size}."
                )
            return opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not decode detection image {path}. {exc}") from exc


def detection_image_tensor(image: Image.Image) -> Tensor:
    """Convert decoded RGB pixels to contiguous CHW float32 in [0, 1]."""
    packed = try_pack_rgb8(image, contiguous=True)
    return img2tensor(np.array(image)).contiguous() if packed is None else packed

"""Optional packing of decoded RGB views for the built-in still-image tasks."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from traiNNer.data.paired_video_pack import normalized_bytes

_LIBRARY = (
    Path(__file__).resolve().parents[2]
    / "native/studio/data/build/clang/task_sample_pack.so"
)


@cache
def _kernel() -> Callable[..., None] | None:
    if not _LIBRARY.is_file():
        return None
    try:
        pack = ctypes.CDLL(str(_LIBRARY)).pack_rgb8_sample_v1
    except (OSError, AttributeError):
        return None
    pack.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
        ctypes.c_ssize_t,
        ctypes.c_ssize_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    pack.restype = None
    return pack


def try_pack_rgb8(
    image: np.ndarray | Image.Image, *, contiguous: bool = False
) -> Tensor | None:
    """Write a valid uint8 HWC view directly to the caller's CHW tensor layout."""
    pack = _kernel()
    if pack is None:
        return None
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            return None
        image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return None
    height, width, _ = image.shape
    if not height or not width:
        return None

    if contiguous:
        output = torch.empty((3, height, width), dtype=torch.float32, device="cpu")
    else:
        output = torch.empty_strided(
            (3, height, width), (1, width * 3, 3), dtype=torch.float32, device="cpu"
        )
    pack(
        image.ctypes.data,
        height,
        width,
        *image.strides,
        int(contiguous),
        normalized_bytes().data_ptr(),
        output.data_ptr(),
    )
    return output

"""Optional CPU assembly of cropped paired-video LQ frames."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from traiNNer.utils.img_util import tensor2float32
from traiNNer.utils.logger import get_root_logger

_LIBRARY = (
    Path(__file__).resolve().parents[2]
    / "native"
    / "studio"
    / "data"
    / "build"
    / "clang"
    / "paired_video_pack.so"
)


class _RgbFrame(ctypes.Structure):
    _fields_ = [
        ("pixels", ctypes.c_void_p),
        ("row_stride", ctypes.c_size_t),
        ("pixel_stride", ctypes.c_size_t),
    ]


@cache
def load_library() -> ctypes.CDLL | None:
    if not _LIBRARY.is_file():
        return None
    try:
        library = ctypes.CDLL(str(_LIBRARY))
        pack = library.pack_rgb8_tchw
    except (OSError, AttributeError) as exc:
        get_root_logger().warning("Could not load native paired-video packing: %s", exc)
        return None
    pack.argtypes = [
        ctypes.POINTER(_RgbFrame),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    pack.restype = None
    return library


@cache
def normalized_bytes() -> Tensor:
    return tensor2float32(torch.from_numpy(np.arange(256, dtype=np.uint8)))


def try_pack_lq_clip(frames: Sequence[np.ndarray]) -> Tensor | None:
    """Write uint8 HWC RGB frames directly into a contiguous float32 TCHW tensor."""
    library = load_library()
    if library is None or not frames:
        return None

    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or not (shape[0] and shape[1]):
        return None
    height, width, _ = shape

    views = (_RgbFrame * len(frames))()
    for index, frame in enumerate(frames):
        if frame.shape != shape or frame.dtype != np.uint8:
            return None
        row_stride, pixel_stride, channel_stride = frame.strides
        if (
            channel_stride != 1
            or pixel_stride < 3
            or row_stride < (width - 1) * pixel_stride + 3
        ):
            return None
        views[index] = _RgbFrame(frame.ctypes.data, row_stride, pixel_stride)

    output = torch.empty(
        (len(frames), 3, height, width), dtype=torch.float32, device="cpu"
    )
    library.pack_rgb8_tchw(
        views,
        len(frames),
        height,
        width,
        normalized_bytes().data_ptr(),
        output.data_ptr(),
    )
    return output

"""The work canvas of a still source exactly twice gt_size square, made on the GPU.

Such a source in 8-bit RGB, halved with Hermite and without a separate target,
has one canvas geometry: the halved image, flipped and rotated by the sample's
draws and mirror-padded to the centre of the gt_size + 32 canvas. Its sample
carries the decoded bytes and its orientation flags instead of the float32
canvas, and the model makes the canvas when the batch starts. The kernel
repeats the CPU path's arithmetic (chainner_ext's Hermite halving of the bytes
scaled by 1/255 in float32, then the clip, the flips, the rotation and libvips'
mirror padding), so the canvas is bit-identical to the one the CPU path makes.
"""

import importlib
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
import pyvips
import torch
from torch import Tensor

from traiNNer.data.clip_hr import resize_matrix
from traiNNer.data.dvd_response import collate_dvd
from traiNNer.utils.redux_options import DatasetOptions

_CANVAS_MARGIN = 32  # the canvas is gt_size + 32 square, as the dataset pads
_TAPS = 4  # Hermite halving reads source pixels 2i - 1 .. 2i + 2


@cache
def _triton_available() -> bool:
    try:
        importlib.import_module("triton")
    except ImportError:
        return False
    return True


def uses_still_sources(opt: DatasetOptions) -> bool:
    """Whether the dataset's sources of the right size take the GPU canvas:
    Hermite halving without a separate target, a gt_size whose canvas padding
    one mirror of the halved source covers, and Triton for the canvas kernel."""
    return (
        opt.otf_canvas == "trainer"
        and _triton_available()
        and opt.downsample_input
        and opt.downscale_algorithm == "hermite"
        and opt.target_dataroot_gt is None
        and opt.gt_size is not None
        and opt.gt_size >= _CANVAS_MARGIN // 2
    )


def is_still_source(img: pyvips.Image, opt: DatasetOptions) -> bool:
    """Whether the decoded source takes the GPU canvas: 8-bit RGB, 2 * gt_size
    square, so that the halved source fills the canvas centre exactly."""
    gt_size = opt.gt_size
    return (
        uses_still_sources(opt)
        and gt_size is not None
        and img.format == "uchar"
        and img.bands == 3
        and img.width == img.height == 2 * gt_size
    )


def source_bytes(img: pyvips.Image) -> Tensor:
    """The decoded source as uint8 CHW (a view of its HWC pixels)."""
    return torch.from_numpy(img.numpy()).permute(2, 0, 1)


def orientation_flags(hflip: bool, vflip: bool, rot90: bool) -> int:
    """The sample's flips and clockwise rotation as bits 1, 2 and 4."""
    return int(hflip) | int(vflip) << 1 | int(rot90) << 2


def host_canvas(source: Tensor, flags: int) -> Tensor:
    """The float32 CHW canvas of a source sample, made by the dataset's CPU path."""
    from traiNNer.data.realesrgan_dataset import (
        _downsample_input_vips,
        _reflect_pad_vips,
    )
    from traiNNer.data.transforms import augment_vips
    from traiNNer.utils import img2tensor
    from traiNNer.utils.img_util import img2rgb

    img = _downsample_input_vips(
        pyvips.Image.new_from_array(np.ascontiguousarray(source.permute(1, 2, 0))),
        "hermite",
    )
    img = augment_vips(
        img, bool(flags & 1), bool(flags & 2), bool(flags & 4), randomize=False
    )
    img = _reflect_pad_vips(img, source.shape[-1] // 2 + _CANVAS_MARGIN)
    return img2tensor(img2rgb(img.numpy()), from_bgr=False, float32=True)


def collate_still_source(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """collate_dvd for a dataset whose samples may carry source bytes.

    A batch of source samples only stays bytes, with its flags, for the GPU
    canvas; in a mixed batch the source samples get their canvas here, so the
    batch is the one the CPU path delivers.
    """
    if not all("gt_orientation" in sample for sample in samples):
        samples = [
            {
                **{
                    key: value
                    for key, value in sample.items()
                    if key != "gt_orientation"
                },
                "gt": host_canvas(sample["gt"], sample["gt_orientation"]),
            }
            if "gt_orientation" in sample
            else sample
            for sample in samples
        ]
    return collate_dvd(samples)


@cache
def _hermite_taps(gt_size: int, device: torch.device) -> Tensor:
    """The float32 (gt_size, 4) weights of chainner_ext's Hermite halving of
    2 * gt_size pixels, for source pixels 2i - 1 .. 2i + 2; a tap outside the
    image weighs 0."""
    matrix = resize_matrix(2 * gt_size, gt_size, "hermite")
    first = 2 * np.arange(gt_size) - 1
    columns = first[:, None] + np.arange(_TAPS)
    inside = (columns >= 0) & (columns < 2 * gt_size)
    taps = np.where(
        inside,
        np.take_along_axis(matrix, np.clip(columns, 0, 2 * gt_size - 1), axis=1),
        0.0,
    )
    if not np.array_equal(taps.sum(axis=1), matrix.sum(axis=1)):
        raise ValueError("Hermite halving reads outside its four taps")
    return torch.tensor(taps, dtype=torch.float32, device=device)


def device_canvas(
    source: Tensor, flags: Tensor, memory_format: torch.memory_format
) -> Tensor:
    """The float32 (b, 3, gt_size + 32, gt_size + 32) canvases of a CUDA batch
    of uint8 (b, 3, 2 * gt_size, 2 * gt_size) sources and their flags."""
    from traiNNer.data.still_source_kernel import fill_canvas

    batch, channels, height, width = source.shape
    if (
        source.dtype != torch.uint8
        or channels != 3
        or height != width
        or height % 2
        or flags.shape != (batch,)
    ):
        raise ValueError(
            f"Expected uint8 RGB square sources and one flag each, got "
            f"{tuple(source.shape)} {source.dtype} and {tuple(flags.shape)}"
        )
    size = height // 2
    canvas_size = size + _CANVAS_MARGIN
    canvas = torch.empty(
        (batch, channels, canvas_size, canvas_size),
        dtype=torch.float32,
        device=source.device,
        memory_format=torch.contiguous_format
        if memory_format is torch.preserve_format
        else memory_format,
    )
    fill_canvas(source, flags, _hermite_taps(size, source.device), canvas)
    return canvas

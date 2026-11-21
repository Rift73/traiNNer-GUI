"""Native paired patch assembly using the trainer's decoded libvips source."""
from __future__ import annotations

import ctypes
import random
from functools import cache
from pathlib import Path

import numpy as np
import pyvips
import torch
from torch.utils.data import default_collate, get_worker_info

from traiNNer.data.image_luma import luma_tensor
from traiNNer.data.paired_video_pack import normalized_bytes
from traiNNer.data.transforms import draw_augment_vips
from traiNNer.utils.img_util import img2rgb

_LIB = Path(__file__).resolve().parents[2] / "native/studio/data/build/clang/paired_crop.so"


@cache
def kernel(*, luma: bool = False):
    try:
        library = ctypes.CDLL(str(_LIB))
        pack = library.pack_oriented_plane_f32 if luma else library.pack_oriented_crop
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Native paired crops require the LLVM data tools. Run ./trainner-gui prepare.") from exc
    if luma:
        pack.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)
    else:
        pack.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                         ctypes.c_void_p, ctypes.c_void_p)
    pack.restype = ctypes.c_int
    return pack


def try_crop_pair(gt, lq, opt, gt_path: str, lq_path: str, destination=None, *, rng=random):
    """Return None before consuming RNG for formats handled by the ordinary loader."""
    color = opt.color
    formats = ("uchar",) if color is None else ("uchar", "ushort")
    if any(image.format not in formats or image.bands not in (1,3,4) for image in (lq,gt)):
        return None
    scale, high = opt.scale, opt.gt_size
    if not scale or not high or high % scale:
        raise ValueError("The GT patch side must be a positive multiple of the restoration scale.")
    low = high // scale
    width, height = lq.width, lq.height
    if (gt.width,gt.height) != (scale*width,scale*height):
        raise ValueError(f"Expected {scale}x registered source dimensions for {lq_path} and {gt_path}.")
    if min(width,height) < low:
        raise ValueError(f"Source image {lq_path} is smaller than the {low}-pixel patch.")
    flip_x,flip_y,rotate = draw_augment_vips(opt.use_hflip,opt.use_rot,opt.use_rot,rng=rng)
    # Match the existing draws on the augmented full canvas, y then x.
    transformed_w,transformed_h = (height,width) if rotate else (width,height)
    y = rng.randint(0,transformed_h-low)
    x = rng.randint(0,transformed_w-low)
    if rotate:
        x,y = y,height-x-low
    if flip_x:
        x = width-x-low
    if flip_y:
        y = height-y-low
    flags = int(flip_x) | (int(flip_y)<<1) | (int(rotate)<<2)
    channels = 3 if color is None else 1
    pack = kernel(luma=color is not None)
    lookup = normalized_bytes() if color is None else None

    def read(image, side, factor, path, output):
        try:
            buf = pyvips.Region.new(image).fetch(x*factor, y*factor, side, side)
        except pyvips.Error as error:
            raise RuntimeError(f"Unable to read image {path}") from error
        shape = (channels, side, side)
        if output is None:
            output = torch.empty(shape, dtype=torch.float32, device="cpu")
        elif (output.shape != shape or not output.is_contiguous() or
              output.dtype != torch.float32 or output.device.type != "cpu"):
            raise ValueError(f"Expected a contiguous CPU float32 patch buffer of shape {shape}.")
        if color is None:
            assert lookup is not None
            data = np.frombuffer(buf, dtype=np.uint8)
            status = pack(data.ctypes.data, data.nbytes, side, image.bands, flags, lookup.data_ptr(), output.data_ptr())
        else:
            dtype = np.uint16 if image.format == "ushort" else np.uint8
            pixels = np.frombuffer(buf, dtype=dtype).reshape(side, side, image.bands)
            plane = luma_tensor(img2rgb(pixels), color).contiguous()
            status = pack(plane.data_ptr(), plane.nbytes, side, flags, output.data_ptr())
        if status:
            raise ValueError(f"Invalid crop buffer for {path}.")
        return output

    gt_out,lq_out = destination if destination is not None else (None,None)
    return read(gt,high,scale,gt_path,gt_out),read(lq,low,1,lq_path,lq_out)


class CropSamples(list):
    """Ordinary sample list for external callers, with its two private batch buffers."""

    def __init__(self, count: int, high: int, low: int, *, channels: int = 3):
        super().__init__()
        self.gt = self._buffer(count, high, channels)
        self.lq = self._buffer(count, low, channels)

    @staticmethod
    def _buffer(count: int, side: int, channels: int) -> torch.Tensor:
        shape = (count, channels, side, side)
        if get_worker_info() is None:
            return torch.empty(shape, dtype=torch.float32, device="cpu")
        # Use the same worker sharing strategy as torch's default_collate.
        # The native kernel writes into this mapping before it is sent.
        elem = torch.empty(0, dtype=torch.float32, device="cpu")
        storage = elem._typed_storage()._new_shared(count * channels * side * side, device=elem.device)
        return elem.new(storage).resize_(shape)


def collate_crop_samples(samples, *, release_shared: bool):
    from .shared_release import RELEASE_MIN_BYTES, SharedBatch, collate_shared
    if not isinstance(samples, CropSamples):
        return collate_shared(samples, default_collate) if release_shared else default_collate(samples)
    batch = {"gt": samples.gt, "lq": samples.lq,
             "gt_path": [sample["gt_path"] for sample in samples],
             "lq_path": [sample["lq_path"] for sample in samples]}
    if release_shared:
        # Allocated for this batch alone. No dataset cache or other batch owns a view.
        return SharedBatch(batch, [t for t in (samples.gt, samples.lq) if t.nbytes >= RELEASE_MIN_BYTES])
    return batch

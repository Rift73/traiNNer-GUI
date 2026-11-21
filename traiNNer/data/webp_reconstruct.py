"""Lossy WebP (method 0) pixels without writing and reading a WebP file.

Pillow's WebP round trip of an opaque RGB image converts it to YUV420 with
libwebp's integer formulas, encodes, decodes and converts back to RGB with
libwebp's "fancy" chroma upsampling. Here the two colour conversions are
Triton kernels on the GPU that reproduce libwebp's integer arithmetic, and the
encode and decode in between are replaced by libwebp_reconstruct.so (built
from traiNNer/csrc/webp_reconstruct with its build.sh), which returns the
decoded YUV420 planes of the encoder's own decisions. The result equals
otf_degradations.compress_webp value for value.

The route is used when the library has been built and the batch is opaque
RGB float32 on CUDA, with sides up to 2048 and quality 40 to 100 (supports).
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor

from traiNNer.data.compress_video_batch import Finish, RoundTrip, stage_to_host
from traiNNer.data.otf_degradations import _parallel_pool

_LIBRARY = (
    Path(__file__).resolve().parent.parent
    / "csrc"
    / "webp_reconstruct"
    / "build"
    / "libwebp_reconstruct.so"
)
_BLOCK = 256
_MAX_SIDE = 2048
_QUALITIES = range(40, 101)


@cache
def load_library() -> ctypes.CDLL | None:
    """libwebp_reconstruct.so, or None when it has not been built."""
    if not _LIBRARY.is_file():
        return None
    library = ctypes.CDLL(str(_LIBRARY))
    library.webp_reconstruct.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [
        ctypes.c_int
    ] * 3
    library.webp_reconstruct.restype = ctypes.c_int
    return library


def supports(tensor: Tensor, quality: int) -> bool:
    """Whether prepare can take this batch: built library, opaque RGB
    float32 on CUDA, sides 1 to 2048, quality 40 to 100."""
    if tensor.dim() != 4 or tensor.shape[1] != 3:
        return False
    if not tensor.is_cuda or tensor.dtype != torch.float32:
        return False
    if not 1 <= min(tensor.shape[-2:]) <= max(tensor.shape[-2:]) <= _MAX_SIDE:
        return False
    return quality in _QUALITIES and load_library() is not None


def _planar_size(height: int, width: int) -> int:
    return height * width + 2 * ((height + 1) // 2) * ((width + 1) // 2)


@cache
def _gamma_tables(device: torch.device) -> tuple[Tensor, Tensor]:
    """libwebp's tables for averaging chroma in linear light: 8-bit value →
    12-bit linear, and 33 interpolation points back to 8 bits."""
    values = np.arange(256, dtype=np.float64)
    to_linear = ((values / 255) ** 0.8 * 4095 + 0.5).astype(np.int32)
    points = np.arange(33, dtype=np.float64)
    from_linear = (255 * (128 * points / 4095) ** 1.25 + 0.5).astype(np.int32)
    return (
        torch.from_numpy(to_linear).to(device),
        torch.from_numpy(from_linear).to(device),
    )


@triton.jit
def _from_linear(table: tl.tensor, total: tl.tensor) -> tl.tensor:
    index = total >> 9
    fraction = total & 511
    low = tl.load(table + index)
    high = tl.load(table + index + 1)
    return (low * (512 - fraction) + high * fraction + 64) >> 7


@triton.jit(do_not_specialize=["h", "w", "ch", "cw", "sn", "sc", "sh", "sw"])
def _rgb_to_planes(
    image: tl.tensor,
    planes: tl.tensor,
    to_linear: tl.tensor,
    from_linear: tl.tensor,
    h: int,
    w: int,
    ch: int,
    cw: int,
    sn: int,
    sc: int,
    sh: int,
    sw: int,
    block: tl.constexpr,
) -> None:
    """One program per block of chroma samples of one image: the four luma
    samples of each 2x2 cell (clamped at odd edges) and its U and V."""
    n = tl.program_id(1)
    k = tl.program_id(0) * block + tl.arange(0, block)
    size = h * w + 2 * ch * cw
    valid = k < ch * cw
    cell = tl.minimum(k, ch * cw - 1)
    cy = cell // cw
    cx = cell % cw
    sum_r = tl.zeros((block,), tl.int32)
    sum_g = tl.zeros((block,), tl.int32)
    sum_b = tl.zeros((block,), tl.int32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            y = 2 * cy + dy
            x = 2 * cx + dx
            offset = n * sn + tl.minimum(y, h - 1) * sh + tl.minimum(x, w - 1) * sw
            # The trainer's byte conversion: clamp, scale, truncate.
            r = (tl.clamp(tl.load(image + offset), 0.0, 1.0) * 255.0).to(tl.int32)
            g = (tl.clamp(tl.load(image + offset + sc), 0.0, 1.0) * 255.0).to(tl.int32)
            b = (tl.clamp(tl.load(image + offset + 2 * sc), 0.0, 1.0) * 255.0).to(
                tl.int32
            )
            luma = (16839 * r + 33059 * g + 6420 * b + 1081344) >> 16
            tl.store(planes + n * size + y * w + x, luma, valid & (y < h) & (x < w))
            sum_r += tl.load(to_linear + r)
            sum_g += tl.load(to_linear + g)
            sum_b += tl.load(to_linear + b)
    r = _from_linear(from_linear, sum_r)
    g = _from_linear(from_linear, sum_g)
    b = _from_linear(from_linear, sum_b)
    u = (-9719 * r - 19081 * g + 28800 * b + 33685504) >> 18
    v = (28800 * r - 24116 * g - 4684 * b + 33685504) >> 18
    chroma = planes + n * size + h * w + k
    tl.store(chroma, tl.clamp(u, 0, 255), valid)
    tl.store(chroma + ch * cw, tl.clamp(v, 0, 255), valid)


@triton.jit
def _upsample(
    plane: tl.tensor,
    cy: tl.tensor,
    cx: tl.tensor,
    ny: tl.tensor,
    nx: tl.tensor,
    cw: int,
) -> tl.tensor:
    """libwebp's fancy upsampling: 9/3/3/1 weights of the nearest chroma
    sample and its neighbours towards the luma sample."""
    a = tl.load(plane + cy * cw + cx).to(tl.int32)
    b = tl.load(plane + cy * cw + nx).to(tl.int32)
    c = tl.load(plane + ny * cw + cx).to(tl.int32)
    d = tl.load(plane + ny * cw + nx).to(tl.int32)
    return (9 * a + 3 * (b + c) + d + 8) >> 4


@triton.jit(do_not_specialize=["h", "w", "ch", "cw"])
def _planes_to_rgb(
    planes: tl.tensor,
    image: tl.tensor,
    h: int,
    w: int,
    ch: int,
    cw: int,
    block: tl.constexpr,
) -> None:
    """One program per block of pixels of one image: contiguous NCHW float32 in
    [0, 1], each 8-bit value divided exactly by 255."""
    n = tl.program_id(1)
    k = tl.program_id(0) * block + tl.arange(0, block)
    valid = k < h * w
    pixel = tl.minimum(k, h * w - 1)
    row = pixel // w
    col = pixel % w
    cy = row // 2
    cx = col // 2
    ny = tl.clamp(cy + 2 * (row % 2) - 1, 0, ch - 1)
    nx = tl.clamp(cx + 2 * (col % 2) - 1, 0, cw - 1)
    source = planes + n * (h * w + 2 * ch * cw)
    luma = (tl.load(source + pixel).to(tl.int32) * 19077) >> 8
    u = _upsample(source + h * w, cy, cx, ny, nx, cw)
    v = _upsample(source + h * w + ch * cw, cy, cx, ny, nx, cw)
    r = luma + ((v * 26149) >> 8) - 14234
    g = luma - ((u * 6419) >> 8) - ((v * 13320) >> 8) + 8708
    b = luma + ((u * 33050) >> 8) - 17685
    target = image + n * 3 * h * w + k
    for c in tl.static_range(3):
        channel = r if c == 0 else (g if c == 1 else b)
        value = tl.clamp(channel >> 6, 0, 255).to(tl.float32)
        tl.store(target + c * h * w, tl.div_rn(value, 255.0), valid)


def _to_planes(tensor: Tensor) -> Tensor:
    """BCHW RGB float32 → (B, packed planar YUV420 bytes) uint8 on its device."""
    b, _, h, w = tensor.shape
    ch, cw = (h + 1) // 2, (w + 1) // 2
    planes = torch.empty(
        (b, _planar_size(h, w)), dtype=torch.uint8, device=tensor.device
    )
    to_linear, from_linear = _gamma_tables(tensor.device)
    _rgb_to_planes[(triton.cdiv(ch * cw, _BLOCK), b)](
        tensor,
        planes,
        to_linear,
        from_linear,
        h,
        w,
        ch,
        cw,
        *tensor.stride(),
        block=_BLOCK,
        enable_fp_fusion=False,
    )
    return planes


def _to_rgb(planes: Tensor, height: int, width: int) -> Tensor:
    """(B, packed planar YUV420 bytes) on the device → BCHW RGB float32."""
    b = planes.shape[0]
    image = torch.empty(
        (b, 3, height, width), dtype=torch.float32, device=planes.device
    )
    _planes_to_rgb[(triton.cdiv(height * width, _BLOCK), b)](
        planes,
        image,
        height,
        width,
        (height + 1) // 2,
        (width + 1) // 2,
        block=_BLOCK,
        enable_fp_fusion=False,
    )
    return image


def prepare(tensor: Tensor, quality: int) -> tuple[RoundTrip, Finish]:
    """The WebP round trip of every image of a batch that supports() takes,
    in the three steps of otf_degradations.prepare_webp_batch: this call
    converts to planes and starts their copy to pinned CPU memory, the round
    trip reconstructs each image on the per-image pool, the finish copies the
    planes back and converts them to RGB on the tensor's device."""
    library = load_library()
    if library is None:
        raise RuntimeError(f"{_LIBRARY} has not been built")
    device = tensor.device
    height, width = tensor.shape[-2:]
    source, ready = stage_to_host(_to_planes(tensor))

    def round_trip() -> Tensor:
        if ready is not None:
            ready.synchronize()
        target = torch.empty_like(source, pin_memory=True)
        source_np, target_np = source.numpy(), target.numpy()

        def one(i: int) -> None:
            if not library.webp_reconstruct(
                source_np[i].ctypes.data,
                target_np[i].ctypes.data,
                width,
                height,
                quality,
            ):
                raise RuntimeError(
                    f"WebP reconstruction failed ({width}x{height}, quality {quality})"
                )

        for future in [
            _parallel_pool().submit(one, i) for i in range(source_np.shape[0])
        ]:
            future.result()
        return target

    def finish(planes: Tensor) -> Tensor:
        return _to_rgb(planes.to(device, non_blocking=True), height, width)

    return round_trip, finish

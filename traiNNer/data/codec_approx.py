"""GPU approximations of the video codec and WebP round trips.

approximate_video stands in for one prepare_video_batch call with its round
trip and finish (compress_video_batch.py), approximate_webp for the Pillow
WebP round trip of compress_webp applied to every image. Both are pixel models
with constants fitted against the genuine codecs (codec_approx_constants.json),
not encoders: their damage is of the same kind and of a similar size, the
pixels differ from the codec's.

- Independent frames (stills) are coded intra-only, with the still
  calibration and, for subsampled chroma, the nearest-sample integer colour
  conversion of a software decoder.
- Clips (temporal=True, batch dimension = time) are coded in time order: the
  mirrored lead-in frames first, then every frame predicted from the previous
  decoded frame with per-block motion, with the clip calibration. The lead-in
  frames are dropped from the result, as in the genuine round trip.

Both functions take and return float32 BCHW RGB on CUDA. The input is
truncated to 8 bits as the genuine round trip does, and the result is on the
8-bit grid (level / 255), contiguous, on the input's device, computed on the
current CUDA stream. can_approximate says whether a call is covered; the
kernels need Triton and compile on first use.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

from traiNNer.data.compress_video_batch import _filter_sampling
from traiNNer.utils.rng import RNG

VIDEO_CODECS = ("h264", "hevc", "mpeg2", "mpeg4", "vp9")

# Transform parameters of the model where the calibration has no entry.
_DEFAULT_SKIP = 0.22
_DEFAULT_CHROMA_STEP = 1.12

# Colour reconstruction modes named by the still calibration, as
# decoder_colors flags of the unpack kernel.
_COLOR_MODES = {"smooth_float": False, "nearest_decoder": True}

# Kernel index arithmetic is 32-bit and the frame axis is a CUDA grid axis.
_MAX_ELEMENTS = 2**31
_MAX_GRID_FRAMES = 65535 // 2
_MAX_HEIGHT = 65535 * 8


@dataclass(frozen=True)
class ClipTuning:
    """Clip transform parameters of one codec."""

    skip: float  # residual skip threshold, in quantizer steps
    chroma: float  # chroma quantizer step relative to luma
    bias_by_sampling: dict[str, float]  # level offset before the final rounding


@dataclass(frozen=True)
class Calibration:
    """Fitted constants of the approximations."""

    still_strength: dict[str, float]  # quantizer scale per codec, stills
    still_decoder_colors: dict[str, bool]  # per chroma sampling, stills
    clip_strength: dict[str, float]  # quantizer scale per codec, clips
    clip_tuning: dict[str, ClipTuning]  # codecs without an entry use defaults
    webp_scale: float  # quantizer scale of the WebP model


@cache
def calibration() -> Calibration:
    """The constants of codec_approx_constants.json."""
    data = json.loads(
        Path(__file__).with_name("codec_approx_constants.json").read_text()
    )
    names = data["color_mode_names"]
    return Calibration(
        still_strength=data["still"]["strength"],
        still_decoder_colors={
            sampling: _COLOR_MODES[names[mode]]
            for sampling, mode in data["still"]["color_mode"].items()
        },
        clip_strength=data["clip"]["strength"],
        clip_tuning={
            codec: ClipTuning(item["skip"], item["chroma"], item["bias_by_sampling"])
            for codec, item in data["clip"]["tuning"].items()
        },
        webp_scale=data["webp"]["scale"],
    )


@cache
def _triton_available() -> bool:
    try:
        importlib.import_module("triton")
    except ImportError:
        return False
    return True


def can_approximate(
    frames: Tensor, codec: str, temporal: bool = False, lead_in: int = 0
) -> bool:
    """Whether approximate_video (a video codec) or approximate_webp
    (codec "webp") covers this call.

    It needs Triton, float32 BCHW RGB on CUDA, a codec in the calibration and
    sizes within the kernels' limits. For video, temporal=True needs
    0 <= lead_in < frames.shape[0] and temporal=False needs lead_in == 0, as
    prepare_video_batch does. The WebP model codes every image on its own and
    ignores temporal and lead_in.
    """
    if not (
        frames.is_cuda
        and frames.dtype == torch.float32
        and frames.ndim == 4
        and frames.shape[1] == 3
        and frames.numel() > 0
    ):
        return False
    t, _, h, w = frames.shape
    if codec == "webp":
        lead_in = 0
    elif codec not in VIDEO_CODECS:
        return False
    elif temporal:
        if not 0 <= lead_in < t:
            return False
    elif lead_in != 0:
        return False
    reach = sum(
        (size - 1) * stride
        for size, stride in zip(frames.shape, frames.stride(), strict=True)
    )
    return (
        _triton_available()
        and reach < _MAX_ELEMENTS
        and (t + lead_in) * 3 * h * w < _MAX_ELEMENTS
        and t + lead_in <= _MAX_GRID_FRAMES
        and h <= _MAX_HEIGHT
    )


def _quantizer_step(codec: str, quality: int) -> float:
    """Base quantizer step (in [0, 1] pixel units) of a codec's quality value:
    qscale for MPEG-2/MPEG-4, CRF otherwise (VP9 mapped onto the H.264 scale)."""
    if codec in ("mpeg2", "mpeg4"):
        return max(1.0, 1.7 * quality + 2.3) / 255
    scale = 9.0 + 0.62 * quality if codec == "vp9" else quality
    return (2 ** ((scale - 4) / 6)) / 255


def approximate_video(
    frames: Tensor,
    codec: str = "h264",
    quality: int = 28,
    video_sampling: list[str] | None = None,
    temporal: bool = False,
    lead_in: int = 0,
) -> Tensor:
    """GPU stand-in for finish(round_trip()) of prepare_video_batch with the
    same arguments.

    The chroma sampling is drawn from the project RNG exactly as the genuine
    path draws it (one choice among the codec's supported options when
    video_sampling is given, 4:2:0 and no draw otherwise); there are no other
    draws. The call must satisfy can_approximate.
    """
    if not can_approximate(frames, codec, temporal, lead_in):
        raise ValueError(
            f"approximate_video does not cover codec {codec!r} with temporal="
            f"{temporal}, lead_in={lead_in} for a {frames.dtype} tensor of shape "
            f"{tuple(frames.shape)} on {frames.device}"
        )
    from traiNNer.data import codec_approx_kernels as kernels

    sampling = (
        str(RNG.get_rng().choice(_filter_sampling(video_sampling, codec)))
        if video_sampling
        else "420"
    )
    sy = 2 if sampling == "420" else 1
    sx = 1 if sampling == "444" else 2
    constants = calibration()
    if temporal:
        strength = constants.clip_strength[codec]
        tuning = constants.clip_tuning.get(codec)
        if tuning is None:
            skip, chroma, bias = _DEFAULT_SKIP, _DEFAULT_CHROMA_STEP, 0.0
        else:
            skip = tuning.skip
            chroma = tuning.chroma
            bias = tuning.bias_by_sampling.get(sampling, 0.0)
        decoder_colors = False
    else:
        strength = constants.still_strength[codec]
        skip, chroma, bias = _DEFAULT_SKIP, _DEFAULT_CHROMA_STEP, 0.0
        decoder_colors = constants.still_decoder_colors[sampling]
    mpeg = codec in ("mpeg2", "mpeg4")
    block = 8 if mpeg or codec in ("vp9", "hevc") else 4
    step = strength * _quantizer_step(codec, quality)

    y, uv = kernels.pack(frames, lead_in, sy, sx, 0.0)
    field = kernels.motion(y) if temporal else None
    y = kernels.transform(y, field, step, block, mpeg, 1, 1, skip)
    uv = kernels.transform(uv, field, step * chroma, block, mpeg, sy, sx, skip)
    return kernels.unpack(y, uv, lead_in, sy, sx, 0.0, bias, decoder_colors)


def approximate_webp(image: Tensor, quality: int) -> Tensor:
    """GPU stand-in for the WebP round trip (compress_webp, Pillow method=0)
    of every image of the batch, 4:2:0 chroma. No random draws. The call must
    satisfy can_approximate(image, "webp")."""
    if not can_approximate(image, "webp"):
        raise ValueError(
            f"approximate_webp does not cover a {image.dtype} tensor of shape "
            f"{tuple(image.shape)} on {image.device}"
        )
    from traiNNer.data import codec_approx_kernels as kernels

    step = calibration().webp_scale * (
        0.007 + 0.115 * ((100.0 - float(quality)) / 100.0) ** 1.35
    )
    y, uv = kernels.pack(image, 0, 2, 2, 0.5)
    y = kernels.webp_plane(y, step)
    uv = kernels.webp_plane(uv, step * 0.85)
    return kernels.unpack(y, uv, 0, 2, 2, 0.5, 0.0, False)

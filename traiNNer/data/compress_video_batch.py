"""Batch video compression — optimized implementations per codec.

Operates on BCHW GPU tensors directly. No apply_per_image bridge.

Strategy (from exhaustive benchmarking):
- TorchCodec BATCH for: libx264, libx265, mpeg4
  (native tensor I/O, single encode call for entire batch)
- PyAV BATCH for: libvpx-vp9 (needs cpu-used option), mpeg2video
  (single container, all frames, in-process)

IMPORTANT: All frames are encoded as intra-only (gop_size=1, no B-frames).
This ensures each frame is independently compressed — batch-as-video is purely
a performance optimization, not a semantic change from per-image encoding.
One exception: x264 and x265 in CRF mode set each frame's QP from the running
complexity of the frames before it in the batch, so the same frame can get a
different QP depending on its neighbours (prediction stays intra).

With temporal=True the batch is instead one clip in time order and the codec
predicts between frames in a stream shaped like a real encode (x264 veryfast,
x265 ultrafast, mpeg2 and mpeg4 a GOP of 12 with 2 B-frames, vp9 its defaults); lead_in frames
mirrored from the clip's start are encoded first and dropped after decoding,
so the clip need not open on a keyframe.

Thread counts come from _thread_counts. None of them changes the decoded
frames. Outside clip mode the vp9 and mpeg2 batches are encoded as chunks of
_CHUNK_FRAMES frames in parallel: their frames are intra-only at a quality
that does not look across frames, so the chunks decode to the same frames as
one stream does (x264 and x265 in CRF mode do look across frames and stay one
stream).
"""

from __future__ import annotations

import io
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from traiNNer.utils.rng import RNG

# Codec → TorchCodec format mapping
_TC_CODEC_MAP = {
    "h264": ("libx264", "mp4"),
    "hevc": ("libx265", "mp4"),
    "mpeg4": ("mpeg4", "mp4"),
}

# Codec → PyAV format/options mapping
_PYAV_CODEC_MAP = {
    "vp9": ("libvpx-vp9", "webm"),
    "mpeg2": ("mpeg2video", "mpegts"),
}

# Video subsampling map (matches traiNNer-redux VIDEO_SUBSAMPLING)
_VIDEO_SUBSAMPLING = {
    "444": "yuv444p",
    "422": "yuv422p",
    "420": "yuv420p",
}

# Supported pixel formats per codec (subset of _VIDEO_SUBSAMPLING keys)
# Codecs reject unsupported formats at encode time — filter before selection.
_CODEC_SUPPORTED_SAMPLING: dict[str, set[str]] = {
    "h264": {"444", "422", "420"},
    "hevc": {"444", "422", "420"},
    "mpeg4": {"420"},              # only yuv420p
    "vp9": {"444", "422", "420"},
    "mpeg2": {"422", "420"},       # no yuv444p
}


def _filter_sampling(
    video_sampling: list[str], codec: str
) -> list[str]:
    """Filter sampling options to only those supported by the codec."""
    supported = _CODEC_SUPPORTED_SAMPLING.get(codec)
    if supported is None:
        return video_sampling
    filtered = [s for s in video_sampling if s in supported]
    return filtered if filtered else ["420"]  # fallback to 420 if nothing matches


# The steps of one compression after the work on the calling thread: the
# encode/decode round trip, then the finish that takes its frames.
RoundTrip = Callable[[], Tensor]
Finish = Callable[[Tensor], Tensor]


def usable_cpus() -> int:
    """The CPUs this process may run on: its affinity mask where the OS has
    one (training may be pinned), otherwise os.cpu_count()."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class _ThreadCounts(NamedTuple):
    x265_pools: int
    x265_frame_threads: int
    vp9_encoder: int
    decoder: int


def _thread_counts(
    batch: int, height: int, width: int, jobs: int = 1
) -> _ThreadCounts:
    """Encoder and decoder thread counts for one call, one of `jobs` calls
    running side by side, which share the cores for decoding.

    The codec defaults suit long videos, not a batch of small intra-only
    frames: x265 runs one frame thread next to a pool of one worker per core,
    and libvpx starts one thread per core, whose start-up costs more than it
    saves on frames up to 256². Intra-only frames are independent, so frame
    threads (x265 encode, vp9 and TorchCodec decode) spread the batch over the
    cores.

    The counts derive from the batch size and from usable_cpus(). On a
    16-core/32-thread CPU, batch 8: x265 pool 8 and 8 frame threads, libvpx 1
    thread up to 256² and 4 above, 8 decode threads (vp9, h264, hevc, mpeg4);
    with 8 jobs side by side, 4 decode threads each.
    """
    cpus = usable_cpus()
    pools = min(8, cpus)
    return _ThreadCounts(
        x265_pools=pools,
        x265_frame_threads=min(batch, pools),
        vp9_encoder=1 if height * width <= 256 * 256 else min(4, cpus),
        decoder=max(1, min(8, cpus // jobs)),
    )


# Frames per chunk of a vp9 or mpeg2 batch encoded outside clip mode.
_CHUNK_FRAMES = 8
_CHUNK_POOL: list[ThreadPoolExecutor] = []
_CHUNK_POOL_LOCK = threading.Lock()


def _chunk_pool(chunks: int) -> ThreadPoolExecutor:
    """The pool the chunks of a batch run in, created on first use with room
    for two batches' chunks at once. It is not the caller's pool: a round
    trip waiting there for its chunks could hold every worker they need."""
    with _CHUNK_POOL_LOCK:
        if not _CHUNK_POOL:
            _CHUNK_POOL.append(
                ThreadPoolExecutor(
                    max_workers=min(2 * chunks, usable_cpus()),
                    thread_name_prefix="video-chunk",
                )
            )
        return _CHUNK_POOL[0]


def _to_unit_float(frames: Tensor, device: torch.device) -> Tensor:
    """BCHW uint8 CPU frames → contiguous float32 [0, 1] on `device`.

    The frames cross to the device as uint8 (a quarter of the float32 bytes).
    The divisor is a 0-dim device tensor because CUDA turns a Python-scalar
    divisor into a multiply by its reciprocal, which is 1 ulp off true division
    for about half of the values 0..255.
    """
    frames = frames.to(device)
    divisor = torch.full((), 255.0, device=device)
    return frames.to(torch.float32, memory_format=torch.contiguous_format).div_(divisor)


_COPY_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def stage_to_host(frames: Tensor) -> tuple[Tensor, torch.cuda.Event | None]:
    """Start copying uint8 `frames` to the CPU and return the CPU tensor and
    the event that marks the end of the copy.

    CUDA frames go to pinned memory by a non-blocking copy on a copy stream
    of their device, after the work queued so far on the current stream, so
    the calling thread does not wait for the device. Whoever reads the CPU
    tensor first waits on the event (ready.synchronize()). CPU frames are
    returned as they are, with no event.
    """
    if not frames.is_cuda:
        return frames, None
    device = frames.device
    copy = _COPY_STREAMS.get(device)
    if copy is None:
        copy = _COPY_STREAMS[device] = torch.cuda.Stream(device=device)
    host = torch.empty_like(frames, device="cpu", pin_memory=True)
    copy.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(copy):
        host.copy_(frames, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(copy)
    # The current stream may reuse the frames' memory once it is freed; the
    # copy stream still reads it.
    frames.record_stream(copy)
    return host, ready


def _with_lead_in(tensor: Tensor, temporal: bool, lead_in: int) -> Tensor:
    """Prepend frames lead_in, ..., 1 of the clip before frame 0."""
    if lead_in == 0:
        return tensor
    if not temporal:
        raise ValueError("lead_in needs temporal=True")
    if not 0 < lead_in < tensor.shape[0]:
        raise ValueError(
            f"lead_in must be in [0, {tensor.shape[0] - 1}] for a clip of "
            f"{tensor.shape[0]} frames, got {lead_in}"
        )
    return torch.cat([tensor[1 : lead_in + 1].flip(0), tensor])


def _finish(lead_in: int, height: int, width: int, device: torch.device) -> Finish:
    """The step after the round trip: crop the padding, drop the lead-in frames
    and convert back to float on `device`.

    For a CUDA device the kept frames cross as one dense block, without a
    wait when they are pinned, and the padding is cropped on the device.
    """

    def finish(frames: Tensor) -> Tensor:
        if device.type == "cuda":
            kept = frames[lead_in:].to(device, non_blocking=frames.is_pinned())
            return _to_unit_float(kept[:, :, :height, :width], device)
        return _to_unit_float(frames[lead_in:, :, :height, :width], device)

    return finish


def _prepare_torchcodec(
    tensor: Tensor,
    codec: str,
    quality: int,
    video_sampling: list[str] | None,
    temporal: bool,
    lead_in: int,
    jobs: int,
) -> tuple[RoundTrip, Finish]:
    """prepare_video_batch for the TorchCodec codecs (h264, hevc, mpeg4)."""
    from torchcodec.decoders import VideoDecoder
    from torchcodec.encoders import VideoEncoder

    if codec not in _TC_CODEC_MAP:
        raise ValueError(
            f"TorchCodec batch: unsupported codec '{codec}'. "
            f"Use one of: {list(_TC_CODEC_MAP.keys())}"
        )

    device = tensor.device
    codec_name, fmt = _TC_CODEC_MAP[codec]
    tensor = _with_lead_in(tensor, temporal, lead_in)
    b, _, orig_h, orig_w = tensor.shape

    # Chroma subsampling selection (filter to codec-supported formats)
    if video_sampling:
        filtered = _filter_sampling(video_sampling, codec)
        sampling_key = RNG.get_rng().choice(filtered)
        pix_fmt = _VIDEO_SUBSAMPLING.get(sampling_key, "yuv420p")
    else:
        pix_fmt = None  # codec default

    # Pad to even dims if needed by chroma subsampling (trainer generates
    # arbitrary resize dims — odd sizes are expected, not edge cases)
    pad_h = pad_w = 0
    if pix_fmt and pix_fmt != "yuv444p":
        need_w = 2 if "420" in pix_fmt or "422" in pix_fmt else 1
        need_h = 2 if "420" in pix_fmt else 1
        pad_w = (need_w - orig_w % need_w) % need_w
        pad_h = (need_h - orig_h % need_h) % need_h

    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # BCHW float [0,1] → BCHW uint8, on its way to the CPU
    inp, ready = stage_to_host((tensor.clamp(0, 1) * 255).byte())
    pinned = ready is not None

    # Build encode kwargs — gop=1 ensures every frame is an I-frame. Temporal
    # streams are shaped like real encodes: x264 veryfast and x265 ultrafast
    # (B-frames, deblocking; x265 veryfast takes about twice as long), mpeg4
    # FFmpeg's GOP of 12 with 2 B-frames.
    if not temporal:
        extra_options: dict[str, str] = {"g": "1", "bf": "0"}
    elif codec_name == "mpeg4":
        extra_options = {"g": "12", "bf": "2"}
    else:
        extra_options = {}
    preset = "veryfast" if temporal and codec_name == "libx264" else "ultrafast"
    if codec_name == "libx264":
        extra_options["preset"] = preset
    elif codec_name == "libx265":
        threads = _thread_counts(b, orig_h, orig_w)
        extra_options["preset"] = preset
        extra_options["x265-params"] = (
            f"log-level=0:pools={threads.x265_pools}"
            f":frame-threads={threads.x265_frame_threads}"
        )

    encode_kwargs: dict = {
        "format": fmt,
        "codec": codec_name,
        "extra_options": extra_options,
    }
    if codec_name == "mpeg4":
        # The mpeg4 encoder has no CRF option and ignores `crf`; its quality is
        # a fixed qscale, as in compress_video.
        extra_options["qmin"] = extra_options["qmax"] = str(quality)
    else:
        encode_kwargs["crf"] = quality

    if pix_fmt:
        encode_kwargs["pixel_format"] = pix_fmt
    decode_threads = _thread_counts(b, orig_h, orig_w, jobs).decoder

    def round_trip() -> Tensor:
        if ready is not None:
            ready.synchronize()
        # Encode the entire batch as one video, decode all frames back in one
        # pass on FFmpeg's threads (seeking to each frame decodes serially).
        encoded = VideoEncoder(frames=inp, frame_rate=1).to_tensor(**encode_kwargs)
        dec = VideoDecoder(encoded, num_ffmpeg_threads=decode_threads)
        frames = dec.get_frames_in_range(0, len(dec)).data
        # Pinned frames return to the device without a wait.
        out = torch.empty(frames.shape, dtype=torch.uint8, pin_memory=pinned)
        return out.copy_(frames)

    return round_trip, _finish(lead_in, orig_h, orig_w, device)


def _prepare_pyav(
    tensor: Tensor,
    codec: str,
    quality: int,
    video_sampling: list[str] | None,
    temporal: bool,
    lead_in: int,
) -> tuple[RoundTrip, Finish]:
    """prepare_video_batch for the PyAV codecs (vp9, mpeg2)."""
    import av
    from av.video.reformatter import VideoReformatter

    if codec not in _PYAV_CODEC_MAP:
        raise ValueError(
            f"PyAV batch: unsupported codec '{codec}'. "
            f"Use one of: {list(_PYAV_CODEC_MAP.keys())}"
        )

    device = tensor.device
    codec_name, fmt = _PYAV_CODEC_MAP[codec]
    # PyAV encodes RGB24. Interleave the source pixels on the GPU once so each
    # CPU frame can borrow that buffer, including mirrored lead-in frames.
    packed_rgb = (
        temporal and tensor.is_cuda and tensor.dtype == torch.float32
        and hasattr(av.VideoFrame, "from_numpy_buffer")
    )
    if packed_rgb:
        tensor = tensor.clamp(0, 1).mul_(255).to(torch.uint8, memory_format=torch.channels_last)
    tensor = _with_lead_in(tensor, temporal, lead_in)
    b, _, h, w = tensor.shape

    # Keep the staged layout so the codec worker can borrow interleaved frames.
    inp, ready = stage_to_host(tensor if packed_rgb else (tensor.clamp(0, 1) * 255).byte())
    pinned = ready is not None

    # Chroma subsampling (filter to codec-supported formats)
    if video_sampling:
        filtered = _filter_sampling(video_sampling, codec)
        sampling_key = RNG.get_rng().choice(filtered)
        pix_fmt = _VIDEO_SUBSAMPLING.get(sampling_key, "yuv420p")
    else:
        pix_fmt = "yuv420p"

    # Pad to even dims if needed by chroma subsampling (in the round trip,
    # once the frames are on the CPU)
    pad_h = h % 2 if "420" in pix_fmt else 0
    pad_w = w % 2 if "420" in pix_fmt or "422" in pix_fmt else 0
    h_padded, w_padded = h + pad_h, w + pad_w

    # Build codec options and encoder thread count
    threads = _thread_counts(b, h_padded, w_padded)
    if codec == "vp9":
        options = {"cpu-used": "8", "crf": str(quality), "row-mt": "1"}
        enc_threads = threads.vp9_encoder
    else:
        # Fixed qscale. The slice encoder's default of one thread per core
        # (capped at 16) is several times slower than 1 thread on these frames.
        options = {"qmax": str(quality), "qmin": str(quality)}
        enc_threads = 1

    def encode_decode(frames_np: np.ndarray) -> list[np.ndarray]:
        """Encode the frames into one container and decode them back, HWC."""
        buf = io.BytesIO()
        container = av.open(buf, mode="w", format=fmt)
        stream = container.add_stream(codec_name, rate=1)
        stream.width = w_padded
        stream.height = h_padded
        stream.pix_fmt = pix_fmt
        if not temporal:
            stream.gop_size = 1  # Every frame is an I-frame
        elif codec == "mpeg2":
            # Shaped like a real encode: FFmpeg's GOP of 12 with 2 B-frames.
            stream.gop_size = 12
            stream.codec_context.max_b_frames = 2
        stream.options = options
        stream.codec_context.thread_count = enc_threads

        for frame_hwc in frames_np:
            if packed_rgb and frame_hwc.strides[1:] == (3, 1):
                vf = av.VideoFrame.from_numpy_buffer(frame_hwc, format="rgb24")
            else:
                vf = av.VideoFrame.from_ndarray(frame_hwc, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

        # Decode all frames back
        buf.seek(0)
        dec_container = av.open(buf)
        dec_ctx = dec_container.streams.video[0].codec_context
        if codec == "vp9":
            # Intra-only frames decode in parallel on frame threads; predicted
            # frames wait for their references.
            dec_ctx.thread_type = "FRAME"
            dec_ctx.thread_count = threads.decoder
        else:
            dec_ctx.thread_count = 1
        # One converter for the batch: frame.to_ndarray(format=...) builds a new
        # swscale context for every frame.
        to_rgb = VideoReformatter()
        frames = [
            to_rgb.reformat(frame, format="rgb24").to_ndarray()
            for frame in dec_container.decode(video=0)
        ]
        dec_container.close()
        return frames

    def round_trip() -> Tensor:
        if ready is not None:
            ready.synchronize()
        inp_np = inp.permute(0, 2, 3, 1).numpy()
        if pad_h or pad_w:
            inp_np = np.pad(
                inp_np, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
            )
        if temporal or b <= _CHUNK_FRAMES:
            frames = encode_decode(inp_np)
        else:
            starts = range(0, b, _CHUNK_FRAMES)
            pool = _chunk_pool(len(starts))
            chunks = [
                pool.submit(encode_decode, inp_np[s : s + _CHUNK_FRAMES])
                for s in starts
            ]
            frames = [frame for chunk in chunks for frame in chunk.result()]
        # Pinned frames return to the device without a wait.
        out = torch.empty(
            (len(frames), *frames[0].shape), dtype=torch.uint8, pin_memory=pinned
        )
        np.stack(frames, out=out.numpy())
        # HWC → CHW
        return out.permute(0, 3, 1, 2)

    return round_trip, _finish(lead_in, h, w, device)


def prepare_video_batch(
    tensor: Tensor,
    codec: str = "h264",
    quality: int = 28,
    video_sampling: list[str] | None = None,
    temporal: bool = False,
    lead_in: int = 0,
    jobs: int = 1,
) -> tuple[RoundTrip, Finish]:
    """compress_video_batch in three steps, so the codec work can run elsewhere.

    This call does what comes before the encode, on the calling thread: the
    lead-in frames, the chroma sampling draw from the project RNG, padding and
    the conversion to uint8, whose copy to the CPU is only started for CUDA
    tensors (stage_to_host). It returns the round trip, which waits for that
    copy, then only encodes and decodes CPU frames (no random draws, no device
    work, no shared state, so it may run in a worker thread) into pinned
    memory for CUDA tensors, and the finish, which crops the padding, drops
    the lead-in and converts back to float on the tensor's device; call it on
    the thread that called this. finish(round_trip()) is
    compress_video_batch(...) with the same arguments. jobs is the number of
    round trips that run side by side, which share the TorchCodec decode
    threads; it does not change the frames.
    """
    if codec in _TC_CODEC_MAP:
        return _prepare_torchcodec(
            tensor, codec, quality, video_sampling, temporal, lead_in, jobs
        )
    elif codec in _PYAV_CODEC_MAP:
        return _prepare_pyav(tensor, codec, quality, video_sampling, temporal, lead_in)
    else:
        raise ValueError(
            f"Unknown codec '{codec}'. "
            f"Supported: {list(_TC_CODEC_MAP.keys()) + list(_PYAV_CODEC_MAP.keys())}"
        )


def compress_video_batch(
    tensor: Tensor,
    codec: str = "h264",
    quality: int = 28,
    video_sampling: list[str] | None = None,
    temporal: bool = False,
    lead_in: int = 0,
) -> Tensor:
    """Unified batch video compression — dispatches to best backend per codec.

    By default all frames are encoded as intra-only (gop_size=1, no B-frames)
    to ensure semantic equivalence with per-image compression. Chroma
    subsampling is randomly sampled from video_sampling per batch.

    Args:
        tensor: BCHW float32 tensor on GPU, values in [0, 1].
        codec: "h264", "hevc", "mpeg4", "vp9", or "mpeg2".
        quality: CRF/quality value.
        video_sampling: List of chroma subsampling options, e.g. ["444", "422", "420"].
            One is randomly chosen per batch. If None, uses codec default.
        temporal: Treat the batch as one clip in time order and let the codec
            predict between frames, in a stream shaped like a real encode.
        lead_in: With temporal, number of time-mirrored frames (lead_in, ...,
            1) encoded before frame 0 and dropped after decoding; less than
            the batch size.

    Returns:
        BCHW float32 tensor on same device, after codec roundtrip, frames in
        display order.
    """
    round_trip, finish = prepare_video_batch(
        tensor, codec, quality, video_sampling, temporal, lead_in
    )
    return finish(round_trip())


def compress_video_batch_torchcodec(
    tensor: Tensor,
    codec: str = "h264",
    quality: int = 28,
    video_sampling: list[str] | None = None,
    temporal: bool = False,
    lead_in: int = 0,
) -> Tensor:
    """Run the TorchCodec video codec round trip synchronously."""
    round_trip, finish = _prepare_torchcodec(
        tensor, codec, quality, video_sampling, temporal, lead_in, jobs=1
    )
    return finish(round_trip())


def compress_video_batch_pyav(
    tensor: Tensor,
    codec: str = "vp9",
    quality: int = 28,
    video_sampling: list[str] | None = None,
    temporal: bool = False,
    lead_in: int = 0,
) -> Tensor:
    """Run the PyAV video codec round trip synchronously."""
    round_trip, finish = _prepare_pyav(
        tensor, codec, quality, video_sampling, temporal, lead_in
    )
    return finish(round_trip())

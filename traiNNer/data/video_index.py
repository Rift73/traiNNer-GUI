"""Episode index for clip training: frame timestamps, motion, shot cuts and the
re-upscale error of sampled frames, from one decoding pass with PyAV.

Also the RGB conversion of the decoded frames: 8-bit yuv420p, limited range,
BT.709 matrix and left-sited chroma, set explicitly because the source carries
no colour tags (FFmpeg would assume BT.601).
"""

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
from av.video.frame import VideoFrame

from traiNNer.data.descale import descale, upscale

KR, KB = 0.2126, 0.0722
KG = 1 - KR - KB
MOTION_REDUCTION = 8
CUT_FLOOR = 6.0
CUT_RATIO = 2.0
CUT_WINDOW = 6
ERROR_EVERY = 8


def open_video(path: str | Path, threads: int) -> av.container.InputContainer:
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = threads
    return container


def _plane(frame: VideoFrame, i: int) -> np.ndarray:
    p = frame.planes[i]
    return np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, : p.width]


def _check_format(frame: VideoFrame) -> None:
    if frame.format.name != "yuv420p":
        raise ValueError(f"Expected yuv420p frames, got {frame.format.name}")


def luma(frame: VideoFrame) -> np.ndarray:
    """The Y plane as H,W,1 float32, limited range scaled to 0..1."""
    _check_format(frame)
    return ((_plane(frame, 0).astype(np.float32) - 16) / 219)[:, :, None]


def _upsample_chroma(c: np.ndarray) -> np.ndarray:
    """4:2:0 chroma sited left: co-sited with the even luma columns, centred
    between luma rows. Linear interpolation, edges repeated."""
    c = c.astype(np.float32)
    h, w = c.shape
    padded = np.concatenate([c[:1], c, c[-1:]], axis=0)
    rows = np.empty((2 * h, w), np.float32)
    rows[0::2] = 0.75 * padded[1:-1] + 0.25 * padded[:-2]
    rows[1::2] = 0.75 * padded[1:-1] + 0.25 * padded[2:]
    right = np.concatenate([rows[:, 1:], rows[:, -1:]], axis=1)
    out = np.empty((2 * h, 2 * w), np.float32)
    out[:, 0::2] = rows
    out[:, 1::2] = 0.5 * (rows + right)
    return out


def to_rgb(frame: VideoFrame) -> np.ndarray:
    """H,W,3 float32 RGB, not clipped."""
    _check_format(frame)
    y = (_plane(frame, 0).astype(np.float32) - 16) / 219
    cb = (_upsample_chroma(_plane(frame, 1)) - 128) / 224
    cr = (_upsample_chroma(_plane(frame, 2)) - 128) / 224
    r = y + 2 * (1 - KR) * cr
    g = y - (2 * (1 - KB) * KB / KG) * cb - (2 * (1 - KR) * KR / KG) * cr
    b = y + 2 * (1 - KB) * cb
    return np.stack([r, g, b], axis=-1)


def yuv_window(
    frame: VideoFrame, top: int, left: int, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copies of the uint8 Y, U and V planes of the luma window at (top, left);
    all four values even."""
    _check_format(frame)
    y = _plane(frame, 0)[top : top + height, left : left + width]
    rows = slice(top // 2, (top + height) // 2)
    cols = slice(left // 2, (left + width) // 2)
    u = _plane(frame, 1)[rows, cols]
    v = _plane(frame, 2)[rows, cols]
    return y.copy(), u.copy(), v.copy()


def reduced_luma(frame: VideoFrame) -> np.ndarray:
    """The Y plane averaged over MOTION_REDUCTION-pixel blocks, 0..255 units."""
    y = _plane(frame, 0)
    n = MOTION_REDUCTION
    h, w = y.shape[0] // n * n, y.shape[1] // n * n
    blocks = y[:h, :w].reshape(h // n, n, w // n, n)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def find_cuts(
    motion: list[float],
    floor: float = CUT_FLOOR,
    ratio: float = CUT_RATIO,
    window: int = CUT_WINDOW,
) -> list[int]:
    """Frames that start a shot. Frame i is a cut when its motion is at least
    floor and at least ratio times the mean of its neighbours: the two largest
    values among the window frames before it and the two largest among the
    window frames after. Taking the largest compares the change of a drawing
    animated on twos or threes with the other drawing changes of the shot, not
    with the near-still frames between them."""
    values = np.asarray(motion, dtype=np.float64)
    cuts = [0] if len(values) else []
    for i in np.flatnonzero(values >= floor):
        if i == 0:
            continue
        before = np.sort(values[max(0, i - window) : i])[-2:]
        after = np.sort(values[i + 1 : i + 1 + window])[-2:]
        near = np.concatenate([before, after])
        reference = near.mean() if len(near) else 0.0
        if values[i] >= ratio * reference:
            cuts.append(int(i))
    return cuts


def descale_error(
    frame: VideoFrame, width: int, height: int, kernel: str, border: str
) -> float:
    """99.9th percentile of the luma re-upscale error (0..1), descaled in the
    stored domain. Frames with 1080p-native text or overlays score high."""
    y = luma(frame)
    back = upscale(
        descale(y, width, height, kernel, border=border),
        y.shape[1],
        y.shape[0],
        kernel,
        border=border,
    )
    error = np.abs(back - y).ravel()
    k = int(0.999 * (error.size - 1))
    return float(np.partition(error, k)[k])


def index_video(
    path: str | Path,
    width: int = 1488,
    height: int = 837,
    kernel: str = "bilinear",
    border: str = "zero",
    threads: int = 4,
) -> dict[str, Any]:
    pts: list[int] = []
    motion: list[float] = []
    errors: list[dict[str, float]] = []
    with open_video(path, threads) as container:
        stream = container.streams.video[0]
        previous = None
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"{path}: frame {i} has no pts")
            pts.append(int(frame.pts))
            small = reduced_luma(frame)
            motion.append(
                0.0 if previous is None else float(np.abs(small - previous).mean())
            )
            previous = small
            if i % ERROR_EVERY == 0:
                p999 = descale_error(frame, width, height, kernel, border)
                errors.append({"frame": i, "p999": p999})
        time_base = stream.time_base
    if np.any(np.diff(pts) <= 0):
        raise ValueError(f"{path}: pts are not strictly increasing")
    return {
        "frames": len(pts),
        "time_base": [time_base.numerator, time_base.denominator],
        "pts": pts,
        "motion": [round(m, 4) for m in motion],
        "cuts": find_cuts(motion),
        "cut_rule": {"floor": CUT_FLOOR, "ratio": CUT_RATIO, "window": CUT_WINDOW},
        "descale": {
            "width": width,
            "height": height,
            "kernel": kernel,
            "border": border,
        },
        "descale_error": errors,
    }


def index_path(video: str | Path) -> Path:
    video = Path(video)
    return video.parent / "index" / f"{video.stem}.json"


def write_index(video: str | Path, **options: Any) -> Path:
    out = index_path(video)
    out.parent.mkdir(exist_ok=True)
    data = index_video(video, **options)
    out.write_text(json.dumps(data))
    return out

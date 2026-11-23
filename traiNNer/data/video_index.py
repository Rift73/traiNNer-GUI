"""Episode index for clip training: frame timestamps, motion, shot cuts and the
re-upscale error of sampled frames, from one decoding pass with PyAV.

Also the RGB conversion of 8-bit yuv420p frames with limited range, a BT.709
matrix and left-sited chroma. Untagged video uses these same conventions.
"""

import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import av
import numpy as np
from av.video.frame import VideoFrame
from trainner_data.file_version import artifact_version
from trainner_data.video_indexes import (
    INDEX_VERSION,
    check_color,
    clip_settings,
    clip_stream,
    index_path,
    inspect_clips,
    source_record,
    video_stream,
)

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
    try:
        stream = video_stream(container)
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
    except (ValueError, av.FFmpegError):
        container.close()
        raise
    return container


def _plane(frame: VideoFrame, i: int) -> np.ndarray:
    p = frame.planes[i]
    return np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, : p.width]


def _check_format(frame: VideoFrame) -> None:
    check_color(frame.format.name, frame.color_range, frame.colorspace)


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
    n = min(MOTION_REDUCTION, *y.shape)
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
    frame: VideoFrame,
    width: int,
    height: int,
    kernel: str,
    border: str,
    descale_linear_light: str | None = None,
) -> float:
    """99.9th percentile of the luma re-upscale error in stored values."""
    y = luma(frame)
    back = upscale(
        descale(
            y, width, height, kernel, border=border, linear_light=descale_linear_light
        ),
        y.shape[1],
        y.shape[0],
        kernel,
        border=border,
        linear_light=descale_linear_light,
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
    *,
    descale_linear_light: str | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> dict[str, Any]:
    settings = clip_settings(
        {
            "width": width,
            "height": height,
            "kernel": kernel,
            "border": border,
            "descale_linear_light": descale_linear_light,
        }
    )
    if type(threads) is not int or threads < 1:
        raise ValueError("Decoder threads must be a positive integer.")
    if check_cancel is not None:
        check_cancel()

    source = Path(path).stat()
    pts: list[int] = []
    motion: list[float] = []
    errors: list[dict[str, int | float]] = []
    with open_video(path, threads) as container:
        stream = container.streams.video[0]
        metadata = clip_stream(stream, settings)
        previous = None
        for i, frame in enumerate(container.decode(stream)):
            if check_cancel is not None:
                check_cancel()
            _check_format(frame)
            if (frame.width, frame.height) != metadata.size:
                raise ValueError(f"Frame dimensions changed at frame {i}.")
            if frame.time_base != stream.time_base:
                raise ValueError(f"Time base changed at frame {i}.")
            if frame.pts is None or (pts and frame.pts <= pts[-1]):
                raise ValueError(f"Missing or non-increasing timestamp at frame {i}.")
            pts.append(frame.pts)
            small = reduced_luma(frame)
            motion.append(
                0.0 if previous is None else float(np.abs(small - previous).mean())
            )
            previous = small
            if i % ERROR_EVERY == 0:
                p999 = descale_error(frame, **settings)
                errors.append({"frame": i, "p999": p999})
    if check_cancel is not None:
        check_cancel()
    if not pts:
        raise ValueError("The video stream has no decoded frames.")
    return {
        "version": INDEX_VERSION,
        "source": source_record(source, *metadata.size),
        "frames": len(pts),
        "time_base": list(metadata.time_base),
        "pts": pts,
        "motion": [round(m, 4) for m in motion],
        "cuts": find_cuts(motion),
        "cut_rule": {"floor": CUT_FLOOR, "ratio": CUT_RATIO, "window": CUT_WINDOW},
        "descale": settings,
        "descale_error": errors,
    }


def _index_entry_version(path: Path) -> str | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise ValueError(f"{path} must be a regular index file.")
    return artifact_version(info)


def _write_index(
    video: Path, *, check_cancel: Callable[[], None] | None = None, **options: Any
) -> tuple[Path, int]:
    if check_cancel is not None:
        check_cancel()
    out = index_path(video)
    source_version = artifact_version(video.stat())
    previous_index = _index_entry_version(out)
    out.parent.mkdir(exist_ok=True)
    temporary = None
    try:
        data = index_video(video, check_cancel=check_cancel, **options)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=out.parent, prefix=".writing-", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(data, file, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        if source_version != artifact_version(video.stat()):
            raise ValueError(f"{video} changed during indexing. Prepare it again.")
        if previous_index != _index_entry_version(out):
            raise ValueError(
                f"{out} changed during indexing. Check the current index and retry."
            )
        if check_cancel is not None:
            check_cancel()
        os.replace(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return out, data["frames"]


def write_index(video: str | Path, **options: Any) -> Path:
    """Build and atomically replace the selected video's sidecar."""
    return _write_index(Path(video), **options)[0]


def prepare_clips(
    roots: Sequence[Path],
    settings: Mapping[str, Any],
    progress: Callable[[dict[str, Any]], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Reuse ready indexes and rebuild missing or stale ones, one episode at a time."""
    options = clip_settings(settings)
    report = inspect_clips(roots, options, check_cancel=check_cancel, sample_limit=None)
    if report["errors"]:
        raise ValueError(report["errors"][0])
    episodes = report["episodes"]
    for row in episodes:
        if row["status"] == "incompatible":
            raise ValueError(row["message"])

    total, skipped = report["total"], report["ready"]
    for i, row in enumerate(episodes):
        if check_cancel is not None:
            check_cancel()
        event = {"stage": "indexing", "total": total, "path": row["path"]}
        if row["status"] == "ready":
            action = "skipped"
        else:
            if progress is not None:
                progress(dict(event, completed=i, action="indexing"))
            path = Path(row["path"])
            try:
                _, frames = _write_index(path, check_cancel=check_cancel, **options)
            except av.FFmpegError as exc:
                raise ValueError(f"{path}: indexing failed: {exc}") from exc
            row.pop("message", None)
            row.update(status="ready", frames=frames, legacy=False)
            action = "indexed"
        if progress is not None:
            progress(dict(event, completed=i + 1, action=action))
    if progress is not None:
        progress({"stage": "done", "completed": total, "total": total, "path": None})
    report.update(
        ready=total, missing=0, stale=0, indexed=total - skipped, skipped=skipped
    )
    report["episodes"] = episodes[:25]
    return report

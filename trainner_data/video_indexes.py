"""Episode selection and index validation shared by Studio and clip training."""

import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, TypedDict, cast

from trainner_data.file_version import artifact_version

VIDEO_SUFFIXES = (".mkv", ".mp4", ".m2ts", ".mov")
PIXEL_FORMAT = "yuv420p"
INDEX_VERSION = 1


class ClipSettings(TypedDict):
    width: int
    height: int
    kernel: str
    border: str
    descale_linear_light: str | None


def clip_settings(settings: Mapping[str, Any]) -> ClipSettings:
    width, height = settings.get("width", 1488), settings.get("height", 837)
    kernel, border = settings.get("kernel", "bilinear"), settings.get("border", "zero")
    light = settings.get("descale_linear_light")
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Descale width and height must be positive integers.")
    if kernel not in (
        "bilinear",
        "bicubic",
        "lanczos",
        "spline16",
        "spline36",
        "spline64",
        "point",
    ):
        raise ValueError(f"Unsupported descale kernel {kernel!r}.")
    if border not in ("mirror", "zero", "extend"):
        raise ValueError(f"Unsupported descale border {border!r}.")
    if light not in (None, "gamma24", "srgb", "gamma22", "bt709"):
        raise ValueError(f"Unsupported descale linear-light transfer {light!r}.")
    return {
        "width": width,
        "height": height,
        "kernel": kernel,
        "border": border,
        "descale_linear_light": light,
    }


def index_path(video: str | Path) -> Path:
    video = Path(video)
    return video.parent / "index" / f"{video.stem}.json"


def clip_paths(roots: Sequence[Path]) -> list[Path]:
    """Select direct folder members and explicit files, in recipe order."""
    selected: dict[Path, None] = {}
    folders: dict[Path, dict[str, Path]] = {}

    def members(folder: Path) -> dict[str, Path]:
        if folder not in folders:
            folders[folder] = {
                p.name: p
                for p in sorted(folder.iterdir())
                if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()
            }
        return folders[folder]

    for selection in roots:
        root = Path(selection).expanduser()
        if root.is_dir():
            videos = members(root.resolve())
            if not videos:
                raise ValueError(
                    f"No supported videos in {root}. Select a folder with MKV, MP4, M2TS or MOV files."
                )
            selected.update(dict.fromkeys(videos.values()))
        elif root.suffix.lower() in VIDEO_SUFFIXES and root.is_file():
            # Keep the filename for sidecar lookup, using the spelling on disk.
            folder = root.parent.resolve()
            videos = members(folder)
            path = videos.get(root.name, folder / root.name)
            if root.name not in videos:
                for name, video in videos.items():
                    if name.casefold() == root.name.casefold() and video.samefile(path):
                        path = video
                        break
            selected[path] = None
        else:
            raise ValueError(
                f"{root}: select an existing video folder or a regular MKV, MP4, M2TS or MOV file."
            )
    if not selected:
        raise ValueError("Select at least one video file or video folder.")

    # A legacy sidecar belongs to a stem, so even an unselected sibling can
    # claim the same name. Also check folders that share an index directory.
    selected_stems: dict[Path, set[str]] = {}
    for path in selected:
        selected_stems.setdefault(path.parent, set()).add(path.stem.casefold())
    destinations: dict[tuple[Path, str], Path] = {}
    for folder, stems in selected_stems.items():
        destination = (folder / "index").resolve()
        for video in members(folder).values():
            stem = video.stem.casefold()
            if stem not in stems:
                continue
            key = (destination, stem)
            if key in destinations:
                raise ValueError(
                    f"{destinations[key]} and {video} share an index name in {destination}. "
                    "Rename one of the videos or put it in a different folder."
                )
            destinations[key] = video
    return list(selected)


@dataclass(frozen=True)
class ClipStream:
    size: tuple[int, int]  # width, height
    time_base: tuple[int, int]
    pixel_format: str | None


def video_stream(container: Any) -> Any:
    if not container.streams.video:
        raise ValueError("The file has no video stream.")
    stream = container.streams.video[0]
    if stream.disposition & stream.disposition.attached_pic:
        raise ValueError(
            "The first video stream is an attached image. Select a file with the episode as the first video stream."
        )
    if stream.codec_context is None:
        raise ValueError("The video stream has no decoder.")
    return stream


def check_color(pixel_format: str | None, color_range: int, colorspace: int) -> None:
    # FFmpeg enum values for unspecified/limited range and BT.709/unspecified.
    if pixel_format is not None and pixel_format != PIXEL_FORMAT:
        raise ValueError(
            f"Unsupported pixel format {pixel_format}; clips require 8-bit {PIXEL_FORMAT}."
        )
    if color_range not in (0, 1):
        raise ValueError(
            "Clips require limited-range video; this stream signals a different color range."
        )
    if colorspace not in (1, 2):
        raise ValueError(
            "Clips require the BT.709 color matrix; this stream signals a different matrix."
        )


def clip_stream(stream: Any, settings: ClipSettings) -> ClipStream:
    width, height = stream.width, stream.height
    if min(width, height) < 2 or width % 2 or height % 2:
        raise ValueError(
            f"Clip training requires positive, even source dimensions; got {width}x{height}."
        )
    if settings["width"] > width or settings["height"] > height:
        raise ValueError(
            f"Descale size {settings['width']}x{settings['height']} exceeds source size {width}x{height}."
        )
    if stream.time_base is None or stream.time_base <= 0:
        raise ValueError("The video stream has no usable time base.")
    codec = stream.codec_context
    check_color(codec.pix_fmt, codec.color_range, codec.colorspace)
    return ClipStream(
        (width, height),
        (stream.time_base.numerator, stream.time_base.denominator),
        codec.pix_fmt,
    )


def probe_clip(video: Path, settings: ClipSettings) -> ClipStream:
    """Read stream metadata with a bounded probe and no episode decode."""
    import av

    try:
        with av.open(
            str(video), options={"probesize": "1048576", "analyzeduration": "1000000"}
        ) as container:
            return clip_stream(video_stream(container), settings)
    except (av.FFmpegError, ValueError) as exc:
        raise ValueError(f"{video}: {exc}") from exc


class ClipIndexError(ValueError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _is_frame(value: object, frames: int) -> bool:
    return type(value) is int and 0 <= value < frames


def _structure_error(index: dict[str, Any]) -> str | None:
    frames = index.get("frames")
    if type(frames) is not int or frames < 1:
        return "the frame count must be a positive integer"
    pts = index.get("pts")
    if (
        not isinstance(pts, list)
        or len(pts) != frames
        or any(type(p) is not int or not -(2**63) < p < 2**63 for p in pts)
        or any(a >= b for a, b in pairwise(pts))
    ):
        return (
            "frame timestamps must be strictly increasing integers with one per frame"
        )
    motion = index.get("motion")
    if (
        not isinstance(motion, list)
        or len(motion) != frames
        or any(type(m) not in (int, float) or not 0 <= m <= 255 for m in motion)
    ):
        return "motion must contain one finite score per frame"
    cuts = index.get("cuts")
    if (
        not isinstance(cuts, list)
        or not cuts
        or cuts[0] != 0
        or any(not _is_frame(c, frames) for c in cuts)
        or any(a >= b for a, b in pairwise(cuts))
    ):
        return "shot boundaries must start at zero and increase within the frame count"
    samples = index.get("descale_error")
    if not isinstance(samples, list) or not samples:
        return "descale error samples are missing"
    prev = -1
    for sample in samples:
        if not isinstance(sample, dict):
            return "a descale error sample is invalid"
        frame, error = sample.get("frame"), sample.get("p999")
        if not _is_frame(frame, frames) or frame <= prev:
            return "descale error sample positions must increase within the frame count"
        if type(error) not in (int, float) or not 0 <= error < math.inf:
            return "descale error samples must be finite, nonnegative scores"
        prev = frame
    return None


def read_clip_index(
    video: Path, settings: ClipSettings, stream: ClipStream
) -> dict[str, Any]:
    """Return an index the loader can consume, or a classified preparation issue."""
    path = index_path(video)
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise ClipIndexError(
                "incompatible", f"{path} must be a regular index file."
            )
        with path.open(encoding="utf-8") as file:
            info = os.fstat(file.fileno())
            index = json.load(file)
    except FileNotFoundError as exc:
        raise ClipIndexError(
            "missing", f"{video}: missing {path}. Prepare video indexes."
        ) from exc
    except ClipIndexError:
        raise
    except OSError as exc:
        raise ClipIndexError("incompatible", f"{path}: {exc}") from exc
    except (ValueError, RecursionError) as exc:
        raise ClipIndexError(
            "stale", f"{path}: invalid index JSON. Prepare video indexes again."
        ) from exc

    def stale(reason: str) -> ClipIndexError:
        return ClipIndexError(
            "stale", f"{path}: {reason}. Prepare video indexes again."
        )

    if not isinstance(index, dict):
        raise stale("the index must be a JSON object")
    version = index.get("version")
    if "version" in index and (type(version) is not int or version != INDEX_VERSION):
        raise stale(f"unsupported index version {version!r}")
    descale = index.get("descale")
    if not isinstance(descale, dict) or any(
        descale.get(k) != v for k, v in settings.items()
    ):
        raise stale("descale settings differ from the recipe")
    time_base = index.get("time_base")
    if (
        not isinstance(time_base, list)
        or any(type(n) is not int for n in time_base)
        or tuple(time_base) != stream.time_base
    ):
        raise stale("the stream time base differs from the index")
    source = video.stat()
    if version is None:
        if source.st_mtime_ns > info.st_mtime_ns:
            raise stale("the video is newer than its legacy index")
    else:
        width, height = stream.size
        if index.get("source") != source_record(source, width, height):
            raise stale("the video size, modification time or frame dimensions changed")
    if error := _structure_error(index):
        raise stale(error)
    return index


def source_record(info: os.stat_result, width: int, height: int) -> dict[str, Any]:
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "width": width,
        "height": height,
        "pixel_format": PIXEL_FORMAT,
    }


@dataclass(frozen=True)
class _EpisodeSummary:
    stream: ClipStream | None
    status: str
    message: str | None = None
    frames: int = 0
    legacy: bool = False


@lru_cache(maxsize=256)
def _episode_summary(
    video: Path,
    _source_version: str,
    _index_version: str | None,
    settings_items: tuple[tuple[str, Any], ...],
) -> _EpisodeSummary:
    # The file versions are cache keys. Full arrays only live during validation.
    settings = cast(ClipSettings, dict(settings_items))
    stream = None
    try:
        stream = probe_clip(video, settings)
        index = read_clip_index(video, settings, stream)
    except ClipIndexError as exc:
        return _EpisodeSummary(stream, exc.status, str(exc))
    except (OSError, ValueError) as exc:
        return _EpisodeSummary(stream, "incompatible", str(exc))
    return _EpisodeSummary(
        stream, "ready", frames=index["frames"], legacy="version" not in index
    )


def _sidecar_version(video: Path) -> str | None:
    try:
        return artifact_version(index_path(video).stat())
    except FileNotFoundError:
        return None


def inspect_clips(
    roots: Sequence[Path],
    settings: Mapping[str, Any],
    *,
    check_cancel: Callable[[], None] | None = None,
    sample_limit: int | None = 25,
) -> dict[str, Any]:
    """Summarize source metadata and sidecars. Problems precede ready samples."""
    report: dict[str, Any] = {
        "total": 0,
        "ready": 0,
        "missing": 0,
        "stale": 0,
        "incompatible": 0,
        "frameSize": None,
        "episodes": [],
        "errors": [],
    }
    if check_cancel is not None:
        check_cancel()
    try:
        options = clip_settings(settings)
        paths = clip_paths(roots)
    except (OSError, ValueError, RuntimeError) as exc:
        report["errors"].append(str(exc))
        return report

    settings_items = tuple(options.items())
    common_size: tuple[int, int] | None = None
    first: Path | None = None
    episodes = []
    for path in paths:
        if check_cancel is not None:
            check_cancel()
        row: dict[str, Any] = {"path": str(path), "indexPath": str(index_path(path))}
        try:
            source = path.stat()
            row["sourceBytes"] = source.st_size
            summary = _episode_summary(
                path, artifact_version(source), _sidecar_version(path), settings_items
            )
        except OSError as exc:
            summary = _EpisodeSummary(None, "incompatible", str(exc))
        stream = summary.stream
        if stream is not None:
            row.update(frameSize=list(stream.size), pixelFormat=stream.pixel_format)
            if common_size is None:
                common_size, first = stream.size, path
            elif stream.size != common_size:
                summary = _EpisodeSummary(
                    stream,
                    "incompatible",
                    f"{path}: frame size {stream.size[0]}x{stream.size[1]} differs from "
                    f"{common_size[0]}x{common_size[1]} in {first}. Select videos with one frame size.",
                )
        row["status"] = summary.status
        if summary.message is not None:
            row["message"] = summary.message
        if summary.status == "ready":
            row.update(frames=summary.frames, legacy=summary.legacy)
        report[summary.status] += 1
        episodes.append(row)
    report["total"] = len(paths)
    if common_size is not None and not report["incompatible"]:
        report["frameSize"] = list(common_size)
    report["episodes"] = (
        episodes
        if sample_limit is None
        else sorted(episodes, key=lambda row: row["status"] == "ready")[:sample_limit]
    )
    return report

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import av
import numpy as np
import torch
from av.video.frame import VideoFrame
from chainner_ext import ResizeFilter, resize
from numpy.random import Generator
from trainner_data.video_indexes import (
    clip_paths,
    clip_settings,
    probe_clip,
    read_clip_index,
)

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.clip_hr import clip_axes
from traiNNer.data.descale import descale, encode_light, linearize
from traiNNer.data.video_index import open_video, to_rgb, yuv_window
from traiNNer.utils import RNG
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY

T = TypeVar("T")

_OPEN_CONTAINERS = 2  # per worker process
_HR_FILTERS = {
    "catrom": ResizeFilter.CubicCatrom,
    "mitchell": ResizeFilter.CubicMitchell,
    "hermite": ResizeFilter.Hermite,
}


@dataclass
class _Video:
    path: str
    name: str
    pts: list[int]
    starts: np.ndarray  # first frame of each shot
    ends: np.ndarray  # one past the last frame of each shot
    clean: np.ndarray  # shot has no sampled frame above the error threshold
    factor: np.ndarray  # motion weight of each shot in [floor, 1]


def _intervals(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Merge sorted inclusive [start, end] intervals; empty ones are dropped."""
    merged: list[list[int]] = []
    for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
        if b < a:
            continue
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.array(merged, dtype=np.int64).reshape(-1, 2)


@DATASET_REGISTRY.register()
class VideoClipDataset(BaseDataset):
    """Clips of consecutive frames from indexed videos, with the HR made on the
    fly: decode, convert to RGB (BT.709, limited range), descale, downscale to
    hr_size, then one random crop, flip and rotation for the whole clip.

    The items carry the frames as the uint8 YUV window that the HR crop reads
    (gt_y T,1,h,w; gt_u and gt_v T,1,h/2,w/2) and gt_geometry, the frame size,
    crop, window position and flips; clip_hr.hr_clip makes the HR crop from
    them on the training device. hr_frame is the same HR making on the CPU for
    one whole frame. All videos must share one frame size.

    Each video needs the index written by scripts/data_preparation/
    index_video_clips.py in an index/ folder next to the video. The data roots
    can be folders or selected videos. Clips stay inside one shot,
    except a cut_clip_prob share that crosses a cut between clean shots; shots
    with a sampled frame whose descale error is above descale_error_threshold
    are never used. One clip length from clip_lengths is drawn per batch.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        if opt.phase != "train":
            raise ValueError("VideoClipDataset supports the train phase only")
        if opt.lq_size is None or opt.scale is None:
            raise ValueError("VideoClipDataset needs lq_size and scale")
        if opt.downscale_algorithm not in _HR_FILTERS:
            raise ValueError(
                f"VideoClipDataset does not support downscale_algorithm "
                f"{opt.downscale_algorithm!r}"
            )
        if opt.io_backend.get("type", "disk") != "disk":
            raise ValueError("VideoClipDataset requires video files on disk.")
        roots = opt.dataroot_gt
        if roots is None:
            raise ValueError("VideoClipDataset needs dataroot_gt")
        if isinstance(roots, str):
            roots = [roots]
        self.clip_lengths = list(opt.clip_lengths)
        if not self.clip_lengths or any(length < 1 for length in self.clip_lengths):
            raise ValueError("VideoClipDataset needs positive clip_lengths.")
        self.crop = opt.lq_size * opt.scale
        if self.crop < 1:
            raise ValueError("VideoClipDataset needs a positive crop size.")
        hr_w, hr_h = opt.hr_size
        if self.crop > min(hr_w, hr_h):
            raise ValueError(f"Crop {self.crop} is larger than hr_size {hr_w}x{hr_h}")
        width, height = opt.descale_size
        settings = clip_settings(
            {
                "width": width,
                "height": height,
                "kernel": opt.descale_kernel,
                "border": opt.descale_border,
                "descale_linear_light": opt.descale_linear_light,
            }
        )
        paths = clip_paths([Path(root) for root in roots])
        self.videos: list[_Video] = []
        motions: list[np.ndarray] = []
        frame_size = None
        for path in paths:
            stream = probe_clip(path, settings)
            if frame_size is None:
                frame_size = stream.size
            elif stream.size != frame_size:
                raise ValueError(
                    f"{path}: frame size {stream.size[0]}x{stream.size[1]} differs from "
                    f"{frame_size[0]}x{frame_size[1]} in {paths[0]}. Select videos with one frame size."
                )
            index = read_clip_index(path, settings, stream)
            motions.append(self._load(path, index, opt.descale_error_threshold))
        assert frame_size is not None
        self.frame_size = frame_size[1], frame_size[0]
        rows, cols = clip_axes(*self.frame_size, opt)
        self.window = (rows.size, cols.size)
        self.window_origins = (rows.origins, cols.origins)
        clean_motion = np.concatenate(
            [m[v.clean] for v, m in zip(self.videos, motions, strict=True)]
        )
        if clean_motion.size == 0:
            raise ValueError("Every shot is above descale_error_threshold")
        median = float(np.median(clean_motion))
        for video, motion in zip(self.videos, motions, strict=True):
            relative = motion / median if median > 0 else np.ones_like(motion)
            video.factor = np.clip(relative, opt.motion_weight_floor, 1.0)
        self.eligible = sum(
            int((v.ends - v.starts)[v.clean].sum()) for v in self.videos
        )
        self._shot_tables = {t: self._shot_table(t) for t in self.clip_lengths}
        self._cut_tables = {t: self._cut_table(t) for t in self.clip_lengths}
        self._containers: dict[str, av.container.InputContainer] = {}
        self._pid = os.getpid()

    def _load(self, path: Path, index: dict, threshold: float) -> np.ndarray:
        """Adds the video; returns the mean motion of each shot."""
        frames = index["frames"]
        starts = np.asarray(index["cuts"], dtype=np.int64)
        ends = np.append(starts[1:], frames)
        shot_of = np.repeat(np.arange(len(starts)), ends - starts)
        clean = np.ones(len(starts), dtype=bool)
        for sample in index["descale_error"]:
            if sample["p999"] > threshold:
                clean[shot_of[sample["frame"]]] = False
        motion = np.asarray(index["motion"], dtype=np.float64)
        # A shot's first value is the cut itself.
        shot_motion = np.array(
            [
                motion[a + 1 : b].mean() if b - a > 1 else 0.0
                for a, b in zip(starts, ends, strict=True)
            ]
        )
        self.videos.append(
            _Video(str(path), path.name, index["pts"], starts, ends, clean, shot_motion)
        )
        return shot_motion

    def _shot_table(self, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(video, shot, cumulative weight) of every shot that holds a clip."""
        rows = []
        for v, video in enumerate(self.videos):
            windows = np.maximum(video.ends - video.starts - length + 1, 0)
            weight = windows * video.factor * video.clean
            for s in np.flatnonzero(weight > 0):
                rows.append((v, s, weight[s]))
        if not rows:
            raise ValueError(f"No clean shot holds a clip of {length} frames")
        table = np.array(rows)
        return table[:, 0].astype(int), table[:, 1].astype(int), np.cumsum(table[:, 2])

    def _cut_table(self, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(video, first start, cumulative count) of the start frames whose clip
        crosses a cut between two clean shots."""
        rows = []
        for v, video in enumerate(self.videos):
            # Shot k + 1 starts at a crossable cut when shots k and k + 1 are clean.
            joined = video.clean[:-1] & video.clean[1:]
            segment_start = video.starts.copy()
            segment_end = video.ends.copy()
            for k in range(1, len(video.starts)):
                if joined[k - 1]:
                    segment_start[k] = segment_start[k - 1]
            for k in range(len(video.starts) - 2, -1, -1):
                if joined[k]:
                    segment_end[k] = segment_end[k + 1]
            cuts = video.starts[1:][joined]
            seg_a, seg_b = segment_start[1:][joined], segment_end[1:][joined]
            first = np.maximum(seg_a, cuts - length + 1)
            last = np.minimum(cuts - 1, seg_b - length)
            for a, b in _intervals(first, last):
                rows.append((v, a, b - a + 1))
        if not rows:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        table = np.array(rows)
        return table[:, 0], table[:, 1], np.cumsum(table[:, 2])

    def draw(self, length: int, rng: Generator) -> tuple[int, int]:
        """(video, first frame) of a random clip."""
        videos, firsts, total = self._cut_tables[length]
        if total.size and rng.random() < self.opt.cut_clip_prob:
            i = int(np.searchsorted(total, rng.random() * total[-1], side="right"))
            offset = int(rng.integers(total[i] - (total[i - 1] if i else 0)))
            return int(videos[i]), int(firsts[i]) + offset
        videos, shots, total = self._shot_tables[length]
        i = int(np.searchsorted(total, rng.random() * total[-1], side="right"))
        video = self.videos[videos[i]]
        shot = shots[i]
        windows = int(video.ends[shot] - video.starts[shot]) - length + 1
        return int(videos[i]), int(video.starts[shot]) + int(rng.integers(windows))

    def _container(self, path: str) -> av.container.InputContainer:
        if self._pid != os.getpid():
            self._containers = {}
            self._pid = os.getpid()
        # Most recently used last; the least recently used one is closed.
        container = self._containers.pop(path, None)
        if container is None:
            container = open_video(path, threads=2)
        self._containers[path] = container
        if len(self._containers) > _OPEN_CONTAINERS:
            self._containers.pop(next(iter(self._containers))).close()
        return container

    def _decode(
        self, video: int, first: int, count: int, convert: Callable[[VideoFrame], T]
    ) -> list[T]:
        """Frames first .. first + count - 1 through convert, by seeking to the
        keyframe before the first and decoding forward."""
        entry = self.videos[video]
        wanted = entry.pts[first : first + count]
        if len(wanted) != count:
            raise ValueError(f"{entry.name}: frames {first}+{count} out of range")
        container = self._container(entry.path)
        stream = container.streams.video[0]
        container.seek(wanted[0], stream=stream, backward=True, any_frame=False)
        frames: list[T] = []
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < wanted[0]:
                continue
            if frame.pts != wanted[len(frames)]:
                raise ValueError(
                    f"{entry.name}: decoded pts {frame.pts}, index has "
                    f"{wanted[len(frames)]} for frame {first + len(frames)}. "
                    "Prepare the video index again."
                )
            if (frame.height, frame.width) != self.frame_size:
                raise ValueError(
                    f"{entry.name}: frame dimensions changed during decoding."
                )
            frames.append(convert(frame))
            if len(frames) == count:
                return frames
        raise ValueError(
            f"{entry.name}: stream ended before frame {first + count - 1}. "
            "Prepare the video index again."
        )

    def decode_frames(self, video: int, first: int, count: int) -> list[np.ndarray]:
        """RGB float32 frames first .. first + count - 1."""
        return self._decode(video, first, count, to_rgb)

    def hr_frame(self, rgb: np.ndarray) -> np.ndarray:
        opt = self.opt
        width, height = opt.descale_size
        image = descale(
            rgb,
            width,
            height,
            opt.descale_kernel,
            border=opt.descale_border,
            linear_light=opt.descale_linear_light,
        )
        light = opt.hr_linear_light
        if light is not None:
            image = linearize(image, light)
        image = resize(
            image, tuple(opt.hr_size), _HR_FILTERS[opt.downscale_algorithm], False
        )
        if light is not None:
            image = encode_light(image, light)
        np.clip(image, 0.0, 1.0, out=image)
        return image

    def _clip(self, length: int, rng: Generator) -> dict[str, torch.Tensor | str]:
        v, first = self.draw(length, rng)
        video = self.videos[v]
        hr_w, hr_h = self.opt.hr_size
        top = int(rng.integers(hr_h - self.crop + 1))
        left = int(rng.integers(hr_w - self.crop + 1))
        hflip = self.opt.use_hflip and rng.random() < 0.5
        vflip = self.opt.use_rot and rng.random() < 0.5
        transpose = self.opt.use_rot and rng.random() < 0.5
        reverse = self.opt.use_time_reverse and rng.random() < 0.5
        row = int(self.window_origins[0][top])
        col = int(self.window_origins[1][left])
        height, width = self.window
        frames = self._decode(
            v, first, length, lambda f: yuv_window(f, row, col, height, width)
        )
        shot_ids = (
            np.searchsorted(
                video.starts, np.arange(first, first + length), side="right"
            )
            - 1
        )
        if reverse:
            frames.reverse()
            shot_ids = shot_ids[::-1]
        y, u, v_plane = (
            torch.from_numpy(np.stack(planes)[:, None])
            for planes in zip(*frames, strict=True)
        )
        flips = (int(hflip), int(vflip), int(transpose))
        geometry = (*self.frame_size, top, left, row, col, *flips)
        return {
            "gt_y": y,
            "gt_u": u,
            "gt_v": v_plane,
            "gt_geometry": torch.tensor(geometry, dtype=torch.int64),
            "shot_ids": torch.from_numpy(
                np.ascontiguousarray(shot_ids, dtype=np.int64)
            ),
            "gt_path": f"{video.name}:{first}",
        }

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor | str]]:
        rng = RNG.get_rng()
        length = int(rng.choice(self.clip_lengths))
        return [self._clip(length, rng) for _ in indices]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self.__getitems__([index])[0]

    def __len__(self) -> int:
        return self.eligible // round(np.mean(self.clip_lengths))

    @property
    def label(self) -> str:
        return "video clips"

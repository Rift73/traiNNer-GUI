"""A scene-local index of complete frame windows, shared by training and inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

_EXTENSIONS = (".png", ".jpg", ".jpeg")
_MANIFEST_FORMAT = "paired-video-scenes-v1"
_NUMBERED = re.compile(r"(.*?)([0-9]+)$")


_T = TypeVar("_T")


def _polled(items: Iterable[_T], check: Callable[[], None] | None) -> Iterable[_T]:
    if check is None:
        return items

    def checked() -> Iterator[_T]:
        for i, item in enumerate(items):
            if i % 4096 == 0:
                check()
            yield item

    return checked()


def _filenames(folder: str, check: Callable[[], None] | None) -> list[str]:
    with os.scandir(folder) as entries:
        return sorted(
            e.name
            for e in _polled(entries, check)
            if e.name.lower().endswith(_EXTENSIONS) and e.is_file()
        )


def _scenes(
    filenames: Iterable[str], selected: dict | None = None
) -> Iterator[tuple[tuple[bool, str], list[str]]]:
    groups = {}
    for name in filenames:
        stem = os.path.splitext(name)[0]
        match = _NUMBERED.fullmatch(stem)
        # An entire numbered folder may be one sequence. Keeping the full
        # prefix also prevents shot_01_001 and shot_02_001 from sharing state.
        key = (True, match[1]) if match else (False, name.partition("_")[0])
        if selected is not None and key not in selected:
            continue
        groups.setdefault(key, []).append((int(match[2]) if match else 0, name))
    for key, group in groups.items():
        if key[0]:
            group.sort()
            for (first, a), (second, b) in pairwise(group):
                if first == second:
                    raise ValueError(
                        f"Frame number {first} is ambiguous between {a!r} and {b!r}."
                    )
        yield key, [name for _, name in group]


@dataclass(slots=True)
class FrameScene:
    slot: int  # one-based position in the ordered root lists
    key: tuple[bool, str]
    root: str
    target_root: str | None
    frames: list[str]
    starts: range | array

    def manifest_entry(self, check: Callable[[], None] | None = None) -> dict:
        # Names cannot contain NUL. Ordered names plus starts specify every
        # LQ context and exact GT center, including gaps in sparse references.
        digest = hashlib.sha256()
        for name in _polled(self.frames, check):
            digest.update(os.fsencode(name))
            digest.update(b"\0")
        digest.update(b"\0")
        if isinstance(self.starts, range):
            digest.update(b"all")
        else:
            for start in _polled(self.starts, check):
                digest.update(start.to_bytes(8, "big"))
        return {
            "root": self.slot,
            "numbered": self.key[0],
            "key": self.key[1],
            "membershipSha256": digest.hexdigest(),
        }


def _read_selection(path: str | Path, width: int, root_count: int) -> dict:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("format") != _MANIFEST_FORMAT:
        raise ValueError(f"Unsupported paired video scene manifest: {path}")
    if (
        type(data.get("clipSize")) is not int
        or data["clipSize"] != width
        or type(data.get("rootCount")) is not int
        or data["rootCount"] != root_count
    ):
        raise ValueError(
            f"Scene manifest requires matching clip_size and ordered root count: {path}"
        )
    entries = data.get("scenes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Scene manifest must select at least one scene: {path}")
    selected = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid scene manifest entry: {entry!r}")
        slot, numbered, key = entry.get("root"), entry.get("numbered"), entry.get("key")
        digest = entry.get("membershipSha256")
        if (
            type(slot) is not int
            or not 1 <= slot <= root_count
            or type(numbered) is not bool
            or not isinstance(key, str)
            or not isinstance(digest, str)
            or not re.fullmatch("[a-f0-9]{64}", digest)
        ):
            raise ValueError(f"Invalid scene manifest entry: {entry!r}")
        by_root = selected.setdefault(slot, {})
        if (numbered, key) in by_root:
            raise ValueError(f"Duplicate scene in root {slot}: {key!r}")
        by_root[numbered, key] = digest
    return selected


class FrameWindowIndex:
    """Only sparse target selections need a stored integer for each window.

    Windows never cross source roots or scene prefixes. Their center uses the
    trainer convention width // 2, including even-width windows.
    """

    def __init__(
        self,
        roots: list[str],
        width: int,
        target_roots: list[str] | None = None,
        *,
        scene_manifest: str | Path | None = None,
        check: Callable[[], None] | None = None,
    ) -> None:
        if type(width) is not int or width < 1:
            raise ValueError("A frame window must contain at least one frame.")
        if target_roots is not None and len(roots) != len(target_roots):
            raise ValueError("Each video input root needs its own target root.")
        selected = None
        if scene_manifest is not None:
            if target_roots is None:
                raise ValueError("A paired video scene manifest requires target roots.")
            selected = _read_selection(scene_manifest, width, len(roots))
        self.root_count = len(roots)
        self.width = width
        self.frame_count = 0
        self.scenes: list[FrameScene] = []
        self.ends: list[int] = []
        total = 0
        for index, root in enumerate(roots):
            slot = index + 1
            expected = None if selected is None else selected.get(slot, {})
            if expected is not None and not expected:
                continue
            names = _filenames(root, check)
            self.frame_count += len(names)
            target = target_roots[index] if target_roots is not None else None
            references = set(_filenames(target, check)) if target is not None else None
            for key, frames in _scenes(_polled(names, check), expected):
                all_starts = range(max(0, len(frames) - width + 1))
                if references is None:
                    starts = all_starts
                else:
                    sparse = array(
                        "Q",
                        (
                            i
                            for i in _polled(all_starts, check)
                            if frames[i + width // 2] in references
                        ),
                    )
                    starts = all_starts if len(sparse) == len(all_starts) else sparse
                if starts:
                    scene = FrameScene(slot, key, root, target, frames, starts)
                    if expected is not None and scene.manifest_entry(check)[
                        "membershipSha256"
                    ] != expected.pop(key):
                        raise ValueError(
                            f"Frame or center membership changed in root {slot}, scene {key!r}."
                        )
                    self.scenes.append(scene)
                    total += len(starts)
                    self.ends.append(total)

            if expected:
                raise ValueError(
                    f"Missing scene or eligible windows in root {slot}: {next(iter(expected))!r}"
                )

    def manifest(
        self, scenes: Iterable[FrameScene], *, check: Callable[[], None] | None = None
    ) -> dict:
        return {
            "format": _MANIFEST_FORMAT,
            "clipSize": self.width,
            "rootCount": self.root_count,
            "scenes": [scene.manifest_entry(check) for scene in scenes],
        }

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def window(self, index: int) -> tuple[list[str], str | None]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect_right(self.ends, index)
        scene = self.scenes[group]
        local = index - (self.ends[group - 1] if group else 0)
        start = scene.starts[local]
        names = scene.frames[start : start + self.width]
        target = (
            os.path.join(scene.target_root, names[self.width // 2])
            if scene.target_root
            else None
        )
        return [os.path.join(scene.root, name) for name in names], target

    def centers(self) -> Iterator[Path]:
        for scene in self.scenes:
            for start in scene.starts:
                yield Path(scene.root, scene.frames[start + self.width // 2])

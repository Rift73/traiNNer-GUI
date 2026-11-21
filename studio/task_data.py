"""Read-only CPU dataset inventories for Studio's label-aware providers."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import sqlite3
import struct
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, closing
from functools import lru_cache
from pathlib import Path
from random import Random
from time import monotonic
from typing import Any, TypeVar

from PIL import Image, ImageMode
from trainner_data.coco import (
    CocoAnnotation,
    CocoImage,
    clip_box,
    image_path,
    index_coco,
    read_coco,
)
from trainner_data.supervised_manifest import folder_members
from trainner_data.task_folders import (
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
    class_entries,
    files_by_stem,
    image_files,
)

from studio.paths import artifact_version, local_path

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE = _ROOT / "native/studio/data"
_TASKS = {
    "semantic_segmentation": "imagemaskdataset",
    "image_classification": "classfolderdataset",
    "object_detection": "cocodetectiondataset",
    "tabular_regression": "numericcsvdataset",
}
_ISSUE_LIMIT = 200
_SAMPLE_LIMIT = 100
_VALUE_LIMIT = 256
_BOX_LIMIT = 20
_DECODE_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)
_STOP = ctypes.CFUNCTYPE(ctypes.c_int)
_BINS = ctypes.c_uint64 * 65536
_VALUES = ctypes.c_uint16 * 65536
_FLOAT32_BOX = struct.Struct("4f")
_Row = TypeVar("_Row")
Json = dict[str, Any]


class TaskDataCancelled(RuntimeError):
    """The job was stopped before completing dataset inspection."""


class _Work:
    def __init__(
        self,
        root: Path,
        audit_limit: int | None,
        sample_limit: int,
        progress: Callable[[Json], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        self.root = root
        self.audit_limit = audit_limit
        self.sample_limit = sample_limit
        self.progress = progress
        self.cancelled = cancelled
        self.split = ""
        self.issue_counts = {"error": 0, "warning": 0}
        self.details: dict[str, list[Json]] = {"error": [], "warning": []}
        self._next_cancel = self._next_progress = 0.0

    def check(self, *, force: bool = False) -> None:
        if self.cancelled is not None and (force or monotonic() >= self._next_cancel):
            self._next_cancel = monotonic() + 0.15
            if self.cancelled():
                raise TaskDataCancelled("Dataset inspection was cancelled.")

    def update(
        self, stage: str, done: int, total: int | None, *, force: bool = False
    ) -> None:
        self.check(force=force)
        if self.progress is not None and (force or monotonic() >= self._next_progress):
            self._next_progress = monotonic() + 0.5
            self.progress(
                {
                    "stage": stage,
                    "splitId": self.split,
                    "completed": done,
                    "total": total,
                }
            )
            self.check(force=True)

    def issue(self, path: str | Path, message: str, *, level: str = "error") -> None:
        self.issue_counts[level] += 1
        if len(self.details[level]) < _ISSUE_LIMIT:
            self.details[level].append(
                {
                    "level": level,
                    "path": str(path),
                    "message": message,
                    "splitId": self.split,
                }
            )

    def audit(
        self,
        rows: Iterable[_Row],
        inspect: Callable[[_Row, bool], tuple[bool, Json | None]],
        method: str,
    ) -> Json:
        limit = self.audit_limit
        eligible = attempted = completed = 0
        samples = []

        def visit(row: _Row, total: int | None) -> None:
            nonlocal attempted, completed
            self.check()
            success, sample = inspect(row, len(samples) < self.sample_limit)
            attempted += 1
            completed += success
            if sample is not None:
                samples.append(sample)
            self.update("auditing", attempted, total)

        if limit is None:
            self.update("auditing", 0, None, force=True)
            for row in rows:
                visit(row, None)
            eligible = attempted
        else:
            # A fixed-seed reservoir covers the whole selection without saving every path.
            rng = Random(0)
            reservoir = []
            for eligible, row in enumerate(rows, 1):
                if eligible <= limit:
                    reservoir.append((eligible, row))
                elif limit and (slot := rng.randrange(eligible)) < limit:
                    reservoir[slot] = (eligible, row)
                self.update("inventory", eligible, None)
            self.update("auditing", 0, len(reservoir), force=True)
            for _, row in sorted(reservoir):
                visit(row, len(reservoir))

        self.update("auditing", attempted, attempted, force=True)
        return {
            "audit": {
                "method": method,
                "eligible": eligible,
                "attempted": attempted,
                "completed": completed,
                "complete": completed == eligible,
            },
            "samples": samples,
            "samplesTruncated": len(samples) < attempted,
        }


def _roots(dataset: Mapping[str, Any], key: str, root: Path) -> list[Path]:
    values = dataset.get(key)
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError(f"Provide {key} as a path or nonempty array of paths.")
    return [local_path(value, root) for value in values]


def _class_names(settings: Mapping[str, Any], *, folders: bool) -> list[str] | None:
    names = settings.get("class_names")
    if names is None:
        return None
    if (
        not isinstance(names, list)
        or len(names) < 2
        or any(
            not isinstance(name, str)
            or not name.strip()
            or (folders and (name in (".", "..") or "/" in name or "\\" in name))
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("Provide at least two distinct nonempty class_names.")
    return names


def _size(path: Path, work: _Work, *, still: bool) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            if still and getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame image.")
            return image.size
    except _DECODE_ERRORS as exc:
        work.issue(path, f"Could not read image header. {exc}")
        return None


@lru_cache(maxsize=1)
def _png_histogram() -> Callable[..., int] | None:
    if sys.platform != "linux":
        return None
    try:
        subprocess.run(
            ["make", "--no-print-directory", "-C", str(_NATIVE), "mask-histogram"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=True,
        )
        kernel = ctypes.CDLL(
            str(_NATIVE / "build/clang/mask_histogram.so")
        ).studio_png_histogram_v1
    except (OSError, AttributeError, subprocess.SubprocessError):
        return None
    kernel.argtypes = [
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint16),
        _STOP,
    ]
    kernel.restype = ctypes.c_int
    return kernel


class _MaskHistogram:
    def __init__(self, work: _Work) -> None:
        self._work = work
        self._scratch: tuple[Any, Any] | None = None
        self._failure: Exception | None = None
        self._stop = _STOP(self._check) if work.cancelled is not None else _STOP()

    def _check(self) -> int:
        try:
            self._work.check()
        except Exception as exc:
            self._failure = exc
            return 1
        return 0

    def count(self, mask: Image.Image) -> tuple[dict[int, int], str]:
        mode = ImageMode.getmode(mask.mode)
        if len(mode.bands) != 1 or mode.typestr[1] not in "biu":
            raise ValueError("Expected one integer class ID per pixel.")

        fp = getattr(mask, "fp", None)
        if (
            fp is not None
            and mask.format == "PNG"
            and not mask.info.get("interlace")
            and getattr(mask, "n_frames", 1) == 1
            and (kernel := _png_histogram()) is not None
        ):
            if self._scratch is None:
                self._scratch = _BINS(), _VALUES()
            bins, values = self._scratch
            count = kernel(fp.fileno(), *mask.size, bins, values, self._stop)
            if isinstance(self._failure, TaskDataCancelled):
                raise self._failure
            if self._failure is not None:
                raise RuntimeError(
                    "The dataset cancellation callback failed."
                ) from self._failure
            if count > 0:
                return {values[i]: bins[values[i]] for i in range(count)}, "clangPng"
        if mask.mode in {"P", "L"}:
            return {i: n for i, n in enumerate(mask.histogram()) if n}, "pillow"
        if mask.mode == "1":
            # Pillow's histogram addresses the white bin as 255, while its boolean array uses 1.
            bins = mask.histogram()
            return {i: n for i, n in enumerate((bins[0], bins[255])) if n}, "pillow"

        import numpy as np

        values, bins = np.unique(np.asarray(mask), return_counts=True)
        return {
            int(value): int(n) for value, n in zip(values, bins, strict=True)
        }, "pillow"


class _ObservedValues:
    """Keep exact pixel totals and exact counts for at most 256 observed IDs."""

    def __init__(self) -> None:
        self.pixels = 0
        self.values: Counter[int] = Counter()
        self.truncated = False

    def add(self, label: int, pixels: int) -> None:
        self.pixels += pixels
        if label in self.values or len(self.values) < _VALUE_LIMIT:
            self.values[label] += pixels
        else:
            self.truncated = True

    def result(self) -> Json:
        return {
            "pixels": self.pixels,
            "values": [
                {"id": label, "pixels": pixels}
                for label, pixels in sorted(self.values.items())
            ],
            "unlistedPixels": self.pixels - self.values.total(),
            "valuesTruncated": self.truncated,
        }


def _segmentation(
    work: _Work, dataset: Mapping[str, Any], names: list[str] | None, ignore_label: int
) -> Json:
    images = _roots(dataset, "dataroot_lq", work.root)
    masks = _roots(dataset, "dataroot_gt", work.root)
    if len(images) != len(masks):
        raise ValueError("Provide one mask root per image root.")
    crop = None
    if work.split.split("_", 1)[0] == "train":
        crop = dataset.get("gt_size")
        if crop is None:
            crop = dataset.get("lq_size")
        if type(crop) is not int or crop <= 0:
            work.issue(
                f"datasets.{work.split}",
                "ImageMaskDataset training requires a positive lq_size or gt_size.",
            )
            crop = None

    roots = []
    counts = dict.fromkeys(
        ("images", "masks", "pairs", "missingMasks", "missingImages", "duplicateStems"),
        0,
    )
    inventory_complete = True
    pixel_counts = [0] * len(names or [])
    other = _ObservedValues()
    ignored = 0
    engines: Counter[str] = Counter()
    geometry = dict.fromkeys(("checked", "matched", "mismatched", "tooSmall"), 0)
    histogram = _MaskHistogram(work)

    def inventory(
        folder: Path, extensions: frozenset[str], ambiguous: set[Path]
    ) -> tuple[dict[Path, Path], int, bool]:
        duplicates = 0
        readable = True

        def duplicate(stem: Path, path: Path) -> None:
            nonlocal duplicates
            duplicates += 1
            ambiguous.add(stem)
            work.issue(
                path, f"Multiple files match the relative stem {stem} in {folder}."
            )

        def unreadable(path: Path, exc: OSError) -> None:
            nonlocal readable
            readable = False
            work.issue(path, str(exc))

        try:
            selected = files_by_stem(
                folder,
                extensions,
                duplicate=duplicate,
                check=work.check,
                unreadable=unreadable,
            )
        except OSError as exc:
            unreadable(folder, exc)
            return {}, 0, False
        return selected, len(selected) + duplicates, readable

    def rows() -> Iterator[tuple[str, str, Path, Path]]:
        nonlocal inventory_complete
        if dataset.get("meta_info") is not None:
            manifest = local_path(dataset["meta_info"], work.root)
            for i, (image_root, mask_root) in enumerate(
                zip(images, masks, strict=True), 1
            ):
                roots.append(
                    {
                        "id": str(i),
                        "images": str(image_root),
                        "masks": str(mask_root),
                        "counts": dict.fromkeys(counts, 0),
                        "inventoryComplete": True,
                    }
                )
            try:
                for group, image, mask in folder_members(manifest, images, masks):
                    work.check()
                    assert mask is not None
                    for key in ("images", "masks", "pairs"):
                        roots[group - 1]["counts"][key] += 1
                        counts[key] += 1
                    yield (
                        str(group),
                        str(image.with_suffix("")),
                        images[group - 1] / image,
                        masks[group - 1] / mask,
                    )
            except (OSError, ValueError) as exc:
                inventory_complete = False
                for record in roots:
                    record["inventoryComplete"] = False
                work.issue(manifest, str(exc))
            return
        for index, (image_root, mask_root) in enumerate(
            zip(images, masks, strict=True), 1
        ):
            work.check()
            root_id = str(index)
            ambiguous: set[Path] = set()
            inputs = inventory(image_root, IMAGE_EXTENSIONS, ambiguous)
            labels = inventory(mask_root, MASK_EXTENSIONS, ambiguous)
            readable = inputs[2] and labels[2]
            inventory_complete &= readable
            root_counts = dict.fromkeys(counts, 0)
            root_counts.update(
                images=inputs[1], masks=labels[1], duplicateStems=len(ambiguous)
            )
            roots.append(
                {
                    "id": root_id,
                    "images": str(image_root),
                    "masks": str(mask_root),
                    "counts": root_counts,
                    "inventoryComplete": readable,
                }
            )
            image_paths, mask_paths = inputs[0], labels[0]
            for stem, path in image_paths.items():
                work.check()
                if stem in mask_paths:
                    if readable and stem not in ambiguous:
                        root_counts["pairs"] += 1
                        yield root_id, str(stem), path, mask_paths[stem]
                elif labels[2]:
                    work.issue(path, f"Missing mask for {stem} in {mask_root}.")
                    root_counts["missingMasks"] += 1
            for stem, path in mask_paths.items():
                work.check()
                if inputs[2] and stem not in image_paths:
                    work.issue(path, f"Missing image for {stem} in {image_root}.")
                    root_counts["missingImages"] += 1
            for key, value in root_counts.items():
                counts[key] += value

    def inspect(
        row: tuple[str, str, Path, Path], sample: bool
    ) -> tuple[bool, Json | None]:
        nonlocal ignored
        root_id, stem, image_pathname, mask_pathname = row
        input_size = _size(image_pathname, work, still=False)
        mask_size = mode = None
        decoded = False
        mask_values = {}
        try:
            with Image.open(mask_pathname) as mask:
                mask_size, mode = mask.size, mask.mode
                mask_values, engine = histogram.count(mask)
                engines[engine] += 1
                decoded = True
        except _DECODE_ERRORS as exc:
            work.issue(mask_pathname, f"Could not read class mask. {exc}")

        ignored_pixels = mask_values.pop(ignore_label, 0)
        ignored += ignored_pixels
        unknown_pixels = 0
        examples = []
        for value, pixels in mask_values.items():
            if names is not None and 0 <= value < len(names):
                pixel_counts[value] += pixels
            else:
                other.add(value, pixels)
                if names is not None:
                    unknown_pixels += pixels
                    if len(examples) < 5:
                        examples.append(value)
        if unknown_pixels:
            work.issue(
                mask_pathname,
                f"{unknown_pixels} pixels have unknown class IDs, including {sorted(examples)}.",
            )

        matches = None
        if input_size is not None and mask_size is not None:
            matches = input_size == mask_size
            geometry["checked"] += 1
            geometry["matched" if matches else "mismatched"] += 1
            if not matches:
                work.issue(
                    mask_pathname, f"Mask is {mask_size}; its image is {input_size}."
                )
        if crop is not None and any(
            min(size) < crop for size in (input_size, mask_size) if size is not None
        ):
            geometry["tooSmall"] += 1
            work.issue(
                mask_pathname,
                f"The pair is smaller than the training crop of {crop} pixels.",
            )
        result = None
        if sample:
            result = {
                "id": f"{root_id}:{stem}",
                "rootId": root_id,
                "image": str(image_pathname),
                "mask": str(mask_pathname),
                "imageSize": input_size,
                "maskSize": mask_size,
                "maskMode": mode,
                "geometryMatches": matches,
                "ignoredPixels": ignored_pixels if decoded else None,
                "unknownPixels": unknown_pixels
                if names is not None and decoded
                else None,
            }
        return input_size is not None and decoded, result

    result = work.audit(rows(), inspect, "image-headers-and-mask-pixels")
    if not counts["pairs"] and inventory_complete:
        work.issue(
            f"datasets.{work.split}", "No unambiguous image/mask pairs were found."
        )
    audited_masks = engines.total()
    eligible = result["audit"]["eligible"]
    result.update(
        roots=roots,
        counts=counts,
        inventoryComplete=inventory_complete,
        geometry={**geometry, "complete": geometry["checked"] == eligible},
        maskPixels={
            "auditedMasks": audited_masks,
            "complete": audited_masks == eligible,
            "engines": dict(engines),
            "pixels": ignored + other.pixels + sum(pixel_counts),
            "classes": [
                {"id": index, "name": name, "pixels": pixel_counts[index]}
                for index, name in enumerate(names or ())
            ],
            "ignored": {"id": ignore_label, "pixels": ignored},
            ("unknown" if names is not None else "unmapped"): other.result(),
        },
    )
    return result


def _classification(
    work: _Work, dataset: Mapping[str, Any], names: list[str] | None
) -> Json:
    paths = _roots(dataset, "dataroot_lq", work.root)
    roots = [
        {"id": str(index), "path": str(path)} for index, path in enumerate(paths, 1)
    ]
    inventory_complete = True
    class_ids = {name: index for index, name in enumerate(names or ())}
    classes: dict[str, Json] = {
        name: {
            "name": name,
            "id": index,
            "images": 0,
            "headersRead": 0,
            "directories": [],
        }
        for name, index in class_ids.items()
    }
    counts = dict.fromkeys(
        ("images", "classFolders", "emptyFolders", "unlabeledImages"), 0
    )

    def select(folder: Path) -> tuple[list[Path], bool]:
        readable = True

        def unreadable(path: Path, exc: OSError) -> None:
            nonlocal readable
            readable = False
            work.issue(path, str(exc))

        return image_files(folder, check=work.check, unreadable=unreadable), readable

    def class_record(name: str, path: Path) -> Json:
        record = classes.get(name)
        if record is None:
            record = classes[name] = {
                "name": name,
                "id": None,
                "images": 0,
                "headersRead": 0,
                "directories": [],
            }
            if names is not None:
                work.issue(path, f"Unknown class folder {name!r}.")
        return record

    def rows() -> Iterator[tuple[str, str, Path]]:
        nonlocal inventory_complete
        if dataset.get("meta_info") is not None:
            manifest = local_path(dataset["meta_info"], work.root)
            directories = {}
            try:
                for group, image, _ in folder_members(manifest, paths):
                    work.check()
                    name = image.parts[0]
                    folder = paths[group - 1] / name
                    record = class_record(name, folder)
                    key = (group, name)
                    if key not in directories:
                        directory = {
                            "rootId": str(group),
                            "path": str(folder),
                            "images": 0,
                            "inventoryComplete": True,
                        }
                        record["directories"].append(directory)
                        directories[key] = directory
                        counts["classFolders"] += 1
                    directories[key]["images"] += 1
                    record["images"] += 1
                    counts["images"] += 1
                    yield str(group), name, paths[group - 1] / image
            except (OSError, ValueError) as exc:
                inventory_complete = False
                for directory in directories.values():
                    directory["inventoryComplete"] = False
                work.issue(manifest, str(exc))
            return
        for root, folder in zip(roots, paths, strict=True):
            try:
                for path, is_class in class_entries(folder):
                    work.check()
                    if not is_class:
                        counts["unlabeledImages"] += 1
                        work.issue(path, "Image has no class folder.")
                        continue
                    name = path.name
                    record = class_record(name, path)
                    counts["classFolders"] += 1
                    directory = {
                        "rootId": root["id"],
                        "path": str(path),
                        "images": None,
                    }
                    record["directories"].append(directory)
                    files, readable = select(path)
                    inventory_complete &= readable
                    directory.update(images=len(files), inventoryComplete=readable)
                    if not files and readable:
                        counts["emptyFolders"] += 1
                        work.issue(path, "Class folder contains no supported images.")
                    record["images"] += len(files)
                    counts["images"] += len(files)
                    yield from ((root["id"], name, file) for file in files)
            except OSError as exc:
                inventory_complete = False
                work.issue(folder, str(exc))

    def inspect(row: tuple[str, str, Path], sample: bool) -> tuple[bool, Json | None]:
        root_id, name, path = row
        size = _size(path, work, still=True)
        if size is not None:
            classes[name]["headersRead"] += 1
        return size is not None, {
            "rootId": root_id,
            "path": str(path),
            "className": name,
            "classId": class_ids.get(name),
            "size": size,
        } if sample else None

    result = work.audit(rows(), inspect, "still-image-headers")
    present = {name for name, record in classes.items() if record["directories"]}
    missing = sorted(class_ids.keys() - present) if names is not None else None
    unknown = sorted(present - class_ids.keys()) if names is not None else None
    if inventory_complete and missing:
        phase = work.split.split("_", 1)[0]
        work.issue(
            f"datasets.{work.split}",
            f"No class folders for {missing}.",
            level="error" if phase == "train" else "warning",
        )
    if not counts["images"] and inventory_complete:
        work.issue(f"datasets.{work.split}", "No labeled images were found.")
    result.update(
        roots=roots,
        counts=counts,
        inventoryComplete=inventory_complete,
        classes=[
            classes[name] for name in [*class_ids, *sorted(present - class_ids.keys())]
        ],
        missingClasses=missing if inventory_complete else None,
        unknownClasses=unknown,
    )
    return result


def _category_names(settings: Mapping[str, Any]) -> dict[int, str] | None:
    categories = settings.get("categories")
    if categories is None:
        return None
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            "Provide categories as an ordered list of COCO {id, name} records."
        )
    names: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            raise ValueError("Each COCO category must provide id and name.")
        category_id, name = category.get("id"), category.get("name")
        if (
            type(category_id) is not int
            or category_id < 0
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError(
                "Each COCO category needs a nonnegative integer ID and nonempty name."
            )
        if category_id in names:
            raise ValueError(f"Repeated configured COCO category ID {category_id}.")
        names[category_id] = name
    if len(set(names.values())) != len(names):
        raise ValueError("Configured COCO category names must be distinct.")
    return names


def _box(ann: CocoAnnotation, image: CocoImage) -> tuple[tuple[float, ...], bool]:
    visible = clip_box(ann, image)
    rounded = _FLOAT32_BOX.unpack(_FLOAT32_BOX.pack(*visible))
    left, top, right, bottom = rounded
    if not all(math.isfinite(coordinate) for coordinate in rounded):
        raise ValueError(f"COCO annotation {ann.id} exceeds float32 coordinate range.")
    if left >= right or top >= bottom:
        raise ValueError(
            f"COCO annotation {ann.id} has zero extent at float32 precision."
        )
    x, y, width, height = ann.bbox
    return rounded, visible != (x, y, x + width, y + height)


_CocoRow = tuple[Path, CocoImage, list[CocoAnnotation], int, int, int]


def _detection(
    work: _Work, dataset: Mapping[str, Any], configured: dict[int, str] | None
) -> Json:
    roots = _roots(dataset, "dataroot_lq", work.root)
    if len(roots) != 1:
        raise ValueError("CocoDetectionDataset needs exactly one dataroot_lq.")
    annotation_path = local_path(dataset.get("annotation_file"), work.root)
    root = roots[0]
    source = {"imageRoot": str(root), "annotationFile": str(annotation_path)}
    try:
        if not annotation_path.is_file():
            raise FileNotFoundError("Expected a regular COCO JSON file.")
        document = read_coco(annotation_path)
    except (OSError, ValueError) as exc:
        work.issue(annotation_path, str(exc))
        return {
            **source,
            "counts": None,
            "categories": [],
            "inventoryComplete": False,
            "audit": {
                "method": "still-image-headers",
                "eligible": None,
                "attempted": 0,
                "completed": 0,
                "complete": False,
            },
            "samples": [],
            "samplesTruncated": False,
        }

    listing = index_coco(
        document,
        annotation_path,
        configured,
        report=lambda message: work.issue(annotation_path, message),
        check=work.check,
    )
    inventory_complete = True
    try:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"COCO image root does not exist: {root}")
    except (OSError, RuntimeError) as exc:
        work.issue(roots[0], str(exc))
        inventory_complete = False
    expected_names = configured or {}
    declared_names = {
        category_id: category.name
        for category_id, category in listing.categories.items()
    }
    labels = {
        category_id: index
        for index, (category_id, name) in enumerate(expected_names.items(), 1)
        if declared_names.get(category_id, name) == name
    }
    category_counts = {
        category_id: {
            "id": category_id,
            "name": declared_names.get(category_id, expected_names.get(category_id)),
            "configuredName": expected_names.get(category_id),
            "label": labels.get(category_id),
            "declared": category_id in declared_names,
            "boxes": 0,
            "ignoredBoxes": 0,
        }
        for category_id in dict.fromkeys((*expected_names, *declared_names))
    }
    counts = {
        "images": len(document.images),
        "distinctImages": len(listing.images),
        "categories": len(document.categories),
        "annotations": len(document.annotations),
        "boxes": 0,
        "ignoredBoxes": 0,
        "crowdBoxes": 0,
        "explicitIgnoreBoxes": 0,
        "clippedBoxes": 0,
        "emptyImages": 0,
        "ignoredOnlyImages": 0,
        "imagesWithoutUsableBoxes": 0,
        "pathsChecked": 0,
        "pathErrors": 0,
    }
    referenced = set()
    for ann in document.annotations:
        work.check()
        referenced.add(ann.image_id)
    del document

    def rows() -> Iterator[_CocoRow]:
        for image_id, image in listing.images.items():
            work.check()
            records = listing.annotations.get(image_id, [])
            boxes = ignored = clipped = 0
            for ann in records:
                work.check()
                try:
                    _, changed = _box(ann, image)
                except (OverflowError, ValueError) as exc:
                    work.issue(
                        annotation_path, f"Invalid box for annotation {ann.id}. {exc}"
                    )
                    continue
                boxes += 1
                ignored += ann.ignored
                clipped += changed
                counts["crowdBoxes"] += ann.iscrowd
                counts["explicitIgnoreBoxes"] += ann.ignore
                category = category_counts[ann.category_id]
                category["boxes"] += 1
                category["ignoredBoxes"] += ann.ignored
            counts["boxes"] += boxes
            counts["ignoredBoxes"] += ignored
            counts["clippedBoxes"] += clipped
            counts["emptyImages"] += image_id not in referenced
            counts["ignoredOnlyImages"] += boxes == ignored and ignored > 0
            counts["imagesWithoutUsableBoxes"] += boxes == ignored
            if inventory_complete:
                counts["pathsChecked"] += 1
                try:
                    path = image_path(root, image.file_name)
                except (OSError, RuntimeError, ValueError) as exc:
                    counts["pathErrors"] += 1
                    work.issue(annotation_path, f"COCO image {image_id}: {exc}")
                else:
                    yield path, image, records, boxes, ignored, clipped

    def inspect(row: _CocoRow, sample: bool) -> tuple[bool, Json | None]:
        path, image, records, boxes, ignored, clipped = row
        declared_size = (image.width, image.height)
        size = _size(path, work, still=True)
        if size is not None and size != declared_size:
            work.issue(path, f"File is {size}; COCO declares {declared_size}.")
        if not sample:
            return size is not None, None
        examples = []
        for ann in records:
            if len(examples) == _BOX_LIMIT:
                break
            work.check()
            try:
                xyxy, changed = _box(ann, image)
            except (OverflowError, ValueError):
                continue
            examples.append(
                {
                    "id": ann.id,
                    "categoryId": ann.category_id,
                    "label": labels.get(ann.category_id),
                    "xyxy": xyxy,
                    "ignored": ann.ignored,
                    "clipped": changed,
                }
            )
        return size is not None, {
            "id": image.id,
            "path": str(path),
            "fileName": image.file_name,
            "size": size,
            "declaredSize": declared_size,
            "boxes": boxes,
            "ignoredBoxes": ignored,
            "clippedBoxes": clipped,
            "empty": image.id not in referenced,
            "boxSamples": examples,
            "boxesTruncated": len(examples) < boxes,
        }

    result = work.audit(rows(), inspect, "still-image-headers")
    counts["invalidAnnotations"] = counts["annotations"] - counts["boxes"]
    counts["usableBoxes"] = counts["boxes"] - counts["ignoredBoxes"]
    result.update(
        **source,
        counts=counts,
        inventoryComplete=inventory_complete,
        categories=list(category_counts.values()),
    )
    return result


def _compare_class_splits(work: _Work, splits: list[Json], discover: bool) -> None:
    def present(split: Json) -> set[str]:
        return {record["name"] for record in split["classes"] if record["directories"]}

    train = next(
        (
            split
            for split in splits
            if split["id"] == "train" and split["inventoryComplete"]
        ),
        None,
    )
    if train is None:
        return
    train_classes = present(train)
    for split in splits:
        if split is train or not split["inventoryComplete"]:
            continue
        names = present(split)
        missing, extra = sorted(train_classes - names), sorted(names - train_classes)
        split["classComparison"] = {
            "comparedTo": "train",
            "missingClasses": missing,
            "extraClasses": extra,
        }
        if discover and (missing or extra):
            work.split = split["id"]
            work.issue(
                f"datasets.{work.split}",
                f"Class folders differ from train. Missing: {missing}. Extra: {extra}.",
                level="warning",
            )


def _compare_numeric_inputs(work: _Work, splits: list[Json], db: sqlite3.Connection) -> None:
    train = next(split for split in splits if split["id"] == "train")
    if work.cancelled is not None:
        db.set_progress_handler(work.cancelled, 32768)
    for split in splits:
        if split["id"].partition("_")[0] not in {"val", "test"}:
            continue
        # Scan the holdout features once; the training lookup uses the primary key.
        try:
            distinct, matches, first = db.execute(
                """SELECT count(*), coalesce(sum(v.occurrences), 0), min(v.first_row)
                   FROM seen v CROSS JOIN seen t
                   WHERE v.split=? AND t.split='train' AND t.features=v.features""",
                (split["id"],),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT:
                raise TaskDataCancelled("Dataset inspection was cancelled.") from exc
            raise
        seen_train = train["counts"]["recordsInspected"]
        seen_holdout = split["counts"]["recordsInspected"]
        split["trainingOverlap"] = {
            "trainingRecordsInspected": seen_train, "holdoutRecordsInspected": seen_holdout,
            "matchedRecords": matches, "distinctMatchedInputs": distinct,
            "firstMatchingRecord": first, "complete": train["inventoryComplete"] and split["inventoryComplete"],
        }
        if matches:
            work.split = split["id"]
            work.issue(split["source"], f"{matches} of the {seen_holdout} inspected holdout records "
                       f"share their complete input feature values with a record in the {seen_train} "
                       f"inspected training records. The first match is holdout record {first}.",
                       level="warning")


def _numeric_records(work: _Work, dataset: Mapping[str, Any], settings: Mapping[str, Any],
                     record_index: sqlite3.Connection | None) -> Json:
    import msgspec
    import numpy as np
    from trainner_config.tabular_task import RegressionOptions
    from trainner_data.numeric_records import read_values, selected_columns

    columns = msgspec.convert(settings, type=RegressionOptions)
    names = columns.features + columns.targets
    path = local_path(dataset.get("meta_info"), work.root)
    count = 0
    complete = False
    samples = []
    minima, maxima = np.full(len(names), np.inf), np.full(len(names), -np.inf)
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            original_version = artifact_version(os.fstat(stream.fileno()))
            positions = selected_columns(stream, path, names)
            while work.audit_limit is None or count < work.audit_limit:
                take = 4096 if work.audit_limit is None else min(4096, work.audit_limit - count)
                block = read_values(stream, path, positions, take)
                if len(block):
                    minima = np.minimum(minima, block.min(axis=0))
                    maxima = np.maximum(maxima, block.max(axis=0))
                    if record_index is not None:
                        # Match the numerical inputs the model receives. A signed zero has
                        # the same value as positive zero. Targets and extra columns play no role.
                        features = np.array(block[:, :len(columns.features)], dtype="<f4", order="C", copy=True)
                        features[features == 0] = 0
                        record_index.executemany(
                            "INSERT INTO seen VALUES(?,?,?,1) ON CONFLICT(split,features) "
                            "DO UPDATE SET occurrences=occurrences+1",
                            ((work.split, hashlib.sha256(row).digest(), index)
                             for index, row in enumerate(features, count + 1)),
                        )
                    for index, row in enumerate(block[:work.sample_limit - len(samples)], count + 1):
                        samples.append({"record": index, "values": dict(zip(names, row.tolist(), strict=True))})
                count += len(block)
                work.update("auditing", count, None)
                if len(block) < take:
                    complete = True
                    break
            if not complete:
                complete = not any(line.strip() for line in stream)
            if artifact_version(path.stat()) != original_version:
                raise ValueError("The CSV changed while it was being inspected.")
            if complete and not count:
                work.issue(path, "The CSV contains no numeric records.")
            elif complete and count > 1:
                for name in columns.targets:
                    i = names.index(name)
                    if minima[i] == maxima[i]:
                        work.issue(path, f"Target {name!r} is constant in this split.", level="warning")
    except (OSError, ValueError) as exc:
        complete = False
        work.issue(path, str(exc))
    return {
        "counts": {"recordsInspected": count}, "inventoryComplete": complete,
        "audit": {"method": "numeric-csv-prefix", "eligible": count if complete else None,
                  "attempted": count, "completed": count, "complete": complete},
        "samples": samples, "samplesTruncated": len(samples) < count or not complete,
        "source": str(path), "columns": [
            {"name": name, "role": "feature" if i < len(columns.features) else "target",
             "min": float(minima[i]) if count else None, "max": float(maxima[i]) if count else None}
            for i, name in enumerate(names)
        ],
    }


def inspect_task_data(
    body: Mapping[str, Any],
    *,
    root: Path = _ROOT,
    progress: Callable[[Json], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Json:
    """Inventory all provider records and audit a bounded set of file contents."""
    task = body.get("task")
    if not isinstance(task, str) or task not in _TASKS:
        raise ValueError(
            "Choose a supported task: " + ", ".join(_TASKS)
        )
    datasets = body.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("Provide at least one named dataset split in datasets.")
    settings = body.get("task_options")
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("task_options must be an object.")
    audit_limit, sample_limit = body.get("auditLimit", 100), body.get("sampleLimit", 25)
    if audit_limit is not None and (type(audit_limit) is not int or audit_limit < 0):
        raise ValueError(
            "auditLimit must be a nonnegative integer or null for a full audit."
        )
    if type(sample_limit) is not int or not 0 <= sample_limit <= _SAMPLE_LIMIT:
        raise ValueError(f"sampleLimit must be an integer from 0 to {_SAMPLE_LIMIT}.")

    names = (
        None
        if task in ("object_detection", "tabular_regression")
        else _class_names(settings, folders=task == "image_classification")
    )
    categories = _category_names(settings) if task == "object_detection" else None
    ignore_label = settings.get("ignore_label", 255)
    if task == "semantic_segmentation" and (
        type(ignore_label) is not int
        or (names is not None and 0 <= ignore_label < len(names))
    ):
        raise ValueError(
            "ignore_label must be an integer outside the known class ID range."
        )
    work = _Work(root, audit_limit, sample_limit, progress, cancelled)
    with ExitStack() as resources:
        record_index = None
        if task == "tabular_regression" and "train" in datasets and any(
            isinstance(key, str) and key.partition("_")[0] in {"val", "test"} for key in datasets
        ):
            # An unnamed SQLite file bounds RAM on full audits and disappears on close.
            record_index = resources.enter_context(closing(sqlite3.connect("")))
            record_index.execute("CREATE TABLE seen(split TEXT, features BLOB, first_row INTEGER, "
                                 "occurrences INTEGER, PRIMARY KEY(split,features)) WITHOUT ROWID")
        splits = []
        for split, dataset in datasets.items():
            if not isinstance(split, str) or not split or not isinstance(dataset, Mapping):
                raise ValueError("Each dataset split needs a name and an options object.")
            kind = dataset.get("type", _TASKS[task])
            if not isinstance(kind, str) or kind.lower() != _TASKS[task]:
                raise ValueError(f"datasets.{split} requires dataset type {_TASKS[task]}.")
            backend = dataset.get("io_backend", {"type": "disk"})
            if not isinstance(backend, Mapping) or backend.get("type") != "disk":
                raise ValueError(f"datasets.{split} requires the disk backend.")
            work.split = split
            work.update("inventory", 0, None, force=True)
            if task == "semantic_segmentation":
                report = _segmentation(work, dataset, names, ignore_label)
            elif task == "tabular_regression":
                report = _numeric_records(work, dataset, settings, record_index)
            elif task == "image_classification":
                report = _classification(work, dataset, names)
            else:
                report = _detection(work, dataset, categories)
            name = dataset.get("name")
            splits.append(
                {"id": split, "name": name if isinstance(name, str) else split, **report}
            )
        if record_index is not None:
            _compare_numeric_inputs(work, splits, record_index)

    if task == "image_classification":
        _compare_class_splits(work, splits, names is None)
    work.check(force=True)
    issues = (work.details["error"] + work.details["warning"])[:_ISSUE_LIMIT]
    issue_total = sum(work.issue_counts.values())
    return {
        "task": task,
        "ok": work.issue_counts["error"] == 0,
        "complete": all(
            split["inventoryComplete"] and split["audit"]["complete"]
            for split in splits
        ),
        "limits": {
            "auditPerSplit": audit_limit,
            "samplesPerSplit": sample_limit,
            "issues": _ISSUE_LIMIT,
            "maskValues": _VALUE_LIMIT,
            "boxesPerSample": _BOX_LIMIT,
        },
        "splits": splits,
        "issues": issues,
        "totals": {"issues": issue_total, **work.issue_counts},
        "issuesTruncated": len(issues) < issue_total,
    }

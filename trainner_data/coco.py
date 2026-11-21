"""COCO metadata, validated file selection and box geometry shared with the trainer."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from msgspec import DecodeError, Meta, Struct, json


class CocoImage(Struct):
    id: int
    file_name: str
    width: Annotated[int, Meta(ge=1)]
    height: Annotated[int, Meta(ge=1)]


class CocoCategory(Struct):
    id: int
    name: str


class CocoAnnotation(Struct):
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    iscrowd: Literal[0, 1] = 0
    ignore: Literal[0, 1] = 0

    @property
    def ignored(self) -> bool:
        return bool(self.iscrowd or self.ignore)


class CocoDocument(Struct):
    images: list[CocoImage]
    categories: list[CocoCategory]
    annotations: list[CocoAnnotation]


@dataclass(slots=True)
class CocoIndex:
    categories: dict[int, CocoCategory]
    images: dict[int, CocoImage]
    annotations: dict[int, list[CocoAnnotation]]


def read_coco(annotation_path: Path) -> CocoDocument:
    try:
        return json.decode(annotation_path.read_bytes(), type=CocoDocument)
    except DecodeError as exc:
        raise ValueError(f"Invalid COCO JSON in {annotation_path}. {exc}") from exc


def index_coco(
    document: CocoDocument,
    annotation_path: Path,
    configured: Mapping[int, str] | None,
    *,
    report: Callable[[str], None] | None = None,
    check: Callable[[], None] | None = None,
) -> CocoIndex:
    """Keep the first record for each ID when reporting recoverable errors."""

    def invalid(message: str) -> None:
        if report is None:
            raise ValueError(message)
        report(message)

    categories = {}
    for category in document.categories:
        if check is not None:
            check()
        if category.id in categories:
            invalid(f"Repeated COCO category ID {category.id} in {annotation_path}.")
            continue
        if configured is not None and configured.get(category.id) != category.name:
            invalid(
                f"COCO category {category.id} ({category.name!r}) does not match task_options.categories."
            )
        categories[category.id] = category

    images = {}
    for image in document.images:
        if check is not None:
            check()
        if image.id in images:
            invalid(f"Repeated COCO image ID in {annotation_path}: {image.id}.")
            continue
        images[image.id] = image
    if not images:
        invalid(f"No images in {annotation_path}.")

    annotations: dict[int, list[CocoAnnotation]] = {}
    annotation_ids: set[int] = set()
    for ann in document.annotations:
        if check is not None:
            check()
        if ann.id in annotation_ids:
            invalid(f"Repeated COCO annotation ID {ann.id}.")
            continue
        annotation_ids.add(ann.id)
        if ann.image_id not in images or ann.category_id not in categories:
            invalid(
                f"COCO annotation {ann.id} references an unknown image or category."
            )
            continue
        annotations.setdefault(ann.image_id, []).append(ann)
    return CocoIndex(categories, images, annotations)


def image_path(root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"COCO file_name {filename!r} must be relative to the image root."
        )
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"COCO file_name {filename!r} resolves outside the image root."
        )
    if not path.is_file():
        raise FileNotFoundError(f"COCO image does not exist: {path}")
    return path


def clip_box(
    ann: CocoAnnotation, image: CocoImage
) -> tuple[float, float, float, float]:
    x, y, width, height = ann.bbox
    if not all(math.isfinite(value) for value in ann.bbox) or width <= 0 or height <= 0:
        raise ValueError(
            f"COCO annotation {ann.id} needs a finite box with positive width and height."
        )
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = (
        min(float(image.width), x + width),
        min(float(image.height), y + height),
    )
    if left >= right or top >= bottom:
        raise ValueError(
            f"COCO annotation {ann.id} has no visible box in image {image.id}."
        )
    return left, top, right, bottom

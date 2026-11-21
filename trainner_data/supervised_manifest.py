"""Explicit disk membership for ClassFolderDataset and ImageMaskDataset."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path, PureWindowsPath

from trainner_data.task_folders import IMAGE_EXTENSIONS, MASK_EXTENSIONS


def _relative(value: object, extensions: frozenset[str]) -> Path:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("Expected a POSIX relative filename.")
    path = Path(value)
    if (
        PureWindowsPath(value).anchor
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or path.suffix.lower() not in extensions
    ):
        raise ValueError(f"Invalid relative filename {value!r}.")
    return path


def folder_members(
    manifest: Path, images: Sequence[Path], masks: Sequence[Path] | None = None
) -> Iterator[tuple[int, Path, Path | None]]:
    """Read ordered (one-based root, relative image, relative mask) records."""
    if masks is not None and len(images) != len(masks):
        raise ValueError("Provide one mask root per image root.")
    fields = {"root", "image"} if masks is None else {"root", "image", "mask"}
    with manifest.expanduser().open(encoding="utf-8") as stream:
        count = 0
        for count, line in enumerate(stream, 1):
            try:
                data = json.loads(line)
                if not isinstance(data, dict) or data.keys() != fields:
                    raise ValueError(f"Expected fields {sorted(fields)}.")
                group = data["root"]
                if type(group) is not int or not 1 <= group <= len(images):
                    raise ValueError("Root index is outside the declared image roots.")
                image = _relative(data["image"], IMAGE_EXTENSIONS)
                mask = None
                if masks is None:
                    if len(image.parts) < 2:
                        raise ValueError("Image has no class directory.")
                else:
                    mask = _relative(data["mask"], MASK_EXTENSIONS)
                    if image.with_suffix("") != mask.with_suffix(""):
                        raise ValueError("Image and mask relative stems differ.")
                    if not (masks[group - 1] / mask).is_file():
                        raise ValueError(f"Missing mask {mask} in root {group}.")
                if not (images[group - 1] / image).is_file():
                    raise ValueError(f"Missing image {image} in root {group}.")
            except ValueError as exc:
                raise ValueError(f"{manifest} line {count}: {exc}") from exc
            yield group, image, mask
    if count == 0:
        raise ValueError(f"Empty supervised membership file: {manifest}")

"""Disk selectors for whole-image classes and aligned class masks."""

import os
from collections.abc import Callable, Collection, Iterator
from pathlib import Path

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
)
MASK_EXTENSIONS = frozenset({".png", ".tiff", ".tif"})


def image_files(
    root: Path,
    extensions: Collection[str] = IMAGE_EXTENSIONS,
    *,
    check: Callable[[], None] | None = None,
    unreadable: Callable[[Path, OSError], None] | None = None,
) -> list[Path]:
    # As in Path.rglob, follow a symlinked root but do not recurse into
    # symlinked subdirectories. A symlink to an image file is eligible.
    def entries(folder: Path) -> Iterator[os.DirEntry[str]]:
        try:
            with os.scandir(folder) as stream:
                yield from stream
        except OSError as exc:
            if unreadable is not None:
                unreadable(folder, exc)

    pending = [root]
    selected = []
    while pending:
        folder = pending.pop()
        for entry in entries(folder):
            if check is not None:
                check()
            path = folder / entry.name
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif path.suffix.lower() in extensions and path.is_file():
                    selected.append(path)
            except OSError as exc:
                if unreadable is None:
                    raise
                unreadable(path, exc)
    return sorted(selected)


def files_by_stem(
    root: Path,
    extensions: Collection[str],
    *,
    duplicate: Callable[[Path, Path], None] | None = None,
    check: Callable[[], None] | None = None,
    unreadable: Callable[[Path, OSError], None] | None = None,
) -> dict[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"ImageMaskDataset directory does not exist: {root}")
    files = {}
    for path in image_files(root, extensions, check=check, unreadable=unreadable):
        stem = path.relative_to(root).with_suffix("")
        if stem not in files:
            files[stem] = path
        elif duplicate is None:
            raise ValueError(f"Multiple files match {stem} in {root}.")
        else:
            duplicate(stem, path)
    return files


def class_entries(root: Path) -> Iterator[tuple[Path, bool]]:
    """Yield top-level class directories and supported files without a class."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Classification dataset directory does not exist: {root}"
        )
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path, True
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path, False

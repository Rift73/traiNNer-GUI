"""Manifest of completed validation image exports. Opening the trainer UI is optional."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from trainner_data.file_version import artifact_version

FILE_NAME = "validation-images.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, image TEXT NOT NULL,
    output TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
    sources TEXT NOT NULL, PRIMARY KEY (dataset, step, image)
);
CREATE INDEX IF NOT EXISTS gallery_order ON images(step DESC, dataset, image);
CREATE INDEX IF NOT EXISTS image_history ON images(dataset, image, step DESC);
CREATE TABLE IF NOT EXISTS image_version (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, image TEXT NOT NULL,
    capture_id TEXT, version TEXT NOT NULL, PRIMARY KEY(dataset,step,image)
);
"""


def source_roots(dataset, kind: str) -> tuple[Path, ...]:
    roots = getattr(dataset, "dataroot_" + kind) or ()
    if isinstance(roots, str):
        roots = (roots,)
    return tuple(Path(os.path.abspath(root)) for root in roots)


def relative_image(path: str, roots: tuple[Path, ...]) -> Path:
    """Root slots separate independent sources with identical relative names."""
    source = Path(os.path.abspath(path))
    if source.stem in ("", ".", ".."):
        raise ValueError(f"Validation input has no safe image stem: {path!r}.")
    for index, root in enumerate(roots, 1):
        if source != root and source.is_relative_to(root):
            relative = source.relative_to(root)
            return Path(f"root-{index}") / relative if len(roots) > 1 else relative
    if roots:
        raise ValueError(f"Validation input {path} does not belong to a configured source root.")
    return Path(source.name)


def _member_path(key: str, database: str | None) -> str:
    if database is None:
        return key
    # The still providers display LMDB members as <database>/<exact key>.png.
    # Keys are opaque to LMDB. Only canonical relative keys name an export.
    relative = Path(key)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != key:
        raise ValueError(f"Cannot name a validation export from LMDB key {key!r}.")
    return os.path.join(database, key + ".png")


def selected_image_names(dataset, kind: str, roots: tuple[Path, ...], *, flat: bool) -> dict[str, Path] | None:
    """Plan from already selected members, without decoding images or scanning folders."""
    from traiNNer.data.paired_image_dataset import PairedImageDataset
    from traiNNer.data.single_image_dataset import SingleImageDataset

    if type(dataset) is PairedImageDataset:
        paths = (_member_path(row[kind + "_path"], row.get(kind + "_db")) for row in dataset.paths)
    elif type(dataset) is SingleImageDataset:
        paths = (_member_path(path, dataset.storage[i] if dataset.lmdb is not None else None)
                 for i, path in enumerate(dataset.paths))
    else:
        return None

    def legacy_name(relative: Path) -> Path:
        if flat:
            return Path(relative.stem)
        stem = relative.with_suffix("")
        return Path(*stem.parts[1:]) if len(roots) > 1 else stem

    members = {}
    legacy = Counter()
    qualified = Counter()
    for path in paths:
        if path in members:
            raise ValueError(f"Validation selects the same {kind.upper()} member twice: {path}.")
        relative = relative_image(path, roots)
        members[path] = relative
        legacy[legacy_name(relative)] += 1
        qualified[relative.with_suffix("")] += 1

    names = set()
    for path, relative in members.items():
        name = legacy_name(relative)
        if legacy[name] > 1:
            name = relative.with_suffix("")
            if qualified[name] > 1:
                # Two codecs in one scene can share a stem. Retain their suffixes.
                name = relative
        if name in names:
            raise ValueError(f"Validation inputs still share the export name {name}. Select distinct members.")
        members[path] = name
        names.add(name)
    return members


def selected_files_in_output(dataset, output: Path) -> set[Path]:
    """Guard selected inputs when a visualization directory overlaps a disk dataset."""
    from traiNNer.data.paired_image_dataset import PairedImageDataset
    from traiNNer.data.single_image_dataset import SingleImageDataset

    if type(dataset) not in (PairedImageDataset, SingleImageDataset) or dataset.lmdb is not None:
        return set()
    output = output.resolve()
    roots = source_roots(dataset.opt, "lq") + source_roots(dataset.opt, "gt")
    if not any(output.is_relative_to(root) or root.is_relative_to(output) for root in map(Path.resolve, roots)):
        return set()
    if type(dataset) is PairedImageDataset:
        paths = (row[kind] for row in dataset.paths for kind in ("gt_path", "lq_path"))
    else:
        paths = dataset.paths
    return {path for item in paths if (path := Path(item).resolve()).is_relative_to(output)}


def sample_id(path: str, roots: tuple[Path, ...]) -> str:
    image = Path(path).absolute()
    for number, root in enumerate(roots):
        if image.is_relative_to(root):
            return f"{number}/{image.relative_to(root).as_posix()}"
    return image.as_posix()


def _source(batch, kind: str):
    paths = batch.get(kind + "_path")
    if not paths:
        return None
    path = Path(paths[0]).absolute()
    db = batch.get(kind + "_db")
    if db:
        database = Path(db[0]).absolute()
        observed_keys = batch.get(kind + "_key")
        key = observed_keys[0] if observed_keys else path.relative_to(database).as_posix().removesuffix(".png")
        location = {"database":str(database), "key":key}
        storage = database / "data.mdb"
    else:
        location = {"path":str(path)}
        storage = path
    try:
        info = storage.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return {**location, "size":info.st_size, "mtimeNs":str(info.st_mtime_ns), "version":artifact_version(info)}


class ValidationImages:
    """Publish only after the existing writer futures have finished successfully."""

    def __init__(self, root: Path, dataset, step: int, source_kind: str = "lq"):
        self.root = root.absolute()
        self.dataset = dataset.name
        self.source_kind = source_kind
        roots = getattr(dataset,"dataroot_"+source_kind) or ()
        self.roots = tuple(Path(p).absolute() for p in ([roots] if isinstance(roots, str) else roots))
        self.step = step
        self.rows = []

    def add(self, batch, output: str, width: int, height: int) -> None:
        path = Path(output).absolute()
        if not path.is_relative_to(self.root):
            # The gallery serves images contained in the experiment directory.
            return
        name = sample_id(batch[self.source_kind+"_path"][0], self.roots)
        sources = {k: ref for k in ("lq","gt") if (ref := _source(batch, k)) is not None}
        self.rows.append((self.dataset, self.step, name, path.relative_to(self.root).as_posix(),
                          width, height, json.dumps(sources, separators=(",",":"))))

    def publish(self, capture_id: str | None = None) -> None:
        if not self.rows:
            return
        db = sqlite3.connect(self.root / FILE_NAME, timeout=10)
        try:
            db.executescript(_SCHEMA)
            with db:
                db.execute("DELETE FROM images WHERE dataset=? AND step=?", (self.dataset,self.step))
                db.execute("DELETE FROM image_version WHERE dataset=? AND step=?", (self.dataset,self.step))
                db.executemany("INSERT INTO images VALUES(?,?,?,?,?,?,?)", self.rows)
                db.executemany("INSERT INTO image_version VALUES(?,?,?,?,?)",
                               ((*row[:3], capture_id, artifact_version((self.root / row[3]).stat()))
                                for row in self.rows))
        finally:
            db.close()

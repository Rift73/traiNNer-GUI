"""Membership and protocol for a pooled supervised validation result."""

import sqlite3
from pathlib import Path

from trainner_config.options import DatasetOptions

from .validation_images import sample_id
from .validation_records import _json, _source

SCHEMA = """
CREATE TABLE IF NOT EXISTS supervised_evidence (
    evaluation INTEGER PRIMARY KEY REFERENCES evaluation ON DELETE CASCADE,
    sample_count INTEGER NOT NULL,
    protocol TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS supervised_member (
    evaluation INTEGER NOT NULL REFERENCES supervised_evidence ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    sample_id TEXT NOT NULL,
    input_path TEXT NOT NULL,
    target_path TEXT,
    identity TEXT,
    PRIMARY KEY(evaluation, ordinal)
) WITHOUT ROWID;
"""


class SupervisedValidationEvidence:
    """Record actual loader order and join to the existing pooled scalar results."""

    def __init__(self, dataset: DatasetOptions, protocol: dict[str, object]) -> None:
        if dataset.phase == "train":
            raise ValueError("Supervised validation evidence requires unaugmented data.")
        self.dataset = dataset.name
        roots = dataset.dataroot_lq
        if isinstance(roots, str):
            roots = [roots]
        self.roots = tuple(Path(root).expanduser().absolute() for root in roots or ())
        self.protocol = _json({"version": 1, **protocol})
        self.members: list[tuple[str, str, str | None, str | None]] = []

    def add(
        self,
        image_path: str,
        size: tuple[int, int],
        *,
        target_path: str | None = None,
        target_identity: str | int | None = None,
    ) -> None:
        """Size is the observed width and height before the model runs."""
        image_path = str(Path(image_path).expanduser().absolute())
        image_stamp = _source(image_path)
        if target_path is not None:
            target_path = str(Path(target_path).expanduser().absolute())
        target = (
            _source(target_path)
            if target_path is not None and target_identity is None
            else target_identity
        )
        identity = (
            _json({"image": image_stamp, "size": size, "target": target})
            if image_stamp is not None and target is not None
            else None
        )
        self.members.append(
            (sample_id(image_path, self.roots), image_path, target_path, identity)
        )

    def add_record(self, source: str, index: int, content_digest: str) -> None:
        """Store the ordinal and digest of the selected numeric input and target fields."""
        self.members.append((str(index), source, None, content_digest))

    def write(self, db: sqlite3.Connection, evaluation: int) -> None:
        """Publish within the transaction that owns the matching scalar rows."""
        db.execute(
            "INSERT INTO supervised_evidence VALUES(?,?,?)",
            (evaluation, len(self.members), self.protocol),
        )
        db.executemany(
            "INSERT INTO supervised_member VALUES(?,?,?,?,?,?)",
            (
                (evaluation, ordinal, *member)
                for ordinal, member in enumerate(self.members)
            ),
        )

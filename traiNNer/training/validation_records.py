"""Persist per-image validation values that were already computed by the trainer."""
from __future__ import annotations

import json
import hashlib
from uuid import uuid4
import math
import os
import stat
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
from trainner_config.metrics import metric_direction

from .validation_images import sample_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS capture (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, capture_id TEXT NOT NULL,
    PRIMARY KEY(dataset,step), FOREIGN KEY(dataset,step) REFERENCES evaluations ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS evaluations (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, input_protocol TEXT NOT NULL,
    sample_count INTEGER NOT NULL, PRIMARY KEY(dataset, step)
);
CREATE TABLE IF NOT EXISTS inference (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, settings TEXT NOT NULL,
    PRIMARY KEY(dataset,step),
    FOREIGN KEY(dataset,step) REFERENCES evaluations ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS metrics (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, metric TEXT NOT NULL,
    definition TEXT NOT NULL, better TEXT NOT NULL,
    PRIMARY KEY(dataset, step, metric),
    FOREIGN KEY(dataset, step) REFERENCES evaluations ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS samples (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, sample_id TEXT NOT NULL,
    input_path TEXT NOT NULL, target_path TEXT, identity TEXT,
    PRIMARY KEY(dataset, step, sample_id),
    FOREIGN KEY(dataset, step) REFERENCES evaluations ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS scores (
    dataset TEXT NOT NULL, step INTEGER NOT NULL, metric TEXT NOT NULL,
    sample_id TEXT NOT NULL, value REAL, nonfinite TEXT,
    PRIMARY KEY(dataset, step, metric, sample_id),
    FOREIGN KEY(dataset, step, sample_id) REFERENCES samples ON DELETE CASCADE,
    FOREIGN KEY(dataset, step, metric) REFERENCES metrics ON DELETE CASCADE
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source(path: str) -> tuple[str, int, int] | None:
    try:
        info = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (os.path.realpath(path), info.st_size, info.st_mtime_ns) if stat.S_ISREG(info.st_mode) else None


@dataclass(slots=True)
class Sample:
    key: str
    input: str
    target: str | None
    identity: str | None
    scores: dict[str, float] = field(default_factory=dict)


class ValidationRecords:
    """One complete evaluation, published atomically after its existing GPU reduction."""

    def __init__(self, opt: Any, dataset: Any, step: int, tensor_metrics: set[str], source_kind: str = "lq",
                 *, inference: dict):
        destination = opt.path.experiments_root if opt.is_train else opt.path.results_root
        if not destination:
            raise ValueError("Validation sample recording requires an experiment or result directory.")
        from .validation_summary import attempt_id
        file_key = hashlib.sha256(attempt_id().encode()).hexdigest()
        self.path = Path(destination) / "validation-observations" / (file_key + ".sqlite3")
        self.capture_id = uuid4().hex
        self.dataset = dataset.name
        self.step = step
        self.source_kind = source_kind
        self.inference = _json(inference)
        self.roots = tuple(Path(p).absolute() for p in (getattr(dataset,"dataroot_"+source_kind) or ()))
        self.samples: list[Sample] = []
        self.tensor_sample_indices: list[int] = []
        fields = msgspec.to_builtins(dataset)
        # File identity is tracked per pair. Loader scheduling and saved image
        # names do not change the pixels a model sees.
        for key in ("name", "phase", "dataroot_lq", "dataroot_gt", "num_worker_per_gpu",
                    "batch_size_per_gpu", "persistent_workers", "pin_memory",
                    "prefetch_factor", "prefetch_mode", "num_prefetch_queue", "native_paired_crop",
                    "video_scene_manifest"):
            fields.pop(key, None)
        from trainner_config.tasks import task_name
        self.reconstruction = task_name(opt) == "autoencoder"
        mono = dataset.color and str(dataset.type).lower() in ("pairedimagedataset", "singleimagedataset")
        plane_policy = ({"complementaryPlanes": "input_bicubic"} if opt.val.complementary_planes == "input_bicubic" else {})
        self.protocol = _json({"task":task_name(opt), "scale":opt.scale, **plane_policy,
                               "dataset":fields, "evaluator":"single_channel_reference_v1" if mono else "rgb_reference_v2",
                               "inputPixelFormat":opt.input_pixel_format,"outputPixelFormat":opt.output_pixel_format})
        self.metrics: dict[str, tuple[str, str]] = {}
        for name, raw in opt.val.metrics.items():
            kwargs = dict(raw)
            better = metric_direction(kwargs)
            kwargs.pop("better", None)
            # Preserve the recorded scoring protocol for pretrained metrics.
            # The direct device path uses the same 8-bit inputs and weights as
            # their NumPy entry point, including gray expansion and alpha removal.
            implementation = "quantized_rgb_tensor" if name in tensor_metrics else "rgb_numpy"
            if mono:
                implementation = "quantized_single_channel_tensor" if name in tensor_metrics else "single_channel_numpy"
            self.metrics[name] = (_json({"implementation":implementation, "options":kwargs}), better)

    def record_sample(self, batch: Mapping[str, Any]) -> Sample:
        lq = batch[self.source_kind + "_path"]
        target_kind = self.source_kind if self.reconstruction else "gt"
        gt = batch.get(target_kind + "_path")
        if len(lq) != 1 or (gt is not None and len(gt) != 1):
            raise ValueError("Per-image reports require validation batches with one image.")
        path, target = os.fspath(lq[0]), os.fspath(gt[0]) if gt else None
        def identity(image_path: str, db_field: str):
            storage = batch.get(db_field)
            if storage:
                stamp = _source(os.path.join(storage[0], "data.mdb"))
                return [stamp, image_path] if stamp else None
            return _source(image_path)
        stamps = [identity(path, self.source_kind+"_db")]
        # A centre prediction depends on its neighbours. Changing one frame
        # invalidates that comparison even if the centre and its GT are intact.
        for frame in batch.get("context_paths", ()):
            stamps.append(_source(os.fspath(frame[0])))
        if target is not None:
            stamps.append(identity(target, target_kind + "_db"))
        has_paths = target is not None or batch.get("gt") is None
        identity = _json(stamps) if has_paths and all(stamps) else None
        sample = Sample(sample_id(path, self.roots), path, target, identity)
        self.samples.append(sample)
        return sample

    def expect_tensor_scores(self, size: int) -> None:
        self.tensor_sample_indices.extend([len(self.samples) - 1] * size)

    def _score_rows(self):
        for sample in self.samples:
            for name, value in sample.scores.items():
                finite = math.isfinite(value)
                yield (self.dataset, self.step, name, sample.key, value if finite else None,
                       None if finite else str(value))

    def publish(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"Unsupported validation record format {version}.")
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(_SCHEMA)
            prefix = (self.dataset, self.step)
            with db:
                db.execute("PRAGMA user_version=1")
                db.execute("DELETE FROM evaluations WHERE dataset=? AND step=?", prefix)
                db.execute("INSERT INTO evaluations VALUES(?,?,?,?)",
                           (*prefix,self.protocol,len(self.samples)))
                db.execute("INSERT INTO inference VALUES(?,?,?)", (*prefix,self.inference))
                db.execute("INSERT INTO capture VALUES(?,?,?)", (*prefix,self.capture_id))
                db.executemany("INSERT INTO metrics VALUES(?,?,?,?,?)",
                               ((*prefix, name, *values) for name,values in self.metrics.items()))
                db.executemany("INSERT INTO samples VALUES(?,?,?,?,?,?)",
                               ((*prefix, s.key, s.input, s.target, s.identity) for s in self.samples))
                db.executemany("INSERT INTO scores VALUES(?,?,?,?,?,?)", self._score_rows())
        finally:
            db.close()

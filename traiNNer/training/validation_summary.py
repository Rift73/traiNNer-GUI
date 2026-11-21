"""Publish actual dataset-level observations after validation, outside the update loop."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import uuid4

FILE = "validation-summary.sqlite3"
_PROCESS_KEY = uuid4().hex

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation (
    id INTEGER PRIMARY KEY, attempt TEXT NOT NULL, dataset TEXT NOT NULL, step INTEGER NOT NULL,
    task TEXT NOT NULL, unit TEXT, settings TEXT NOT NULL, UNIQUE(attempt,dataset,step));
CREATE TABLE IF NOT EXISTS scalar (
    evaluation INTEGER NOT NULL REFERENCES evaluation ON DELETE CASCADE,
    metric TEXT NOT NULL, value REAL, nonfinite TEXT, PRIMARY KEY(evaluation, metric)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS label (
    evaluation INTEGER NOT NULL REFERENCES evaluation ON DELETE CASCADE,
    label_id INTEGER NOT NULL, name TEXT NOT NULL, support INTEGER NOT NULL,
    predicted INTEGER NOT NULL, correct INTEGER NOT NULL, PRIMARY KEY(evaluation, label_id)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cell (
    evaluation INTEGER NOT NULL REFERENCES evaluation ON DELETE CASCADE, target INTEGER NOT NULL,
    prediction INTEGER NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(evaluation,target,prediction)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS evaluation_lookup ON evaluation(dataset,step DESC);
"""


def attempt_id() -> str:
    managed = os.environ.get("TRAINNER_STUDIO_ATTEMPT")
    return Path(managed).name if managed else _PROCESS_KEY


def capture(model: Any, dataset_name: str, step: int) -> None:
    opt = model.opt
    directory = opt.path.experiments_root if opt.is_train else opt.path.results_root
    if opt.rank not in (None, 0) or not directory:
        return
    managed = os.environ.get("TRAINNER_STUDIO_ATTEMPT")
    if not managed and not (opt.val and (opt.val.record_samples or opt.is_train and opt.val.save_best is not None)):
        return
    from trainner_config.tasks import task_name
    task = task_name(opt)
    class_task = task in ("semantic_segmentation", "image_classification")
    store_matrix = class_task and opt.val is not None and opt.val.record_samples
    matrix = getattr(model, "confusion_matrices", {}).get(dataset_name) if store_matrix else None
    unit = {"image_classification": "image", "semantic_segmentation": "annotated_pixel"}.get(task)
    if task == "object_detection":
        unit = "box"
    settings = {"classNames": list(model.class_names)} if class_task or task == "object_detection" else {}
    if class_task or task == "object_detection":
        settings["taskOptions"] = opt.task_options
    elif task == "tabular_regression":
        unit = "record"
        settings["taskOptions"] = opt.task_options
    elif opt.val is not None:
        settings.update(metrics=opt.val.metrics, scale=opt.scale, outputPixelFormat=opt.output_pixel_format)
    tracker = getattr(model, "best_weight_tracker", None)
    if opt.val:
        from trainner_config.metrics import validation_snapshot_metrics

        directions = validation_snapshot_metrics(opt)
        if directions:
            settings["snapshotDirections"] = {name: directions[name] for name in model.metric_results if name in directions}
    if tracker is not None and tracker.metric in model.metric_results:
        # Archive the evaluator's actual retention direction, even for external providers.
        settings.setdefault("snapshotDirections", {})[tracker.metric] = tracker.direction
    evidence = (
        getattr(model, "validation_evidence", None)
        if opt.val and opt.val.record_samples
        else None
    )
    if evidence is not None and evidence.dataset != dataset_name:
        raise ValueError("Validation evidence does not match the evaluated dataset.")
    sample_record = getattr(model, "validation_sample_record", None)
    if sample_record is not None and (sample_record.dataset, sample_record.step) != (dataset_name, step):
        raise ValueError("Sample record and validation summary identify different observations.")
    attempt = attempt_id()
    dest = Path(directory) / FILE
    with closing(sqlite3.connect(dest, timeout=10)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError(f"Unsupported validation summary format {version}.")
        db.executescript(_SCHEMA)
        if evidence is not None:
            from .supervised_validation import SCHEMA

            db.executescript(SCHEMA)
        if sample_record is not None:
            db.execute("""CREATE TABLE IF NOT EXISTS restoration_record(
                evaluation INTEGER PRIMARY KEY REFERENCES evaluation ON DELETE CASCADE,
                file_key TEXT NOT NULL, capture_id TEXT NOT NULL, sample_count INTEGER NOT NULL,
                metric_directions TEXT NOT NULL)""")
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("PRAGMA user_version=1")
            key = (attempt, dataset_name, step)
            # A replacement must invalidate old client selections even when it
            # replaces the highest row ID in a legacy, non-AUTOINCREMENT table.
            eid = db.execute("SELECT coalesce(max(id),0)+1 FROM evaluation").fetchone()[0]
            db.execute("DELETE FROM evaluation WHERE attempt=? AND dataset=? AND step=?", key)
            db.execute("INSERT INTO evaluation(id,attempt,dataset,step,task,unit,settings) VALUES(?,?,?,?,?,?,?)",
                       (eid,*key,task,unit,json.dumps(settings,allow_nan=False)))
            for metric, observed in model.metric_results.items():
                value = float(observed)
                finite = math.isfinite(value)
                db.execute("INSERT INTO scalar VALUES(?,?,?,?)", (eid, metric, value if finite else None,
                           None if finite else str(value)))
            if matrix is not None:
                # These task evaluators have already transferred the complete confusion matrix to CPU.
                counts = matrix.numpy()
                names = model.class_names
                if counts.shape != (len(names), len(names)) or counts.dtype.kind not in "iu":
                    raise ValueError("Validation class counts disagree with the declared class map.")
                support, predicted, correct = counts.sum(axis=1), counts.sum(axis=0), counts.diagonal()
                db.executemany("INSERT INTO label VALUES(?,?,?,?,?,?)",
                               ((eid,i,name,int(support[i]),int(predicted[i]),int(correct[i]))
                                for i,name in enumerate(names)))
                targets, predictions = counts.nonzero()
                db.executemany("INSERT INTO cell VALUES(?,?,?,?)",
                               ((eid,int(t),int(p),int(counts[t,p])) for t,p in zip(targets,predictions,strict=True)))
            if evidence is not None:
                evidence.write(db, eid)
            if sample_record is not None:
                db.execute("INSERT INTO restoration_record VALUES(?,?,?,?,?)",
                           (eid, sample_record.path.stem, sample_record.capture_id, len(sample_record.samples),
                            json.dumps({k:v[1] for k,v in sample_record.metrics.items()})))
            if opt.is_train and opt.val and opt.val.record_samples:
                from .checkpoint_association import record_evaluation

                record_evaluation(model, db, eid, step)
    if evidence is not None:
        model.validation_evidence = None
    if sample_record is not None:
        model.validation_sample_record = None

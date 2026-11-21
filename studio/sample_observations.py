"""Locate a particular attempt's recorded per-sample validation scores."""
from __future__ import annotations

import json
import re
from pathlib import Path


def catalog(root: Path, dataset: str | None = None, metric: str | None = None):
    from .comparison import _connection, _file

    rows = []
    filters, params = [], []
    for column, value in (("e.dataset", dataset), ("m.metric", metric)):
        if value is not None:
            filters.append(column + "=?")
            params.append(value)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    legacy = _file(root)
    if legacy is not None:
        with _connection(legacy) as db:
            for row in db.execute("""
                SELECT e.dataset,e.step,e.sample_count AS sampleCount,m.metric,m.better
                FROM a.evaluations e JOIN a.metrics m USING(dataset,step)""" + where, params):
                rows.append({**dict(row), "evaluation":0, "aggregation":"meanPerSample",
                             "ledger":legacy, "capture":None})
    summary = root / "validation-summary.sqlite3"
    if not summary.is_file() or summary.resolve() != summary:
        return rows
    with _connection(summary) as db:
        if not db.execute("SELECT 1 FROM a.sqlite_master WHERE type='table' AND name='restoration_record'").fetchone():
            return rows
        for row in db.execute("""
            SELECT e.id,e.attempt,e.dataset,e.step,v.file_key,v.capture_id,v.sample_count,m.metric,v.metric_directions
            FROM a.restoration_record v JOIN a.evaluation e ON e.id=v.evaluation
            JOIN a.scalar m ON m.evaluation=e.id""" + where, params):
            if not re.fullmatch(r"[a-f0-9]{64}", row["file_key"]) or not re.fullmatch(r"[a-f0-9]{32}", row["capture_id"]):
                raise ValueError("Invalid validation observation identity.")
            directions = json.loads(row["metric_directions"])
            better = directions.get(row["metric"])
            if better in ("higher", "lower"):
                rows.append({"dataset":row["dataset"],"metric":row["metric"],"step":row["step"],
                             "sampleCount":row["sample_count"],"evaluation":row["id"],"attempt":row["attempt"],
                             "better":better,"aggregation":"meanPerSample","capture":row["capture_id"],
                             "ledger":root/"validation-observations"/(row["file_key"]+".sqlite3")})
    return rows


def selected_steps(groups, steps, ids, offline):
    from .comparison import choose_steps

    selected = list(steps)
    for i, evaluation in enumerate(ids):
        if evaluation is not None:
            groups[i] = [row for row in groups[i] if row["evaluation"] == evaluation]
            if not groups[i]:
                raise ValueError("The selected evaluation ID does not record this dataset and metric.")
            if selected[i] is None and evaluation != 0:
                selected[i] = groups[i][0]["step"]
    if not any(offline) and (selected[0] is None) != (selected[1] is None):
        step = selected[1] if selected[0] is None else selected[0]
        selected = [step, step]
    return choose_steps([[r["step"] for r in group] for group in groups], *selected, *offline)


def pick(group: list[dict], step: int) -> dict:
    matches = [row for row in group if row["step"] == step]
    if len(matches) != 1:
        if matches:
            raise ValueError("Multiple attempts recorded this iteration. Select the evaluation ID (0 for the legacy per-step ledger).")
        raise ValueError("The selected dataset, iteration and metric were not recorded.")
    source = matches[0]
    path = source["ledger"]
    if not path.is_file() or path.resolve() != path:
        raise ValueError("The observed validation score ledger is unavailable.")
    return source


def verify_capture(db, schema: str, source: dict) -> None:
    if source["capture"] is None:
        return
    row = db.execute(f"SELECT capture_id FROM {schema}.capture WHERE dataset=? AND step=?",
                     (source["dataset"],source["step"])).fetchone()
    if row is None or row[0] != source["capture"]:
        raise ValueError("The selected observation and its sample ledger no longer identify the same evaluation.")

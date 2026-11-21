"""Find jointly non-dominated validation observations for two recorded metrics."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from .checkpoint_selection import (
    _DIRECTIONS, _SUMMARY_FILE, _Receipt, _choose_direction, _database,
    _evaluator_direction, _observed, _recorded_checkpoint,
)
from .comparison import _connection, _spec
from .sample_observations import catalog, pick, verify_capture
from .study_results import population_key


def _population(protocol, members, count):
    observed = 0
    complete = True

    def rows():
        nonlocal observed, complete
        for key, identity in members:
            observed += 1
            if identity is None:
                complete = False
            yield key, identity

    digest = population_key(protocol, rows())
    return digest if complete and observed == count and observed > 0 else None


def _sample_population(row, metrics, source):
    source = pick([source], row["step"])
    with _connection(source["ledger"]) as db:
        verify_capture(db, "a", source)
        selection = row["dataset"], row["step"]
        specs = [_spec(db, "a", (*selection, metric)) for metric in metrics]
        members = db.execute(
            """SELECT s.sample_id,s.identity FROM a.samples s
               JOIN a.scores x USING(dataset,step,sample_id)
               JOIN a.scores y USING(dataset,step,sample_id)
               WHERE s.dataset=? AND s.step=? AND x.metric=? AND y.metric=?
               ORDER BY s.sample_id""", (*selection, *metrics)
        )
        protocol = {"input": specs[0]["input_protocol"], "inference": specs[0]["inference"],
                    "definitions": [spec["definition"] for spec in specs]}
        return _population(protocol, members, source["sampleCount"]), [spec["better"] for spec in specs]


def _front(points, directions):
    # Sorting a tied X group by its best Y makes it sufficient to compare
    # the group winner with all preceding groups' best Y once. Exact ties survive.
    signs = [1 if d == "higher" else -1 for d in directions]
    ordered = sorted(points, key=lambda p: (-signs[0]*p[0], -signs[1]*p[1], p[2]["id"]))
    best_y = None
    frontier = []
    for _, group in groupby(ordered, key=lambda p: p[0]):
        tied_x = list(group)
        y = signs[1] * tied_x[0][1]
        if best_y is None or y > best_y:
            frontier.extend(p for p in tied_x if signs[1]*p[1] == y)
            best_y = y
    return frontier


def checkpoint_frontier(runs, run_id, dataset, x, y, *, x_direction=None, y_direction=None,
                        available_only=False, limit=50):
    """Keep attempts and evaluator configurations distinct, with optional file availability filtering."""
    if not all(isinstance(v, str) and v.strip() for v in (dataset, x, y)) or x == y:
        raise ValueError("Choose a dataset and two different validation metrics.")
    if any(v is not None and v not in _DIRECTIONS for v in (x_direction, y_direction)):
        raise ValueError("A metric direction must be higher or lower.")
    if type(limit) is not int or not 1 <= limit <= 250 or type(available_only) is not bool:
        raise ValueError("Use a limit from 1 to 250 and a boolean availability filter.")
    root = runs.experiment_dir(run_id)
    metrics = x, y
    inputs = [row for row in catalog(root, dataset, x) if row["evaluation"] > 0]
    ledgers = {row["evaluation"]: row for row in inputs}
    cohorts = defaultdict(list)
    excluded = Counter()
    observations = 0
    with _database(root / _SUMMARY_FILE) as db:
        if db is not None:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
            entries = db.execute(
                """SELECT e.*,x.value AS vx,x.nonfinite AS nx,y.value AS vy,y.nonfinite AS ny
                   FROM evaluation e LEFT JOIN scalar x ON x.evaluation=e.id AND x.metric=?
                   LEFT JOIN scalar y ON y.evaluation=e.id AND y.metric=?
                   WHERE e.dataset=? AND e.step>=0 AND (x.metric IS NOT NULL OR y.metric IS NOT NULL)
                   ORDER BY e.id""", (x, y, dataset)
            )
            for row in entries:
                observations += 1
                values = _observed(row["vx"], row["nx"]), _observed(row["vy"], row["ny"])
                if any(v is None or math.isnan(v) for v in values):
                    excluded["missingOrNaN"] += 1
                    continue
                recorded = [_evaluator_direction(row["settings"], m) for m in metrics]
                population = None
                if row["id"] in ledgers:
                    population, recorded = _sample_population(row, metrics, ledgers[row["id"]])
                elif "supervised_evidence" in tables:
                    pooled = db.execute("SELECT sample_count,protocol FROM supervised_evidence WHERE evaluation=?",
                                        (row["id"],)).fetchone()
                    if pooled:
                        protocol = json.loads(pooled["protocol"])
                        definitions = protocol["metrics"]
                        recorded = [definitions.get(m, {}).get("better") for m in metrics]
                        order = "ordinal" if row["task"] == "object_detection" else "sample_id,identity"
                        members = db.execute(f"SELECT sample_id,identity FROM supervised_member "
                                             f"WHERE evaluation=? ORDER BY {order}", (row["id"],))
                        population = _population(protocol, members, pooled["sample_count"])
                directions = tuple(_choose_direction({(known, "evaluator")} if known in _DIRECTIONS else set(), explicit)[0]
                                   for known, explicit in zip(recorded, (x_direction, y_direction), strict=True))
                if None in directions:
                    excluded["directionUnknown"] += 1
                    continue
                receipt = None
                if "validation_checkpoint" in tables:
                    proof = db.execute("SELECT role,weight_path,save_id,reason FROM validation_checkpoint WHERE evaluation=?",
                                       (row["id"],)).fetchone()
                    if proof:
                        receipt = _Receipt(*proof)
                proof = None
                if available_only:
                    proof = _checkpoint(root, row, receipt)
                    if proof["status"] != "available":
                        excluded["weightsUnavailable"] += 1
                        continue
                settings = json.dumps(json.loads(row["settings"]), sort_keys=True, separators=(",", ":"))
                key = row["attempt"], row["task"], settings, population, directions
                cohorts[key].append((*values, row, receipt, proof))

    visible = []
    for (attempt, task, settings, population, directions), points in cohorts.items():
        frontier = _front(points, directions)
        observations_out = []
        for vx, vy, row, receipt, proof in frontier[:limit]:
            values = {m: v if math.isfinite(v) else str(v) for m, v in zip(metrics, (vx, vy), strict=True)}
            observations_out.append({"step": row["step"], "evaluation": row["id"], "values": values,
                                     "checkpoint": proof or _checkpoint(root, row, receipt)})
        visible.append({"attempt": attempt, "task": task, "settings": json.loads(settings),
                        "populationId": population, "populationVerified": population is not None,
                        "directions": dict(zip(metrics, directions, strict=True)),
                        "observedCount": len(points), "frontierCount": len(frontier),
                        "points": observations_out})
    return {"runId": run_id, "dataset": dataset, "metrics": list(metrics), "availableOnly": available_only,
            "observedCount": observations, "excluded": dict(excluded), "cohorts": visible}


def _checkpoint(root: Path, row, receipt):
    if receipt is None:
        return {"status": "unverified", "reason": "noEvaluationReceipt"}
    return _recorded_checkpoint(root, row["step"], receipt)

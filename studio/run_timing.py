"""Summarize existing trainer measurements at their original logged iterations."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .history import source_index, step_range

# The recent-window timer and the elapsed-run timer have different denominators.
_RECENT = "performance/iterations_per_second"
_SERIES = (_RECENT, "performance/wall_updates_per_second", "performance/data_wait_percent",
           "memory/peak_allocated_GiB", "memory/peak_reserved_GiB", "memory/device_total_GiB")


def _summary(db: sqlite3.Connection, table: str, bounds: tuple[int, int]) -> dict:
    placeholders = ",".join("?" for _ in _SERIES)
    rows = db.execute(f"""
        WITH ranked AS (
          SELECT metric,step,value,
                 row_number() OVER (PARTITION BY metric ORDER BY value,step) AS r,
                 count(*) OVER (PARTITION BY metric) AS n,
                 max(step) OVER (PARTITION BY metric) AS latest
          FROM {table}.points
          WHERE metric IN ({placeholders}) AND step BETWEEN ? AND ? AND value >= 0)
        SELECT metric,count(*),min(step),max(step),min(value),max(value),
               avg(value) FILTER (WHERE r IN ((n+1)/2,(n+2)/2)),
               max(value) FILTER (WHERE step=latest)
        FROM ranked GROUP BY metric ORDER BY metric""", (*_SERIES, *bounds))
    return {key: {"loggedPoints": n, "firstStep": first, "lastStep": last, "minimum": low,
                  "maximum": high, "median": median, "last": end}
            for key,n,first,last,low,high,median,end in rows}


def _matched(db: sqlite3.Connection, bounds: tuple[int, int]) -> dict:
    # Keep the median over matched ratios, including both central values when even.
    row = db.execute("""
        WITH matched(step,factor) AS (
          SELECT a.step,b.value/a.value FROM main.points a JOIN peer.points b
            ON b.metric=a.metric AND b.step=a.step
          WHERE a.metric=? AND a.step BETWEEN ? AND ? AND a.value>0 AND b.value>0),
        ranked AS (
          SELECT step,factor,row_number() OVER (ORDER BY factor,step) r,count(*) OVER () n FROM matched)
        SELECT count(*),min(step),max(step),avg(factor) FILTER (WHERE r IN ((n+1)/2,(n+2)/2))
        FROM ranked""", (_RECENT, *bounds)).fetchone()
    count, first, last, ratio = row
    return {"loggedStepsInCommon": count, "firstStep": first, "lastStep": last,
            "medianAgainstOverBase": ratio if count >= 2 else None}


def timing(runs, state: Path, run_id: str, *, against: str | None = None,
           source: str | None = None, against_source: str | None = None,
           from_step: int | None = None, to_step: int | None = None) -> dict:
    bounds = step_range(from_step, to_step)
    if against_source is not None and against is None:
        raise ValueError("Choose the comparison run for its log source.")
    result = {"fromStep": from_step, "toStep": to_step}
    indexes = {}
    selections = [("base", run_id, source)]
    if against is not None:
        selections.append(("against", against, against_source))
    for side, identity, selected in selections:
        index, path = source_index(runs.experiment_dir(identity), state, selected, trainer_only=True)
        result[side] = {"runId": identity, **index, "series": {}}
        indexes[side] = path
    primary = indexes["base"] or indexes.get("against")
    if primary is None:
        return result
    with closing(sqlite3.connect(primary.as_uri()+"?mode=ro", uri=True, timeout=15)) as db:
        if indexes["base"] and indexes.get("against"):
            db.execute("ATTACH DATABASE ? AS peer", (indexes["against"].as_uri()+"?mode=ro",))
        db.execute("BEGIN")
        for side, path in indexes.items():
            if path:
                table = "main" if side == "base" or not indexes["base"] else "peer"
                result[side]["series"] = _summary(db, table, bounds)
        if indexes["base"] and indexes.get("against"):
            if result["base"]["more"] or result["against"]["more"]:
                result["comparisonPendingIndex"] = True
            else:
                result["matchedRecentRates"] = _matched(db, bounds)
    return result

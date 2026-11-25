"""Trainer log segments belonging to the selected checkpoint lineage."""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from itertools import groupby
from pathlib import Path

from . import run_observe as observe
from .history import history, history_source_files, index_file, step_range
from .run_process import read_record
from .tensorboard_history import Writer, writer_order


_START = re.compile(observe.LOG_PREFIX + r"(?:Start training from|Resuming training from) epoch: \d+, iter: ([\d,]+)\.")
_MAX_STEP = 2**63 - 1


@lru_cache(maxsize=256)
def _start(path: Path, inode: int, size: int, modified: int) -> tuple[int | None, bool]:
    # A growing log can acquire its startup record on the next refresh.
    with path.open("rb") as stream:
        while stream.tell() < 4 * 1024 * 1024:
            raw = stream.readline(65536)
            if not raw:
                return None, False
            line = raw.decode(errors="replace")
            match = _START.match(line) or observe.NEO_START.match(line)
            if match:
                return int(match[1].replace(",", "")), True
            if observe.NEO_UPDATE.match(line) or ("[epoch:" in line and observe.ITERATION.search(line)):
                break
    return None, True


@dataclass(frozen=True)
class Segment:
    file: Path
    first: int
    last: int


def _log_time(path: Path) -> float:
    # Native filenames record creation in trainer-local time and survive a later touch.
    try:
        return datetime.strptime(path.stem[-15:], "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return path.stat().st_mtime


def _lineage(runs, queue, run_id):
    choices, segments, resumes = [], [], []
    visited = set()
    last, before = _MAX_STEP, None
    partial = False
    parents = None
    while run_id not in visited:
        visited.add(run_id)
        try:
            experiment = runs.experiment_dir(run_id)
        except FileNotFoundError:
            if len(visited) == 1:
                raise
            partial = True
            break
        files = history_source_files(experiment)
        trainer = sorted((p for name, p in files.items() if not name.startswith("TensorBoard/")),
                         key=_log_time, reverse=True)
        if before is not None:
            trainer = [p for p in trainer if _log_time(p) <= before]
        choices.extend({"runId": run_id, "source": p.name} for p in trainer)
        choices.extend({"runId": run_id, "source": key} for key, p in files.items()
                       if key.startswith("TensorBoard/") and (before is None or writer_order(p)[0] <= before))
        earliest = None
        for path in trainer:
            stat = path.stat()
            start, started = _start(path, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if not started:
                continue
            earliest = start
            if start is None:
                # No verified join for a legacy or still-starting log.
                segments.append(Segment(path, 0, last))
                return choices, segments, resumes, len(trainer) > 1 or before is not None
            if start <= last:
                segments.append(Segment(path, start + 1 if start else 0, last))
                if start:
                    resumes.append(start)
            if start == 0:
                return choices, segments, resumes, False
            last = min(last, start)

        # Queue replay alone is not a checkpoint link. Use only the pinned full-state continuation.
        if parents is None:
            parents = queue.history_parents()
        parent = parents.get(run_id)
        if not parent or (earliest is not None and earliest != parent[2]) or parent[0] in visited:
            partial = earliest is not None or before is not None
            break
        last = min(last, parent[2])
        run_id, attempt_id, _ = parent
        try:
            attempt = runs.attempt_dir(run_id, attempt_id)
            result = read_record(attempt / "result.json") or {}
            if not result.get("updatedAt"):
                partial = True
                break
            before = datetime.fromisoformat(result["updatedAt"]).timestamp()
        except FileNotFoundError:
            partial = True
            break
    return choices, segments, resumes, partial


def comparison_writers(runs, queue, run_id):
    """Keep writers from the same attempts and checkpoint intervals as the training chart."""
    from bisect import bisect_right

    choices, segments, resumes, partial = _lineage(runs, queue, run_id)
    selected = {seg.file: seg for seg in segments}
    owners = list(dict.fromkeys(choice["runId"] for choice in choices))
    writers = []
    for owner in reversed(owners):
        experiment = runs.experiment_dir(owner)
        files = history_source_files(experiment)
        available = [files[c["source"]] for c in choices if c["runId"] == owner and c["source"] in files]
        logs = sorted((p for p in available if p.name.startswith("train_")), key=_log_time)
        times = [_log_time(log) for log in logs]
        for path in sorted((p for p in available if p.name.startswith("events.out.tfevents.")), key=writer_order):
            # The trainer opens its TensorBoard writer after creating the corresponding log.
            # The next trainer log ends that attempt, even when it wrote no event file.
            pos = bisect_right(times, writer_order(path)[0]) - 1
            if logs and pos >= 0:
                seg = selected.get(logs[pos])
                if seg is None:
                    continue
                writers.append(Writer(path, seg.first, seg.last))
            elif not logs or (len(logs) == 1 and logs[0] in selected):
                seg = selected.get(logs[0]) if logs else None
                writers.append(Writer(path, seg.first, seg.last) if seg else Writer(path))
            else:
                partial = True
    return writers, sorted(set(resumes)), partial


def _sample(rows, low, high, count, budget):
    if count <= budget:
        return list(rows)
    buckets = budget // 4
    width = max(1, (high - low + buckets) // buckets)
    points = []
    for _, group in groupby(rows, key=lambda row: (row[0] - low) // width):
        first = last = minimum = maximum = next(group)
        for last in group:
            # Same ordering as SQLite: NULL sorts first. Retain a logged gap.
            if minimum[1] is not None and (last[1] is None or last[1] < minimum[1]):
                minimum = last
            if last[1] is not None and (maximum[1] is None or last[1] > maximum[1]):
                maximum = last
        points.extend(sorted({first, last, minimum, maximum}))
    return points


def training_history(runs, queue, state, run_id, metric, budget, from_step, to_step):
    if type(budget) is not int or not 8 <= budget <= 8192:
        raise ValueError("The plotted point budget must be an integer from 8 to 8192.")
    first, last = step_range(from_step, to_step)
    choices, segments, resumes, partial = _lineage(runs, queue, run_id)
    extra = {"attemptSources": choices, "resumeSteps": sorted(resumes), "partialHistory": partial}
    if not segments or (len(segments) == 1 and not resumes and segments[0].last == _MAX_STEP):
        source = segments[0].file if segments else None
        experiment = source.parent if source else runs.experiment_dir(run_id)
        return {**history(experiment, state, source=source.name if source else None, metric=metric, budget=budget,
                          from_step=from_step, to_step=to_step), **extra}

    reply = {"source": None, "series": [], "points": [], "indexedBytes": 0, "sourceBytes": 0,
             "more": False, **extra}
    totals = {}
    intervals = []
    count, low, high = 0, _MAX_STEP, 0
    for seg in reversed(segments):
        meta, index = index_file(state, seg.file, seg.file.name)
        for key in ("indexedBytes", "sourceBytes"):
            reply[key] += meta[key]
        reply["more"] |= meta["more"]
        lower, upper = max(first, seg.first), min(last, seg.last)
        with closing(sqlite3.connect(index)) as db:
            for key, n, lo, hi in db.execute(
                "SELECT metric,count(*),min(step),max(step) FROM points WHERE step BETWEEN ? AND ? GROUP BY metric",
                (seg.first, seg.last),
            ):
                if key not in totals:
                    totals[key] = {"key": key, "count": 0, "first": lo, "last": hi}
                totals[key]["count"] += n
                totals[key]["last"] = hi
                if key == metric and lower <= upper:
                    if lower > lo or upper < hi:
                        n, lo, hi = db.execute(
                            "SELECT count(*),min(step),max(step) FROM points WHERE metric=? AND step BETWEEN ? AND ?",
                            (key, lower, upper),
                        ).fetchone()
                    if n:
                        intervals.append((index, lo, hi))
                        count, low, high = count + n, min(low, lo), max(high, hi)
    reply["series"] = [totals[k] for k in sorted(totals)]
    if len(segments) == 1:
        reply.update(source=meta["source"], sourcePaths=meta["sourcePaths"])
    if metric is None:
        return reply
    if metric not in totals:
        raise ValueError("The selected metric was not logged in this training history.")

    def rows():
        for index, lower, upper in intervals:
            with closing(sqlite3.connect(index)) as db:
                yield from db.execute("SELECT step,value FROM points WHERE metric=? AND step BETWEEN ? AND ? ORDER BY step",
                                      (metric, lower, upper))

    reply.update(metric=metric, matchedPoints=count, downsampled=count > budget,
                 points=_sample(rows(), low, high, count, budget))
    return reply

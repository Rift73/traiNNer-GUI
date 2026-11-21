"""Incremental, restartable history of the trainer's existing per-update file logs."""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

from .native import executable

BUDGET = 4 * 1024 * 1024


def _logs(experiment: Path) -> list[Path]:
    paths = [p for p in experiment.glob("train_*.log") if p.is_file() and not p.is_symlink()]
    return sorted(paths, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)


def _stamp(stream, offset: int) -> str:
    # Catch truncate-and-replace on an existing inode even if the log has
    # already regrown since our previous observation.
    h = hashlib.sha256()
    stream.seek(0)
    h.update(stream.read(min(512, offset)))
    stream.seek(max(0, offset - 512))
    h.update(stream.read(min(512, offset)))
    return h.hexdigest()


def _consume(db, stream, start: int, end: int) -> int:
    stream.seek(start)
    result = subprocess.run([str(executable("history", "history")), str(start), str(end), str(BUDGET)],
                            stdin=stream, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Could not index the trainer log. " + result.stderr.decode(errors="replace"))
    lines = result.stdout.decode("ascii").splitlines()
    if not lines or not lines[-1].startswith("C\t"):
        raise RuntimeError("Incomplete response from the trainer log reader.")
    consumed = int(lines.pop()[2:])
    if not start <= consumed <= end:
        raise RuntimeError("Invalid cursor from the trainer log reader.")

    def points():
        for line in lines:
            tag, raw_step, key, raw_value = line.split("\t")
            if tag != "P":
                raise RuntimeError("Unexpected record from the trainer log reader.")
            value = float(raw_value)
            yield (key, int(raw_step), value if math.isfinite(value) else None)

    db.executemany("""INSERT INTO points(metric, step, value) VALUES(?,?,?)
                      ON CONFLICT(metric, step) DO UPDATE SET value=excluded.value""", points())
    return consumed


def _refresh(db, log: Path, *, tensorboard: bool = False) -> tuple[int, int, bool]:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cursor (
          slot INTEGER PRIMARY KEY CHECK(slot=1), dev INTEGER, inode INTEGER, size INTEGER,
          modified INTEGER, consumed INTEGER, stamp TEXT);
        CREATE TABLE IF NOT EXISTS points (
          metric TEXT NOT NULL, step INTEGER NOT NULL, value REAL, PRIMARY KEY(metric,step)
        ) WITHOUT ROWID;
    """)
    with log.open("rb", buffering=0) as stream, db:
        db.execute("BEGIN IMMEDIATE")
        stat = os.fstat(stream.fileno())
        row = db.execute("SELECT dev,inode,size,modified,consumed,stamp FROM cursor WHERE slot=1").fetchone()
        consumed = 0
        if row:
            dev, inode, size, modified, old, stamp = row
            if (dev,inode) == (stat.st_dev,stat.st_ino) and size <= stat.st_size and (
                stat.st_size != size or modified == stat.st_mtime_ns
            ) and _stamp(stream, old) == stamp:
                consumed = old
                if size == stat.st_size and modified == stat.st_mtime_ns and old == size:
                    return old,size,False
            else:
                db.execute("DELETE FROM points")
        before = consumed
        more = False
        if consumed < stat.st_size:
            if tensorboard:
                from .tensorboard_history import consume
                consumed, more = consume(db, stream, consumed, stat.st_size, BUDGET)
            else:
                consumed = _consume(db,stream,consumed,stat.st_size)
                more = consumed - before >= BUDGET and consumed < stat.st_size
        db.execute("INSERT OR REPLACE INTO cursor VALUES(1,?,?,?,?,?,?)",
                   (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,consumed,_stamp(stream,consumed)))
        return consumed, stat.st_size, more


def step_range(from_step: int | None, to_step: int | None) -> tuple[int, int]:
    for step in (from_step, to_step):
        if step is not None and (type(step) is not int or not 0 <= step <= 2**63-1):
            raise ValueError("Iteration bounds must be non-negative integers.")
    if from_step is not None and to_step is not None and from_step > to_step:
        raise ValueError("The first iteration must be at or before the last.")
    return from_step or 0, to_step if to_step is not None else 2**63-1


def source_index(experiment: Path, state: Path, source: str | None, *, trainer_only: bool = False) -> tuple[dict, Path | None]:
    files = {p.name: p for p in _logs(experiment)}
    if not trainer_only:
        from .tensorboard_history import sources

        for path in sources(experiment):
            location = "experiment" if path.parent == experiment else "tb_logger"
            files[f"TensorBoard/{location}/{path.name}"] = path
    reply = {"sources": list(files), "source": None}
    if not files and source is None:
        return reply, None
    source = source or next(iter(files), None)
    if source not in files:
        raise FileNotFoundError("The selected history source does not belong to this run.")
    log = files[source]
    tensorboard = source.startswith("TensorBoard/")
    reply["sourceFormat"] = "tensorboard" if tensorboard else "trainer-log"
    folder = state.resolve() / "history"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = b"tensorboard-scalars-v1\0" if tensorboard else b"trainer-log-v4\0"
    tag = hashlib.sha256(prefix + os.fsencode(log)).hexdigest()
    index = folder / (tag+".sqlite3")
    with closing(sqlite3.connect(index, timeout=15)) as db:
        consumed,size,more = _refresh(db,log,tensorboard=tensorboard)
    reply.update(source=source, indexedBytes=consumed, sourceBytes=size, more=more)
    return reply, index


def history(experiment: Path, state: Path, source: str | None = None,
            metric: str | None = None, budget: int = 1000,
            from_step: int | None = None, to_step: int | None = None) -> dict:
    if type(budget) is not int or not 8 <= budget <= 8192:
        raise ValueError("The plotted point budget must be an integer from 8 to 8192.")
    first_step, last_step = step_range(from_step, to_step)
    reply, index = source_index(experiment, state, source)
    reply.update(series=[], points=[])
    if index is None:
        return reply
    with closing(sqlite3.connect(index.as_uri() + "?mode=ro", uri=True, timeout=15)) as db:
        db.execute("BEGIN")
        series = [{"key": key, "count": n, "first": first, "last": last}
                  for key,n,first,last in db.execute(
                      "SELECT metric,count(*),min(step),max(step) FROM points GROUP BY metric ORDER BY metric")]
        reply["series"] = series
        if metric is None:
            return reply
        if metric not in {s["key"] for s in series}:
            raise ValueError("The selected metric was not logged in this attempt.")
        bounds = (metric, first_step, last_step)
        where = "metric=? AND step BETWEEN ? AND ?"
        n, low, high = db.execute(f"SELECT count(*),min(step),max(step) FROM points WHERE {where}",
                                  bounds).fetchone()
        reply.update(metric=metric, matchedPoints=n, downsampled=n>budget)
        if n <= budget:
            rows = db.execute(f"SELECT step,value FROM points WHERE {where} ORDER BY step", bounds)
        else:
            # Each nonempty bucket retains both extrema and its endpoints.
            # A missing/non-finite value remains a line break when selected.
            buckets = max(1, budget//4)
            width = max(1, (high-low+1+buckets-1)//buckets)
            rows = db.execute(f"""
                WITH marked AS (
                  SELECT step,value,(step-?)/? bucket FROM points WHERE {where}
                ), picks AS (
                  SELECT step,value,
                     row_number() OVER (PARTITION BY bucket ORDER BY step) s,
                     row_number() OVER (PARTITION BY bucket ORDER BY step DESC) e,
                     row_number() OVER (PARTITION BY bucket ORDER BY value,step) lo,
                     row_number() OVER (PARTITION BY bucket ORDER BY value DESC,step) hi
                  FROM marked
                )
                SELECT step,value FROM picks WHERE s=1 OR e=1 OR lo=1 OR hi=1 ORDER BY step
            """, (low,width,*bounds))
        reply["points"] = [[s,v] for s,v in rows]
        return reply

"""Read dataset metrics and class diagnostics from this run's actual evaluator output."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path


@contextmanager
def _open(root: Path):
    path = root / "validation-summary.sqlite3"
    if not path.exists():
        yield None
        return
    if not path.is_file() or path.resolve() != path:
        raise ValueError("Invalid validation summary path.")
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unknown validation summary format.")
        db.execute("BEGIN")
        yield db


def summaries(root: Path, query):
    limit = int(query.get("limit", 50))
    before = int(query.get("before", 2**63-1))
    if not 1 <= limit <= 250 or not 0 <= before <= 2**63-1:
        raise ValueError("Use a positive evaluation cursor and a limit from 1 to 250.")
    filters, args = ["id < ?"], [before]
    for field in ("dataset", "attempt"):
        if field in query:
            filters.append(field+"=?")
            args.append(query[field])
    with _open(root) as db:
        if db is None:
            return {"evaluations": [], "next": None}
        rows = db.execute(
            "SELECT id,attempt,dataset,step,task,unit,EXISTS(SELECT 1 FROM label WHERE evaluation=e.id) AS hasClassCounts "
            "FROM evaluation AS e WHERE " + " AND ".join(filters) + " ORDER BY id DESC LIMIT ?",
            (*args,limit+1),
        ).fetchall()
        visible = [dict(row) for row in rows[:limit]]
        if visible:
            by_id = {item["id"]: item for item in visible}
            for item in visible:
                item.update(metrics={}, nonfinite={}, hasClassCounts=bool(item["hasClassCounts"]))
            placeholders = ",".join("?" for _ in visible)
            for eid,key,val,invalid in db.execute(
                f"SELECT evaluation,metric,value,nonfinite FROM scalar WHERE evaluation IN ({placeholders})",
                tuple(by_id),
            ):
                by_id[eid]["metrics"][key] = val
                if invalid is not None:
                    by_id[eid]["nonfinite"][key] = invalid
        return {"evaluations": visible, "next": visible[-1]["id"] if len(rows)>limit else None}


def classes(root: Path, query):
    eid = int(query["evaluation"])
    offset = int(query.get("offset", 0))
    limit = int(query.get("limit", 50))
    if not 1 <= eid < 2**63 or not 0 <= offset < 2**63 or not 1 <= limit <= 250:
        raise ValueError("Invalid evaluation, offset or page length.")
    with _open(root) as db:
        header = db.execute("SELECT task,unit,settings FROM evaluation WHERE id=?",(eid,)).fetchone() if db else None
        if header is None:
            raise FileNotFoundError("The selected evaluation was not recorded.")
        labels = []
        for row in db.execute(
            "SELECT label_id AS id,name,support,predicted,correct FROM label WHERE evaluation=? "
            "ORDER BY label_id LIMIT ? OFFSET ?", (eid,limit,offset),
        ):
            label = dict(row)
            hits, support, predicted = label["correct"],label["support"],label["predicted"]
            union = support + predicted - hits
            label.update(precision=hits/predicted if predicted else None, recall=hits/support if support else None,
                         iou=hits/union if union else None)
            labels.append(label)
        count, total, correct = db.execute(
            "SELECT count(*),sum(support),sum(correct) FROM label WHERE evaluation=?", (eid,),
        ).fetchone()
        report = {"evaluation": eid, "task": header["task"], "unit": header["unit"], "settings": json.loads(header["settings"]),
                  "recorded": bool(count), "labelCount": count, "total":total, "correct":correct, "labels": labels,
                  "next": offset+len(labels) if offset+len(labels)<count else None}
        if "target" in query:
            target = int(query["target"])
            if not 0 <= target < count:
                raise ValueError("The target class ID is outside this evaluation's recorded class map.")
            report["target"] = target
            report["predictions"] = [dict(row) for row in db.execute(
                "SELECT c.prediction,l.name,c.count FROM cell c JOIN label l "
                "ON l.evaluation=c.evaluation AND l.label_id=c.prediction WHERE c.evaluation=? AND c.target=? "
                "ORDER BY c.count DESC,c.prediction LIMIT ? OFFSET ?", (eid,target,limit,offset))]
            n = db.execute("SELECT count(*) FROM cell WHERE evaluation=? AND target=?",(eid,target)).fetchone()[0]
            report["nextPrediction"] = offset+limit if offset+limit<n else None
        return report

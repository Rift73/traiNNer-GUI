"""Compare dataset-level supervised scores only when their scored populations agree."""
from __future__ import annotations

import json
from pathlib import Path

from .comparison import _connection


def _record(root: Path):
    path = root / "validation-summary.sqlite3"
    if path.resolve() != path or not path.is_file():
        return None
    return path


def _has_evidence(db, schema):
    return bool(db.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' "
                           "AND name='supervised_evidence'").fetchone())


def observations(root: Path):
    path = _record(root)
    if path is None:
        return []
    with _connection(path) as db:
        if not _has_evidence(db, "a"):
            return []
        result = []
        rows = db.execute("""
            SELECT e.dataset,e.step,e.id,e.attempt,v.sample_count AS sampleCount,v.protocol,s.metric
            FROM a.supervised_evidence v JOIN a.evaluation e ON e.id=v.evaluation
            JOIN a.scalar s ON s.evaluation=e.id ORDER BY e.step DESC,e.dataset,e.id DESC,s.metric
        """)
        # Each protocol is attached to the evaluation, not to each class metric.
        last, definitions = None, {}
        for row in rows:
            if row["id"] != last:
                definitions = json.loads(row["protocol"]).get("metrics", {})
                last = row["id"]
            definition = definitions.get(row["metric"])
            if definition:
                result.append({"dataset":row["dataset"], "step":row["step"], "evaluation":row["id"],
                               "attempt":row["attempt"], "sampleCount":row["sampleCount"],
                               "metric":row["metric"], "better":definition.get("better"),
                               "aggregation":"pooledDataset"})
        return result


def _choices(db, schema, dataset, metric, evaluation):
    if not _has_evidence(db, schema):
        return []
    filter_id = "" if evaluation is None else " AND e.id=?"
    return db.execute(f"""
        SELECT e.id,e.step,e.task,e.attempt,s.value,s.nonfinite,v.sample_count
        FROM {schema}.evaluation e JOIN {schema}.supervised_evidence v ON e.id=v.evaluation
        JOIN {schema}.scalar s ON s.evaluation=e.id
        WHERE e.dataset=? AND s.metric=? {filter_id} ORDER BY e.step DESC,e.id DESC
    """, (dataset,metric) if evaluation is None else (dataset,metric,evaluation)).fetchall()


def _protocol(db, schema, row, metric):
    recorded = db.execute(f"SELECT protocol FROM {schema}.supervised_evidence WHERE evaluation=?", (row["id"],)).fetchone()
    protocol = json.loads(recorded["protocol"])
    metrics = protocol.pop("metrics")
    if metric not in metrics:
        raise ValueError("The pooled score has no recorded metric definition.")
    return {**protocol, "metric":metrics[metric]}


def compare_pooled(left: Path, right: Path, dataset, metric, left_step, right_step, *,
                   left_offline=False, right_offline=False, left_evaluation=None, right_evaluation=None,
                   study_population=False):
    from .comparison import choose_steps

    a, b = _record(left), _record(right)
    if a is None or b is None:
        raise ValueError("Both evaluations need validation evidence recorded with val.record_samples.")
    with _connection(a,b) as db:
        groups = [_choices(db,schema,dataset,metric,evaluation) for schema,evaluation in
                  (("a",left_evaluation),("b",right_evaluation))]
        if not all(groups):
            raise ValueError("Both sides need pooled scores and recorded membership for this dataset and metric.")
        if left_evaluation and left_step is None:
            left_step = groups[0][0]["step"]
        if right_evaluation and right_step is None:
            right_step = groups[1][0]["step"]
        if left_step is None and right_step is not None and left_evaluation is None and right_evaluation and not (left_offline or right_offline):
            left_step = right_step
        if right_step is None and left_step is not None and right_evaluation is None and left_evaluation and not (left_offline or right_offline):
            right_step = left_step
        steps = choose_steps([{row["step"] for row in choices} for choices in groups],
                             left_step,right_step,left_offline,right_offline)
        rows = []
        for choices, step in zip(groups, steps, strict=True):
            at_step = [row for row in choices if row["step"] == step]
            if len(at_step) != 1:
                if at_step:
                    raise ValueError("Multiple attempts recorded this iteration. Select its evaluation ID.")
                raise ValueError("The selected dataset, iteration and metric were not recorded.")
            rows.append(at_step[0])
        first, second = rows
        left_protocol, right_protocol = [_protocol(db, schema, row, metric) for schema,row in zip(("a","b"),rows,strict=True)]
        inference = {}
        for side, protocol in (("left", left_protocol), ("right", right_protocol)):
            details = protocol.pop("inference", None)
            if details is None and "forwardPrecision" in protocol:
                details = {"precision": protocol.pop("forwardPrecision")}
            inference[side] = details
        reply = {"dataset":dataset, "metric":metric, "aggregation":"pooledDataset", "inference":inference,
                 "task":first["task"], "leftStep":steps[0], "rightStep":steps[1],
                 "leftCount":first["sample_count"], "rightCount":second["sample_count"],
                 "leftEvaluation":first["id"], "rightEvaluation":second["id"],
                 "better":left_protocol["metric"].get("better"), "comparable":False}
        differences = {k:{"left":left_protocol.get(k), "right":right_protocol.get(k)}
                       for k in left_protocol.keys() | right_protocol.keys()
                       if left_protocol.get(k) != right_protocol.get(k)}
        if differences:
            return {**reply,"differences":differences}
        params = (first["id"],second["id"])
        # Compare multisets. Two copies of a source on one side must not be
        # silently treated as one. File and target identities are writer evidence.
        counts = db.execute("""
            WITH grouped AS (
              SELECT sample_id,identity,count(*) AS nl,0 AS nr FROM a.supervised_member
              WHERE evaluation=? GROUP BY sample_id,identity
              UNION ALL
              SELECT sample_id,identity,0,count(*) FROM b.supervised_member
              WHERE evaluation=? GROUP BY sample_id,identity
            ), totals AS (
              SELECT identity,sum(nl) AS l,sum(nr) AS r FROM grouped GROUP BY sample_id,identity
            )
            SELECT coalesce(sum(CASE WHEN identity IS NOT NULL THEN min(l,r) ELSE 0 END),0) AS matched,
                   coalesce(sum(CASE WHEN identity IS NULL THEN l ELSE 0 END),0) AS leftUnverified,
                   coalesce(sum(CASE WHEN identity IS NULL THEN r ELSE 0 END),0) AS rightUnverified,
                   coalesce(sum(l),0) AS recordedLeft,coalesce(sum(r),0) AS recordedRight FROM totals
        """,params).fetchone()
        reply["counts"] = dict(counts)
        matched = counts["matched"]
        if not matched or any(matched != count for count in
                              (first["sample_count"],second["sample_count"],counts["recordedLeft"],counts["recordedRight"])):
            return {**reply,"reason":"The scored populations or recorded source identities differ."}
        if first["task"] == "object_detection":
            changed = db.execute("""
                SELECT 1 FROM a.supervised_member l JOIN b.supervised_member r ON l.ordinal=r.ordinal
                WHERE l.evaluation=? AND r.evaluation=? AND (l.sample_id!=r.sample_id OR l.identity!=r.identity)
                LIMIT 1
            """,params).fetchone()
            if changed:
                return {**reply,"reason":"Detection score ties depend on evaluation order, which differs."}
        if first["value"] is None or second["value"] is None:
            return {**reply,"reason":"The pooled metric did not produce two finite values.",
                    "nonfinite":{"left":first["nonfinite"],"right":second["nonfinite"]}}
        reply.update(comparable=True, pooled={"left":first["value"],"right":second["value"],
                                              "delta":second["value"]-first["value"]})
        if study_population:
            from .study_results import population_key
            order = "ordinal" if first["task"] == "object_detection" else "sample_id,identity"
            members = db.execute(f"SELECT sample_id,identity FROM a.supervised_member WHERE evaluation=? ORDER BY {order}",
                                 (first["id"],))
            reply["populationId"] = population_key(
                {"dataset":dataset,"metric":metric,"protocol":left_protocol,"inference":inference,
                 "aggregation":"pooledDataset"},members)
        return reply

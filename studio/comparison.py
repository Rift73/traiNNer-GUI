"""Compare recorded validation scores on matched images, outside the training process."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

RECORD_FILE = "validation-samples.sqlite3"


def _file(root: Path) -> Path | None:
    path = root / RECORD_FILE
    if path.is_symlink():
        raise ValueError("The validation record must be a regular file in the experiment.")
    return path if path.is_file() else None


@contextmanager
def _connection(*files: Path):
    # Keep both snapshots in one read transaction. A concurrently completing
    # validation must not change the membership halfway through a comparison.
    with closing(sqlite3.connect("file::memory:", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA trusted_schema=OFF")
        for name, path in zip(("a", "b"), files, strict=False):
            db.execute(f"ATTACH DATABASE ? AS {name}", (path.as_uri() + "?mode=ro",))
            version = db.execute(f"PRAGMA {name}.user_version").fetchone()[0]
            if version != 1:
                raise ValueError(f"Unsupported validation sample record version {version}.")
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        try:
            yield db
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Could not read the recorded validation samples: {exc}") from exc


def evaluations(root: Path) -> list[dict]:
    from .sample_observations import catalog
    from .pooled_comparison import observations
    records = [{key: value for key, value in row.items() if key not in ("capture","ledger")}
               for row in catalog(root)]
    records.extend(observations(root))
    return sorted(records, key=lambda row:(-row["step"],row["dataset"],-row["evaluation"],row["metric"]))


def _spec(db, schema: str, selection: tuple) -> dict:
    row = db.execute(f"""
        SELECT e.input_protocol, e.sample_count AS sampleCount, m.definition, m.better
        FROM {schema}.metrics m JOIN {schema}.evaluations e USING(dataset, step)
        WHERE m.dataset=? AND m.step=? AND m.metric=?""", selection).fetchone()
    if row is None:
        raise ValueError("The selected dataset, iteration and metric were not recorded.")
    # Older version-1 ledgers predate recording the inference environment.
    runtime = None
    if db.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name='inference'").fetchone():
        observation = db.execute(f"SELECT settings FROM {schema}.inference WHERE dataset=? AND step=?",
                                 selection[:2]).fetchone()
        if observation:
            runtime = json.loads(observation[0])
    protocol = json.loads(row["input_protocol"])
    dataset = protocol.get("dataset", {})
    dataset.pop("native_paired_crop", None)
    # The scene list selects members. Every neighbour of a scored centre is
    # already part of that member's identity, including on older ledgers.
    dataset.pop("video_scene_manifest", None)
    # Older observations predate these options. Only their historical default
    # is equivalent to omission. Keep explicit changes and evaluator versions.
    for name, default in (("annotation_file", None), ("otf_canvas", "trainer")):
        if dataset.get(name) == default:
            dataset.pop(name, None)
    return {**dict(row), "inference": runtime, "input_protocol": protocol,
            "definition": json.loads(row["definition"])}


# Each matching key retains its source-file identity and score status. This
# makes a renamed dataset, changed GT, and non-finite metric distinguishable.
_PAIRS = """
WITH
aa AS (SELECT s.sample_id, s.identity, v.value, v.nonfinite
       FROM a.samples s LEFT JOIN a.scores v
       ON v.dataset=s.dataset AND v.step=s.step AND v.sample_id=s.sample_id AND v.metric=:metric
       WHERE s.dataset=:dataset AND s.step=:left),
bb AS (SELECT s.sample_id, s.identity, v.value, v.nonfinite
       FROM b.samples s LEFT JOIN b.scores v
       ON v.dataset=s.dataset AND v.step=s.step AND v.sample_id=s.sample_id AND v.metric=:metric
       WHERE s.dataset=:dataset AND s.step=:right),
ids AS (SELECT sample_id FROM aa UNION SELECT sample_id FROM bb),
pairs AS MATERIALIZED (
    SELECT ids.sample_id AS sample, aa.value AS a, bb.value AS b, bb.value-aa.value AS delta,
      CASE WHEN aa.sample_id IS NULL THEN 'rightOnly'
           WHEN bb.sample_id IS NULL THEN 'leftOnly'
           WHEN aa.identity IS NULL OR bb.identity IS NULL THEN 'unverifiedSource'
           WHEN aa.identity != bb.identity THEN 'changedSource'
           WHEN aa.nonfinite IS NOT NULL OR bb.nonfinite IS NOT NULL THEN 'nonfinite'
           WHEN aa.value IS NULL OR bb.value IS NULL THEN 'missingScore'
           ELSE 'matched' END AS status
    FROM ids LEFT JOIN aa USING(sample_id) LEFT JOIN bb USING(sample_id)
)
"""


def _measure_pairs(db, params, higher):
    # Resolve both datasets and each sample's identity once. The two bounded
    # extremum lists share the same rows as the reported aggregate.
    predicate = ">" if higher else "<"
    rows = db.execute(_PAIRS + f"""
        , positive AS (SELECT * FROM pairs WHERE status='matched' AND delta>0
                       ORDER BY delta DESC,sample LIMIT 10),
          negative AS (SELECT * FROM pairs WHERE status='matched' AND delta<0
                       ORDER BY delta ASC,sample LIMIT 10)
        SELECT 'group',status,count(*),avg(a),avg(b),avg(delta),min(delta),max(delta),
               sum(b {predicate} a),sum(a=b),sum(a {predicate} b) FROM pairs GROUP BY status
        UNION ALL
        SELECT '+',sample,NULL,a,b,delta,NULL,NULL,NULL,NULL,NULL FROM positive
        UNION ALL
        SELECT '-',sample,NULL,a,b,delta,NULL,NULL,NULL,NULL,NULL FROM negative
        """, params)
    counts, summary = {}, {}
    examples = {"+": [], "-": []}
    for kind, name, count, a, b, delta, low, high, improved, tied, regressed in rows:
        if kind == "group":
            counts[name] = count
            if name == "matched":
                summary = {"meanLeft": a, "meanRight": b, "meanDelta": delta,
                           "minDelta": low, "maxDelta": high, "improved": improved,
                           "tied": tied, "regressed": regressed}
        else:
            examples[kind].append({"sample": name, "left": a, "right": b, "delta": delta})
    # The outer compound SELECT has no ordering contract.
    positive = sorted(examples["+"], key=lambda x: (-x["delta"], x["sample"]))
    negative = sorted(examples["-"], key=lambda x: (x["delta"], x["sample"]))
    return counts, summary, (positive, negative) if higher else (negative, positive)


def choose_steps(available, left_step, right_step, left_offline, right_offline):
    if left_offline or right_offline:
        selected = []
        for steps, step, offline in zip(available, (left_step,right_step), (left_offline,right_offline), strict=True):
            if not steps:
                raise ValueError("The selected dataset and metric have no recorded observations.")
            selected.append(-1 if offline else max(steps) if step is None else step)
        return tuple(selected)
    if left_step is None and right_step is None:
        common = set(available[0]).intersection(available[1])
        if not common:
            raise ValueError("These runs have no common recorded iteration. Choose two existing iterations.")
        return (max(common),) * 2
    return left_step, right_step


def compare(left: Path, right: Path, dataset: str, metric: str,
            left_step: int | None = None, right_step: int | None = None,
            *, left_offline: bool = False, right_offline: bool = False,
            left_evaluation: int | None = None, right_evaluation: int | None = None,
            study_population: bool = False) -> dict:
    if not dataset or not metric:
        raise ValueError("Select a validation dataset and metric.")
    for step, offline in ((left_step, left_offline), (right_step, right_offline)):
        if step is None:
            continue
        if type(step) is not int or not -1 <= step < 1 << 63:
            raise ValueError("The selected validation iteration is invalid.")
        if offline and step != -1:
            raise ValueError("A standalone evaluation has a single observation at step -1.")
        if not offline and step < 0:
            raise ValueError("Training evaluation iterations must be nonnegative.")
    selected_ids = (left_evaluation, right_evaluation)
    if any(v is not None and (type(v) is not int or not 0 <= v < 1 << 63) for v in selected_ids):
        raise ValueError("Evaluation IDs must be non-negative integers.")
    if not (left_offline or right_offline or any(v is not None for v in selected_ids)) and (left_step is None) != (right_step is None):
        raise ValueError("Select an iteration for both runs, or use the latest common iteration.")
    from .sample_observations import catalog, pick, selected_steps, verify_capture
    groups = [catalog(root, dataset, metric) for root in (left,right)]
    if not any(groups):
        from .pooled_comparison import compare_pooled
        return compare_pooled(left, right, dataset, metric, left_step, right_step,
                              left_offline=left_offline, right_offline=right_offline,
                              left_evaluation=left_evaluation, right_evaluation=right_evaluation,
                              study_population=study_population)
    if not all(groups):
        raise ValueError("Both sides need per-sample scores for this dataset and metric.")
    left_step,right_step = selected_steps(groups,(left_step,right_step),selected_ids,(left_offline,right_offline))
    first,second = (pick(group, step) for group,step in zip(groups,(left_step,right_step),strict=True))
    with _connection(first["ledger"], second["ledger"]) as db:
        verify_capture(db, "a", first)
        verify_capture(db, "b", second)
        a = _spec(db, "a", (dataset, left_step, metric))
        b = _spec(db, "b", (dataset, right_step, metric))
        reply = {"dataset": dataset, "metric": metric, "leftStep": left_step, "rightStep": right_step,
                 "leftEvaluation":first["evaluation"], "rightEvaluation":second["evaluation"],
                 "leftCapture":first["capture"], "rightCapture":second["capture"],
                 "leftCount": a["sampleCount"], "rightCount": b["sampleCount"], "better": a["better"],
                 "inference": {"left": a["inference"], "right": b["inference"]}}
        mismatches = [key for key in ("definition", "input_protocol", "better") if a[key] != b[key]]
        if mismatches:
            return {**reply, "comparable": False, "differences": {
                key: {"left": a[key], "right": b[key]} for key in mismatches}}
        params = {"dataset": dataset, "metric": metric, "left": left_step, "right": right_step}
        counts, summary, (improvements, regressions) = _measure_pairs(db, params, a["better"] == "higher")
        reply.update(counts=counts, comparable=bool(counts.get("matched")))
        if not reply["comparable"]:
            return reply
        # These are deltas across matched examples only. No independence or
        # uncertainty assumption is needed for the observed descriptive values.
        reply.update(summary=summary, improvements=improvements, regressions=regressions)
        if study_population and counts.get("matched") == a["sampleCount"] == b["sampleCount"] and len(counts) == 1:
            from .study_results import population_key
            members = db.execute("SELECT sample_id,identity FROM a.samples WHERE dataset=? AND step=? ORDER BY sample_id",
                                 (dataset, left_step))
            reply["populationId"] = population_key(
                {"dataset": dataset, "metric": metric, "definition": a["definition"], "input": a["input_protocol"],
                 "inference": reply["inference"], "aggregation": "meanPerSample"}, members)
        return reply

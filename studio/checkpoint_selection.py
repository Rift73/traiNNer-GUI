"""Rank recorded validation observations and scoped, numbered inference weights."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from traiNNer.training.checkpoint_association import (
    ROLES,
    SAVE_ID,
    SAVE_ID_KEY,
    pth_save_id,
)
from trainner_config.metrics import best_weight_filename, metric_direction, supervised_metric_direction

from . import run_observe as observe
from .paths import artifact_version
from .recipes import task_from_config
from .runs import RUN_ID, RunManager
from .yaml_edit import read_recipe

LOG_BYTE_LIMIT = 4 * 1024 * 1024
_SUMMARY_FILE = "validation-summary.sqlite3"
_SAMPLE_FILE = "validation-samples.sqlite3"
_DIRECTIONS = ("higher", "lower")
_BUILTIN_METRICS = {
    "calculate_psnr",
    "calculate_ssim",
    "calculate_lpips",
    "calculate_dists",
    "calculate_topiq",
    "calculate_neosr_topiq",
}
_START = re.compile(
    observe.LOG_PREFIX
    + r"(Start training from|Resuming training from) epoch: \d+, iter: ([\d,]+)\."
)
_UPDATE = re.compile(observe.LOG_PREFIX + r"\[epoch:\s*[\d,]+, iter:\s*([\d,]+),")
_SAVE = re.compile(observe.LOG_PREFIX + r"Saving models and training states to ")
_NEO_SAVE = re.compile(observe.LOG_PREFIX + r"saving update ([\d,]+)\s*$")
_SAVE_STEP = re.compile(r" for epoch: \d+, iter: ([\d,]+)\.\s*$")
_SAVE_FAILURE = re.compile(
    r"(?:Periodic checkpoint (\d+) could not be written|Checkpoint (\d+) is incomplete)"
)
_VALIDATION = re.compile(observe.LOG_PREFIX + r"Validation (.+)")
_SCALAR = re.compile(
    r"^\s*# (.+?)\s*:\s*(" + observe.NUMBER + r"|[+-]?(?:inf(?:inity)?|nan))(?=\s|$|\.{8} Best:)",
    re.IGNORECASE,
)
_END = re.compile(observe.LOG_PREFIX + r"End of training\. Time consumed:")
_MODEL = re.compile(
    observe.LOG_PREFIX + r"Model (?:\[bold\]|\[)?(\w+)(?:\[/bold\]|\])? is created\."
)
_FORMATS = (".safetensors", ".pth")
_GENERATORS = {
    "SRModel",
    "RealESRGANModel",
    "RealESRGANPairedModel",
    "RDUALModel",
    "RSRModel",
}


@dataclass(frozen=True, slots=True)
class _Receipt:
    role: str | None
    path: str | None
    identity: str | None
    reason: str | None


@dataclass(slots=True)
class _Point:
    dataset: str
    metric: str
    step: int
    value: float | None
    source: str
    attempt: str | None = None
    evaluation: int | None = None
    direction: str | None = None
    direction_source: str | None = None
    log: str | None = None
    line: int | None = None
    receipt: _Receipt | None = None

    def public(self) -> dict[str, Any]:
        value = self.value
        nonfinite = (
            str(value) if value is not None and not math.isfinite(value) else None
        )
        return {
            "step": self.step,
            "value": value if nonfinite is None else None,
            "nonfinite": nonfinite,
            "source": self.source,
            **({"attempt": self.attempt} if self.attempt is not None else {}),
            **({"evaluation": self.evaluation} if self.evaluation is not None else {}),
            **({"log": self.log, "line": self.line} if self.log is not None else {}),
        }


@dataclass(slots=True)
class _Validation:
    dataset: str
    line: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _Log:
    points: list[_Point] = field(default_factory=list)
    unplaced: Counter[tuple[str, str]] = field(default_factory=Counter)
    saves: Counter[int] = field(default_factory=Counter)
    failures: set[int] = field(default_factory=set)
    fresh: bool = False
    ended: bool = False
    complete: bool = False
    model: str | None = None
    bytes_read: int = 0


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _json_object(value: str | None) -> dict:
    return _mapping(json.loads(value)) if value else {}


def _direction(options: Any) -> str | None:
    options = _mapping(options)
    kind = options.get("type")
    if (
        ("better" in options and options["better"] not in _DIRECTIONS)
        or not isinstance(kind, str)
        or (kind not in _BUILTIN_METRICS and options.get("better") not in _DIRECTIONS)
    ):
        return None
    return metric_direction(options)


def _recipe_direction(recipe: dict, metric: str) -> str | None:
    metrics = _mapping(_mapping(recipe.get("val")).get("metrics"))
    return _direction(metrics.get(metric))


def _observed(value: float | None, nonfinite: str | None) -> float | None:
    if nonfinite in ("nan", "inf", "+inf", "-inf"):
        return float(nonfinite)
    return value if nonfinite is None else None


@contextmanager
def _database(path: Path) -> Iterator[sqlite3.Connection | None]:
    if not path.exists():
        yield None
        return
    if not observe.regular_file(path):
        raise ValueError(
            "Validation records must be regular files inside the experiment."
        )
    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        ) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("PRAGMA query_only=ON")
            if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError(
                    f"Unsupported validation record format in {path.name}."
                )
            db.execute("BEGIN")
            yield db
    except sqlite3.DatabaseError as error:
        raise ValueError(f"Could not read {path.name}: {error}") from error


def _query(path: Path, sql: str, *args: Any) -> list[sqlite3.Row]:
    with _database(path) as db:
        return [] if db is None else db.execute(sql, args).fetchall()


def _evaluator_direction(settings: str, metric: str, task: str) -> str | None:
    saved = _json_object(settings)
    retained = _mapping(saved.get("snapshotDirections")).get(metric)
    if retained in _DIRECTIONS:
        return retained
    configured = _direction(_mapping(saved.get("metrics")).get(metric))
    names = saved.get("classNames")
    return configured or (supervised_metric_direction(task, metric, names) if isinstance(names, list) else None)


def _summaries(experiment: Path, dataset: str, metric: str) -> list[_Point]:
    with _database(experiment / _SUMMARY_FILE) as db:
        if db is None:
            return []
        has_receipts = db.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='validation_checkpoint'"
        ).fetchone()
        fields = (
            ", c.evaluation, c.role, c.weight_path, c.save_id, c.reason"
            if has_receipts
            else ""
        )
        join = (
            "LEFT JOIN validation_checkpoint c ON c.evaluation=e.id"
            if has_receipts
            else ""
        )
        rows = db.execute(
            f"""SELECT e.id, e.attempt, e.step, e.task, e.settings, s.value, s.nonfinite {fields}
                FROM evaluation e JOIN scalar s ON s.evaluation=e.id {join}
                WHERE e.dataset=? AND s.metric=? ORDER BY e.id""",
            (dataset, metric),
        ).fetchall()
    points = []
    for row in rows:
        direction = _evaluator_direction(row["settings"], metric, row["task"])
        receipt = None
        if has_receipts and row["evaluation"] is not None:
            receipt = _Receipt(
                row["role"], row["weight_path"], row["save_id"], row["reason"]
            )
        points.append(
            _Point(
                dataset,
                metric,
                row["step"],
                _observed(row["value"], row["nonfinite"]),
                "summary",
                attempt=row["attempt"],
                evaluation=row["id"],
                direction=direction,
                direction_source="evaluator" if direction else None,
                receipt=receipt,
            )
        )
    return points


def _sample_direction_column(database: Path) -> str:
    columns = {row["name"] for row in _query(database, "PRAGMA table_info(metrics)")}
    return "m.better" if "better" in columns else "NULL"


def _indexed_catalog(experiment: Path) -> dict[tuple[str, str], set[tuple[str, str]]]:
    series: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for dataset, metric, settings, task in _query(
        experiment / _SUMMARY_FILE,
        """SELECT DISTINCT e.dataset, s.metric, e.settings, e.task
           FROM evaluation e JOIN scalar s ON s.evaluation=e.id""",
    ):
        entry = series[dataset, metric]
        if direction := _evaluator_direction(settings, metric, task):
            entry.add((direction, "evaluator"))
    database = experiment / _SAMPLE_FILE
    column = _sample_direction_column(database)
    for dataset, metric, direction in _query(
        database, f"SELECT DISTINCT m.dataset, m.metric, {column} FROM metrics m"
    ):
        entry = series[dataset, metric]
        if direction in _DIRECTIONS:
            entry.add((direction, "stored"))
    return series


def _sample_directions(experiment: Path, dataset: str, metric: str) -> dict[int, str]:
    database = experiment / _SAMPLE_FILE
    column = _sample_direction_column(database)
    rows = _query(
        database,
        f"SELECT m.step, {column} FROM metrics m WHERE m.dataset=? AND m.metric=?",
        dataset,
        metric,
    )
    return {step: direction for step, direction in rows if direction in _DIRECTIONS}


def _sample_means(experiment: Path, dataset: str, metric: str) -> list[_Point]:
    # The native evaluator's older records contain the per-image observations.
    # The caller gets a derived mean and its direction from the same read.
    database = experiment / _SAMPLE_FILE
    column = _sample_direction_column(database)
    rows = _query(
        database,
        f"""SELECT s.step, e.input_protocol, {column} AS better, AVG(s.value) AS mean,
                   MAX(s.nonfinite = 'nan') AS nan,
                   MAX(s.nonfinite IN ('inf', '+inf')) AS pos_inf,
                   MAX(s.nonfinite = '-inf') AS neg_inf
            FROM scores s JOIN evaluations e USING(dataset, step)
            JOIN metrics m USING(dataset, step, metric)
            WHERE s.dataset=? AND s.metric=? AND (s.value IS NOT NULL OR s.nonfinite IS NOT NULL)
            GROUP BY s.step, e.input_protocol, better""",
        dataset,
        metric,
    )
    points = []
    for row in rows:
        if _json_object(row["input_protocol"]).get("evaluator") != "rgb_reference_v2":
            continue
        if row["nan"] or (row["pos_inf"] and row["neg_inf"]):
            value = math.nan
        elif row["pos_inf"]:
            value = math.inf
        elif row["neg_inf"]:
            value = -math.inf
        else:
            value = row["mean"]
        direction = row["better"] if row["better"] in _DIRECTIONS else None
        points.append(
            _Point(
                dataset,
                metric,
                row["step"],
                value,
                "sampleMean",
                direction=direction,
                direction_source="stored" if direction else None,
            )
        )
    return points


def _positive(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _only_step(
    first: int | None, last: int | None, cadence: int | None, final: int | None
) -> int | None:
    if first is None or last is None or first > last:
        return None
    if first == last:
        return first
    if cadence is None:
        return None
    scheduled = ((first + cadence - 1) // cadence) * cadence
    steps = {scheduled} if scheduled <= last else set()
    if scheduled + cadence <= last:
        return None
    if final is not None and first <= final <= last:
        steps.add(final)
    return next(iter(steps)) if len(steps) == 1 else None


def _scan_log(
    path: Path, recipe: dict, budget: int, selected: tuple[str, str] | None, whole: bool
) -> _Log:
    result = _Log()
    total = _positive(_mapping(recipe.get("train")).get("total_iter"))
    save_freq = _positive(_mapping(recipe.get("logger")).get("save_checkpoint_freq"))
    val_freq = _positive(_mapping(recipe.get("val")).get("val_freq"))
    step = None
    trajectory = True
    starts = 0
    explicit_end = None
    validations: list[_Validation] = []
    unnumbered_saves = 0
    active_validation = None

    def close_interval(last: int | None, final: int | None = None) -> None:
        nonlocal unnumbered_saves
        val_step = _only_step(step, last, val_freq, final)
        for block in validations:
            for name, value in block.metrics.items():
                if val_step is None:
                    result.unplaced[block.dataset, name] += 1
                else:
                    result.points.append(
                        _Point(
                            block.dataset,
                            name,
                            val_step,
                            value,
                            "log",
                            log=path.name,
                            line=block.line,
                        )
                    )
        save_step = _only_step(step, last, save_freq, final)
        if save_step is not None and unnumbered_saves:
            result.saves[save_step] += unnumbered_saves
        validations.clear()
        unnumbered_saves = 0

    try:
        with path.open("rb", buffering=0) as stream:
            before = os.fstat(stream.fileno())
            if whole and before.st_size > budget:
                return result
            data = stream.read(min(budget, before.st_size))
            result.bytes_read = len(data)
            after = os.fstat(stream.fileno())
    except OSError:
        return result
    result.complete = result.bytes_read == before.st_size and artifact_version(
        before
    ) == artifact_version(after)
    for number, raw in enumerate(data.splitlines(keepends=True), start=1):
        if not raw.endswith(b"\n"):
            result.complete = False
            break
        if not raw.strip():
            continue
        if (
            b" INFO: " not in raw
            and b"# " not in raw
            and b"checkpoint " not in raw
            and b"Checkpoint " not in raw
        ):
            active_validation = None
            continue
        line = observe.ESCAPE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
        if active_validation is not None and (scalar := _SCALAR.match(line)):
            if selected is None or selected == (active_validation.dataset, scalar[1]):
                active_validation.metrics[scalar[1]] = float(scalar[2])
            continue
        active_validation = None
        if start := _START.match(line) or observe.NEO_START.match(line):
            close_interval(None)
            local_start = start.re is _START
            step = int(start[2 if local_start else 1].replace(",", ""))
            explicit_end = None
            starts += 1
            trajectory = (
                trajectory
                and starts == 1
                and (not local_start or start[1] == "Start training from")
                and step == 0
            )
            result.ended = False
            if not trajectory:
                save_freq = val_freq = None
        elif update := _UPDATE.match(line) or observe.NEO_UPDATE.match(line):
            observed_step = int(update[1].replace(",", ""))
            forward = step is not None and observed_step > step
            close_interval(observed_step - 1 if forward else None)
            trajectory = trajectory and forward and not result.ended
            step = observed_step
            if not trajectory:
                save_freq = val_freq = None
        elif validation := _VALIDATION.match(line):
            active_validation = _Validation(validation[1], number)
            validations.append(active_validation)
        elif saved := _NEO_SAVE.match(line):
            result.saves[int(saved[1].replace(",", ""))] += 1
        elif _SAVE.match(line):
            explicit = _SAVE_STEP.search(line)
            if explicit:
                result.saves[int(explicit[1].replace(",", ""))] += 1
            else:
                unnumbered_saves += 1
        elif failure := _SAVE_FAILURE.search(line):
            result.failures.add(int(failure[1] or failure[2]))
        elif ending := observe.NEO_END.match(line):
            explicit_end = int(ending[1].replace(",", ""))
            result.ended = True
        elif _END.match(line):
            result.ended = True
        elif model := _MODEL.match(line):
            result.model = model[1]
    final = (explicit_end if explicit_end is not None else total) if result.complete and result.ended and trajectory else None
    close_interval(final, final)
    result.fresh = trajectory and starts == 1
    return result


def _recipe(path: Path) -> dict:
    text = observe.recipe_text(path) if observe.regular_file(path) else None
    try:
        if text is None:
            return {}
        if path.suffix == ".toml":
            # NeoSR archives attest cadence and metric names, independently of
            # this trainer's raw/EMA weight naming and validation policy.
            original = tomllib.loads(text)
            return {name: _mapping(original.get(name)) for name in ("train", "val", "logger")}
        return read_recipe(text)
    except ValueError:
        # An unreadable archived recipe can't vouch for cadence or model role.
        return {}


def _context(
    runs: RunManager, run_id: str, experiment: Path
) -> tuple[dict, str | None, bool]:
    if RUN_ID.fullmatch(run_id):
        attempt = runs.attempt_dir(run_id)
        attempts = [
            p
            for p in observe.files(attempt.parent)
            if RUN_ID.fullmatch(p.name) and p.is_dir()
        ]
        if attempts != [attempt]:
            return {}, None, False
        return _recipe(attempt / "options.yml"), attempt.name, True
    recipes = [
        path
        for path in observe.files(experiment)
        if path.suffix in (".yml", ".yaml", ".toml") and observe.regular_file(path)
    ]
    return (_recipe(recipes[0]) if len(recipes) == 1 else {}), None, len(recipes) <= 1


def _inference_role(recipe: dict) -> str | None:
    task = task_from_config(recipe)
    train = _mapping(recipe.get("train"))
    if (
        task == "autoencoder"
        and recipe.get("network_ae")
        and not recipe.get("network_g")
    ):
        role = "net_ae"
    elif (
        task in ("image_restoration", "recurrent_restoration")
        and recipe.get("network_g")
        and not recipe.get("network_ae")
    ):
        role = "net_g"
    else:
        return None
    decay = train.get("ema_decay", 0)
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        return None
    if decay > 0:
        return role + "_ema"
    # Legacy AE observations can precede validation in schedule-free eval
    # weights. A recipe alone cannot attest which representation was used.
    optimizer = _mapping(train.get("optim_ae")).get("type", "")
    if role == "net_ae" and "schedulefree" in str(optimizer).casefold().replace(
        "_", ""
    ):
        return None
    return role


def _artifact(
    experiment: Path, keys: tuple[str, ...], role: str, expected_id: str | None = None
) -> dict:
    from traiNNer.training_recovery import _complete_tensor_file

    paths = []
    for key in keys:
        try:
            paths.append((key, observe.artifact_path(experiment, key)))
        except FileNotFoundError:
            continue
    if not paths:
        return {"status": "missing", "reason": "numberedWeightsMissing"}
    if len(paths) != 1:
        return {"status": "unverified", "reason": "ambiguousWeightFormat"}
    key, path = paths[0]
    embedded_id = None
    try:
        before = path.stat()
        if path.suffix == ".safetensors":
            from safetensors import SafetensorError, safe_open

            try:
                with safe_open(str(path), framework="numpy", device="cpu") as weights:
                    valid = bool(weights.keys())
                    embedded_id = (weights.metadata() or {}).get(SAVE_ID_KEY)
            except SafetensorError:
                valid = False
        else:
            valid = _complete_tensor_file(path)
            if valid:
                embedded_id = pth_save_id(path)
        after = path.stat()
    except OSError:
        return {"status": "unverified", "reason": "weightsChanged"}
    except (ValueError, BadZipFile):
        return {"status": "invalid", "reason": "invalidWeightContainer"}
    version = artifact_version(after)
    if artifact_version(before) != version:
        return {"status": "unverified", "reason": "weightsChanged"}
    if not valid or (
        embedded_id is not None and SAVE_ID.fullmatch(embedded_id) is None
    ):
        return {"status": "invalid", "reason": "invalidWeightContainer"}
    if embedded_id != expected_id:
        reason = (
            "weightIdentityMismatch" if expected_id else "weightIdentityUnassociated"
        )
        return {"status": "unverified", "reason": reason}
    artifact = {"name": path.name, "path": key, "role": role, "version": version}
    if expected_id is not None:
        artifact["saveId"] = expected_id
    return {
        "status": "available",
        "evidence": "recordedSaveId" if expected_id is not None else "uniqueLoggedSave",
        "artifact": artifact,
    }


def _recorded_checkpoint(experiment: Path, step: int, receipt: _Receipt) -> dict:
    role, key, identity = receipt.role, receipt.path, receipt.identity
    if identity is None:
        return {
            "status": "unverified",
            "reason": receipt.reason or "numberedSaveUnmatched",
        }
    if (
        step <= 0
        or role not in ROLES
        or not isinstance(identity, str)
        or SAVE_ID.fullmatch(identity) is None
        or not isinstance(key, str)
    ):
        return {"status": "unverified", "reason": "invalidWeightAssociation"}
    path = Path(key)
    folders = (
        ("models",) if role.endswith("_ema") else ("models", "models/resume_models")
    )
    if (
        path.parent.as_posix() not in folders
        or path.stem != f"{role}_{step}"
        or path.suffix not in _FORMATS
    ):
        return {"status": "unverified", "reason": "invalidWeightAssociation"}
    return _artifact(experiment, (key,), role, identity)


def _choose_direction(
    evidence: set[tuple[str, str]], explicit: str | None
) -> tuple[str | None, str]:
    known = {direction for direction, _ in evidence}
    if len(known) > 1:
        return None, "conflicting"
    if known:
        direction = next(iter(known))
        if explicit is not None and explicit != direction:
            raise ValueError(f"The recorded metric direction is {direction}.")
        sources = {source for _, source in evidence}
        source = next(
            name for name in ("stored", "evaluator", "recipe") if name in sources
        )
        return direction, source
    return (explicit, "explicit") if explicit is not None else (None, "unknown")


def _ranking(
    points: list[_Point], checkpoints: list[dict], direction: str | None, limit: int
) -> dict:
    sortable = [
        (point, checkpoint)
        for point, checkpoint in zip(points, checkpoints, strict=True)
        if point.value is not None and not math.isnan(point.value)
    ]
    if direction is None:
        ordered = sorted(
            zip(points, checkpoints, strict=True), key=lambda row: row[0].step
        )
    else:
        ascending = direction == "lower"
        sortable.sort(
            key=lambda row: (row[0].value if ascending else -row[0].value, row[0].step)
        )
        invalid = [
            (p, c)
            for p, c in zip(points, checkpoints, strict=True)
            if p.value is None or math.isnan(p.value)
        ]
        ordered = sortable + invalid

    def public(row: tuple[_Point, dict]) -> dict:
        point, checkpoint = row
        return {**point.public(), "checkpoint": checkpoint}

    def best(rows: list[tuple[_Point, dict]]) -> dict | None:
        if not rows or direction is None:
            return None
        return {
            **public(rows[0]),
            "tieCount": sum(row[0].value == rows[0][0].value for row in rows),
        }

    return {
        "points": [public(row) for row in ordered[:limit]],
        "observedCount": len(points),
        "rankableCount": len(sortable) if direction is not None else 0,
        "availableCount": sum(
            checkpoint["status"] == "available" for checkpoint in checkpoints
        ),
        "bestObserved": best(sortable),
        "bestAvailable": best(
            [row for row in sortable if row[1]["status"] == "available"]
        ),
    }


def _log_paths(experiment: Path) -> list[Path]:
    return sorted(
        path
        for path in observe.files(experiment)
        if path.name.startswith("train_")
        and path.suffix == ".log"
        and observe.regular_file(path)
    )


def _log_history(
    paths: list[Path],
    recipe: dict,
    selected: tuple[str, str] | None,
    *,
    proof_only: bool = False,
) -> list[_Log]:
    remaining = LOG_BYTE_LIMIT
    logs = []
    for path in paths:
        if not remaining:
            break
        log = _scan_log(path, recipe, remaining, selected, proof_only)
        logs.append(log)
        remaining -= log.bytes_read
    return logs


def _coverage(paths: list[Path] | None, logs: list[_Log], source: str) -> dict:
    unread = None if paths is None else len(paths) - sum(log.complete for log in logs)
    if paths is None:
        coverage = "notSearched"
    elif not paths:
        coverage = "unavailable"
    else:
        coverage = "partial" if unread else "complete"
    return {
        "source": source,
        "logCoverage": coverage,
        "logPurpose": "saveEvidence"
        if source in ("summary", "sampleMean")
        else "observations",
        "logBytesRead": sum(log.bytes_read for log in logs),
        "logByteLimit": LOG_BYTE_LIMIT,
        "unindexedLogs": unread,
    }


def retained_validation_weights(experiment: Path, attempt: str, dataset: str, metric: str) -> tuple | None:
    from safetensors import SafetensorError, safe_open

    key = "models/" + best_weight_filename(attempt, dataset, metric)
    try:
        path = observe.artifact_path(experiment, key)
        version = artifact_version(path.stat())
        with safe_open(str(path), framework="numpy", device="cpu") as file:
            receipt = json.loads((file.metadata() or {}).get("trainner_best_validation", "null"))
            populated = bool(file.keys())
        if not populated or artifact_version(path.stat()) != version or not isinstance(receipt, dict):
            return None
    except (OSError, ValueError, SafetensorError):
        return None
    if (receipt.get("version") != 1 or receipt.get("attempt") != attempt or receipt.get("dataset") != dataset
            or receipt.get("metric") != metric or receipt.get("role") not in ROLES
            or receipt.get("direction") not in _DIRECTIONS or type(receipt.get("step")) is not int
            or type(receipt.get("value")) not in (int, float) or not math.isfinite(receipt["value"])):
        return None
    return receipt, {"status": "available", "evidence": "retainedValidationWeights",
                     "artifact": {"name": path.name, "path": key, "role": receipt["role"], "version": version}}


def _checkpoints(
    experiment: Path,
    points: list[_Point],
    scan: _Log | None,
    recipe: dict,
    role: str | None,
) -> list[dict]:
    configured_format = _mapping(recipe.get("logger")).get("save_checkpoint_format")
    extension = (
        "." + configured_format if configured_format in ("pth", "safetensors") else None
    )
    family = {"AEModel"} if role and role.startswith("net_ae") else _GENERATORS
    repeats = Counter(point.step for point in points)
    logged = (
        Counter(point.step for point in scan.points) if scan is not None else Counter()
    )
    checkpoints = []
    winners = {}
    for point in points:
        step = point.step
        if point.source == "summary" and point.attempt:
            lookup = (point.attempt, point.dataset, point.metric)
            if lookup not in winners:
                winners[lookup] = retained_validation_weights(experiment, *lookup)
            winner = winners[lookup]
            if (winner is not None and winner[0]["step"] == step and winner[0]["value"] == point.value
                    and (point.receipt is None or point.receipt.role == winner[0]["role"])):
                checkpoints.append(winner[1])
                continue
        if point.receipt is not None:
            result = _recorded_checkpoint(experiment, step, point.receipt)
        elif (
            repeats[step] > 1
            or logged[step] > 1
            or (scan is not None and scan.saves[step] > 1)
        ):
            result = {"status": "unverified", "reason": "repeatedStep"}
        elif scan is not None and not scan.complete:
            result = {"status": "unverified", "reason": "saveHistoryUnindexed"}
        elif scan is None or not scan.fresh or not scan.ended:
            result = {"status": "unverified", "reason": "noSingleCompletedTrajectory"}
        elif role is None or scan.model not in family:
            result = {"status": "unverified", "reason": "inferenceWeightsUnknown"}
        elif step in scan.failures:
            result = {"status": "unverified", "reason": "saveFailed"}
        elif step <= 0 or scan.saves[step] != 1:
            result = {"status": "unverified", "reason": "numberedSaveUnrecorded"}
        else:
            suffixes = (extension,) if extension is not None else _FORMATS
            keys = tuple(f"models/{role}_{step}{suffix}" for suffix in suffixes)
            result = _artifact(experiment, keys, role)
        checkpoints.append(result)
    return checkpoints


def _catalog(runs: RunManager, run_id: str, experiment: Path) -> dict[str, Any]:
    groups = _indexed_catalog(experiment)
    paths = None
    logs: list[_Log] = []
    if groups:
        source = "indexed"
    else:
        recipe, _, single = _context(runs, run_id, experiment)
        paths = _log_paths(experiment)
        logs = _log_history(paths, recipe if single and len(paths) == 1 else {}, None)
        for log in logs:
            for key in log.unplaced:
                groups[key]
            for point in log.points:
                groups[point.dataset, point.metric]
        if single and len(logs) == 1 and logs[0].fresh:
            for key, evidence in groups.items():
                if declared := _recipe_direction(recipe, key[1]):
                    evidence.add((declared, "recipe"))
        source = "log" if groups else "none"

    series = []
    for (dataset, metric), evidence in sorted(groups.items()):
        direction, source_of_direction = _choose_direction(evidence, None)
        series.append(
            {
                "dataset": dataset,
                "metric": metric,
                "direction": direction,
                "directionSource": source_of_direction,
            }
        )
    return {
        "runId": run_id,
        "series": series,
        "history": _coverage(paths, logs, source),
    }


def select_checkpoints(
    runs: RunManager,
    run_id: str,
    dataset: str | None = None,
    metric: str | None = None,
    direction: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List validation series or rank the selected pair's recorded observations."""
    if (dataset is None) != (metric is None) or any(
        name is not None and (not isinstance(name, str) or not name.strip())
        for name in (dataset, metric)
    ):
        raise ValueError("Select both a validation dataset and metric.")
    if direction is not None and (dataset is None or direction not in _DIRECTIONS):
        raise ValueError(
            "Direction requires a selected metric and must be higher or lower."
        )
    if type(limit) is not int or not 1 <= limit <= 250:
        raise ValueError("Use a checkpoint selection limit from 1 to 250.")

    experiment = runs.experiment_dir(run_id)
    if dataset is None or metric is None:
        return _catalog(runs, run_id, experiment)

    points = _summaries(experiment, dataset, metric)
    source = "summary"
    if not points:
        points = _sample_means(experiment, dataset, metric)
        source = "sampleMean" if points else "log"

    needs_legacy = source != "summary" or any(point.receipt is None for point in points)
    recipe, attempt, single = (
        _context(runs, run_id, experiment) if needs_legacy else ({}, None, False)
    )
    paths = _log_paths(experiment) if needs_legacy else None
    single = single and paths is not None and len(paths) <= 1
    if single and points:
        attempts = {
            row[0]
            for row in _query(
                experiment / _SUMMARY_FILE,
                "SELECT DISTINCT attempt FROM evaluation LIMIT 2",
            )
        }
        single = len(attempts) <= 1 and (attempt is None or not attempts - {attempt})
    role = _inference_role(recipe) if single else None
    logs: list[_Log] = []
    if paths is not None and (source == "log" or (single and role is not None)):
        logs = _log_history(
            paths,
            recipe if single else {},
            (dataset, metric),
            proof_only=source != "log",
        )
    scan = logs[0] if single and len(logs) == 1 else None
    if source == "log":
        points = [point for log in logs for point in log.points]

    evidence = {
        (point.direction, point.direction_source)
        for point in points
        if point.direction is not None and point.direction_source is not None
    }
    if single and (scan is None or scan.fresh):
        if declared := _recipe_direction(recipe, metric):
            if not evidence:
                evidence.add((declared, "recipe"))
        if source != "sampleMean" and (
            not evidence or any(point.direction is None for point in points)
        ):
            stored = _sample_directions(experiment, dataset, metric)
            evidence.update(
                (stored[point.step], "stored")
                for point in points
                if point.step in stored
            )
        for point in points:
            if point.attempt is None:
                point.attempt = attempt

    resolved, origin = _choose_direction(evidence, direction)
    checkpoints = _checkpoints(experiment, points, scan, recipe, role)
    return {
        "runId": run_id,
        "dataset": dataset,
        "metric": metric,
        "direction": resolved,
        "directionSource": origin,
        "unplacedLogCount": sum(log.unplaced[dataset, metric] for log in logs),
        "history": _coverage(paths, logs, source if points else "none"),
        **_ranking(points, checkpoints, resolved, limit),
    }

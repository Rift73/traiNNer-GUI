"""Read-only capacity estimates for files in the trainer's checkpoint directories."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import tomllib
from collections import Counter
from contextlib import closing
from heapq import nlargest
from pathlib import Path

from . import run_observe as observe
from .paths import artifact_version
from .recipes import safe_name
from .run_checkpoints import NUMBERED_STATE, _checkpoint_paths
from .runs import RUN_ID, RunManager
from .yaml_edit import read_recipe

_WEIGHT = re.compile(r"net_(?:g|ae|d)(?:_ema)?_([1-9][0-9]{0,18}|latest)\.(?:pth|safetensors)")
_BEST_WEIGHT = re.compile(r"best-[A-Za-z0-9_-]{0,54}-[0-9a-f]{24}\.safetensors")
MAX_CANDIDATES = 16
MAX_SAMPLES = 3


def _integer(value, minimum=0):
    return value if type(value) is int and value >= minimum else None


def _recipe(root, attempt, notes):
    try:
        if attempt:
            path = attempt / "options.yml"
        else:
            choices = [p for p in observe.files(root) if p.suffix.lower() in (".yaml", ".yml", ".toml")
                       and observe.regular_file(p)]
            path = choices[0] if len(choices) == 1 else None
        if path and observe.regular_file(path) and (text := observe.recipe_text(path)):
            if path.suffix.lower() == ".toml":
                return tomllib.loads(text), "neosrToml"
            return read_recipe(text), "managedAttempt" if attempt else "experimentYaml"
        notes.append("A single readable training recipe was unavailable.")
    except (OSError, ValueError) as error:
        notes.append(f"Could not read the training recipe: {str(error)[:200]}")
    return {}, None


def _policy(config, source, notes):
    logger = config.get("logger")
    logger = logger if isinstance(logger, dict) else {}
    train = config.get("train")
    train = train if isinstance(train, dict) else {}
    policy = {"source": source, "saveEvery": _integer(logger.get("save_checkpoint_freq"), 1),
              "targetStep": _integer(train.get("total_iter"), 1), "keepRegularSets": None,
              "retention": "unknown", "format": None}
    if source == "neosrToml":
        policy.update(targetStep=_integer(logger.get("total_iter"), 1), retention="separateNeoSRPolicies",
                      keepExports=_integer(logger.get("keep_exports")), keepStates=_integer(logger.get("keep_states")))
        notes.append("NeoSR retains numbered exports per network separately from numbered states. Native set projections are unavailable.")
    elif source:
        keep = _integer(logger.get("save_checkpoint_keep", 0))
        fmt = logger.get("save_checkpoint_format", "safetensors")
        policy.update(keepRegularSets=keep, format=fmt if fmt in ("safetensors", "pth") else None)
        if keep == 0:
            policy["retention"] = "disabled"
        elif keep and config.get("network_g"):
            policy["retention"] = "completeRegularSets"
        elif keep and config.get("network_ae"):
            policy["retention"] = "unsupported"
            notes.append("This trainer rejects positive save_checkpoint_keep for AE. No retention savings are assumed.")
    return policy


def _inventory(root, notes):
    files, volumes = {}, {}
    totals = Counter()
    complete = True
    for folder, suffixes in observe.ARTIFACT_FOLDERS.items():
        path = root / folder
        try:
            if path.resolve() != path:
                raise ValueError("directory is redirected")
            existing = path
            while not existing.exists():
                existing = existing.parent
            dev = existing.stat().st_dev
            if dev not in volumes:
                fs = os.statvfs(existing)
                volumes[dev] = {"directories": [], "availableBytes": fs.f_bavail * fs.f_frsize}
            volumes[dev]["directories"].append(folder)
            if existing != path:
                continue
            with os.scandir(path) as entries:
                for entry in entries:
                    name = entry.name
                    tmp = name.startswith(".") and name.endswith(".tmp") and any(
                        suffix + "." in name for suffix in suffixes)
                    if not tmp and (name.startswith(".") or Path(name).suffix not in suffixes):
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        complete = False
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        complete = False
                        continue
                    key = f"{folder}/{name}"
                    files[key] = info
                    weight = _WEIGHT.fullmatch(name)
                    if tmp:
                        category = "temporary"
                    elif weight and weight[1] == "latest":
                        category = "latest"
                    elif weight or NUMBERED_STATE.fullmatch(name):
                        category = "numbered"
                    elif folder == "models" and _BEST_WEIGHT.fullmatch(name):
                        category = "bestValidation"
                    else:
                        category = "other"
                    totals[category] += info.st_size
        except (OSError, ValueError, RuntimeError) as error:
            notes.append(f"Could not inspect {folder}: {error}")
            complete = False
    if not complete:
        notes.append("Some entries changed or could not be inspected. Inventory totals cover the files observed.")
    inventory = {"bytes": sum(totals.values()), "fileCount": len(files), "complete": complete,
                 **{key + "Bytes": totals[key] for key in ("numbered", "latest", "bestValidation", "other", "temporary")}}
    measured_folders = sum(len(v["directories"]) for v in volumes.values())
    unique_volume = next(iter(volumes.values())) if len(volumes) == 1 and measured_folders == len(observe.ARTIFACT_FOLDERS) else None
    filesystem = {"availableBytes": unique_volume["availableBytes"] if unique_volume else None,
                  "volumes": list(volumes.values())}
    return files, inventory, filesystem


def _samples(root, config, policy, files, notes):
    states = {}
    weight_counts = Counter()
    for key, info in files.items():
        name = Path(key).name
        if key.startswith("training_states/") and (state := NUMBERED_STATE.fullmatch(name)):
            states[int(state[1])] = info.st_mtime_ns
        elif (weight := _WEIGHT.fullmatch(name)) and weight[1] != "latest":
            weight_counts[int(weight[1])] += 1
    recent = []
    checked = 0
    if policy["format"] and isinstance(config.get("train"), dict) and (
        config.get("network_g") or config.get("network_ae")
    ):
        # Input-only resume exclusions have no effect on the files the trainer saves.
        layout = {**config, "path": {}}
        for step in nlargest(MAX_CANDIDATES, states, key=lambda s: (states[s], s)):
            checked += 1
            try:
                paths = _checkpoint_paths(root, root / "training_states" / f"{step}.state", layout)
                if weight_counts[step] != len(paths) - 1:
                    raise ValueError("Extra numbered weights make this save's role or format history ambiguous.")
                parts = []
                for role, path in paths.items():
                    key = path.relative_to(root).as_posix()
                    if role != "resume_state" and path.suffix != "." + policy["format"]:
                        raise ValueError("Weight format differs from the recorded recipe.")
                    before = files.get(key)
                    if before is None or artifact_version(before) != artifact_version(path.lstat()):
                        raise ValueError("File changed during measurement.")
                    parts.append({"role": role, "path": key, "bytes": before.st_size})
                recent.append({"step": step, "bytes": sum(part["bytes"] for part in parts),
                               "onCadence": step % policy["saveEvery"] == 0 if policy["saveEvery"] else None,
                               "files": parts})
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                if len(notes) < 6:
                    notes.append(f"Excluded step {step}: {error}")
            if len(recent) == MAX_SAMPLES:
                break
    if not recent:
        notes.append("No recent complete set matching the recorded model roles and weight format was measured.")
    return recent, len(states), checked


def _receipts(root, samples, files, notes):
    """Corroborate individual sampled weights with bounded, existing validation-save receipts."""
    wanted = {(s["step"], f["path"]): f for s in samples for f in s["files"] if f["role"] != "resume_state"}
    if not wanted:
        return
    dbfile = root / "validation-summary.sqlite3"
    if not observe.regular_file(dbfile):
        return
    from .checkpoint_selection import _Receipt, _recorded_checkpoint

    # Immutable mode cannot create WAL/SHM files in a read-only archive.
    # Only consume a stable main database with no outstanding journal.
    def stable_version():
        if any(Path(str(dbfile) + suffix).exists() for suffix in ("-wal", "-journal")):
            return None
        return artifact_version(dbfile.stat())

    try:
        version = stable_version()
        if version is None:
            notes.append("Validation save receipts were skipped while a database journal was present.")
            return
        with closing(sqlite3.connect(dbfile.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=0)) as db:
            db.execute("PRAGMA trusted_schema=OFF")
            if db.execute("PRAGMA user_version").fetchone()[0] != 1 or not db.execute(
                "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='validation_checkpoint'"
            ).fetchone():
                return
            # Keep work independent of the full validation history. Multiple
            # attempts can reuse a step/path; match the on-disk save identity.
            rows = db.execute("""
                SELECT e.step,e.attempt,c.role,c.weight_path,c.save_id
                FROM (SELECT id,step,attempt FROM evaluation ORDER BY id DESC LIMIT 128) e
                JOIN validation_checkpoint c ON c.evaluation=e.id WHERE c.save_id IS NOT NULL
            """)
            records = rows.fetchall()
        if version != stable_version():
            notes.append("The validation receipt database changed during the observation.")
            return
        seen = set()
        for step, attempt, role, key, identity in records:
            file = wanted.get((step, key))
            if file is None or "validationSave" in file or (key, identity) in seen:
                continue
            seen.add((key, identity))
            match = _recorded_checkpoint(root, step, _Receipt(role, key, identity, None))
            if match["status"] == "available" and match["artifact"]["version"] == artifact_version(files[key]):
                file["validationSave"] = {"attempt": attempt, "saveId": identity}
    except (OSError, ValueError, sqlite3.Error) as error:
        notes.append(f"Validation save receipts unavailable: {error}")


def _projection(policy, samples, monitor, managed):
    estimate = max((s["bytes"] for s in samples), default=None)
    largest = max((f["bytes"] for s in samples for f in s["files"]), default=None)
    target, every, keep = policy["targetStep"], policy["saveEvery"], policy["keepRegularSets"]
    step = _integer(monitor.get("completedIteration"))
    basis = "completionReceipt" if step is not None else None
    if step is None and monitor.get("status") not in ("stopped", "failed"):
        step = _integer(monitor.get("iteration"))
        basis = "reportedIteration" if step is not None else None
    if step is None and samples:
        step, basis = samples[0]["step"], "mostRecentlyWrittenCompleteSet"
    regular = final = None
    completed = managed and monitor.get("status") == "completed"
    if completed:
        regular = final = 0
    elif every and target and step is not None and step <= target and policy["source"] != "neosrToml":
        regular = target // every - step // every
        final = int(managed and step < target and target % every != 0)
    count = regular + final if regular is not None else None
    retained = regular
    if regular and policy["retention"] == "completeRegularSets":
        retained = min(keep, regular)
    elif regular and policy["retention"] != "disabled":
        retained = None

    def bytes_for(n):
        if n == 0:
            return 0
        if n is None or estimate is None:
            return None
        return n * estimate

    peak = None
    if retained is not None:
        # A new complete set lands before an older regular set can be pruned.
        before_pruning = min(regular, keep + 1) if policy["retention"] == "completeRegularSets" else regular
        peak = max(before_pruning, retained + final)
    return {"sampleSteps": [s["step"] for s in samples], "numberedSaveBytes": estimate,
            "atomicTemporaryFileBytes": largest, "newNumberedSaveHeadroomBytes": estimate,
            "fromStep": step, "fromStepSource": basis, "runStatus": monitor.get("status"),
            "scenario": "completedAttempt" if completed else "ifContinuedToTarget",
            "remainingRegularSaves": regular, "additionalFinalNumberedSaves": final,
            "remainingNumberedWriteBytes": bytes_for(count), "peakAdditionalNumberedBytes": bytes_for(peak),
            "futureRetainedNumberedBytes": bytes_for(retained + final if retained is not None else None),
            "regularRetentionWindowBytes": bytes_for(keep) if policy["retention"] == "completeRegularSets" else None}


def checkpoint_storage(runs: RunManager, run_id: str) -> dict:
    managed = bool(RUN_ID.fullmatch(run_id))
    attempt = runs.attempt_dir(run_id) if managed else None
    notes = []
    try:
        monitor = runs.monitor(run_id)
    except PermissionError:
        monitor = {}
        notes.append("The run progress record was unreadable.")
    try:
        root = runs.experiment_dir(run_id)
    except FileNotFoundError:
        if not attempt or "name" not in monitor:
            raise
        root = attempt.parent.parent / "experiments" / safe_name(monitor["name"])
        if root.exists() or root.resolve() != root:
            raise
    config, source = _recipe(root, attempt, notes)
    policy = _policy(config, source, notes)
    files, observed, filesystem = _inventory(root, notes)
    samples, count, checked = _samples(root, config, policy, files, notes)
    _receipts(root, samples, files, notes)
    observed.update(recentCompleteSets=samples, numberedStateFiles=count, candidatesChecked=checked)
    projection = _projection(policy, samples, monitor, managed)
    available, headroom = filesystem["availableBytes"], projection["newNumberedSaveHeadroomBytes"]
    filesystem["wouldFitOneNewSet"] = available >= headroom if available is not None and headroom is not None else None
    peak = projection["peakAdditionalNumberedBytes"]
    filesystem["wouldFitProjectedNumberedPeak"] = available >= peak if available is not None and peak is not None else None
    projection["assumptions"] = [
        "Uses the largest of up to three recent complete numbered sets. Future optimizer state and serialization sizes may change.",
        "Assumes progress to the configured target and new numbered paths. Manual stops, early stops, retries and latest exports can change disk use.",
        "Best-validation exports are included in observed disk use. The numbered-set estimate excludes these optional files and their replacement writes.",
        "Retention assumes successful pruning of eligible future regular sets. No credit is taken for deleting files already on disk.",
        "Fit compares logical bytes with currently available filesystem bytes. Allocation overhead, host capacity and competing writers can affect the outcome.",
    ]
    return {"runId": run_id, "policy": policy, "observed": observed, "projection": projection,
            "filesystem": filesystem, "limitations": notes}

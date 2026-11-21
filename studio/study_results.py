"""Describe paired *training seed* outcomes on one fixed scored population."""
from __future__ import annotations

import hashlib
import json
import math
from statistics import fmean, median, stdev


def population_key(protocol: dict, members) -> str:
    """Order and duplicates are supplied by the evaluator for the task in question."""
    digest = hashlib.sha256()
    digest.update(json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for member in members:
        digest.update(b"\n")
        digest.update(json.dumps(tuple(member), separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


def _describe(values: list[float]) -> dict:
    return {"mean": fmean(values), "median": median(values), "min": min(values), "max": max(values),
            "sampleSd": stdev(values) if len(values) >= 2 else None}


def summarize(pairs: list[dict]) -> dict:
    """Separate incompatible protocols/steps; a validation image is never a training replicate."""
    groups: dict[tuple, list] = {}
    excluded = []
    for row in pairs:
        result = row.get("comparison") or {}
        population = result.get("populationId")
        if row["status"] != "compared" or not population:
            reason = row["status"] if row["status"] != "compared" else "partialPopulation"
            excluded.append({"seed": row["seed"], "reason": reason})
            continue
        if "pooled" in result:
            left, right = (result["pooled"][k] for k in ("left", "right"))
        else:
            left, right = (result["summary"][k] for k in ("meanLeft", "meanRight"))
        delta = right - left
        if not all(map(math.isfinite, (left, right, delta))):
            excluded.append({"seed": row["seed"], "reason": "nonfinite"})
            continue
        group = (result["leftStep"], result["rightStep"], population, result["better"],
                 result.get("aggregation", "meanPerSample"))
        groups.setdefault(group, []).append((row["seed"], left, right, delta))
    cohorts = []
    for (left_step, right_step, population, better, aggregation), observations in groups.items():
        seeds, left, right, deltas = (list(values) for values in zip(*observations, strict=True))
        sign = 1 if better == "higher" else -1
        cohorts.append({"populationId": population, "leftStep": left_step, "rightStep": right_step,
                        "aggregation": aggregation, "better": better, "seeds": seeds, "n":len(seeds),
                        "left": _describe(left), "right": _describe(right), "delta": _describe(deltas),
                        "improved": sum(sign * d > 0 for d in deltas), "tied": deltas.count(0),
                        "regressed": sum(sign * d < 0 for d in deltas)})
    return {"unit": "matchedTrainingSeed", "cohorts": cohorts, "excluded": excluded}

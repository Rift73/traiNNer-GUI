"""Read a numbered trainer state in an isolated CPU process without reading tensor data."""
from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path


def _scheduler_clock(saved: object) -> dict:
    """Keep only scheduler metadata needed to check hard limits."""
    if not isinstance(saved, dict):
        return {}
    clock: dict = {
        key: value
        for key in ("last_epoch", "total_steps")
        if type(value := saved.get(key)) is int
    }
    periods = saved.get("cumulative_period")
    if isinstance(periods, list) and periods and all(
        type(period) is int or (type(period) is float and math.isfinite(period))
        for period in periods
    ):
        # The cosine scheduler accepts any clock at or below a cumulative period.
        clock["period_limit"] = math.floor(max(periods))
    milestones = saved.get("_milestones")
    if isinstance(milestones, list) and all(type(step) is int for step in milestones):
        clock["milestones"] = milestones
    children = saved.get("_schedulers")
    if isinstance(children, list):
        clock["children"] = [_scheduler_clock(child) for child in children]
    return clock


def describe(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        records = [info for info in archive.infolist() if info.filename.endswith("/data.pkl")]
        if len(records) != 1:
            raise ValueError("The checkpoint does not have one Torch metadata record.")
        if records[0].file_size > 32 * 1024 * 1024:
            raise RuntimeError("Detailed metadata exceeds the interactive inspection budget.")
    import torch

    state = torch.load(path, weights_only=True, map_location="meta", mmap=True)
    if not isinstance(state, dict) or not all(isinstance(state.get(key), list) for key in ("schedulers", "optimizers")):
        raise ValueError("The file does not contain a full trainer state.")
    if type(state.get("iter")) is not int or state["iter"] != int(path.stem):
        raise ValueError("The saved iteration differs from the numbered training state filename.")
    identities = state.get("optimizer_identities")
    optimizers = []
    if identities is not None:
        if not isinstance(identities, list) or len(identities) != len(state["optimizers"]):
            raise ValueError("The saved optimizer identities disagree with the optimizer count.")
        for item in identities:
            roles = {name.partition(".")[0] for group in item["groups"] for name, _shape in group}
            optimizers.append({"type": item["type"], "roles": sorted(roles)})
    cursor = state.get("sampler_cursor") or {}
    streams = state.get("random_streams") or {}
    return {"iteration": state["iter"], "epoch": state.get("epoch"),
            "optimizerCount": len(state["optimizers"]), "optimizers": optimizers,
            "schedulerCount": len(state["schedulers"]), "schedulerSpecs": state.get("scheduler_specs"),
            "schedulerClocks": [_scheduler_clock(scheduler) for scheduler in state["schedulers"]],
            "geometry": cursor.get("geometry"), "nextBatch": cursor.get("next_batch"),
            "randomStreams": bool(streams), "seed":streams.get("seed"),
            "workerAugmentation":streams.get("worker_augmentation"), "pairedTraining": streams.get("paired_training"),
            "supervisedTraining":streams.get("supervised_training"),
            "recipe": state.get("resume_recipe")}


if __name__ == "__main__":
    try:
        result = describe(Path(sys.argv[1]))
        print(json.dumps(result, allow_nan=False))
    except (KeyError, TypeError, ValueError, AttributeError, OSError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

"""Report full-state resume compatibility from the user's selected checkpoint."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

from trainner_config.resume import recipe_learning_rate, resume_recipe
from trainner_config.schedules import COMPOSITES

from .paths import artifact_version
from .run_checkpoints import import_sources


_HARD_LIMITS = {
    "onecyclelr": "total_steps",
    "cosineannealingrestartlr": "period_limit",
}
_COMPOSITE_TYPES = {name.casefold() for name in COMPOSITES}
_probe_lock = threading.Lock()


@lru_cache(maxsize=32)
def _probe(path: Path, version: str) -> tuple[dict | None, str | None]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "studio.state_inspect", str(path)],
            cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=25,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    except subprocess.TimeoutExpired:
        return None, "The CPU metadata reader timed out. The trainer will check the state during startup."
    if artifact_version(path.stat()) != version:
        raise ValueError("The selected training state changed during inspection. Select the completed save.")
    if result.returncode == 2:
        raise ValueError(result.stderr.strip()[-500:] or "Invalid training state.")
    if result.returncode:
        return None, "Detailed state metadata could not be read. The trainer will check the state during startup."
    return json.loads(result.stdout), None


def state_metadata(state: Path):
    version = artifact_version(state.stat())
    # On an explicit state selection import Torch in one short-lived child.
    # Concurrent form edits share the result without importing Torch in the server.
    with _probe_lock:
        saved, warning = _probe(state, version)
    return saved, warning, version


def _has_hard_limit(spec: dict | None) -> bool:
    if spec is None:
        return False
    kind = spec["type"].casefold()
    return kind in _HARD_LIMITS or (
        kind in _COMPOSITE_TYPES and any(_has_hard_limit(child) for child in spec["schedulers"])
    )


def _extension_limit(
    spec: dict, clock: dict, updates: int, path: str, *, enters_phase: bool = False
) -> tuple[int, str] | None:
    """Return the first hard limit crossed, in updates relative to this scheduler."""
    kind = spec["type"].casefold()
    missing = f"Cannot verify {path}. Select a checkpoint with its saved scheduler clock and limits."
    if kind in _HARD_LIMITS:
        limit = clock.get(_HARD_LIMITS[kind])
        step = -1 if enters_phase else clock.get("last_epoch")
        if limit is None or step is None:
            raise ValueError(missing)
        remaining = max(0, limit - step)
        if updates > remaining:
            position = "phase starts at step 0" if enters_phase else f"saved step {step:,}"
            return remaining, f"{path} ({spec['type']}, {position}, limit {limit:,})"
        return None

    children = spec["schedulers"]
    child_clocks = clock.get("children")
    if child_clocks is None or len(child_clocks) != len(children):
        raise ValueError(missing)
    milestones = []
    parent_step = 0
    if kind == "sequentiallr":
        milestones = clock.get("milestones")
        parent_step = clock.get("last_epoch")
        if milestones is None or len(milestones) != len(children) - 1 or parent_step is None:
            raise ValueError(missing)

    limits = []
    for index, (child, child_clock) in enumerate(zip(children, child_clocks, strict=True)):
        if not _has_hard_limit(child):
            continue
        # Relative parent calls are one-based. A sequential milestone belongs to
        # the incoming child and sets its clock to zero. Chained children run every call.
        phase_start = milestones[index - 1] - parent_step if milestones and index else 0
        first = max(1, phase_start)
        last = min(updates, milestones[index] - parent_step - 1) if index < len(milestones) else updates
        if first <= last and (exceeded := _extension_limit(
            child, child_clock, last - first + 1, f"{path}.schedulers[{index}]",
            enters_phase=first == phase_start,
        )):
            remaining, scheduler = exceeded
            limits.append((first - 1 + remaining, scheduler))
    return min(limits, default=None)


def validate_extension_scheduler_horizons(
    train: dict, state: Path, additional_iterations: int
) -> None:
    """Validate a full-state extension against the original attempt's scheduler limits.

    The caller supplies the resolved numbered state and a positive update count.
    Each optimizer's scheduler can advance at most once per accepted update.
    """
    per_role = train.get("schedulers")
    shared = train.get("scheduler")
    declared = {None: shared} if per_role is None else per_role
    hard = {role: spec for role, spec in declared.items() if _has_hard_limit(spec)}
    if not hard:
        return

    saved, _, _ = state_metadata(state)
    missing = (
        f"Cannot verify hard LR scheduler limits in {state.name}. "
        "Select a complete checkpoint with matching scheduler metadata."
    )
    if saved is None:
        raise ValueError(missing)
    clocks = saved["schedulerClocks"]
    bindings = saved["schedulerSpecs"]
    if bindings is None and per_role is None:
        # Older checkpoints used one shared schedule for every optimizer.
        bindings = [{"target": index + 1, "type": shared["type"]} for index in range(len(clocks))]
    if not clocks or bindings is None or len(clocks) != len(bindings):
        raise ValueError(missing)
    by_target = {
        binding["target"]: (binding["type"], clock)
        for binding, clock in zip(bindings, clocks, strict=True)
    }
    if per_role is None:
        if len(by_target) != saved["optimizerCount"]:
            raise ValueError(missing)
        hard = dict.fromkeys(by_target, shared)

    limits = []
    for role, spec in hard.items():
        path = f"train.scheduler (optimizer {role})" if per_role is None else f"train.schedulers.{role}"
        saved_kind, clock = by_target.get(role, ("", {}))
        if saved_kind.casefold() != spec["type"].casefold():
            raise ValueError(f"{missing} No matching schedule for {path}.")
        if exceeded := _extension_limit(spec, clock, additional_iterations, path):
            limits.append(exceeded)
    if limits:
        remaining, scheduler = min(limits)
        remedy = (
            f"Request at most {remaining:,} additional updates with the saved schedule."
            if remaining else "Start a new run from weights to use a new schedule."
        )
        raise ValueError(
            f"{scheduler} can exceed its hard limit within {additional_iterations:,} additional updates "
            f"if its optimizer steps on every update. {remedy}"
        )


def inspect_resume(config: dict, opt, root: Path) -> tuple[dict, list[dict]]:
    paths = import_sources(config, root)
    state = paths["resume_state"]
    saved, warning, version = state_metadata(state)
    issues = []

    def issue(path: str, message: str, level: str = "error"):
        issues.append({"level": level, "path": path, "message": message, "source": "resume"})

    roles = [role for role in ("g", "ae", "d") if "pretrain_network_" + role in paths]
    summary = {"state": str(state), "iteration": int(state.stem), "version": version,
               "files": {name: str(p) for name, p in paths.items()},
               "metadataVerified": saved is not None}
    if saved is None:
        issue("path.resume_state", warning, "warning")
        return summary, issues
    summary.update(epoch=saved["epoch"], nextBatch=saved["nextBatch"],
                   optimizerCount=saved["optimizerCount"], schedulerCount=saved["schedulerCount"],
                   optimizerIdentities=bool(saved["optimizers"]), randomStreams=saved["randomStreams"],
                   replaySeed=saved["seed"])
    remedy = "Use the checkpoint recipe to resume, or start from network weights for a new optimizer."
    requested = resume_recipe(opt)
    optimizer_roles = [role for role in roles if role in requested["optimizers"]]
    if len(optimizer_roles) != saved["optimizerCount"]:
        issue("path.resume_state", f"The state has {saved['optimizerCount']} optimizers. This recipe uses {len(optimizer_roles)}. {remedy}")
    recorded = saved["recipe"]
    if recorded and recorded.get("version") == 1:
        for key in ("task", "scale"):
            if requested[key] != recorded[key]:
                issue(key, f"The checkpoint records {key}={recorded[key]!r}. {remedy}")
        for role in roles:
            key = "network_" + role
            old, new = recorded["networks"].get(role) or {}, requested["networks"].get(role) or {}
            if str(old.get("type")).casefold() != str(new.get("type")).casefold():
                issue(key + ".type", f"The saved architecture is {old.get('type')!r}. {remedy}")
            elif {k:v for k,v in old.items() if k != "type"} != {k:v for k,v in new.items() if k != "type"}:
                issue(key, "Architecture settings differ from the checkpoint. The trainer checks exact tensor shapes at load.", "warning")
            old, new = recorded["optimizers"].get(role) or {}, requested["optimizers"].get(role) or {}
            if str(old.get("type")).casefold() != str(new.get("type")).casefold():
                issue(f"train.optim_{role}.type", f"The saved optimizer is {old.get('type')!r}. {remedy}")
            else:
                options = sorted(
                    key for key in old.keys() | new.keys()
                    if key not in {"type", "lr"} and old.get(key) != new.get(key)
                )
                if options:
                    issue(
                        f"train.optim_{role}",
                        f"Optimizer options changed ({', '.join(options)}). Resume restores these options "
                        "from the saved parameter groups. Use weights-only initialization to apply them.",
                        "warning",
                    )
                old_rate = recipe_learning_rate(old.get("lr"))
                new_rate = recipe_learning_rate(new.get("lr"))
                if old_rate is not None and new_rate is not None and old_rate != new_rate:
                    issue(
                        f"train.optim_{role}.lr",
                        "Resume applies the edited base learning rate. Restored schedules keep their "
                        "position and use the new base rate.",
                        "warning",
                    )

    for role, identity in zip(optimizer_roles, saved["optimizers"]):
        actual_roles = identity["roles"]
        if actual_roles and actual_roles != ["net_" + role]:
            issue("path.resume_state", f"Saved parameters belong to {actual_roles}, but this optimizer targets net_{role}. {remedy}")
        # Aliased external optimizer registrations are resolved only by the trainer.
        canonical = {"adam": "torch.optim.adam.Adam", "adamw": "torch.optim.adamw.AdamW",
                     "nadam": "torch.optim.nadam.NAdam"}
        expected = canonical.get(str((requested["optimizers"].get(role) or {}).get("type", "")).lower())
        if expected and identity["type"] != expected:
            issue(f"train.optim_{role}.type", f"The state contains {identity['type']}. This recipe selects {expected}. {remedy}")
    if not saved["optimizers"] and saved["optimizerCount"]:
        issue("path.resume_state", "This legacy state has no named optimizer identities. The trainer matches groups by position.", "warning")
    train = opt.train
    if not train.restart_scheduler:
        expected = [(role, train.scheduler if train.schedulers is None else train.schedulers.get(role)) for role in optimizer_roles]
        expected = [(role, spec) for role, spec in expected if spec is not None]
        if len(expected) != saved["schedulerCount"]:
            issue("train.scheduler", "The number of active schedules differs from the checkpoint. Restore its schedules or explicitly restart them.")
        elif saved["schedulerSpecs"] is not None:
            for (role, spec), original in zip(expected, saved["schedulerSpecs"]):
                if (role, spec["type"].casefold(), spec.get("monitor")) != (original.get("target"), original.get("type"), original.get("monitor")):
                    issue("train.schedulers", "The saved scheduler target, class or monitored metric differs. Restore the original or explicitly restart the schedulers.")
                    break
        elif train.schedulers is not None and expected:
            issue("train.schedulers", "This legacy state has no scheduler target information. Use the shared schedule or explicitly restart the schedulers.")
    if opt.checkpoint_rng and (not saved["randomStreams"] or saved["geometry"] is None):
        issue("checkpoint_rng", "Exact replay needs saved random streams and the sampler cursor. This state is missing one or both.")
    elif opt.checkpoint_rng and opt.manual_seed is not None and opt.manual_seed != saved["seed"]:
        issue("manual_seed", f"Exact replay requires the checkpoint's manual_seed of {saved['seed']}.")
    elif opt.checkpoint_rng:
        from trainner_config.devices import replay_worker_mode
        from trainner_data.supervised_identity import (
            supervised_identity_error, supervised_replay_settings, uses_supervised_replay_identity,
        )

        if saved["workerAugmentation"] != replay_worker_mode(opt):
            issue("datasets.train.num_worker_per_gpu", "The saved replay mode and this recipe use different augmentation streams. Keep the original sequential or position-seeded worker mode.")
        elif uses_supervised_replay_identity(opt):
            identity = saved.get("supervisedTraining")
            if error := supervised_identity_error(identity):
                issue("checkpoint_rng", error)
            elif identity.get("settings") != supervised_replay_settings(opt):
                issue("datasets.train", "Restore the original supervised dataset settings for exact replay.")
            else:
                summary["supervisedTraining"] = {"samples": identity["count"], "selectionCheck": "atTrainerStartup"}
        elif saved.get("supervisedTraining") is not None:
            issue("datasets.train.type", "Restore the original supervised task and training dataset for exact replay.")
    ds = opt.datasets.get("train")
    if opt.checkpoint_rng and ds is not None and saved["randomStreams"]:
        from trainner_data.paired_identity import PairedTrainingIdentityError, paired_replay_settings

        identity = saved.get("pairedTraining")
        if ds.type.casefold() == "pairedimagedataset":
            if not isinstance(identity, dict) or identity.get("version") != 1:
                issue("checkpoint_rng", "This state lacks a supported paired training-data identity. Turn off checkpoint_rng to resume with fresh streams.")
            else:
                try:
                    expected = paired_replay_settings(ds, scale=opt.scale)
                except PairedTrainingIdentityError as exc:
                    issue(exc.path, str(exc))
                else:
                    recorded_settings = identity.get("settings") or {}
                    for key, value in expected.items():
                        if recorded_settings.get(key) != value:
                            issue("scale" if key == "scale" else f"datasets.train.{key}",
                                  f"Paired replay records {key}={recorded_settings.get(key)!r}. This recipe uses {value!r}.")
                    summary["pairedTraining"] = {"pairs": identity.get("pairs"), "storage": identity.get("storage"),
                                                 "selectionCheck": "atTrainerStartup"}
        elif identity is not None:
            issue("datasets.train.type", "This replay state requires the original paired image dataset.")
    if ds and (geometry := saved["geometry"]):
        if geometry.get("accum_iters") != ds.accum_iter:
            issue(
                "datasets.train",
                f"Sampler cursor records accum_iters={geometry.get('accum_iters')!r}. "
                f"This recipe uses {ds.accum_iter}. Keep the original accumulation for full-state resume.",
            )
        if type(ds.batch_size_per_gpu) is int and isinstance(opt.num_gpu, int):
            batch = ds.batch_size_per_gpu * (1 if opt.dist else max(1, opt.num_gpu))
            saved_batch = geometry.get("batch_size")
            if type(saved_batch) is not int or saved_batch < 1:
                issue("datasets.train", "The sampler cursor has an invalid saved batch size.")
            elif saved_batch != batch:
                issue(
                    "datasets.train.batch_size_per_gpu",
                    f"Resume changes the batch size from {saved_batch} to {batch} samples per rank. "
                    "This changes how samples are grouped into updates.",
                    "warning",
                )
    return summary, issues

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

from .paths import artifact_version
from .run_checkpoints import import_sources


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

"""Controlled studies backed by the same durable, held training queue."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from . import run_process as process
from .queue import QueueManager
from .recipes import safe_name
from .yaml_edit import MAX_RECIPE_BYTES, read_recipe, render

MAX_STUDY_ENTRIES = 128
MAX_SEED = 2**32 - 1


class StudyRejected(ValueError):
    """One or more proposed cells failed validation before publication."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        super().__init__("The study recipes failed validation.")
        self.issues = issues


def _label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 120
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("Provide a study or arm name of 1 to 120 characters.")
    return value.strip()


def _study_id(request_id: str | None) -> str:
    if request_id is None:
        return uuid4().hex
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or len(request_id) > 128
    ):
        raise ValueError("Provide a request ID of 1 to 128 characters.")
    return hashlib.sha256(("studio-study\0" + request_id).encode()).hexdigest()[:32]


def _request_digest(
    name: str, base_digest: str, arms: list[dict[str, Any]], seeds: list[int]
) -> str:
    try:
        request = json.dumps(
            {"name": name, "baseDigest": base_digest, "arms": arms, "seeds": seeds},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Study edits must contain valid JSON values.") from error
    return hashlib.sha256(request.encode()).hexdigest()


def _entry_id(study_id: str, arm_id: str, seed: int) -> str:
    key = f"studio-study-entry\0{study_id}\0{arm_id}\0{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _arms(arms: object) -> list[dict[str, Any]]:
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("Provide at least two named study arms.")
    prepared = []
    seen = set()
    for index, arm in enumerate(arms, start=1):
        if not isinstance(arm, dict) or set(arm) != {"name", "edits"}:
            raise ValueError("Each study arm needs a name and an edits list.")
        name = _label(arm["name"])
        if name.casefold() in seen:
            raise ValueError(f"Arm name {name!r} is repeated.")
        seen.add(name.casefold())
        edits = arm["edits"]
        if not isinstance(edits, list):
            raise ValueError(f"Provide a list of field edits for arm {name!r}.")
        for edit in edits:
            if (
                isinstance(edit, dict)
                and isinstance(path := edit.get("path"), list)
                and path[:1] == ["manual_seed"]
            ):
                raise ValueError("Set manual_seed through the study's common seeds.")
        prepared.append({"id": f"arm-{index}", "name": name, "edits": edits})
    return prepared


def _cell_error(
    error: ValueError, arm: dict[str, Any], seed: int | None = None
) -> StudyRejected:
    context = {"armId": arm["id"], "armName": arm["name"]}
    if seed is not None:
        context["seed"] = seed
    issues = getattr(error, "issues", None)
    if not issues:
        issues = [{"level": "error", "path": "", "message": str(error)}]
    return StudyRejected([{**issue, **context} for issue in issues])


def _validate_cell(text: str, seed: int) -> None:
    config = read_recipe(text)
    safe_name(config.get("name"))
    if type(config.get("manual_seed")) is not int or config["manual_seed"] != seed:
        raise ValueError("Launch preparation changed the assigned study seed.")
    paths = config.get("path") or {}
    if paths.get("resume_state"):
        raise ValueError(
            "Study arms start a new training run. Use a common pretrained model "
            "in the base recipe to start from existing weights."
        )
    if paths.get("ignore_resume_networks") is not None:
        raise ValueError("Clear path.ignore_resume_networks for managed study runs.")
    if paths.get("results_root") is not None:
        raise ValueError("Clear path.results_root for managed study runs.")

    # Studio starts a single rank. Worker seeding adds its index before calling
    # NumPy, so very high requested seeds can overflow even when valid on their own.
    from trainner_config.devices import auto_workers

    gpus = config.get("num_gpu")
    for phase, dataset in config.get("datasets", {}).items():
        if phase != "train" and not (config.get("val") or {}).get("val_enabled"):
            continue
        workers = dataset.get("num_worker_per_gpu")
        if workers == "auto":
            workers = auto_workers(1)
        elif type(workers) is int and phase == "train" and type(gpus) is int:
            workers *= max(1, gpus)
        if type(workers) is int and workers > 0 and seed + workers - 1 > MAX_SEED:
            raise ValueError(
                f"This seed plus datasets.{phase} worker offsets exceeds the NumPy seed range."
            )


class StudyManager:
    def __init__(self, queue: QueueManager) -> None:
        self.queue = queue

    def create(
        self,
        name: str,
        base_yaml: str,
        arms: list[dict[str, Any]],
        seeds: list[int],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        name = _label(name)
        arms = _arms(arms)
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or not 0 <= seed <= MAX_SEED for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(
                "Provide distinct integer seeds from 0 through 4294967295."
            )
        if len(arms) * len(seeds) > MAX_STUDY_ENTRIES:
            raise ValueError(
                f"A study can contain up to {MAX_STUDY_ENTRIES} arm/seed entries."
            )
        if not isinstance(base_yaml, str) or len(base_yaml.encode()) > MAX_RECIPE_BYTES:
            raise ValueError("Provide a base recipe of up to 2 MiB.")

        study_id = _study_id(request_id)
        base_digest = hashlib.sha256(base_yaml.encode()).hexdigest()
        digest = _request_digest(name, base_digest, arms, seeds)

        if existing := self.queue.study_records().get(study_id):
            if existing["requestDigest"] != digest:
                raise ValueError("This request ID already created a different study.")
            return self.detail(study_id)

        read_recipe(base_yaml)
        study = {
            "id": study_id,
            "name": name,
            "createdAt": process.timestamp(),
            "baseDigest": base_digest,
            "requestDigest": digest,
            "armCount": len(arms),
            "seedCount": len(seeds),
            "entryCount": len(arms) * len(seeds),
        }
        self.queue.publish_study(
            study,
            base_yaml,
            {"arms": arms, "seeds": seeds},
            self._cells(study_id, base_yaml, arms, seeds),
        )
        return self.detail(study_id)

    def _cells(
        self,
        study_id: str,
        base_yaml: str,
        arms: list[dict[str, Any]],
        seeds: list[int],
    ) -> Iterator[tuple[dict[str, Any], str]]:
        for arm in arms:
            try:
                arm_yaml = render(base_yaml, arm["edits"])
            except ValueError as error:
                raise _cell_error(error, arm) from error

            for seed in seeds:
                entry_id = _entry_id(study_id, arm["id"], seed)
                try:
                    seeded = render(
                        arm_yaml, [{"path": ["manual_seed"], "value": seed}]
                    )
                    text = self.queue.prepare(seeded)
                    _validate_cell(text, seed)
                except ValueError as error:
                    raise _cell_error(error, arm, seed) from error
                yield (
                    {
                        "id": entry_id,
                        "armId": arm["id"],
                        "armName": arm["name"],
                        "seed": seed,
                    },
                    text,
                )

    @staticmethod
    def _public(study: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in study.items() if key != "requestDigest"}

    def list_studies(self) -> list[dict[str, Any]]:
        studies = self.queue.study_records().values()
        return [self._public(study) for study in reversed(list(studies))]

    def detail(self, study_id: str) -> dict[str, Any]:
        study, base_yaml, manifest, entries = self.queue.study_snapshot(study_id)
        if hashlib.sha256(base_yaml.encode()).hexdigest() != study["baseDigest"]:
            raise ValueError("The saved study base recipe changed.")
        if (
            _request_digest(
                study["name"], study["baseDigest"], manifest["arms"], manifest["seeds"]
            )
            != study["requestDigest"]
        ):
            raise ValueError("The saved study manifest changed.")
        arms = [
            {
                **arm,
                "entries": [
                    entries[_entry_id(study_id, arm["id"], seed)]
                    for seed in manifest["seeds"]
                ],
            }
            for arm in manifest["arms"]
        ]
        return {
            **self._public(study),
            "baseYaml": base_yaml,
            "seeds": manifest["seeds"],
            "arms": arms,
        }

    def compare(
        self,
        study_id: str,
        left_arm_id: str,
        right_arm_id: str,
        dataset: str,
        metric: str,
        step: int | None = None,
    ) -> dict[str, Any]:
        from .comparison import compare
        from .study_results import summarize

        if not isinstance(left_arm_id, str) or not isinstance(right_arm_id, str):
            raise ValueError("Select two study arm IDs.")
        if left_arm_id == right_arm_id:
            raise ValueError("Select two different study arms.")
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(metric, str)
            or not metric
        ):
            raise ValueError("Select a recorded validation dataset and metric.")
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("Select a non-negative validation iteration.")
        study = self.detail(study_id)
        arms = {arm["id"]: arm for arm in study["arms"]}
        if left_arm_id not in arms or right_arm_id not in arms:
            raise FileNotFoundError("Study arm was not found.")
        results = []
        for left, right in zip(
            arms[left_arm_id]["entries"], arms[right_arm_id]["entries"], strict=True
        ):
            row = {
                "seed": left["seed"],
                "leftEntryId": left["id"],
                "rightEntryId": right["id"],
                "leftStatus": left["status"],
                "rightStatus": right["status"],
            }
            left_id, right_id = left.get("runId"), right.get("runId")
            if left_id is not None:
                row["leftRunId"] = left_id
            if right_id is not None:
                row["rightRunId"] = right_id
            if left_id is None or right_id is None:
                row["status"] = "unpaired"
            else:
                try:
                    result = compare(
                        self.queue.runs.experiment_dir(left_id),
                        self.queue.runs.experiment_dir(right_id),
                        dataset,
                        metric,
                        step,
                        step,
                        study_population=True,
                    )
                except (OSError, ValueError, sqlite3.DatabaseError) as error:
                    row.update(status="unavailable", error=str(error))
                else:
                    row.update(
                        status="compared" if result["comparable"] else "incompatible",
                        comparison=result,
                    )
            results.append(row)
        return {
            "studyId": study_id,
            "leftArmId": left_arm_id,
            "rightArmId": right_arm_id,
            "dataset": dataset,
            "metric": metric,
            "step": step,
            "pairs": results,
            "seedSummary": summarize(results),
        }

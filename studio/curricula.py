"""Immutable training phases over the owner-released Studio queue."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from . import run_process as process
from .queue import QueueManager
from .recipes import safe_name
from .run_checkpoints import phase_weight_roles
from .yaml_edit import read_recipe, render

MAX_PHASES = 32
PHASE_WEIGHT_FIELDS = (
    "pretrain_network_g",
    "pretrain_network_g_ema",
    "pretrain_network_d",
    "pretrain_network_ae",
    "pretrain_network_ae_ema",
    "pretrain_network_ae_decoder",
    "pretrain_network_ae_decoder_ema",
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _request_key(request_id: str | None) -> str:
    if request_id is None:
        return uuid4().hex
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or len(request_id) > 128
    ):
        raise ValueError("Provide a request ID of 1 to 128 characters.")
    return request_id


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Curriculum edits must contain valid JSON values.") from error
    return _hash(encoded)


def _label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 120
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("Provide a curriculum or phase name of 1 to 120 characters.")
    return value.strip()


def _phase_specs(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(phases, list) or not 1 <= len(phases) <= MAX_PHASES:
        raise ValueError(
            f"Provide between 1 and {MAX_PHASES} ordered curriculum phases."
        )
    result = []
    clock = 0
    names = set()
    for index, item in enumerate(phases, start=1):
        required = {"name", "mode", "updates", "edits"}
        if not isinstance(item, dict) or not required <= item.keys() <= required | {
            "weights"
        }:
            raise ValueError("Each phase needs name, mode, updates and edits.")
        name = _label(item["name"])
        if name.casefold() in names:
            raise ValueError(f"Phase name {name!r} is repeated.")
        names.add(name.casefold())
        mode = item["mode"]
        allowed = ("start",) if index == 1 else ("continue", "finetune")
        if mode not in allowed:
            raise ValueError(
                "The first phase uses start. Later phases use continue or finetune."
            )
        if "weights" in item:
            weights = item["weights"]
            best = (isinstance(weights, dict) and weights.keys() == {"bestValidation"}
                    and isinstance(weights["bestValidation"], str) and weights["bestValidation"].strip())
            if mode != "finetune" or not (best or weights in ("raw", "ema")):
                raise ValueError("Fine-tune with raw, EMA, or the retained best weights from one named validation dataset.")
        updates = item["updates"]
        start = clock if mode == "continue" else 0
        if type(updates) is not int or not 1 <= updates <= (2**63 - 1) - start:
            raise ValueError(
                f"Phase {name!r} needs a positive number of additional updates."
            )
        clock = start + updates
        phase = {
            "id": f"phase-{index}",
            "name": name,
            "mode": mode,
            "updates": updates,
            "startStep": start,
            "targetStep": clock,
            "edits": item["edits"],
        }
        if index > 1:
            phase["predecessorPhaseId"] = f"phase-{index - 1}"
        if mode == "finetune":
            phase["weights"] = item.get("weights", "raw")
        result.append(phase)
    return result


class PhaseRejected(ValueError):
    def __init__(self, error: ValueError, phase_id: str) -> None:
        super().__init__(f"{phase_id}: {error}")
        issues = getattr(error, "issues", None) or [
            {"level": "error", "path": "", "message": str(error)}
        ]
        self.issues = [{**issue, "phaseId": phase_id} for issue in issues]


class CurriculumManager:
    def __init__(self, queue: QueueManager) -> None:
        self.queue = queue

    def _recipe(
        self,
        base_yaml: str,
        mode: str,
        target: int,
        edits: list[dict[str, Any]],
    ) -> str:
        if not isinstance(edits, list):
            raise ValueError("Provide a list of phase field edits.")
        reserved = [("train", "total_iter"), ("path", "resume_state")]
        if mode != "start":
            reserved.extend(("path", field) for field in PHASE_WEIGHT_FIELDS)
        if mode == "continue":
            reserved.append(("manual_seed",))
        for edit in edits:
            if isinstance(edit, dict) and isinstance(path := edit.get("path"), list):
                for field in reserved:
                    if tuple(path[: len(field)]) == field:
                        raise ValueError(
                            f"The curriculum sets {'.'.join(field)} for this phase."
                        )

        text = render(base_yaml, edits)
        config = read_recipe(text)
        path = config.get("path")
        paths = path if isinstance(path, dict) else {}
        for name in ("resume_state", "ignore_resume_networks", "results_root"):
            if paths.get(name) is not None:
                raise ValueError(f"Clear path.{name} in the curriculum recipe.")
        train = config.get("train")
        if (
            mode == "continue"
            and isinstance(train, dict)
            and train.get("restart_scheduler")
        ):
            raise ValueError(
                "Continue restores the source scheduler. Clear train.restart_scheduler."
            )

        updates = [{"path": ["train", "total_iter"], "value": target}]
        if mode != "start":
            updates.extend(
                {"path": ["path", field], "value": None}
                for field in PHASE_WEIGHT_FIELDS
                if paths.get(field) is not None
            )
        if mode == "continue" and config.get("manual_seed") is not None:
            updates.append({"path": ["manual_seed"], "value": None})
        prepared = self.queue.prepare(render(text, updates))
        safe_name(read_recipe(prepared).get("name"))
        return prepared

    def _recipes(
        self, base_yaml: str, phases: list[dict[str, Any]], entries: dict[str, str]
    ) -> Iterator[tuple[dict[str, Any], str]]:
        previous = None
        for phase in phases:
            phase_id = phase["id"]
            try:
                text = self._recipe(
                    base_yaml, phase["mode"], phase["targetStep"], phase["edits"]
                )
                config = read_recipe(text)
                if previous is not None:
                    phase_weight_roles(
                        previous, config, 1, phase["mode"], phase.get("weights")
                    )
            except ValueError as error:
                raise PhaseRejected(error, phase_id) from error
            previous = config
            yield (
                {
                    **phase,
                    "id": entries[phase_id],
                    "phaseId": phase_id,
                    "phaseName": phase["name"],
                    "name": config["name"],
                    "revision": 1,
                },
                text,
            )

    def create(
        self,
        name: str,
        base_yaml: str,
        phases: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        name = _label(name)
        phases = _phase_specs(phases)
        read_recipe(base_yaml)
        curriculum_id = _hash("studio-curriculum\0" + _request_key(request_id))[:32]
        base_digest = _hash(base_yaml)
        digest = _digest([name, base_digest, phases])
        if existing := self.queue.curriculum_records().get(curriculum_id):
            if existing["requestDigest"] != digest:
                raise ValueError(
                    "This request ID already created a different curriculum."
                )
            return self.detail(curriculum_id)

        entries = {
            phase["id"]: _hash(
                f"studio-curriculum-entry\0{curriculum_id}\0{phase['id']}"
            )[:32]
            for phase in phases
        }
        record = {
            "id": curriculum_id,
            "name": name,
            "createdAt": process.timestamp(),
            "baseDigest": base_digest,
            "requestDigest": digest,
            "phaseCount": len(phases),
            "phaseEntries": entries,
        }
        self.queue.publish_curriculum(
            record,
            base_yaml,
            {"phases": phases},
            self._recipes(base_yaml, phases, entries),
        )
        return self.detail(curriculum_id)

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "requestDigest"}

    def list_curricula(self) -> list[dict[str, Any]]:
        return [
            self._public(plan)
            for plan in reversed(list(self.queue.curriculum_records().values()))
        ]

    def _snapshot(
        self, curriculum_id: str
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, dict[str, Any]]]:
        plan, base_yaml, manifest, entries = self.queue.curriculum_snapshot(
            curriculum_id
        )
        phases = manifest["phases"]
        if (
            _hash(base_yaml) != plan["baseDigest"]
            or _digest([plan["name"], plan["baseDigest"], phases])
            != plan["requestDigest"]
        ):
            raise ValueError("The saved curriculum base recipe or phase plan changed.")
        return plan, base_yaml, phases, entries

    def detail(self, curriculum_id: str) -> dict[str, Any]:
        plan, base_yaml, phases, entries = self._snapshot(curriculum_id)
        return {
            **self._public(plan),
            "baseYaml": base_yaml,
            "phases": [
                {
                    **phase,
                    "entry": entries[plan["phaseEntries"][phase["id"]]],
                    "revisions": [
                        entry
                        for entry in entries.values()
                        if entry["phaseId"] == phase["id"]
                    ],
                }
                for phase in phases
            ],
        }

    def retry_phase(
        self,
        curriculum_id: str,
        phase_id: str,
        edits: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        plan, base_yaml, _, entries = self._snapshot(curriculum_id)
        if not isinstance(phase_id, str) or phase_id not in plan["phaseEntries"]:
            raise FileNotFoundError("Curriculum phase was not found.")
        entry_id = plan["phaseEntries"][phase_id]
        revision_id = _hash(
            f"studio-curriculum-revision\0{curriculum_id}\0{phase_id}\0{_request_key(request_id)}"
        )[:32]
        digest = _digest(edits)
        if existing := entries.get(revision_id):
            if existing.get("revisionDigest") != digest:
                raise ValueError(
                    "This request ID already saved a different phase revision."
                )
            return existing

        entry = entries[entry_id]
        if entry["status"] not in {"held", "cancelled", "failed", "stopped"}:
            raise ValueError(
                "A phase can be revised while held, cancelled, failed or stopped."
            )
        selected = entry["edits"] if edits is None else edits
        try:
            text = self._recipe(base_yaml, entry["mode"], entry["targetStep"], selected)
        except ValueError as error:
            raise PhaseRejected(error, phase_id) from error
        return self.queue.revise_curriculum_phase(
            curriculum_id, entry_id, revision_id, digest, selected, text
        )

"""Persistent queue for managed WSL experiments, initially ordered by release.

A dispatch claim is saved before calling RunManager. Reconciliation uses the
same reserved request ID after a service interruption, including before a
pause, hold, cancellation, or Stop can act on the claimed entry.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from . import run_observe as observe
from . import run_process as process
from .paths import local_path
from .recipes import RecipeLibrary, atomic_text, safe_name
from .reproduction_pins import checked_files
from .run_checkpoints import (
    import_sources,
    phase_checkpoint,
    phase_options,
    phase_weight_roles,
    prepared_import_plan,
    rdual_options,
)
from .runs import ACTIVE, QUEUE_REQUEST_PREFIX, RUN_ID, RunBusyError, RunManager
from .update_gate import studio_admission
from .yaml_edit import read_recipe, render

WAITING = {"held", "pending"}
ARCHIVE_PROVENANCE = ("sourceRootId", "sourceRoot", "sourceRecipe", "replayIssues")


def _archive_replay_issues(text: str, root: Path) -> list[dict[str, str]]:
    from trainner_config.rdual import rdual_neuflow_required

    from .preflight import input_paths

    config = read_recipe(text)
    issues = []

    def absolute(value: str, _root: Path) -> Path:
        if not value.lstrip().startswith("~") and (path := local_path(value, Path())).is_absolute():
            return path
        raise ValueError("Resolve this archived recipe input to an explicit local path before training.")

    try:
        opt = rdual_options(config)
    except ValueError:
        opt = None
    need_flow = opt is not None
    if opt is not None and opt.path.neuflow_weights:
        try:
            companions = {}
            if opt.path.resume_state:
                absolute(opt.path.resume_state, root)
                companions = import_sources(config, root)
            need_flow = rdual_neuflow_required(
                opt, root, resolve_path=absolute, resume_files=companions
            ) is not False
        except (OSError, ValueError, RuntimeError):
            pass
    for path, value in input_paths(config):
        if not isinstance(value, str) or not value.strip():
            continue
        if path == ["path", "neuflow_weights"] and not need_flow:
            continue
        if opt is not None and path[0] == "path" and (
            (opt.path.resume_state and path[1].startswith("pretrain_network_"))
            or (path[1] == "pretrain_network_g_ema" and (opt.train is None or opt.train.ema_decay <= 0))
        ):
            continue
        try:
            absolute(value, root)
        except ValueError:
            issues.append(
                {
                    "level": "error",
                    "path": ".".join(map(str, path)),
                    "message": "Resolve this archived recipe input to an explicit local path before training.",
                }
            )
    return issues


class _InvalidRecipe(ValueError):
    def __init__(self, issues: list[dict[str, Any]]) -> None:
        super().__init__("The queued recipe no longer passes preflight.")
        self.issues = issues


class QueueManager:
    def __init__(
        self,
        root: Path,
        state: Path,
        run_manager: RunManager,
        prepare: Callable[[str], str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.directory = state.resolve() / "queue"
        self.index = self.directory / "queue.json"
        self.runs = run_manager
        self.prepare = self._preflight if prepare is None else prepare
        self.library = RecipeLibrary(self.root, state, project=run_manager.project)

    def _preflight(self, text: str) -> str:
        from .preflight import launch_text, preflight

        check = preflight(text, self.library)
        if not check["ok"]:
            raise _InvalidRecipe(check["issues"])
        return launch_text(text, check["config"], self.root)

    def _read(self) -> dict[str, Any]:
        state = process.read_record(self.index)
        if state is None:
            return {"paused": False, "activeId": None, "nextOrder": 0, "entries": {}}
        return state

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        with studio_admission(self.root):
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.directory.resolve() != self.directory:
                raise ValueError("The queue directory is redirected.")
            with (self.directory / "control.lock").open("ab") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield self._read()

    def _save(self, state: dict[str, Any]) -> None:
        process.write_record(self.index, state)
        self._sync_directory(self.directory)

    @staticmethod
    def _entry_id(request_id: str | None) -> str:
        if request_id is None:
            return uuid4().hex
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id) > 128
        ):
            raise ValueError("Provide a request ID of 1 to 128 characters.")
        return hashlib.sha256(("studio-enqueue\0" + request_id).encode()).hexdigest()[
            :32
        ]

    @staticmethod
    def _entry(state: dict[str, Any], entry_id: str) -> dict[str, Any]:
        if not isinstance(entry_id, str) or entry_id not in state["entries"]:
            raise FileNotFoundError("Queue entry was not found.")
        return state["entries"][entry_id]

    def _bundle_directory(self, section: str, bundle_id: str) -> Path:
        return self.directory / section / bundle_id

    def _recipe(self, entry: dict[str, Any]) -> Path:
        folder = self.directory
        if study_id := entry.get("studyId"):
            folder = self._bundle_directory("studies", study_id)
        elif curriculum_id := entry.get("curriculumId"):
            folder = self._bundle_directory("curricula", curriculum_id)
        return folder / "recipes" / (entry["id"] + ".yml")

    def _saved_recipe(self, entry: dict[str, Any]) -> str:
        path = self._recipe(entry)
        text = observe.recipe_text(path) if observe.regular_file(path) else None
        if text is None:
            raise FileNotFoundError("The saved queue recipe is unavailable.")
        if hashlib.sha256(text.encode()).hexdigest() != entry["recipeDigest"]:
            raise ValueError(
                "The saved queue recipe changed. Submit the desired recipe as a new entry."
            )
        return text

    @staticmethod
    def _public(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "paused": state["paused"],
            "activeId": state["activeId"],
            "entries": list(state["entries"].values()),
        }

    def snapshot(self) -> dict[str, Any]:
        state = self._read()
        self._reconcile(state)
        return self._public(state)

    def history_parents(self) -> dict[str, tuple[str, str, int]]:
        """Checkpoint parents by launched run, without advancing or reconciling the queue."""
        parents = {}
        for entry in self._read()["entries"].values():
            if "runId" not in entry:
                continue
            if "resumeStep" in entry and entry["sourceRunId"] != entry["runId"]:
                parents[entry["runId"]] = (entry["sourceRunId"], entry["sourceAttemptId"], entry["resumeStep"])
            elif entry.get("mode") == "continue" and (handoff := entry.get("handoff")):
                parents[entry["runId"]] = (handoff["runId"], handoff["attemptId"], handoff["step"])
        return parents

    def detail(self, entry_id: str) -> dict[str, Any]:
        state = self._read()
        self._reconcile(state)
        entry = self._entry(state, entry_id)
        return {**entry, "yaml": self._saved_recipe(entry)}

    def group(self, group_id: str) -> dict[str, Any]:
        entries = [
            entry
            for entry in self.snapshot()["entries"]
            if entry["groupId"] == group_id
        ]
        if not entries:
            raise FileNotFoundError("Replay group was not found.")
        run_ids = list(
            dict.fromkeys(
                entry[key]
                for entry in entries
                for key in ("sourceRunId", "runId")
                if key in entry
            )
        )
        return {"groupId": group_id, "entries": entries, "runIds": run_ids}

    @staticmethod
    def _held_entry(
        entry_id: str,
        text: str,
        *,
        source_run_id: str | None = None,
        group_id: str | None = None,
        recipe_path: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = read_recipe(text)
        name = safe_name(config.get("name"))
        digest = hashlib.sha256(text.encode()).hexdigest()
        timestamp = process.timestamp()
        entry = {
            "id": entry_id,
            "name": name,
            "status": "held",
            "groupId": group_id or entry_id,
            "recipeDigest": digest,
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }
        if source_run_id is not None:
            entry["sourceRunId"] = source_run_id
        if recipe_path is not None:
            entry["recipePath"] = recipe_path
        if provenance is not None:
            entry.update(
                {
                    key: provenance[key]
                    for key in ARCHIVE_PROVENANCE
                    if key in provenance
                }
            )
        return entry

    def _add(
        self,
        state: dict[str, Any],
        entry_id: str,
        text: str,
        *,
        source_run_id: str | None = None,
        group_id: str | None = None,
        recipe_path: str | None = None,
        provenance: dict[str, Any] | None = None,
        input_precondition: dict | None = None,
        continuation: dict | None = None,
    ) -> dict[str, Any]:
        entry = self._held_entry(
            entry_id,
            text,
            source_run_id=source_run_id,
            group_id=group_id,
            recipe_path=recipe_path,
            provenance=provenance,
        )
        if input_precondition is not None:
            entry["inputPrecondition"] = input_precondition
        if continuation is not None:
            entry.update(continuation)
            self._assign_release_order(state, entry)
            entry["status"] = "pending"
        atomic_text(self._recipe(entry), text)
        state["entries"][entry_id] = entry
        self._save(state)
        return entry

    def study_records(self) -> dict[str, dict[str, Any]]:
        return self._read().get("studies", {})

    def study_snapshot(
        self, study_id: str
    ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, dict[str, Any]]]:
        return self._bundle_snapshot("studies", "studyId", study_id, "Study")

    def curriculum_records(self) -> dict[str, dict[str, Any]]:
        return self._read().get("curricula", {})

    def curriculum_snapshot(
        self, curriculum_id: str
    ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, dict[str, Any]]]:
        return self._bundle_snapshot(
            "curricula", "curriculumId", curriculum_id, "Curriculum"
        )

    def _bundle_snapshot(
        self, section: str, reference: str, bundle_id: str, label: str
    ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, dict[str, Any]]]:
        state = self._read()
        bundle = (
            state.get(section, {}).get(bundle_id)
            if isinstance(bundle_id, str)
            else None
        )
        if bundle is None:
            raise FileNotFoundError(f"{label} was not found.")
        self._reconcile(state)
        directory = self._bundle_directory(section, bundle_id)
        base_path = directory / "base.yml"
        manifest_path = directory / "manifest.json"
        base_yaml = (
            observe.recipe_text(base_path) if observe.regular_file(base_path) else None
        )
        manifest = (
            process.read_record(manifest_path)
            if observe.regular_file(manifest_path)
            else None
        )
        if base_yaml is None or manifest is None:
            raise FileNotFoundError(
                f"The committed {label.lower()} files are unavailable."
            )
        entries = {
            entry_id: entry
            for entry_id, entry in state["entries"].items()
            if entry.get(reference) == bundle_id
        }
        return bundle, base_yaml, manifest, entries

    def publish_study(
        self,
        study: dict[str, Any],
        base_yaml: str,
        manifest: dict[str, Any],
        cells: Iterable[tuple[dict[str, Any], str]],
    ) -> None:
        entries = (
            (
                {**cell, "studyId": study["id"], "studyName": study["name"]},
                text,
            )
            for cell, text in cells
        )
        self._stage_bundle(
            "studies", "study", study, base_yaml, manifest, entries, study["entryCount"]
        )

    def publish_curriculum(
        self,
        curriculum: dict[str, Any],
        base_yaml: str,
        manifest: dict[str, Any],
        phases: Iterable[tuple[dict[str, Any], str]],
    ) -> None:
        entries = (
            ({**phase, "curriculumId": curriculum["id"]}, text)
            for phase, text in phases
        )
        self._stage_bundle(
            "curricula",
            "curriculum",
            curriculum,
            base_yaml,
            manifest,
            entries,
            len(curriculum["phaseEntries"]),
        )

    def _stage_bundle(
        self,
        section: str,
        label: str,
        bundle: dict[str, Any],
        base_yaml: str,
        manifest: dict[str, Any],
        recipes: Iterable[tuple[dict[str, Any], str]],
        count: int,
    ) -> None:
        bundle_id = bundle["id"]
        if not isinstance(bundle_id, str) or RUN_ID.fullmatch(bundle_id) is None:
            raise ValueError(f"Invalid {label} ID.")
        destination = self._bundle_directory(section, bundle_id)
        directory = destination.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.resolve() != directory:
            raise ValueError(f"The {label} directory is redirected.")
        with TemporaryDirectory(prefix=".creating-", dir=directory) as temporary:
            staged = Path(temporary) / "bundle"
            entries = {}
            for metadata, text in recipes:
                entry_id = metadata["id"]
                if (
                    not isinstance(entry_id, str)
                    or RUN_ID.fullmatch(entry_id) is None
                    or entry_id in entries
                ):
                    raise ValueError(f"Invalid or duplicate {label} entry ID.")
                entry = self._held_entry(entry_id, text, group_id=bundle_id)
                entry.update(metadata)
                atomic_text(staged / "recipes" / f"{entry_id}.yml", text)
                entries[entry_id] = entry
            if len(entries) != count:
                raise ValueError(f"The staged {label} is missing one or more entries.")
            atomic_text(staged / "base.yml", base_yaml)
            process.write_record(staged / "manifest.json", manifest)
            self._publish_bundle(section, label, bundle, entries, staged)

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _publish_bundle(
        self,
        section: str,
        label: str,
        bundle: dict[str, Any],
        entries: dict[str, dict[str, Any]],
        staged: Path,
    ) -> None:
        """Publish a bundle and all held entries in one queue index replacement."""
        bundle_id = bundle["id"]
        destination = self._bundle_directory(section, bundle_id)
        self._sync_directory(staged / "recipes")
        self._sync_directory(staged)
        with self._locked() as state:
            registered = state.setdefault(section, {})
            if existing := registered.get(bundle_id):
                if existing["requestDigest"] != bundle["requestDigest"]:
                    raise ValueError(
                        f"This request ID already created a different {label}."
                    )
                return
            if entries.keys() & state["entries"].keys():
                raise ValueError(f"A {label} entry ID is already in use.")
            if os.path.lexists(destination):
                if not destination.is_dir() or destination.resolve() != destination:
                    raise FileExistsError(f"The {label} directory is unavailable.")
                destination.rename(staged.parent / "unpublished")
            staged.rename(destination)
            try:
                self._sync_directory(destination.parent)
                registered[bundle_id] = bundle
                state["entries"].update(entries)
                self._save(state)
            except BaseException:
                if bundle_id not in self._read().get(section, {}):
                    destination.rename(staged.parent / "failed")
                raise

    def revise_curriculum_phase(
        self,
        curriculum_id: str,
        expected_entry_id: str,
        revision_id: str,
        digest: str,
        edits: list[dict[str, Any]],
        yaml_text: str,
    ) -> dict[str, Any]:
        """Publish a new held version and its current pointer together."""
        if RUN_ID.fullmatch(revision_id) is None:
            raise ValueError("Invalid phase revision ID.")
        with self._locked() as state:
            self._reconcile(state, recover_unlaunched=True)
            old = self._entry(state, expected_entry_id)
            if old.get("curriculumId") != curriculum_id:
                raise FileNotFoundError("Curriculum phase was not found.")
            if existing := state["entries"].get(revision_id):
                if (
                    existing.get("curriculumId") != curriculum_id
                    or existing.get("phaseId") != old["phaseId"]
                    or existing.get("revisionDigest") != digest
                ):
                    raise ValueError(
                        "This request ID already saved a different phase revision."
                    )
                return existing
            current = state["curricula"][curriculum_id]["phaseEntries"]
            if current[old["phaseId"]] != expected_entry_id:
                raise ValueError(
                    "The current phase revision changed. Reload the curriculum."
                )
            if old["status"] not in {"held", "cancelled", "stopped", "failed"}:
                raise ValueError(
                    "Hold, cancel or stop the current phase before revising it."
                )
            # The adjacent revisions can change since the editor opened. Validate
            # both handoffs against the current recipes under the publication lock.
            candidate = read_recipe(yaml_text)
            if predecessor := old.get("predecessorPhaseId"):
                previous = self._entry(state, current[predecessor])
                phase_weight_roles(
                    read_recipe(self._saved_recipe(previous)), candidate,
                    previous["targetStep"], old["mode"], old.get("weights"),
                )
            for neighbor_id in current.values():
                neighbor = self._entry(state, neighbor_id)
                if neighbor.get("predecessorPhaseId") == old["phaseId"]:
                    phase_weight_roles(
                        candidate, read_recipe(self._saved_recipe(neighbor)),
                        old["targetStep"], neighbor["mode"], neighbor.get("weights"),
                    )
            entry = self._held_entry(revision_id, yaml_text, group_id=curriculum_id)
            entry.update(
                {
                    key: old[key]
                    for key in (
                        "curriculumId",
                        "phaseId",
                        "phaseName",
                        "predecessorPhaseId",
                        "mode",
                        "startStep",
                        "targetStep",
                        "updates",
                        "weights",
                    )
                    if key in old
                }
            )
            entry.update(
                revision=old["revision"] + 1,
                previousEntryId=old["id"],
                revisionDigest=digest,
                edits=edits,
            )
            recipe = self._recipe(entry)
            if recipe.parent.resolve() != recipe.parent:
                raise ValueError("The curriculum recipe directory is redirected.")
            atomic_text(recipe, yaml_text)
            self._sync_directory(recipe.parent)
            state["entries"][revision_id] = entry
            current[old["phaseId"]] = revision_id
            old.update(replacedBy=revision_id, updatedAt=entry["updatedAt"])
            if old["status"] == "held":
                old["status"] = "cancelled"
            self._save(state)
            return entry

    @staticmethod
    def _current_phase_entries(
        state: dict[str, Any], entry: dict[str, Any]
    ) -> dict[str, str]:
        current = state["curricula"][entry["curriculumId"]]["phaseEntries"]
        if current.get(entry["phaseId"]) != entry["id"]:
            raise ValueError(
                "This phase revision has been replaced. Use the current held entry."
            )
        return current

    def _phase_handoff(
        self, state: dict[str, Any], entry: dict[str, Any], yaml_text: str
    ) -> dict[str, Any] | None:
        current = self._current_phase_entries(state, entry)
        if entry["mode"] == "start":
            return None
        previous = self._entry(state, current[entry["predecessorPhaseId"]])
        if previous["status"] != "completed" or not previous.get("runId"):
            raise ValueError(
                "Complete the preceding curriculum phase before releasing this entry."
            )
        source = previous["runId"]
        step = previous["targetStep"]
        experiment, config, attempt_id, seed = self.runs.completed_phase(
            source, previous["id"], step
        )
        if entry["mode"] == "continue" and step != entry["startStep"]:
            raise ValueError(
                "The preceding phase did not finish at this phase's source clock."
            )
        files = phase_checkpoint(
            experiment,
            config,
            read_recipe(yaml_text),
            step,
            entry["mode"],
            entry.get("weights"),
            attempt=attempt_id,
        )
        return {
            "entryId": previous["id"],
            "runId": source,
            "attemptId": attempt_id,
            "step": step,
            "seed": seed,
            "files": files,
        }

    def _released_phase(
        self, state: dict[str, Any], entry: dict[str, Any], yaml_text: str
    ) -> dict[str, Any]:
        handoff = self._phase_handoff(state, entry, yaml_text)
        if handoff != entry.get("handoff"):
            raise ValueError(
                "The pinned phase checkpoint changed. Retry the phase to select an intact source."
            )
        phase = {
            key: entry[key]
            for key in (
                "curriculumId",
                "phaseId",
                "revision",
                "mode",
                "startStep",
                "targetStep",
                "updates",
            )
        }
        if "weights" in entry:
            phase["weights"] = entry["weights"]
        if handoff is not None:
            phase["handoff"] = handoff
        return phase

    def submit(
        self,
        yaml_text: str,
        request_id: str | None = None,
        *,
        recipe_path: str | None = None,
        input_precondition: dict | None = None,
    ) -> dict[str, Any]:
        """Save a held recipe snapshot. Use release to authorize training."""
        if not isinstance(yaml_text, str):
            raise ValueError("Provide a YAML training recipe.")
        if recipe_path is not None and (
            not isinstance(recipe_path, str) or not recipe_path.strip()
        ):
            raise ValueError("Provide a recipe path for this snapshot.")
        pins = checked_files(input_precondition, yaml_text) if input_precondition is not None else None
        entry_id = self._entry_id(request_id)
        with self._locked() as state:
            self._reconcile(state, recover_unlaunched=True)
            if existing := state["entries"].get(entry_id):
                if (
                    "sourceRunId" in existing
                    or "studyId" in existing
                    or "curriculumId" in existing
                    or existing.get("recipePath") != recipe_path
                    or existing.get("inputPrecondition") != input_precondition
                    or existing["recipeDigest"]
                    != hashlib.sha256(yaml_text.encode()).hexdigest()
                ):
                    raise ValueError(
                        "This request ID already saved a different queue entry."
                    )
                return existing
            if pins is not None:
                prepared_import_plan(read_recipe(yaml_text), self.root, pins)
            return self._add(state, entry_id, yaml_text, recipe_path=recipe_path,
                             input_precondition=input_precondition)

    def _replay_info(self, state: dict[str, Any], run_id: str) -> dict[str, Any]:
        run = self.runs.detail(run_id)
        text = run.get("yaml")
        if run.get("sourceRecipeFormat") == "neosr-toml":
            raise ValueError("Import this archived NeoSR TOML as a native recipe before submitting it to the queue.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("This run has no saved training recipe to replay.")
        # The CLI chooses a seed when the recipe omits it. Preserve the
        # recorded choice for managed replays when it is known.
        if type(seed := run.get("seed")) is int:
            text = render(text, [{"path": ["manual_seed"], "value": seed}])
        parent = state["entries"].get(run.get("queueEntryId"))
        info = {
            "sourceRunId": run_id,
            "groupId": parent["groupId"] if parent is not None else run_id,
            "recipeDigest": hashlib.sha256(text.encode()).hexdigest(),
            "yaml": text,
        }
        if "invocation" in run:
            info["sourceInvocation"] = run["invocation"]
        if run.get("source") == "archive":
            info.update({key: run[key] for key in ARCHIVE_PROVENANCE if key in run})
            if issues := _archive_replay_issues(text, self.root):
                info["replayIssues"] = issues
        return info

    def replay_info(self, run_id: str) -> dict[str, Any]:
        """Read provenance and a replay recipe without creating a queue entry."""
        return self._replay_info(self._read(), run_id)

    def replay(self, run_id: str, request_id: str | None = None) -> dict[str, Any]:
        """Snapshot a run recipe in a new held entry, preserving its lineage."""
        entry_id = self._entry_id(request_id)
        with self._locked() as state:
            self._reconcile(state, recover_unlaunched=True)
            if existing := state["entries"].get(entry_id):
                if existing.get("sourceRunId") != run_id or "resumeStep" in existing:
                    raise ValueError(
                        "This request ID already saved a different queue entry."
                    )
                return existing
            info = self._replay_info(state, run_id)
            return self._add(
                state,
                entry_id,
                info["yaml"],
                source_run_id=run_id,
                group_id=info["groupId"],
                provenance=info,
            )

    def resume(
        self, run_id: str, request_id: str | None = None, *, additional_iterations: int | None = None
    ) -> dict[str, Any]:
        """Queue a stopped run's continuation or a completed run's requested extension."""
        if additional_iterations is not None and (
            type(additional_iterations) is not int or additional_iterations < 1
        ):
            raise ValueError("Additional iterations must be a positive whole number.")
        entry_id = self._entry_id(request_id)
        with self._locked() as state:
            if self._reconcile(state, recover_unlaunched=True):
                self._save(state)
            if existing := state["entries"].get(entry_id):
                if (
                    existing.get("sourceRunId") != run_id
                    or "resumeStep" not in existing
                    or existing.get("additionalIterations") != additional_iterations
                ):
                    raise ValueError("This request ID already saved a different queue entry.")
                return existing
            for entry in state["entries"].values():
                if entry.get("runId") == run_id and entry.get("pauseAttempt") and entry["status"] in WAITING | ACTIVE:
                    if additional_iterations is not None:
                        raise ValueError("Resume the paused queue entry before extending this run.")
                    return entry
                if (entry.get("sourceRunId") == run_id and "resumeStep" in entry
                        and entry["status"] in WAITING | ACTIVE):
                    if entry.get("additionalIterations") != additional_iterations:
                        raise FileExistsError("A continuation with a different target is already queued for this run.")
                    return entry
            text, parent_id, checkpoint = self.runs.queued_resume_source(
                run_id, additional_iterations=additional_iterations
            )
            if additional_iterations is not None:
                checkpoint["additionalIterations"] = additional_iterations
            parent = state["entries"].get(parent_id)
            return self._add(
                state, entry_id, text, source_run_id=run_id,
                group_id=parent["groupId"] if parent else run_id, continuation=checkpoint,
            )

    @staticmethod
    def _assign_release_order(state: dict, entry: dict) -> None:
        state["nextOrder"] += 1
        entry["order"] = state["nextOrder"]

    def _transition(
        self, entry_id: str, target: str, allowed: set[str]
    ) -> dict[str, Any]:
        with self._locked() as state:
            changed = self._reconcile(state, recover_unlaunched=True)
            entry = self._entry(state, entry_id)
            if entry["status"] == target:
                if changed:
                    self._save(state)
                return entry
            if entry["status"] not in allowed:
                raise ValueError(
                    f"Cannot change a {entry['status']} queue entry to {target}."
                )
            if target == "pending":
                if entry.get("replayIssues"):
                    raise ValueError(
                        "Resolve the archived replay's flagged inputs to explicit local paths, "
                        "then submit the edited recipe as a new queue entry."
                    )
                if "curriculumId" in entry:
                    handoff = self._phase_handoff(
                        state, entry, self._saved_recipe(entry)
                    )
                    if handoff is not None:
                        if entry.get("handoff", handoff) != handoff:
                            raise ValueError(
                                "The pinned phase checkpoint changed. Retry the phase to select an intact source."
                            )
                        entry.update(handoff=handoff, sourceRunId=handoff["runId"])
                self._assign_release_order(state, entry)
            entry.pop("blockedReason", None)
            entry.update(status=target, updatedAt=process.timestamp())
            self._save(state)
            return entry

    def reorder(self, expected: list[str], order: list[str]) -> dict[str, Any]:
        """Reorder the pending line as observed by the caller, under the dispatch lock."""
        for ids in (expected, order):
            if not isinstance(ids, list) or any(not isinstance(key, str) for key in ids):
                raise ValueError("Provide the waiting queue entry IDs as lists.")
        if len(set(expected)) != len(expected) or sorted(order) != sorted(expected):
            raise ValueError("The new order must include each waiting entry exactly once.")

        with self._locked() as state:
            changed = self._reconcile(state, recover_unlaunched=True)
            pending = sorted(
                (e for e in state["entries"].values() if e["status"] == "pending"),
                key=lambda e: e["order"],
            )
            current = [e["id"] for e in pending]
            if current != order:
                if current != expected:
                    raise FileExistsError("The waiting queue changed. Refresh it and move the entry again.")
                # Reuse the occupied order numbers so subsequent releases still join the tail.
                slots = [e["order"] for e in pending]
                now = process.timestamp()
                for key, slot in zip(order, slots, strict=True):
                    entry = state["entries"][key]
                    if entry["order"] != slot:
                        entry.update(order=slot, updatedAt=now)
                        entry.pop("blockedReason", None)
                changed = True
            if changed:
                self._save(state)
            return self._public(state)

    def release(self, entry_id: str) -> dict[str, Any]:
        """Authorize one held entry to run when it reaches the front."""
        return self._transition(entry_id, "pending", {"held"})

    def hold(self, entry_id: str) -> dict[str, Any]:
        return self._transition(entry_id, "held", {"pending"})

    def cancel(self, entry_id: str) -> dict[str, Any]:
        return self._transition(entry_id, "cancelled", WAITING)

    def stop(self, entry_id: str) -> dict[str, Any]:
        with self._locked() as state:
            changed = self._reconcile(state, recover_unlaunched=True)
            entry = self._entry(state, entry_id)
            if entry["status"] in WAITING:
                entry.pop("blockedReason", None)
                entry.update(status="cancelled", updatedAt=process.timestamp())
                changed = True
            elif entry["status"] in {"running", "stopping"}:
                run = self.runs.stop(entry["runId"])
                changed = self._attach(state, entry, run) or changed
            elif entry["status"] not in {"cancelled", "stopped"}:
                raise ValueError(f"Cannot stop a {entry['status']} queue entry.")
            if changed:
                self._save(state)
            return entry

    def set_paused(self, paused: bool) -> dict[str, Any]:
        if type(paused) is not bool:
            raise ValueError("Provide a boolean pause value.")
        with self._locked() as state:
            changed = self._reconcile(state, recover_unlaunched=True)
            if paused and (active := state["activeId"]):
                entry = self._entry(state, active)
                if entry["status"] == "running" and "pauseAttempt" not in entry:
                    run = self._claimed_run(entry)
                    if run and run["status"] in {"running", "starting"}:
                        entry["pauseAttempt"] = run["attemptId"]
                        changed = True
            if changed or state["paused"] != paused:
                state["paused"] = paused
                self._save(state)
            # Dispatch also delivers the saved request after a service restart.
            return self._public(state)

    @staticmethod
    def _fail(entry: dict[str, Any], error: str, issues: Any = None) -> None:
        entry.pop("blockedReason", None)
        entry.update(status="failed", updatedAt=process.timestamp(), error=error)
        if isinstance(issues, list) and issues:
            entry["issues"] = issues

    def _attach(
        self, state: dict[str, Any], entry: dict[str, Any], run: dict[str, Any]
    ) -> bool:
        if run.get("queueEntryId") != entry["id"]:
            self._fail(
                entry, "The launch request resolved to a run owned by another entry."
            )
            state["activeId"] = None
            return True
        run_status = run["status"]
        status = "running" if run_status in {"starting", "running"} else run_status
        fields = {"runId": run["id"], "status": status, "runStatus": run_status}
        if run.get("attemptId"):
            fields["attemptId"] = run["attemptId"]
        for key in ("completionReason", "completedIteration"):
            if key in run:
                fields[key] = run[key]
        if isinstance(run.get("error"), str):
            fields["error"] = run["error"]
        changed = any(entry.get(key) != value for key, value in fields.items())
        if "pauseAttempt" in entry:
            if (run.get("attemptId") == entry["pauseAttempt"] and run_status == "stopped"
                    and run.get("completionReason") != "forceStopped"):
                # Reuse this entry, its original place, run directory and study/phase identity.
                fields["status"] = "pending"
                changed = True
            elif run.get("attemptId") != entry["pauseAttempt"] or run_status not in ACTIVE:
                entry.pop("pauseAttempt")
                changed = True
                if run_status in ACTIVE:
                    for key in ("completedIteration", "completionReason", "error"):
                        entry.pop(key, None)
        if changed:
            entry.pop("blockedReason", None)
            entry.update(fields, updatedAt=run.get("updatedAt") or process.timestamp())
        if run_status not in ACTIVE and state["activeId"] is not None:
            state["activeId"] = None
            changed = True
        return changed

    def _claimed_run(self, entry: dict) -> dict | None:
        if "runId" in entry:
            return self.runs.launch_status(entry["runId"])
        if "resumeStep" in entry:
            run = self.runs.launch_status(entry["sourceRunId"])
            if run and run.get("queueEntryId") == entry["id"]:
                return run
        # Also recovers claims made by older services that created a separate run.
        return self.runs.requested_run(QUEUE_REQUEST_PREFIX + entry["id"])

    def _reconcile(
        self, state: dict[str, Any], *, recover_unlaunched: bool = False
    ) -> bool:
        if (claimed := state["activeId"]) is None:
            return False
        entry = self._entry(state, claimed)
        run = self._claimed_run(entry)
        if run is not None:
            return self._attach(state, entry, run)
        if not recover_unlaunched:
            return False
        # The queue lock excludes a dispatcher that has claimed the entry and
        # is still creating its run record. With no record, no trainer launched.
        if "runId" in entry:
            self._fail(entry, "The linked managed run record is missing.")
        state["activeId"] = None
        return True

    def dispatch_once(self) -> dict[str, Any] | None:
        """Reconcile the queue and attempt at most one owner-released launch."""
        if not self.index.is_file():
            return None
        with self._locked() as state:
            changed = self._reconcile(state, recover_unlaunched=True)
            if active := state["activeId"]:
                current = self._entry(state, active)
                if current.get("pauseAttempt") and current["status"] == "running":
                    self.runs.pause_queue_attempt(current["runId"], current["pauseAttempt"])
            if state["paused"] or state["activeId"] is not None:
                if changed:
                    self._save(state)
                return None
            entry = min(
                (
                    entry
                    for entry in state["entries"].values()
                    if entry["status"] == "pending"
                ),
                key=lambda entry: entry["order"],
                default=None,
            )
            if entry is None:
                if changed:
                    self._save(state)
                return None
            if blocker := self.runs.queue_blocker():
                if entry.get("blockedReason") != blocker:
                    entry.update(blockedReason=blocker, updatedAt=process.timestamp())
                    changed = True
                if changed:
                    self._save(state)
                return None

            entry_id = entry["id"]
            request_id = QUEUE_REQUEST_PREFIX + entry_id
            if "pauseAttempt" in entry:
                state["activeId"] = entry_id
                self._save(state)
                try:
                    run = self.runs.resume_queue_attempt(entry["runId"], entry["pauseAttempt"])
                except RunBusyError as error:
                    state["activeId"] = None
                    entry["blockedReason"] = str(error)
                except (ValueError, OSError) as error:
                    state.update(paused=True, activeId=None)
                    entry["blockedReason"] = f"Could not resume the paused run: {error}"
                else:
                    self._attach(state, entry, run)
                self._save(state)
                return entry
            try:
                saved = self._saved_recipe(entry)
                launch = {}
                if "curriculumId" in entry:
                    phase = self._released_phase(state, entry, saved)
                    launch["curriculum"] = phase
                    saved = phase_options(saved, phase)
                text = self.prepare(saved)
                if "studyId" in entry and saved != text:
                    raise ValueError(
                        "Launch preflight would change the saved study recipe. Create a new study to use it."
                    )
                if launch and saved != text:
                    raise ValueError(
                        "Launch preflight would change the curriculum recipe. Revise the phase to use it."
                    )
                if "inputPrecondition" in entry:
                    launch["input_precondition"] = entry["inputPrecondition"]
            except (OSError, ValueError) as error:
                self._fail(entry, str(error), getattr(error, "issues", None))
                self._save(state)
                return entry

            state["activeId"] = entry_id
            self._save(state)
            try:
                if "resumeStep" in entry:
                    run = self.runs.launch_queued_continuation(entry, text)
                else:
                    run = self.runs.start(text, request_id=request_id, queue_entry_id=entry_id, **launch)
            except RunBusyError as error:
                state["activeId"] = None
                entry.update(blockedReason=str(error), updatedAt=process.timestamp())
                self._save(state)
                return None
            except (OSError, ValueError) as error:
                # A failure after run publication must still reconcile against
                # the owned run rather than placing a second launch.
                if run := self._claimed_run(entry):
                    self._attach(state, entry, run)
                else:
                    state["activeId"] = None
                    self._fail(entry, str(error), getattr(error, "issues", None))
            else:
                self._attach(state, entry, run)
            self._save(state)
            return entry

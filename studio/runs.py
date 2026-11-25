"""Headless lifecycle for local traiNNer CLI runs."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from trainner_config.resume import recipe_learning_rate
from trainner_data.file_version import artifact_version

from . import run_observe as observe
from . import run_process as process
from .external_runs import ARCHIVE_PREFIX, ExternalRuns
from .paths import local_path
from .recipes import atomic_text, safe_name, task_from_config
from .reproduction_pins import checked_files
from .run_checkpoints import (
    detached_paths,
    import_options,
    latest_recipe,
    phase_checkpoint,
    phase_options,
    prepare_import,
    prepared_import_plan,
    resume_import_plan,
)
from .update_gate import MANAGED_ROOT, is_managed_runtime, studio_admission
from .workspace import PROJECT_DIR_ENV, WindowsProject, require_ext4
from .yaml_edit import read_recipe, render

RUN_ID = re.compile(r"[0-9a-f]{32}")
ACTIVE = {"starting", "running", "stopping"}
QUEUE_REQUEST_PREFIX = "studio-queue:"


class RunBusyError(FileExistsError):
    """Another trainer currently owns the resource needed for this launch."""


def _request_run_id(request_id: str) -> str:
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or len(request_id) > 128
    ):
        raise ValueError("Provide a request ID of 1 to 128 characters.")
    return hashlib.sha256(("studio-start\0" + request_id).encode()).hexdigest()[:32]


@contextmanager
def _control(directory: Path) -> Iterator[None]:
    # Mutations to unrelated runs use unrelated locks. Reads do not wait here.
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "control.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _summary(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    if config.get("network_g") or config.get("network_ae"):
        result["task"] = task_from_config(config)
    train = config.get("train")
    total = train.get("total_iter") if isinstance(train, dict) else None
    if type(total) is int and total > 0:
        result["total"] = total
    return result


def _resume_settings(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config.get("datasets")
    dataset = datasets.get("train") if isinstance(datasets, dict) else None
    batch = dataset.get("batch_size_per_gpu") if isinstance(dataset, dict) else None
    train = config.get("train")
    return {
        "batchSize": batch if type(batch) is int and batch > 0 else None,
        "learningRates": {
            role: recipe_learning_rate(options.get("lr"))
            for role in ("g", "d", "ae")
            if isinstance(train, dict)
            and isinstance(options := train.get(f"optim_{role}"), dict)
        },
    }


def _resume_edits(
    config: dict[str, Any],
    batch_size: int | None,
    learning_rates: dict[str, float] | None,
    original_rates: dict[str, float | None] | None,
) -> tuple[list[dict[str, Any]], dict[str, float | None] | None]:
    settings = _resume_settings(config)
    edits = []
    if batch_size is not None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batchSize must be a positive integer.")
        if batch_size != settings["batchSize"]:
            edits.append({"path": ["datasets", "train", "batch_size_per_gpu"], "value": batch_size})
    if learning_rates is not None:
        if not isinstance(learning_rates, dict):
            raise ValueError("learningRates must be a map of optimizer roles to rates.")
        configured = settings["learningRates"]
        existing = config["train"].get("resume_previous_lrs") or {}
        previous_rates = existing.copy()
        for role, value in learning_rates.items():
            if role not in configured:
                raise ValueError(f"No optimizer is configured for learningRates role {role!r}.")
            rate = recipe_learning_rate(value)
            if rate is None:
                raise ValueError(f"learningRates.{role} must be a finite nonnegative number.")
            if rate == configured[role]:
                continue
            edits.append({"path": ["train", f"optim_{role}", "lr"], "value": rate})
            if original_rates is None:
                original_rates = configured
            if (previous := original_rates.get(role)) is not None:
                previous_rates.setdefault(role, previous)
        if previous_rates != existing:
            edits.append({"path": ["train", "resume_previous_lrs"], "value": previous_rates})
    return edits, original_rates


def _extend_completed_recipe(
    text: str,
    experiment: Path,
    config: dict[str, Any],
    completion: dict[str, Any],
    additional_iterations: int,
) -> str:
    train = config.get("train")
    goal = train.get("total_iter") if isinstance(train, dict) else None
    if type(goal) is not int or goal < 1:
        raise ValueError("The completed run recipe has no valid training goal.")
    reason = completion.get("completionReason")
    history = completion.get("earlyStopping")
    bad_observations = history.get("badObservations") if isinstance(history, dict) else None
    policy = train.get("early_stopping")
    patience = policy.get("patience") if isinstance(policy, dict) else None
    if reason == "earlyStopping" or (
        type(bad_observations) is int and type(patience) is int and bad_observations >= patience
    ):
        raise ValueError(
            "The run exhausted its early-stopping patience. "
            "Start a new run from its model weights to train with fresh patience."
        )

    step = completion.get("completedIteration", goal)
    if (
        type(step) is not int
        or not 0 < step <= goal
        or reason not in (None, "targetReached")
        or (reason == "targetReached" and step != goal)
    ):
        raise ValueError("The run has no valid completed training step.")
    state = experiment / "training_states" / f"{step}.state"
    if not observe.regular_file(state):
        raise ValueError(f"The completed run is missing its numbered checkpoint state at iteration {step}.")

    edits = [{"path": ["train", "total_iter"], "value": step + additional_iterations}]
    if train.get("restart_scheduler"):
        edits.append({"path": ["train", "restart_scheduler"], "value": False})
    if (qat := train.get("qat")) is not None:
        from msgspec import convert
        from trainner_config.options import QATOptions

        # QAT's default phase boundaries depend on the original training goal.
        _, norm, observers = convert(qat, type=QATOptions, strict=True).schedule(goal)
        for field, value in (("freeze_norm_iter", norm), ("freeze_observers_iter", observers)):
            if qat.get(field) is None:
                edits.append({"path": ["train", "qat", field], "value": value})

    extended = render(text, edits)
    files = phase_checkpoint(experiment, config, read_recipe(extended), step, "continue", None)
    from .resume_preflight import validate_extension_scheduler_horizons

    validate_extension_scheduler_horizons(train, state, additional_iterations)
    return import_options(extended, files, experiment, full_state=True)


@lru_cache(maxsize=96)
def _imported_summary(recipe: Path, version: str) -> dict[str, Any]:
    text = observe.recipe_text(recipe)
    if text:
        try:
            return _summary(read_recipe(text))
        except ValueError:
            pass
    return {}


def _attempt(directory: Path, record: dict[str, Any]) -> Path:
    attempt = record.get("attempt")
    if not isinstance(attempt, str) or not RUN_ID.fullmatch(attempt):
        raise ValueError("The managed run record has an invalid attempt ID.")
    return directory / "attempts" / attempt


def _state(attempt: Path, control: Path) -> dict[str, Any]:
    result_path = attempt / "result.json"
    if result := process.read_record(result_path):
        return result
    active = process.lock_held(control / "active.lock")
    # The supervisor may have published its receipt as it released the lease.
    if result := process.read_record(result_path):
        return result
    if not active:
        return {
            "status": "failed",
            "error": "The run ended without a trainer exit record.",
        }
    launched = (control / "launched").exists()
    if launched and not process.lock_held(control / "supervisor.lock"):
        if result := process.read_record(result_path):
            return result
        return {
            "status": "failed",
            "error": "The supervisor exited while the trainer lease was still held.",
        }
    if (control / "stop").exists():
        return {"status": "stopping"}
    return {"status": "running" if launched else "starting"}


class RunManager:
    def __init__(
        self, root: Path, state_dir: Path, *, project: WindowsProject | None = None
    ) -> None:
        self.root = root.resolve()
        self.state = state_dir.resolve()
        self.project = project
        self.control_runs = self.state / "runs"
        self.runs = self.control_runs if project is None else project.root / "runs"
        self.resource_lock = self.root / ".cache" / "studio-trainer.lock"
        if project is not None:
            require_ext4(self.resource_lock.parent)
        self.archives = ExternalRuns(self.root, self.state)

    def _control_path(self, path: Path) -> Path:
        return self.control_runs / path.relative_to(self.runs)

    def _check_project(self, directory: Path) -> None:
        if self.project is not None:
            self.project.require_directory(directory)

    def _read_run(self, directory: Path) -> dict[str, Any] | None:
        path = directory / "run.json"
        if self.project is not None and os.path.lexists(path) and not observe.regular_file(path):
            raise ValueError(f"The managed run record is redirected or is not a file: {path}.")
        return process.read_record(path)

    def add_archive(self, path: str | Path) -> dict[str, Any]:
        return self.archives.add_archive(path)

    def list_archives(self) -> list[dict[str, Any]]:
        return self.archives.list_archives()

    def remove_archive(self, archive_id: str) -> bool:
        return self.archives.remove_archive(archive_id)

    def _serial_blocker(self) -> str | None:
        if any(
            process.lock_held(self._control_path(_attempt(directory, record)) / "active.lock")
            for directory, record in self._managed_records()
        ):
            return "Waiting for an active managed trainer."
        if process.cli_trainer_running():
            return "Waiting for an ordinary WSL train.py process."
        return None

    def queue_blocker(self) -> str | None:
        """Readiness hint. Launch still acquires the authoritative lease."""
        if process.lock_held(self.resource_lock):
            return "Waiting for the managed trainer resource lease."
        return self._serial_blocker()

    @contextmanager
    def execution_slot(self, *, exclusive: bool) -> Iterator[int]:
        with studio_admission(self.root):
            self.resource_lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.resource_lock.resolve() != self.resource_lock:
                raise ValueError("The managed trainer resource lock is redirected.")
            with self.resource_lock.open("ab") as slot:
                mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                try:
                    fcntl.flock(slot, mode | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RunBusyError(
                        "The managed trainer resource lease is occupied."
                    ) from error
                if exclusive and (blocker := self._serial_blocker()):
                    raise RunBusyError(blocker)
                # The detached supervisor and its trainer inherit the open file.
                # Unlocking it explicitly here would release their lease as well.
                yield slot.fileno()

    def requested_run(self, request_id: str) -> dict[str, Any] | None:
        return self.launch_status(_request_run_id(request_id))

    def launch_status(self, run_id: str) -> dict[str, Any] | None:
        """Read the current attempt and its queue owner without scanning the trainer logs."""
        try:
            directory, record = self._managed(run_id)
        except FileNotFoundError:
            return None
        attempt = _attempt(directory, record)
        return {
            "id": record["id"],
            "queueEntryId": record.get("queueEntryId"),
            "attemptId": record["attempt"],
            **_state(attempt, self._control_path(attempt)),
        }

    def _existing_start(
        self,
        directory: Path,
        record: dict[str, Any],
        text: str,
        curriculum: dict | None,
        input_precondition: dict | None,
    ) -> dict[str, Any]:
        if (observe.recipe_text(directory / "recipe.yml") != text
                or record.get("curriculum") != curriculum
                or record.get("inputPrecondition") != input_precondition):
            raise ValueError("This request ID already started a run with different launch inputs.")
        return self._managed_view(directory, record)

    def prepared_start_retry(
        self, yaml_text: str, request_id: str | None, *, input_precondition: dict,
    ) -> dict[str, Any] | None:
        """Resolve an accepted prepared request before preflight touches its sources."""
        if request_id is None:
            return None
        run_id = _request_run_id(request_id)
        if request_id.startswith(QUEUE_REQUEST_PREFIX):
            raise ValueError("This request ID prefix is reserved for queued runs.")
        checked_files(input_precondition, yaml_text)
        try:
            directory, record = self._managed(run_id)
        except FileNotFoundError:
            return None
        return self._existing_start(directory, record, yaml_text, None, input_precondition)

    def start(
        self,
        yaml_text: str,
        request_id: str | None = None,
        *,
        queue_entry_id: str | None = None,
        curriculum: dict[str, Any] | None = None,
        input_precondition: dict | None = None,
    ) -> dict[str, Any]:
        """Start a new isolated run using a preflight-validated recipe."""
        config = read_recipe(yaml_text)
        name = safe_name(config.get("name"))
        paths = config.get("path") or {}
        if paths.get("ignore_resume_networks") is not None:
            raise ValueError(
                "Clear path.ignore_resume_networks to use full managed checkpoints."
            )
        if paths.get("results_root") is not None:
            raise ValueError(
                "Clear path.results_root to keep trainer outputs inside the run."
            )

        if queue_entry_id is not None:
            if (
                not RUN_ID.fullmatch(queue_entry_id)
                or request_id != QUEUE_REQUEST_PREFIX + queue_entry_id
            ):
                raise ValueError("The queue request does not match its entry ID.")
        elif isinstance(request_id, str) and request_id.startswith(
            QUEUE_REQUEST_PREFIX
        ):
            raise ValueError("This request ID prefix is reserved for queued runs.")

        if curriculum is not None and (
            queue_entry_id is None
            or config["train"]["total_iter"] != curriculum["targetStep"]
            or bool(paths.get("resume_state")) != (curriculum["mode"] == "continue")
            or (curriculum.get("handoff") is None) != (curriculum["mode"] == "start")
        ):
            raise ValueError("The launch recipe does not match its owned curriculum phase.")

        pins = checked_files(input_precondition, yaml_text) if input_precondition is not None else None
        if pins is not None and curriculum is not None:
            raise ValueError("A curriculum phase uses its own checkpoint handoff.")
        run_id = uuid4().hex if request_id is None else _request_run_id(request_id)
        self._check_project(self.runs)
        self.runs.mkdir(mode=0o700, parents=self.project is None, exist_ok=True)
        directory = self.runs / run_id
        directory.mkdir(mode=0o700, exist_ok=request_id is not None)
        if directory.resolve() != directory:
            raise ValueError("The managed run directory is redirected.")

        with _control(self._control_path(directory)):
            self._check_project(directory)
            if previous := self._read_run(directory):
                return self._existing_start(directory, previous, yaml_text, curriculum, input_precondition)

            record = {
                "id": run_id,
                "name": name,
                "startedAt": process.timestamp(),
                "seed": config["manual_seed"] if config.get("manual_seed") is not None else 1024 + secrets.randbelow(8977),
                **_summary(config),
            }
            if queue_entry_id is not None:
                record["queueEntryId"] = queue_entry_id
            launch_text, imported = yaml_text, None
            if curriculum is not None:
                record["curriculum"] = curriculum
            elif pins is not None:
                record["inputPrecondition"] = input_precondition
                imported = prepared_import_plan(config, self.root, pins)
            elif paths.get("resume_state"):
                launch_text, imported = prepare_import(
                    yaml_text, config, self.root, self._experiment(directory, record)
                )
            if imported and paths.get("resume_state") and config.get("checkpoint_rng") and config.get("manual_seed") is None:
                from .resume_preflight import state_metadata
                source = next(Path(item["source"]) for item in imported if item["option"] == "resume_state")
                saved, warning, _version = state_metadata(source)
                if not saved or type(saved["seed"]) is not int:
                    raise ValueError(warning or "Select a training state with a recorded replay seed.")
                record["seed"] = saved["seed"]
            atomic_text(directory / "recipe.yml", yaml_text)
            return self._launch(directory, record, launch_text, imported)

    def _launch(
        self,
        directory: Path,
        record: dict[str, Any],
        yaml_text: str,
        imported: list[dict[str, Any]] | None = None,
        *,
        resumed_phase_step: int | None = None,
    ) -> dict[str, Any]:
        with self.execution_slot(
            exclusive=bool(record.get("queueEntryId"))
        ) as resource_fd:
            attempt_id = uuid4().hex
            record = {**record, "attempt": attempt_id, "launchNs": time.time_ns()}
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            experiment = self._experiment(directory, record)
            self._check_project(attempt)
            self._check_project(experiment)
            attempt.parent.mkdir(mode=0o700, exist_ok=True)
            attempt.mkdir(mode=0o700)
            control.mkdir(mode=0o700, parents=True, exist_ok=True)
            phase = record.get("curriculum")
            import_record = {"files": imported} if imported else None
            if imported and "inputPrecondition" in record:
                full_state = any(item["option"] == "resume_state" for item in imported)
                destination = experiment if full_state else attempt / "inputs"
                yaml_text = import_options(yaml_text, imported, destination, full_state=full_state)
                import_record = {"files": imported, "inputs": not full_state}
            if phase is not None and resumed_phase_step is not None:
                # This attempt continues its own checkpoint, including optimizer state.
                phase = {**phase, "startStep": resumed_phase_step}
            elif phase is not None and (handoff := phase.get("handoff")) is not None:
                weights_only = phase["mode"] == "finetune"
                destination = attempt / "inputs" if weights_only else experiment
                yaml_text = phase_options(yaml_text, phase, destination)
                import_record = {"files": handoff["files"], "inputs": weights_only}
            atomic_text(attempt / "options.yml", yaml_text)
            command = process.invocation(self.root, directory, attempt, record["seed"])
            command["env"]["TRAINNER_STUDIO_CONTROL"] = str(control)
            if self.project is not None:
                command["env"].update(self.project.environment())
            if record.get("queueEntryId"):
                command["queueEntryId"] = record["queueEntryId"]
            process.write_record(attempt / "invocation.json", command)
            if phase is not None:
                process.write_record(attempt / "curriculum.json", phase)
            if import_record:
                process.write_record(attempt / "import.json", import_record)
            with (control / "active.lock").open("ab") as lease:
                fcntl.flock(lease, fcntl.LOCK_EX)
                process.write_record(directory / "run.json", record)
                try:
                    process.launch(
                        attempt,
                        lease.fileno(),
                        experiment,
                        command,
                        resource_fd,
                        control=control,
                        state=self.state,
                    )
                except OSError as error:
                    self._check_project(attempt)
                    process.write_record(
                        attempt / "result.json",
                        {
                            "status": "failed",
                            "updatedAt": process.timestamp(),
                            "error": str(error),
                        },
                    )
        return self._managed_view(directory, record)

    @staticmethod
    def _experiment(directory: Path, record: dict[str, Any]) -> Path:
        return directory / "experiments" / safe_name(record["name"])

    def _managed(self, run_id: str) -> tuple[Path, dict[str, Any]]:
        if isinstance(run_id, str) and RUN_ID.fullmatch(run_id):
            directory = self.runs / run_id
            self._check_project(directory)
            if directory.resolve() == directory:
                if record := self._read_run(directory):
                    return directory, record
        raise FileNotFoundError("Run was not found.")

    def _writable(self, run_id: str) -> Path:
        if isinstance(run_id, str) and run_id.startswith(ARCHIVE_PREFIX):
            raise ValueError("External archive runs are read-only.")
        if isinstance(run_id, str) and run_id.startswith("cli-"):
            self._cli(run_id)
            raise ValueError("CLI experiments are read-only imports.")
        directory, _ = self._managed(run_id)
        return directory

    @staticmethod
    def _checkpoint_request(control: Path, status: str) -> dict[str, Any] | None:
        request = process.read_record(control / "checkpoint-request.json")
        if request is None:
            return None
        receipt = process.read_record(control / "checkpoint-result.json")
        if receipt is not None and receipt["id"] == request["id"]:
            return receipt
        if status not in ACTIVE:
            return {**request, "status": "failed", "error": "The attempt ended before confirming this checkpoint."}
        return {**request, "status": "pending"}

    def request_checkpoint(
        self, run_id: str, attempt_id: str, previous_request_id: str | None,
    ) -> dict[str, Any]:
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            if record["attempt"] != attempt_id:
                raise ValueError("This run has a newer attempt. Refresh before requesting a checkpoint.")
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            status = _state(attempt, control)["status"]
            latest = self._checkpoint_request(control, status)
            if latest and (latest["status"] == "pending" or previous_request_id != latest["id"]):
                # A retry or second browser that saw the previous receipt shares this save.
                return latest
            if status != "running" or not (control / "checkpoint-ready").is_file():
                raise ValueError("Wait for an active trainer that supports on-demand checkpoints.")
            request = {"id": uuid4().hex}
            process.write_record(control / "checkpoint-request.json", request)
            return {**request, "status": "pending"}

    def pause_queue_attempt(self, run_id: str, attempt_id: str) -> None:
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            if record["attempt"] != attempt_id:
                return
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            # SIGINT before the trainer installs its handler would lose its startup state.
            if (control / "checkpoint-ready").is_file() and _state(attempt, control)["status"] == "running":
                (control / "stop").touch(exist_ok=True)

    def resume_queue_attempt(self, run_id: str, attempt_id: str) -> dict[str, Any]:
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            if record["attempt"] != attempt_id:
                return self._managed_view(directory, record)
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            if process.lock_held(control / "active.lock"):
                raise RunBusyError("Waiting for the previous attempt to release the trainer.")
            state = _state(attempt, control)
            if state["status"] != "stopped" or state.get("completionReason") != "interrupted":
                raise ValueError("This attempt has no confirmed graceful stop.")
            text, same = self._resume_recipe(attempt, self._experiment(directory, record))
            step = int(Path(read_recipe(text)["path"]["resume_state"]).stem)
            if step != state.get("completedIteration"):
                raise ValueError("The checkpoint for the paused iteration is incomplete. Check the run files.")
            if not same:
                record.pop("resumeOriginalLearningRates", None)
            return self._launch(directory, record, text, resumed_phase_step=step)

    def stop(self, run_id: str, *, force_attempt: str | None = None) -> dict[str, Any]:
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            attempt = _attempt(directory, record)
            if force_attempt is not None and force_attempt != attempt.name:
                raise ValueError("The run has switched attempts. Refresh before forcing it to stop.")
            control = self._control_path(attempt)
            status = _state(attempt, control)["status"]
            if status in ACTIVE:
                if force_attempt is not None:
                    (control / "force-stop").touch(exist_ok=True)
                (control / "stop").touch(exist_ok=True)
            elif status != "stopped":
                raise ValueError("The run has no managed trainer to stop.")
            return self._managed_view(directory, record)

    @staticmethod
    def _worker_replay_count(config: dict) -> int | None:
        import msgspec
        from trainner_config.devices import replay_worker_mode
        from trainner_config.options import ReduxOptions
        from .resume_preflight import state_metadata

        if not config.get("checkpoint_rng"):
            return None
        try:
            opt = msgspec.convert(config, type=ReduxOptions, strict=True)
        except msgspec.ValidationError:
            return None
        if (mode := replay_worker_mode(opt)) is None or not opt.path.resume_state:
            return None
        saved, _, _ = state_metadata(Path(opt.path.resume_state))
        identity_key = "pairedTraining" if mode == "paired-position-v1" else "supervisedTraining"
        if (saved and saved.get("randomStreams") and saved.get("geometry") and saved.get("workerAugmentation") == mode
                and (saved.get(identity_key) or {}).get("version") == 1):
            return opt.datasets["train"].num_worker_per_gpu
        return None

    def fresh_recipe(self, text: str) -> str:
        """Detach a draft from its checkpoint, retaining independent edits and initial weights."""
        config = read_recipe(text)
        paths = config.get("path") or {}
        if not isinstance(paths, dict):
            raise ValueError("Recipe path options must be a mapping.")
        # Older queued resumes copied a numbered checkpoint into a new run. Follow
        # the saved recipe of each known owner back to the original initialization.
        layers = []
        current = paths
        visited = set()
        while current.get("resume_state"):
            state = local_path(current["resume_state"], self.root)
            layers.append((current, state))
            current = {}
            experiment = state.parent.parent
            if state.parent.name != "training_states" or not experiment.is_relative_to(self.runs):
                break
            parts = experiment.relative_to(self.runs).parts
            if len(parts) != 3 or parts[1] != "experiments" or not RUN_ID.fullmatch(parts[0]):
                break
            if parts[0] in visited:
                break
            visited.add(parts[0])
            directory, record = self._managed(parts[0])
            if self._experiment(directory, record) != experiment:
                break
            recipe = directory / "recipe.yml"
            original = observe.recipe_text(recipe) if observe.regular_file(recipe) else None
            if not original:
                raise ValueError("The original run recipe is unavailable.")
            current = read_recipe(original).get("path") or {}

        for layer, state in reversed(layers):
            current = detached_paths(layer, state, current, self.root)
        edits = []
        for field in paths:
            if field not in current:
                edits.append({"path": ["path", field], "remove": True})
            elif current[field] != paths[field]:
                edits.append({"path": ["path", field], "value": current[field]})
        train = config.get("train") or {}
        if "resume_previous_lrs" in train:
            edits.append({"path": ["train", "resume_previous_lrs"], "remove": True})
        if train.get("restart_scheduler"):
            edits.append({"path": ["train", "restart_scheduler"], "value": False})
        if config.get("auto_resume"):
            edits.append({"path": ["auto_resume"], "value": False})
        return render(text, edits) if edits else text

    def _resume_recipe(
        self,
        attempt: Path,
        experiment: Path,
        *,
        extension: tuple[dict[str, Any], int] | None = None,
    ) -> tuple[str, bool]:
        options = attempt / "options.yml"
        text = observe.recipe_text(options) if observe.regular_file(options) else None
        if text is None:
            raise ValueError("The saved run recipe is unavailable.")
        config = read_recipe(text)
        if self.project is not None:
            command = process.read_record(attempt / "invocation.json") or {}
            previous = command.get("env", {}).get(PROJECT_DIR_ENV)
            if previous and Path(previous) != self.project.root:
                from .preflight import input_paths

                source = Path(previous)
                edits = []
                for key, value in input_paths(config):
                    if not isinstance(value, str):
                        continue
                    path = Path(value)
                    if ".." not in path.parts and path.is_relative_to(source):
                        relative = path.relative_to(source)
                        edits.append({"path": key, "value": str(self.project.root / relative)})
                if edits:
                    text = render(text, edits)
                    config = read_recipe(text)
        if extension is None:
            pinned = latest_recipe(text, experiment, config)
        else:
            completion, additional_iterations = extension
            pinned = _extend_completed_recipe(
                text, experiment, config, completion, additional_iterations
            )
        selected = read_recipe(pinned)["path"]["resume_state"]
        same_checkpoint = selected == config["path"].get("resume_state")
        if not same_checkpoint and "resume_previous_lrs" in config["train"]:
            pinned = render(pinned, [{"path": ["train", "resume_previous_lrs"], "remove": True}])
        return pinned, same_checkpoint

    def queued_resume_source(
        self, run_id: str, *, additional_iterations: int | None = None
    ) -> tuple[str, str | None, dict]:
        """Snapshot a stopped run or extend a completed run from its final update."""
        if additional_iterations is not None and (
            type(additional_iterations) is not int or additional_iterations < 1
        ):
            raise ValueError("additionalIterations must be a positive integer.")
        required_status = "stopped" if additional_iterations is None else "completed"
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            if (
                process.lock_held(control / "active.lock")
                or (state := _state(attempt, control))["status"] != required_status
            ):
                raise ValueError(
                    f"Only a {required_status} training run can be queued for continuation."
                )
            if "curriculum" in record:
                raise ValueError("Retry this curriculum phase and release its new queue entry.")
            extension = (state, additional_iterations) if additional_iterations is not None else None
            text, _ = self._resume_recipe(
                attempt, self._experiment(directory, record), extension=extension
            )
            text = render(text, [{"path": ["manual_seed"], "value": record["seed"]}])
            config = read_recipe(text)
            checkpoint = {
                "sourceAttemptId": record["attempt"],
                "resumeStep": int(Path(config["path"]["resume_state"]).stem),
                "resumeFiles": resume_import_plan(config, self.root),
            }
            return text, record.get("queueEntryId"), checkpoint

    def launch_queued_continuation(self, entry: dict, text: str) -> dict[str, Any]:
        """Publish a new attempt in the run whose checkpoint the user queued."""
        run_id = entry["sourceRunId"]
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            if record.get("queueEntryId") == entry["id"]:
                return self._managed_view(directory, record)
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            required = "completed" if "additionalIterations" in entry else "stopped"
            if (attempt.name != entry["sourceAttemptId"] or process.lock_held(control / "active.lock")
                    or _state(attempt, control)["status"] != required):
                raise ValueError("The run changed since this checkpoint was queued. Open its current attempt to continue.")
            config = read_recipe(text)
            if resume_import_plan(config, self.root) != entry["resumeFiles"]:
                raise ValueError("The queued checkpoint files changed. Queue the continuation again.")
            record.update(queueEntryId=entry["id"], **_summary(config))
            if "resume_previous_lrs" not in config["train"]:
                record.pop("resumeOriginalLearningRates", None)
            return self._launch(directory, record, text)

    def resume(
        self,
        run_id: str,
        *,
        train_workers: int | None = None,
        batch_size: int | None = None,
        learning_rates: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            if process.lock_held(control / "active.lock"):
                raise ValueError(
                    "Wait for the trainer processes to exit before resuming."
                )
            if _state(attempt, control)["status"] not in {"stopped", "failed"}:
                raise ValueError("Resume requires a stopped or failed run.")
            if "curriculum" in record:
                raise ValueError(
                    "Retry this curriculum phase and release its new queue entry."
                )
            # Reuse private copies of any teacher or other persistent pretrained
            # inputs when resuming a run launched under a reproduction precondition.
            yaml_text, same_checkpoint = self._resume_recipe(attempt, self._experiment(directory, record))
            if not same_checkpoint:
                record.pop("resumeOriginalLearningRates", None)
            if train_workers is not None or batch_size is not None or learning_rates is not None:
                config = read_recipe(yaml_text)
                edits, original_rates = _resume_edits(
                    config, batch_size, learning_rates, record.get("resumeOriginalLearningRates")
                )
                if original_rates is not None:
                    # Keep missing bases distinct from rates declared by a failed attempt.
                    record["resumeOriginalLearningRates"] = original_rates
                if train_workers is not None:
                    if type(train_workers) is not int or train_workers < 1:
                        raise ValueError("trainWorkers must be a positive integer.")
                    if self._worker_replay_count(config) is None:
                        raise ValueError("Changing workers requires a checkpoint with position-seeded worker replay.")
                    edits.append({"path": ["datasets", "train", "num_worker_per_gpu"], "value": train_workers})
                if edits:
                    yaml_text = render(yaml_text, edits)
            return self._launch(directory, record, yaml_text)

    def _managed_view(
        self,
        directory: Path,
        record: dict[str, Any],
        *,
        detail: bool = False,
        monitoring: bool = False,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        attempt = _attempt(directory, record)
        experiment = self._experiment(directory, record)
        control = self._control_path(attempt)
        state = _state(attempt, control)
        result = {
            key: record[key]
            for key in ("id", "name", "startedAt", "task", "total", "queueEntryId")
            if key in record
        }
        result.update(source="managed", **state)
        result["attemptId"] = record["attempt"]
        console = attempt / "console.log"
        log = observe.engine_log(experiment, since=record["launchNs"])
        last_write = max(
            record["launchNs"], observe.mtime_ns(console), observe.mtime_ns(log)
        )
        if "updatedAt" not in result:
            result["updatedAt"] = process.timestamp(last_write / 1_000_000_000)
        result.update(observe.report(log or console, metrics=detail or monitoring))
        if state["status"] not in ACTIVE and "completedIteration" in state:
            # A stop or completion can fall between log entries. Keep telemetry at its logged step.
            result["iteration"] = state["completedIteration"]
        if detail or monitoring:
            result["canSaveCheckpoint"] = state["status"] == "running" and (control / "checkpoint-ready").is_file()
            result["checkpointRequest"] = self._checkpoint_request(control, state["status"])
            result.update(
                observe.log_page(
                    console,
                    f"{record['id']}:{record['attempt']}",
                    cursor,
                    final=state["status"] not in ACTIVE
                    and not process.lock_held(control / "active.lock"),
                )
            )
        if detail:
            if "inputPrecondition" in record:
                result["inputPrecondition"] = record["inputPrecondition"]
            if "curriculum" in record:
                result["curriculum"] = record["curriculum"]
            result["seed"] = record["seed"]
            result["yaml"] = observe.recipe_text(directory / "recipe.yml")
            options = attempt / "options.yml"
            effective = observe.recipe_text(options) if observe.regular_file(options) else None
            if self.project is not None or effective != result["yaml"]:
                result["launchedYaml"] = effective
            result["resumeWorkerOverride"] = False
            if effective and "curriculum" not in record:
                try:
                    config = read_recipe(effective)
                    result["resumeSettings"] = _resume_settings(config)
                    if (
                        state["status"] in {"stopped", "failed"}
                        and config.get("checkpoint_rng")
                        and (config.get("datasets", {}).get("train") or {}).get("num_worker_per_gpu")
                    ):
                        pinned = latest_recipe(effective, experiment, config)
                        count = self._worker_replay_count(read_recipe(pinned))
                        if count is not None:
                            result.update(resumeWorkerOverride=True, trainWorkers=count)
                except (OSError, ValueError):
                    # Details stay readable when the latest checkpoint is incomplete.
                    pass
            result["artifacts"] = observe.artifacts(experiment, record["id"])
            if command := process.read_record(attempt / "invocation.json"):
                result["invocation"] = command
        return result

    def _cli_directories(self) -> Iterator[tuple[str, Path]]:
        experiments = self.root / "experiments"
        original = MANAGED_ROOT / "experiments"
        if (
            experiments.is_symlink()
            and self.root != MANAGED_ROOT
            and is_managed_runtime(self.root)
            and experiments.readlink() == original
            and original.resolve() == original
        ):
            experiments = original
        for experiment in observe.files(experiments):
            if (
                experiment.name.startswith(".")
                or not experiment.is_dir()
                or experiment.resolve() != experiment
            ):
                continue
            run_id = (
                "cli-" + hashlib.sha256(os.fsencode(experiment.name)).hexdigest()[:32]
            )
            yield run_id, experiment

    @staticmethod
    def _cli_record(
        run_id: str, experiment: Path, *, include_recipe: bool
    ) -> tuple[str, Path, Path | None, Path | None] | None:
        log = observe.engine_log(experiment)
        recipe = (
            observe.cli_recipe(experiment) if include_recipe or log is None else None
        )
        if log is None and (recipe is None or not (experiment / "models").is_dir()):
            return None
        return run_id, experiment, log, recipe if include_recipe else None

    def _cli_experiments(self) -> Iterator[tuple[str, Path, Path | None, Path | None]]:
        for run_id, experiment in self._cli_directories():
            if entry := self._cli_record(run_id, experiment, include_recipe=True):
                yield entry

    def _cli(
        self, run_id: str, *, include_recipe: bool = True
    ) -> tuple[str, Path, Path | None, Path | None]:
        for candidate, experiment in self._cli_directories():
            if candidate == run_id:
                if entry := self._cli_record(
                    run_id, experiment, include_recipe=include_recipe
                ):
                    return entry
                break
        raise FileNotFoundError("Run was not found.")

    @staticmethod
    def _imported_view(
        run_id: str,
        experiment: Path,
        log: Path | None,
        recipe: Path | None,
        *,
        detail: bool = False,
        monitoring: bool = False,
        cursor: str | None = None,
        archive: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "id": run_id,
            "name": experiment.name,
            "source": "cli",
            "status": "imported",
        }
        if archive is not None:
            result.update(
                source="archive", sourceRootId=archive["id"], sourceRoot=archive["path"]
            )
        if modified := max(observe.mtime_ns(log), observe.mtime_ns(recipe)):
            result["updatedAt"] = process.timestamp(modified / 1_000_000_000)
        result.update(observe.report(log, metrics=detail or monitoring))
        if detail or monitoring:
            result.update(observe.log_page(log, run_id, cursor))
        if recipe is not None:
            if detail and (text := observe.recipe_text(recipe)):
                if recipe.suffix.lower() == ".toml":
                    result.update(sourceRecipeFormat="neosr-toml", toml=text, sourceRecipe=recipe.name)
                else:
                    try:
                        result.update(_summary(read_recipe(text)))
                    except ValueError:
                        pass
                    result["yaml"] = text
                    if archive is not None:
                        result["sourceRecipe"] = recipe.name
            elif not detail and recipe.suffix.lower() != ".toml":
                try:
                    version = artifact_version(recipe.stat())
                except FileNotFoundError:
                    pass
                else:
                    result.update(_imported_summary(recipe, version))
        if detail:
            result["artifacts"] = observe.artifacts(experiment, run_id)
        return result

    def _managed_records(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        self._check_project(self.runs)
        for directory in observe.files(self.runs):
            if RUN_ID.fullmatch(directory.name) and directory.resolve() == directory:
                if record := self._read_run(directory):
                    yield directory, record

    def list_runs(self) -> list[dict[str, Any]]:
        runs = [
            self._managed_view(directory, record)
            for directory, record in self._managed_records()
        ]
        runs.extend(self._imported_view(*entry) for entry in self._cli_experiments())
        runs.extend(self.archives.runs())
        return sorted(
            runs, key=lambda run: (run.get("updatedAt", ""), run["id"]), reverse=True
        )

    def detail(self, run_id: str) -> dict[str, Any]:
        if isinstance(run_id, str) and run_id.startswith(ARCHIVE_PREFIX):
            archive, experiment, log, recipe = self.archives.run(
                run_id, include_recipe=True
            )
            return self._imported_view(
                run_id, experiment, log, recipe, detail=True, archive=archive
            )
        if isinstance(run_id, str) and run_id.startswith("cli-"):
            return self._imported_view(*self._cli(run_id), detail=True)
        directory, record = self._managed(run_id)
        return self._managed_view(directory, record, detail=True)

    def monitor(self, run_id: str, cursor: str | None = None) -> dict[str, Any]:
        """Return a bounded log update and reported progress for one run."""
        if isinstance(run_id, str) and run_id.startswith(ARCHIVE_PREFIX):
            archive, experiment, log, _ = self.archives.run(
                run_id, include_recipe=False
            )
            return self._imported_view(
                run_id,
                experiment,
                log,
                None,
                monitoring=True,
                cursor=cursor,
                archive=archive,
            )
        if isinstance(run_id, str) and run_id.startswith("cli-"):
            return self._imported_view(
                *self._cli(run_id, include_recipe=False), monitoring=True, cursor=cursor
            )
        directory, record = self._managed(run_id)
        return self._managed_view(directory, record, monitoring=True, cursor=cursor)

    def progress(self, run_id: str) -> dict[str, str | int | None]:
        """Read one managed counter without loading recipes or trainer logs."""
        attempt = self.attempt_dir(run_id)
        control = self._control_path(attempt)
        state = _state(attempt, control)
        if state["status"] in ACTIVE:
            iteration = process.read_progress(control)
        else:
            step = state.get("completedIteration")
            iteration = step if type(step) is int and step >= 0 else None
        return {"attemptId": attempt.name, "iteration": iteration}

    def attempt_dir(self, run_id: str, attempt_id: str | None = None) -> Path:
        """Return an existing, unredirected directory for a managed attempt."""
        directory, record = self._managed(run_id)
        selected = record["attempt"] if attempt_id is None else attempt_id
        if not isinstance(selected, str) or RUN_ID.fullmatch(selected) is None:
            raise FileNotFoundError("Managed attempt was not found.")
        attempt = _attempt(directory, record).with_name(selected)
        if not attempt.is_dir() or attempt.resolve() != attempt:
            raise FileNotFoundError("Managed attempt was not found.")
        return attempt

    def completed_phase(
        self, run_id: str, entry_id: str, target: int
    ) -> tuple[Path, dict[str, Any], str, int]:
        """Locate a verified curriculum result without scanning its artifacts."""
        _, record = self._managed(run_id)
        attempt = self.attempt_dir(run_id, record["attempt"])
        if (
            record.get("queueEntryId") != entry_id
            or record.get("curriculum", {}).get("targetStep") != target
            or _state(attempt, self._control_path(attempt))["status"] != "completed"
        ):
            raise ValueError("The preceding phase has no verified completed run.")
        recipe = attempt / "options.yml"
        text = observe.recipe_text(recipe) if observe.regular_file(recipe) else None
        if text is None:
            raise FileNotFoundError("The preceding phase's effective recipe is unavailable.")
        return (
            self.experiment_dir(run_id),
            read_recipe(text),
            record["attempt"],
            record["seed"],
        )

    def experiment_dir(self, run_id: str) -> Path:
        """Return an existing, unredirected experiment directory for a known run."""
        if isinstance(run_id, str) and run_id.startswith(ARCHIVE_PREFIX):
            _, experiment, _, _ = self.archives.run(run_id, include_recipe=False)
        elif isinstance(run_id, str) and run_id.startswith("cli-"):
            _, experiment, _, _ = self._cli(run_id)
        else:
            directory, record = self._managed(run_id)
            experiment = self._experiment(directory, record)
        if not experiment.is_dir() or experiment.resolve() != experiment:
            raise FileNotFoundError("The run's experiment directory is unavailable.")
        return experiment

    def provenance(self, run_id: str) -> dict:
        if not RUN_ID.fullmatch(run_id):
            return {"attempts":[], "recorded":False}
        directory, record = self._managed(run_id)
        from .provenance import read_attempt
        attempts = []
        for path in sorted((directory/"attempts").iterdir()):
            if path.is_dir() and path.resolve()==path:
                evidence = read_attempt(path)
                if evidence:
                    attempts.append({"attempt":path.name,"current":path.name==record["attempt"],**evidence})
        return {"attempts":attempts,"recorded":bool(attempts)}

    def artifact(self, run_id: str, path: str) -> Path:
        return observe.artifact_path(self.experiment_dir(run_id), path)

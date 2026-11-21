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

from trainner_data.file_version import artifact_version

from . import run_observe as observe
from . import run_process as process
from .external_runs import ARCHIVE_PREFIX, ExternalRuns
from .recipes import atomic_text, safe_name, task_from_config
from .reproduction_pins import checked_files
from .run_checkpoints import (
    import_options,
    latest_recipe,
    phase_options,
    prepare_import,
    prepared_import_plan,
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
        """Resolve a launch request without rereading recipes or trainer logs."""
        try:
            directory, record = self._managed(_request_run_id(request_id))
        except FileNotFoundError:
            return None
        attempt = _attempt(directory, record)
        return {
            "id": record["id"],
            "queueEntryId": record.get("queueEntryId"),
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
            if phase is not None and (handoff := phase.get("handoff")) is not None:
                weights_only = phase["mode"] == "finetune"
                destination = attempt / "inputs" if weights_only else experiment
                yaml_text = phase_options(yaml_text, phase, destination)
                import_record = {"files": handoff["files"], "inputs": weights_only}
            atomic_text(attempt / "options.yml", yaml_text)
            command = process.invocation(self.root, directory, attempt, record["seed"])
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

    def stop(self, run_id: str) -> dict[str, Any]:
        directory = self._writable(run_id)
        with _control(self._control_path(directory)):
            _, record = self._managed(run_id)
            attempt = _attempt(directory, record)
            control = self._control_path(attempt)
            status = _state(attempt, control)["status"]
            if status in ACTIVE:
                signal_request = control / "stop"
                if not signal_request.exists():
                    signal_request.touch()
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

    def _resume_recipe(self, attempt: Path, experiment: Path) -> str:
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
        return latest_recipe(text, experiment, config)

    def resume(self, run_id: str, *, train_workers: int | None = None) -> dict[str, Any]:
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
            yaml_text = self._resume_recipe(attempt, self._experiment(directory, record))
            if train_workers is not None:
                if type(train_workers) is not int or train_workers < 1:
                    raise ValueError("trainWorkers must be a positive integer.")
                if self._worker_replay_count(read_recipe(yaml_text)) is None:
                    raise ValueError("Changing workers requires a checkpoint with position-seeded worker replay.")
                yaml_text = render(yaml_text, [{"path": ["datasets", "train", "num_worker_per_gpu"], "value": train_workers}])
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
        if state["status"] == "completed" and "completedIteration" in state:
            # Completion can fall between log entries. Keep telemetry at its logged step.
            result["iteration"] = state["completedIteration"]
        if detail or monitoring:
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
            if self.project is not None:
                options = attempt / "options.yml"
                result["launchedYaml"] = observe.recipe_text(options) if observe.regular_file(options) else None
            result["resumeWorkerOverride"] = False
            if state["status"] in {"stopped", "failed"} and "curriculum" not in record:
                effective = observe.recipe_text(attempt / "options.yml")
                try:
                    config = read_recipe(effective) if effective else {}
                    if config.get("checkpoint_rng") and (config.get("datasets", {}).get("train") or {}).get("num_worker_per_gpu"):
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

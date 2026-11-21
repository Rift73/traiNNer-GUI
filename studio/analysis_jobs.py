"""Run explicit analysis requests in isolated processes and retain their results."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from uuid import uuid4

from .desktop import _boot_id, _start_ticks
from .paths import artifact_version, local_path
from .recipes import atomic_text, task_from_config
from .run_observe import regular_file
from .run_process import lock_held, read_record, timestamp, write_record
from .yaml_edit import read_recipe

if TYPE_CHECKING:
    from .runs import RunManager

JOB_ID = re.compile("[0-9a-f]{32}")
KINDS = {"predict", "evaluate", "export", "loss-report", "pack-dataset", "weight-drift", "image-difference", "task-data", "split-leakage", "pair-registration", "split-dataset", "extract-patches", "convert-weights", "average-checkpoints", "network-parameters"}


def _integer(body, key, default, low, high):
    value = body.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}.")
    return value


def _recipe_options(config):
    import msgspec
    from trainner_config.options import ReduxOptions

    try:
        return msgspec.convert(config, type=ReduxOptions, strict=True)
    except msgspec.ValidationError as exc:
        raise ValueError(str(exc)) from exc


@contextmanager
def _admission(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "control.lock").open("ab") as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        yield


class AnalysisJobs:
    def __init__(self, root: Path, state: Path, *, gallery_root=None, runs: RunManager | None = None):
        from .runs import RunManager

        self.root = root
        self.directory = state.resolve() / "analysis"
        self.gallery_root = gallery_root
        self.runs = RunManager(root, state) if runs is None else runs

    def _folder(self, job_id: str):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise FileNotFoundError("Unknown analysis job.")
        folder = self.directory / job_id
        if folder.resolve() != folder or not folder.is_dir():
            raise FileNotFoundError("Unknown analysis job.")
        return folder

    def _inputs(self, body):
        kind = body.get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError("Choose an analysis kind: " + ", ".join(sorted(KINDS)))
        device = body.get("device", "cpu")
        if device not in ("cpu", "cuda") or device == "cuda" and kind != "evaluate":
            raise ValueError("Choose cpu for analysis, or cuda for an offline holdout evaluation.")
        if kind == "image-difference":
            from .gallery import image_path
            from .image_difference import validate_region

            params = {"kind": kind}
            expected = {}
            for side in ("left", "right", "reference"):
                if side == "reference" and side not in body:
                    continue
                selection = body.get(side)
                if isinstance(selection, dict):
                    source = set(selection) - {"path", "version"}
                    key, version = selection.get("path"), selection.get("version")
                    if (source not in ({"run"}, {"job"}, {"analysis"}) or not isinstance(key, str)
                            or not isinstance(version, str) or not version):
                        raise ValueError("A saved PNG selection needs a run, evaluation job or analysis ID, a path and its version.")
                    owner = next(iter(source))
                    if owner == "analysis":
                        path = self.artifact(selection[owner], key)
                    elif self.gallery_root is not None:
                        root, _ = self.gallery_root({owner: selection[owner]})
                        path = image_path(root, key)
                    else:
                        raise ValueError("Gallery selections require the local workbench.")
                    expected[side] = version
                else:
                    path = local_path(selection, self.root).resolve(strict=True)
                if not path.is_file() or path.suffix.lower() != ".png":
                    raise ValueError(f"Select a PNG image for {side}.")
                params[side] = str(path)
            if expected:
                params["expectedVersions"] = expected
            if body.get("region") is not None:
                validate_region(body["region"])
                params["region"] = body["region"]
            return None, params
        if kind == "average-checkpoints":
            return None, self._average_inputs(body)
        if kind == "convert-weights":
            source = local_path(body.get("weights"), self.root).resolve(strict=True)
            if not source.is_file() or source.suffix.lower() != ".pth":
                raise ValueError("Select a PyTorch .pth network checkpoint.")
            payload = body.get("payload")
            if payload is not None and (not isinstance(payload, str) or not payload.strip()):
                raise ValueError("Provide the name of a weights dictionary or omit payload.")
            return None, {"kind": kind, "weights": str(source), "payload": payload}
        if kind == "weight-drift":
            params = {"kind": kind}
            for side in ("left", "right"):
                path = local_path(body.get(side), self.root).resolve(strict=True)
                if not path.is_file() or path.suffix.lower() != ".safetensors":
                    raise ValueError(f"Select a completed .safetensors file for {side}.")
                params[side] = str(path)
            return None, params
        if kind == "pack-dataset":
            dataset = body.get("dataset")
            if not isinstance(dataset, dict):
                raise ValueError("Provide the paired source folders and scale in dataset.")
            params = {"kind": kind, "dataset": dataset}
            if "destination" in body:
                params["destination"] = str(local_path(body["destination"], self.root))
            return None, params
        text = body.get("yaml")
        config = read_recipe(text)
        params = {"kind": kind}
        if kind == "loss-report":
            from .loss_report import check_loss_report_recipe
            check_loss_report_recipe(config)
        capabilities = None
        if kind == "network-parameters":
            from trainner_config.options import initial_g_transfer
            # This action also accepts incomplete training recipes, e.g. before selecting datasets.
            options = _recipe_options(config)
            inspect_transfer = body.get("inspectTransfer", False)
            if type(inspect_transfer) is not bool:
                raise ValueError("inspectTransfer must be true or false.")
            if inspect_transfer:
                options.is_train = True
                transfer = initial_g_transfer(options)
                if transfer is None or options.auto_resume:
                    raise ValueError("A transfer preview requires fresh training with path.transfer_g and path.pretrain_network_g.")
                source = local_path(options.path.pretrain_network_g, self.root).resolve(strict=True)
                if not source.is_file() or source.suffix.lower() not in (".safetensors", ".pth"):
                    raise ValueError("Select a .safetensors or .pth generator checkpoint for the transfer.")
                params["weights"] = str(source)
            return text, params
        if kind in ("predict", "evaluate", "export"):
            from trainner_config.tasks import require_task_action
            options = _recipe_options(config)
            capabilities = require_task_action(options, "holdout" if kind == "evaluate" else kind)
        if kind == "extract-patches":
            from .dataset_patches import validate_patch_options
            from .pair_audit import pair_source_options

            split = body.get("split", "train")
            task = config.get("task") or "image_restoration"
            if task != "image_restoration" or not isinstance(split, str):
                raise ValueError("Select a paired image restoration dataset split.")
            scale, crop, stride = validate_patch_options({"scale": config.get("scale"),
                "cropSize": body.get("cropSize"), "stride": body.get("stride", body.get("cropSize"))})
            source = pair_source_options(config, split, self.root)
            io = source.get("io_backend", {"type": "disk"})
            if not isinstance(io, dict) or io.get("type") != "disk":
                raise ValueError("Patch extraction requires paired PNG folders on disk.")
            params.update(task=task, split=split, scale=scale, cropSize=crop, stride=stride)
            return text, params
        if kind == "split-dataset":
            from .split_audit import _SUPERVISED

            task = task_from_config(config)
            datasets = config.get("datasets")
            dataset = datasets.get("train") if isinstance(datasets, dict) else None
            selection = str(dataset.get("type", "")).lower() if isinstance(dataset, dict) else ""
            video = selection == "pairedvideodataset" and task == "image_restoration"
            if video:
                group = body.get("groupBy", "scene")
                if group != "scene":
                    raise ValueError("Select scene grouping to keep overlapping video windows together.")
            else:
                expected = "pairedimagedataset" if task == "image_restoration" else _SUPERVISED.get(task)
                if not expected or selection != expected:
                    raise ValueError("Select a prepared image pair, video window, class folder, mask or COCO detection dataset.")
                group = body.get("groupBy", "pair" if task == "image_restoration" else "sample")
                if group == "sample" and task == "image_restoration":
                    group = "pair"
            fraction = body.get("validationFraction")
            seed = body.get("seed", 0)
            if group not in ("sample", "pair", "parent", "scene"):
                raise ValueError("groupBy must be sample, pair, parent, or scene.")
            if type(fraction) not in (int, float) or not 0 < fraction < 1:
                raise ValueError("Choose a validationFraction between zero and one.")
            if type(seed) is not int:
                raise ValueError("Provide an integer seed.")
            params.update(task=task, datasetType=selection, groupBy=group, validationFraction=fraction, seed=seed)
            return text, params
        if kind == "pair-registration":
            from .pair_audit import pair_source_options
            split = body.get("split", "train")
            if type(config.get("scale")) is not int or config["scale"] < 1:
                raise ValueError("Use the positive integer scale for this LQ/GT data.")
            pair_source_options(config, split, self.root)
            params.update(split=split, task=config.get("task", "image_restoration"), sampleLimit=_integer(body,"sampleLimit",12,1,64),
                          radius=_integer(body,"radius",6,2,12), seed=_integer(body,"seed",0,0,2**32-1))
            return text, params
        if kind == "split-leakage":
            from .split_audit import split_selection
            params["task"] = split_selection(config)[0]
            params["groupLimit"] = _integer(body, "groupLimit", 100, 0, 200)
            return text, params
        if kind == "task-data":
            if config.get("task") not in ("semantic_segmentation", "image_classification", "object_detection", "tabular_regression"):
                raise ValueError("Choose a classification, segmentation, detection or numeric regression recipe to inspect.")
            params["task"] = config["task"]
            params["auditLimit"] = (None if "auditLimit" in body and body["auditLimit"] is None
                                    else _integer(body, "auditLimit", 100, 0, 2**53-1))
            params["sampleLimit"] = _integer(body, "sampleLimit", 25, 0, 100)
            return text, params
        for key in ("weights", "input", "discriminator"):
            if key == "input" and kind != "predict":
                continue
            if key == "discriminator" and kind != "loss-report":
                continue
            if not body.get(key) and kind == "loss-report":
                continue
            path = local_path(body.get(key), self.root).resolve(strict=True)
            if not path.is_file() and not (key == "input" and path.is_dir()):
                raise ValueError(f"Invalid {key} path.")
            params[key] = str(path)
        if kind == "evaluate":
            split = body.get("split")
            if split is not None and (not isinstance(split, str) or not split.strip()):
                raise ValueError("Choose a validation or test split, or omit split to evaluate all.")
            assert capabilities is not None
            params.update(split=split, saveImages=False)
            if device == "cuda":
                params["device"] = device
            for field in ("saveImages", "saveDetections", "saveClassifications", "saveTemporalErrors", "saveDetectionOperatingPoints"):
                if field in body:
                    flag = body[field]
                    if type(flag) is not bool or (flag and field not in capabilities.optional_exports):
                        raise ValueError(f"{field} must be a boolean and supported by this task's holdout action.")
                    params[field] = flag
        detection = config.get("task") == "object_detection"
        if kind == "predict":
            assert capabilities is not None
            params["predictionOutput"] = capabilities.output
            if capabilities.inputs == ("records",) and not Path(params["input"]).is_file():
                raise ValueError("Select a file containing the model input records.")
            if capabilities.output == "sequence" and not Path(params["input"]).is_dir():
                raise ValueError("Select a directory containing the input frame sequence.")
            if "clipSize" in body:
                if "sequences" not in capabilities.inputs:
                    raise ValueError("Clip size applies to a frame-sequence input.")
                params["clipSize"] = _integer(body, "clipSize", None, 1, 2**31-1)
            if not detection and ("scoreThreshold" in body or "topLimit" in body):
                raise ValueError("Score threshold and detection limit apply to object detection.")
            if "scoreThreshold" in body:
                score = body["scoreThreshold"]
                if type(score) not in (int, float) or not 0 <= score <= 1:
                    raise ValueError("Set the detection score threshold between 0 and 1.")
                params["scoreThreshold"] = score
            if "topLimit" in body:
                params["topLimit"] = _integer(body, "topLimit", 100, 1, 2**31-1)
        if kind == "export":
            if "dynamicSpatial" in body:
                from trainner_config.tasks import supports_dynamic_spatial_export, task_name

                dynamic = body["dynamicSpatial"]
                if type(dynamic) is not bool:
                    raise ValueError("dynamicSpatial must be a boolean.")
                if dynamic and not supports_dynamic_spatial_export(task_name(options), str((options.network_g or {}).get("type", ""))):
                    raise ValueError("Dynamic spatial export is supported for Compact and Aether image restoration recipes.")
                params["dynamicSpatial"] = dynamic
            if "clipSize" in body:
                if capabilities is None or "sequences" not in capabilities.inputs:
                    raise ValueError("A fixed clip length applies to a sequence ONNX export.")
                params["clipSize"] = _integer(body, "clipSize", None, 1, 2**31-1)
            if capabilities.inputs != ("records",):
                for key in ("height", "width"):
                    params[key] = _integer(body, key, None, 1, 32768)
            params["opset"] = _integer(body, "opset", 17, 13, 20)
        if kind == "loss-report":
            params["step"] = _integer(body, "step", 1, 1, 2**53-1)
            params["batches"] = _integer(body, "batches", 1, 1, 32)
            parameter = body.get("parameter")
            if parameter is not None and not isinstance(parameter, str):
                raise ValueError("Provide a parameter name or omit it.")
            params["parameter"] = parameter or None
        return text, params

    def _average_location(self, run_id, role):
        import msgspec
        from trainner_config.options import ReduxOptions
        from trainner_config.tasks import task_name

        if not isinstance(run_id, str) or not JOB_ID.fullmatch(run_id):
            raise ValueError("Select numbered checkpoints from a managed run.")
        if role not in ("net_g", "net_g_ema", "net_ae", "net_ae_ema", "net_d"):
            raise ValueError("Select the network role to average.")
        experiment = self.runs.experiment_dir(run_id)
        recipe = self.runs.attempt_dir(run_id) / "options.yml"
        config = msgspec.convert(read_recipe(recipe.read_text(encoding="utf-8")), type=ReduxOptions, strict=True)
        task = task_name(config)
        supervised = task in {"image_classification", "semantic_segmentation", "object_detection"}
        if supervised and role not in ("net_g", "net_g_ema"):
            raise ValueError(f"{task} saves generator weights.")
        model = experiment / "models"
        if role == "net_d" or (not role.endswith("_ema") and config.train and config.train.ema_decay > 0):
            model = model / "resume_models"
        if model.resolve() != model:
            raise FileNotFoundError("The checkpoint directory has been redirected.")
        return experiment, model, task

    def average_candidates(self, run_id, role, limit=100, before=None):
        from heapq import nlargest

        if not 1 <= limit <= 250 or before is not None and not 1 <= before <= 2**53 - 1:
            raise ValueError("Use a page size of 1 to 250 and a positive prior step.")
        experiment, model, _ = self._average_location(run_id, role)
        pattern = re.compile(re.escape(role) + r"_([1-9][0-9]{0,15})\.safetensors")

        def entries():
            if not model.is_dir():
                return
            with os.scandir(model) as files:
                for file in files:
                    match = pattern.fullmatch(file.name)
                    if not match or not file.is_file(follow_symlinks=False):
                        continue
                    step = int(match[1])
                    if step > 2**53 - 1 or before is not None and step >= before:
                        continue
                    try:
                        info = file.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    yield {"step": step, "name": file.name, "path": str(Path(file.path).relative_to(experiment)),
                           "bytes": info.st_size, "version": artifact_version(info)}

        page = nlargest(limit + 1, entries(), key=lambda entry: entry["step"])
        return {"role": role, "checkpoints": page[:limit],
                "nextBefore": page[limit - 1]["step"] if len(page) > limit else None}

    def _average_inputs(self, body):
        run_id, role, selected = body.get("run"), body.get("role"), body.get("checkpoints")
        if not isinstance(selected, list) or len(selected) < 2:
            raise ValueError("Select at least two distinct numbered checkpoints.")
        policy = body.get("batchNorm", "requireIdentical")
        if policy not in ("requireIdentical", "fromLatest"):
            raise ValueError("Choose requireIdentical or fromLatest for batchNorm.")
        _, model, task = self._average_location(run_id, role)
        inputs = []
        seen = set()
        for entry in selected:
            if not isinstance(entry, dict) or entry.keys() - {"step", "version"}:
                raise ValueError("Each checkpoint needs a step and an optional file version.")
            step = _integer(entry, "step", None, 1, 2**53 - 1)
            if step in seen:
                raise ValueError(f"Step {step} was selected twice.")
            seen.add(step)
            path = model / f"{role}_{step}.safetensors"
            if not regular_file(path) or path.resolve() != path:
                raise FileNotFoundError(f"The numbered {role} safetensors checkpoint at step {step} is unavailable.")
            version = artifact_version(path.stat())
            if "version" in entry and entry["version"] != version:
                raise FileExistsError(f"The checkpoint at step {step} changed. Select its current version.")
            inputs.append({"path": str(path), "step": step, "version": version})
        inputs.sort(key=lambda entry: entry["step"])
        return {"kind": "average-checkpoints", "run": run_id, "role": role, "task": task, "inputs": inputs,
                **({"batchNorm": policy} if policy != "requireIdentical" else {})}

    def submit(self, body):
        request = body.get("requestId")
        if request is not None and (not isinstance(request, str) or not 1 <= len(request.strip()) <= 128):
            raise ValueError("Provide a request ID of 1 to 128 characters.")
        job_id = hashlib.sha256(request.encode()).hexdigest()[:32] if request is not None else uuid4().hex
        intent = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with _admission(self.directory):
            folder = self.directory / job_id
            existing = self.detail(job_id) if folder.exists() else None
            if existing and existing.get("intentDigest") == intent:
                # The recorded job belongs to this submission, even if the
                # user has since moved its inputs or rotated a checkpoint.
                return existing
            text, params = self._inputs(body)
            if "weightsVersion" in body:
                params["weightsVersion"] = body["weightsVersion"]
            digest = hashlib.sha256(json.dumps([text, params], sort_keys=True).encode()).hexdigest()
            if existing:
                if existing["requestDigest"] != digest:
                    raise FileExistsError("That request ID already belongs to another analysis.")
                return existing
            versions = {key: artifact_version(Path(params[key]).stat())
                        for key in ("weights", "discriminator", "left", "right", "reference") if key in params}
            if "input" in params and Path(params["input"]).is_file():
                versions["input"] = artifact_version(Path(params["input"]).stat())
            for role, expected in params.get("expectedVersions", {}).items():
                if expected != versions[role]:
                    raise FileExistsError(f"The selected {role} PNG has changed. Select the current image.")
            if "weightsVersion" in body and ("weights" not in versions or body["weightsVersion"] != versions["weights"]):
                raise FileExistsError("The selected checkpoint has changed. Select its current version.")
            if versions:
                params["fileVersions"] = versions
            device = params.get("device", "cpu")
            slot = nullcontext(None)
            if device == "cuda":
                slot = self.runs.execution_slot(exclusive=True)
            with (self.directory / "worker.lock").open("ab") as capacity, slot as device_fd:
                try:
                    fcntl.flock(capacity, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise FileExistsError("An analysis job is already running in this workspace.") from exc
                with TemporaryDirectory(dir=self.directory, prefix=".new-") as temp:
                    staged = Path(temp) / "job"
                    staged.mkdir(mode=0o700)
                    if text is not None:
                        atomic_text(staged / "options.yml", text)
                    write_record(staged / "job.json", {"id": job_id, "kind": params["kind"], "device": device,
                                 "createdAt": timestamp(), "intentDigest": intent, "requestDigest": digest, "params": params})
                    staged.rename(folder)
                with (folder / "active.lock").open("ab") as active, (folder / "log.txt").open("ab") as log:
                    fcntl.flock(active, fcntl.LOCK_EX)
                    try:
                        worker = subprocess.Popen(
                            [sys.executable, "-B", "-u", "-m", "studio.analysis_jobs", str(folder), str(capacity.fileno())],
                            cwd=self.root, env=None if device == "cuda" else {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                            pass_fds=(active.fileno(), capacity.fileno()) + ((device_fd,) if device_fd is not None else ()),
                        )
                    except OSError as exc:
                        write_record(folder / "exit.json", {"status": "failed", "finishedAt": timestamp(), "error": str(exc)})
                        raise
                    Thread(target=worker.wait, daemon=True).start()
        return self.detail(job_id)

    def detail(self, job_id):
        folder = self._folder(job_id)
        header = read_record(folder / "job.json")
        if header is None:
            raise FileNotFoundError("Analysis request record is missing.")
        done = read_record(folder / "exit.json")
        if done:
            return {**header, **done}
        active = lock_held(folder / "active.lock")
        if done := read_record(folder / "exit.json"):
            return {**header, **done}
        cancelled = (folder / "cancel").is_file()
        if active:
            progress = read_record(folder / "progress.json")
            return {**header, "status": "cancelling" if cancelled else "running", "progress": progress}
        status = {"status": "cancelled"} if cancelled else {"status": "failed", "error": "Worker exited before publishing a result."}
        return {**header, **status}

    def list_jobs(self):
        if not self.directory.is_dir():
            return []
        jobs = [self.detail(p.name) for p in self.directory.iterdir()
                if JOB_ID.fullmatch(p.name) and regular_file(p / "job.json")]
        return sorted(jobs, key=lambda job: job["createdAt"], reverse=True)

    def cancel(self, job_id):
        with _admission(self.directory):
            folder = self._folder(job_id)
            if self.detail(job_id)["status"] not in {"running", "cancelling"}:
                return self.detail(job_id)
            (folder / "cancel").touch()
            # The packer owns a portable directory transaction and cooperatively
            # removes unpublished staging. After publication it reports success.
            if read_record(folder / "job.json")["kind"] in {"pack-dataset", "weight-drift", "image-difference", "task-data", "split-leakage", "pair-registration", "split-dataset", "extract-patches"}:
                return self.detail(job_id)
            identity = read_record(folder / "worker.json")
            if identity and identity["boot"] == _boot_id():
                try:
                    pid = identity["pid"]
                    fd = os.pidfd_open(pid)
                    try:
                        if _start_ticks(pid) == identity["startTicks"]:
                            signal.pidfd_send_signal(fd, signal.SIGTERM)
                    finally:
                        os.close(fd)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        return self.detail(job_id)

    def evaluation_dir(self, job_id: str) -> Path:
        folder = self._folder(job_id)
        job = self.detail(job_id)
        if job["kind"] != "evaluate" or job["status"] != "completed":
            raise ValueError("Select a completed holdout evaluation.")
        root = folder / "outputs" / "results" / "evaluation"
        if root.resolve() != root or not root.is_dir():
            raise FileNotFoundError("The evaluation results directory is missing.")
        return root

    def artifact(self, job_id, key):
        folder = self._folder(job_id)
        if self.detail(job_id)["status"] != "completed":
            raise ValueError("This job has no completed result.")
        return self._output_path(folder, key)

    @staticmethod
    def _output_path(folder, key):
        private = folder / "outputs"
        path = (private / key).resolve()
        if not key or not path.is_relative_to(private) or not regular_file(path):
            raise FileNotFoundError("Analysis output was not found.")
        return path

    def result(self, job_id, *, offset=0, limit=50):
        folder = self._folder(job_id)
        if self.detail(job_id)["status"] != "completed":
            raise ValueError("This job has no completed result.")
        if offset < 0 or not 1 <= limit <= 250:
            raise ValueError("Use a nonnegative offset and a limit from 1 to 250.")
        metadata = read_record(folder / "result.json")
        artifacts = []
        with (folder / "outputs.jsonl").open(encoding="utf-8") as stream:
            from itertools import islice
            for line in islice(stream, offset, offset + limit):
                name = json.loads(line)
                try:
                    path = self._output_path(folder, name)
                    version = artifact_version(path.stat())
                except FileNotFoundError:
                    artifacts.append({"path": name, "available": False})
                    continue
                query = {"id": job_id, "path": name, "version": version}
                artifacts.append({"path": name, "absolutePath": str(path), "available": True, "version": version,
                                  "url": "/api/analysis/artifact?" + urlencode(query)})
        end = offset + len(artifacts)
        return {"result": metadata, "artifacts": artifacts,
                "next": end if end < metadata["outputCount"] else None}

    def log(self, job_id):
        path = self._folder(job_id) / "log.txt"
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - 16384))
            return {"text": stream.read(16384).decode("utf-8", errors="replace"), "truncated": size > 16384}


class AnalysisCancelled(Exception):
    pass


def _work(folder):
    info = read_record(folder / "job.json")["params"]
    config = folder / "options.yml"
    output = folder / "outputs"
    output.mkdir()
    paths = []
    kind = info["kind"]

    def check_versions():
        for key, expected in info.get("fileVersions", {}).items():
            if artifact_version(Path(info[key]).stat()) != expected:
                raise ValueError(f"The {key} file changed after submission. Select its current version.")

    check_versions()
    last_progress = 0.0
    last_stage = None

    def progress(event):
        nonlocal last_progress, last_stage
        now = time.monotonic()
        if event["stage"] != last_stage or now - last_progress >= 0.5:
            write_record(folder / "progress.json", event)
            last_progress, last_stage = now, event["stage"]

    cancelled = (folder / "cancel").exists

    def check_cancel():
        if cancelled():
            raise AnalysisCancelled

    if kind == "image-difference":
        from .image_difference import compare_pngs, ImageDifferenceCancelled
        try:
            result = compare_pngs(info["left"], info["right"], output, reference=info.get("reference"),
                                  region=info.get("region"), versions=info["fileVersions"],
                                  progress=progress, cancelled=cancelled)
        except ImageDifferenceCancelled as exc:
            raise AnalysisCancelled from exc
        paths = list(result.pop("outputs").values())
    elif kind == "convert-weights":
        from .weight_convert import convert_weights

        source = Path(info["weights"])
        destination = output / (source.stem + ".safetensors")
        result = convert_weights(source, destination, info.get("payload"))
        paths = [destination]
    elif kind == "network-parameters":
        from .network_parameters import inspect_network
        destination = output / "network-parameters.jsonl"
        result = inspect_network(read_recipe(config.read_text(encoding="utf-8")), destination, cancel=check_cancel,
                                 transfer_source=Path(info["weights"]) if "weights" in info else None)
        paths = [destination]
        if "transfer" in result:
            paths.append(output / "transfer-receipt.json")
    elif kind == "average-checkpoints":
        from .checkpoint_average import average_checkpoints
        destination = output / "averaged.safetensors"
        result = average_checkpoints(info["inputs"], destination, run_id=info["run"], role=info["role"],
                                     task=info["task"], batch_norm=info.get("batchNorm", "requireIdentical"),
                                     cancel=check_cancel, progress=progress)
        paths = [destination]
    elif kind == "weight-drift":
        from .weight_drift import measure_drift
        report = output / "weight-drift.jsonl"
        result = measure_drift(Path(info["left"]), Path(info["right"]), report,
                               progress=progress, cancel=check_cancel)
        paths = [report]
    elif kind == "pair-registration":
        from .pair_audit import audit_pairs
        result = audit_pairs(read_recipe(config.read_text(encoding="utf-8")), Path.cwd(),
                             split=info["split"], sample_limit=info["sampleLimit"], radius=info["radius"],
                             seed=info["seed"], progress=progress, cancel=check_cancel)
    elif kind == "extract-patches":
        from .dataset_patches import DatasetPatchesCancelled, extract_paired_patches

        dataset = read_recipe(config.read_text(encoding="utf-8"))["datasets"][info["split"]]
        try:
            result = extract_paired_patches({**info, "dataset": dataset}, output / "patches",
                                            root=Path.cwd(), progress=progress, cancelled=cancelled)
        except DatasetPatchesCancelled as exc:
            raise AnalysisCancelled from exc
        paths = [Path(result[key]) for key in ("reportPath", "manifestPath", "metaInfoPath")]
    elif kind == "split-dataset":
        from .dataset_split import DatasetSplitCancelled, split_dataset
        text = config.read_text(encoding="utf-8")
        try:
            if info.get("datasetType") == "pairedvideodataset":
                from .video_split import split_video_dataset
                result = split_video_dataset({**info, "yaml": text}, output / "manifests",
                                             root=Path.cwd(), progress=progress, cancelled=cancelled)
            elif info["task"] == "image_restoration":
                dataset = read_recipe(text)["datasets"]["train"]
                result = split_dataset({**info, "dataset": dataset, "yaml": text}, output / "manifests",
                                       root=Path.cwd(), progress=progress, cancelled=cancelled)
            else:
                from .supervised_split import split_supervised_dataset

                result = split_supervised_dataset({**info, "yaml": text}, output / "manifests",
                                                  root=Path.cwd(), progress=progress, cancelled=cancelled)
        except DatasetSplitCancelled as exc:
            raise AnalysisCancelled from exc
        paths = [Path(result["reportPath"]), *(Path(p) for p in result["manifests"].values())]
        if result.get("recipePath"):
            paths.append(Path(result["recipePath"]))
    elif kind == "split-leakage":
        from .split_audit import SplitAuditCancelled, audit_split_leakage
        body = read_recipe(config.read_text(encoding="utf-8"))
        try:
            result = audit_split_leakage({**body, "groupLimit": info["groupLimit"]},
                                         root=Path.cwd(), scratch_dir=folder,
                                         progress=progress, cancelled=cancelled)
        except SplitAuditCancelled as exc:
            raise AnalysisCancelled from exc
    elif kind == "task-data":
        from .task_data import TaskDataCancelled, inspect_task_data
        body = read_recipe(config.read_text(encoding="utf-8"))
        try:
            result = inspect_task_data({**body, "auditLimit": info["auditLimit"], "sampleLimit": info["sampleLimit"]},
                                       root=Path.cwd(), progress=progress, cancelled=cancelled)
        except TaskDataCancelled as exc:
            raise AnalysisCancelled from exc
    elif kind == "pack-dataset":
        from .dataset_pack import DatasetPackCancelled, pack_dataset
        destination = Path(info.get("destination", output / "dataset"))
        try:
            result = pack_dataset(info["dataset"], destination, progress=progress, cancelled=cancelled)
        except DatasetPackCancelled as exc:
            raise AnalysisCancelled from exc
        inventory = Path(result["inventoryPath"])
        if inventory.is_relative_to(output):
            paths = [inventory]
    elif kind == "loss-report":
        from .loss_report import loss_report
        result = loss_report(config, Path.cwd(), device="cpu", step=info["step"], batches=info["batches"],
                             parameter=info["parameter"], weights=Path(info["weights"]) if info.get("weights") else None,
                             discriminator=Path(info["discriminator"]) if info.get("discriminator") else None)
    elif kind == "evaluate":
        from .evaluate import run_evaluation
        result, paths = run_evaluation(config, Path(info["weights"]), output, split=info["split"],
                                       device=info.get("device", "cpu"),
                                       save_images=info["saveImages"], save_detections=info.get("saveDetections", False),
                                       save_classifications=info.get("saveClassifications", False),
                                       save_temporal_errors=info.get("saveTemporalErrors", False),
                                       save_detection_operating_points=info.get("saveDetectionOperatingPoints", False),
                                       progress=progress)
    elif kind == "predict":
        from traiNNer.tasks.predict import predict
        source = Path(info["input"])
        task = read_recipe(config.read_text(encoding="utf-8")).get("task")
        if info.get("predictionOutput") == "json" or (
            "predictionOutput" not in info and task in ("image_classification", "object_detection")
        ):
            destination = output / "predictions.json"
        else:
            destination = output / ("images" if source.is_dir() else "prediction.png")
        result = predict(config, info["weights"], source, destination,
                         clip_size=info.get("clipSize"), score_threshold=info.get("scoreThreshold"), top_limit=info.get("topLimit"))
        paths = result.pop("outputs")
    else:
        from traiNNer.tasks.predict import export_onnx
        result = export_onnx(config, info["weights"], output / "model.onnx",
                             height=info.get("height"), width=info.get("width"), opset=info["opset"],
                             clip_size=info.get("clipSize"), dynamic_spatial=info.get("dynamicSpatial", False))
        paths = [result.pop("output")]
    check_versions()
    with (folder / "outputs.jsonl").open("x", encoding="utf-8") as manifest:
        for path in paths:
            name = Path(path).relative_to(output).as_posix()
            manifest.write(json.dumps(name) + "\n")
    write_record(folder / "result.json", {**result, "outputCount": len(paths)})


def main(folder: Path, capacity_fd: int | None = None):
    write_record(folder / "worker.json", {"pid": os.getpid(), "startTicks": _start_ticks(os.getpid()), "boot": _boot_id()})
    status = {"status": "cancelled"}
    if not (folder / "cancel").exists():
        try:
            _work(folder)
        except Exception as exc:
            if isinstance(exc, AnalysisCancelled):
                status = {"status": "cancelled"}
            else:
                import traceback
                traceback.print_exc()
                status = {"status": "failed", "error": str(exc)}
        else:
            status = {"status": "completed"}
    # NumPy/Torch shutdown can take longer than the job. A completed result
    # must allow the next submission to acquire the one-worker lease.
    if capacity_fd is not None:
        os.close(capacity_fd)
    write_record(folder / "exit.json", {**status, "finishedAt": timestamp()})


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)

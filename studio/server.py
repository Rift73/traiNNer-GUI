"""Loopback transport for the same project services used by the CLI."""
from __future__ import annotations

import errno
import json
import signal
import logging
import mimetypes
import os
import shutil
from contextlib import ExitStack
from functools import cached_property, partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .desktop import exclusive_service, stamp, forget
from .preflight import launch_text, phase_snapshot_options, preflight
from .recipes import RecipeLibrary
from .update_gate import UpdateDeferred, remember_state_base, studio_admission
from .workspace import Workspace
from .yaml_edit import render

LOG = logging.getLogger("trainner-gui")
MAX_BODY = 6 * 1024 * 1024
PHONE_ACCESS_ROUTE = "/api/phone-access"


def is_phone_access_route(route: str) -> bool:
    return route == PHONE_ACCESS_ROUTE or route.startswith(PHONE_ACCESS_ROUTE + "/")


def phone_qr_png(url: str) -> bytes:
    import cv2

    modules = cv2.QRCodeEncoder_create().encode(url)
    # Use a white quiet zone and keep module edges sharp at a readable size.
    bordered = cv2.copyMakeBorder(modules, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    image = cv2.resize(bordered, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the phone access QR code as PNG.")
    return png.tobytes()


def required(obj, key, kind=str):
    val = obj.get(key)
    if not isinstance(val, kind):
        raise ValueError(f"Provide a {kind.__name__} value for {key}.")
    return val


class Services:
    def __init__(self, root: Path, state: Path):
        self.workspace = Workspace.from_environment(state)
        self.state = self.workspace.state
        self.root = root.resolve()
        self.library = RecipeLibrary(self.root, self.state, project=self.workspace.project)
        self.ui = self.root / "ui"
        self.queue_wake = None

    @cached_property
    def runs(self):
        from .runs import RunManager
        return RunManager(self.root, self.state, project=self.workspace.project)

    @cached_property
    def queue(self):
        from .queue import QueueManager
        return QueueManager(self.root, self.state, self.runs)

    @cached_property
    def studies(self):
        from .studies import StudyManager
        return StudyManager(self.queue)

    @cached_property
    def curricula(self):
        from .curricula import CurriculumManager
        return CurriculumManager(self.queue)

    @cached_property
    def analyses(self):
        from .analysis_jobs import AnalysisJobs
        return AnalysisJobs(self.root, self.state, gallery_root=self.evaluation_root, runs=self.runs)

    def evaluation_root(self, ref):
        """Resolve only an existing managed training run or completed evaluation."""
        if isinstance(ref, str):
            return self.runs.experiment_dir(ref), False
        if isinstance(ref, dict):
            if ref.keys() == {"job"}:
                return self.analyses.evaluation_dir(required(ref, "job")), True
            if ref.keys() == {"run"}:
                return self.runs.experiment_dir(required(ref, "run")), False
        raise ValueError("Select a training run or a completed evaluation job.")

    def _queue_action(self, route, body):
        actions = {"release": self.queue.release, "hold": self.queue.hold,
                   "cancel": self.queue.cancel, "stop": self.queue.stop}
        if route == "/api/queue":
            result = self.queue.submit(required(body,"yaml"), body.get("requestId"),
                                       recipe_path=body.get("recipePath"),
                                       input_precondition=body.get("inputPrecondition"))
        elif route == "/api/queue/pause":
            result = self.queue.set_paused(required(body,"paused",bool))
        elif route == "/api/queue/replay":
            result = self.queue.replay(required(body,"runId"), body.get("requestId"))
        elif route.removeprefix("/api/queue/") in actions:
            result = actions[route.removeprefix("/api/queue/")](required(body,"id"))
        else:
            raise FileNotFoundError("Unknown queue endpoint.")
        if self.queue_wake is not None:
            self.queue_wake.set()
        return result

    def status(self):
        info = {"name":"traiNNer-GUI", "version":1, "trainerRoot":str(self.root),"stateDir":str(self.state)}
        if self.workspace.project is not None:
            info.update(self.workspace.project.status())
        return info

    def get(self, route, query):
        if route == PHONE_ACCESS_ROUTE:
            from . import tailnet

            return tailnet.phone_access(self.root, self.state)
        if route == "/api/curricula":
            return self.curricula.list_curricula()
        if route == "/api/curriculum":
            return self.curricula.detail(required(query, "id"))
        if route == "/api/run/checkpoint-storage":
            from .checkpoint_storage import checkpoint_storage
            return checkpoint_storage(self.runs, required(query, "id"))
        if route == "/api/run/average-candidates":
            return self.analyses.average_candidates(required(query, "id"), required(query, "role"),
                        int(query.get("limit", 100)), int(query["before"]) if "before" in query else None)
        if route == "/api/run/checkpoint-frontier":
            from .checkpoint_frontier import checkpoint_frontier

            available = query.get("availableOnly", "false")
            if available not in ("true", "false"):
                raise ValueError("availableOnly must be true or false.")
            return checkpoint_frontier(self.runs, required(query, "id"), required(query, "dataset"),
                                       required(query, "x"), required(query, "y"),
                                       x_direction=query.get("xDirection"), y_direction=query.get("yDirection"),
                                       available_only=available == "true", limit=int(query.get("limit", 50)))
        if route == "/api/run/checkpoint-selection":
            from .checkpoint_selection import select_checkpoints
            return select_checkpoints(self.runs, required(query, "id"), query.get("dataset"),
                                      query.get("metric"), query.get("direction"), int(query.get("limit", 50)))
        if route == "/api/studies":
            return self.studies.list_studies()
        if route == "/api/study":
            return self.studies.detail(required(query, "id"))
        if route == "/api/analysis":
            return self.analyses.list_jobs()
        if route == "/api/analysis/job":
            return self.analyses.detail(required(query, "id"))
        if route == "/api/analysis/result":
            return self.analyses.result(required(query, "id"), offset=int(query.get("offset", 0)), limit=int(query.get("limit", 50)))
        if route == "/api/analysis/log":
            return self.analyses.log(required(query, "id"))
        if route == "/api/environment":
            from .provenance import environment
            return environment(self.root,self.state)
        if route == "/api/run/provenance":
            return self.runs.provenance(required(query,"id"))
        if route == "/api/archives":
            return self.runs.archives.list_archives()
        if route == "/api/status":
            return self.status()
        if route == "/api/queue":
            return self.queue.snapshot()
        if route == "/api/queue/entry":
            return self.queue.detail(required(query, "id"))
        if route == "/api/queue/group":
            return self.queue.group(required(query, "id"))
        if route == "/api/run/replay":
            return self.queue.replay_info(required(query, "id"))
        if route == "/api/recipes":
            return self.library.saved()
        if route == "/api/recipe":
            return self.library.get(required(query,"path"))
        if route == "/api/templates":
            return self.library.templates()
        if route == "/api/catalog":
            return self.library.catalog()
        if route == "/api/schema":
            return self.library.schema()
        if route == "/api/runs":
            return self.runs.list_runs()
        if route == "/api/run/monitor":
            return self.runs.monitor(required(query,"id"),query.get("cursor"))
        if route == "/api/run/timing":
            from .run_timing import timing

            return timing(self.runs, self.state, required(query, "id"), against=query.get("against"),
                          source=query.get("source"), against_source=query.get("againstSource"),
                          from_step=int(query["from"]) if query.get("from") is not None else None,
                          to_step=int(query["to"]) if query.get("to") is not None else None)
        if route == "/api/run/history":
            from .history import history
            def integer(name, default=None):
                return int(query[name]) if name in query else default
            return history(self.runs.experiment_dir(required(query, "id")), self.state,
                           query.get("source"), query.get("metric"), integer("budget", 1000),
                           integer("from"), integer("to"))
        if route == "/api/run/training-samples":
            from .training_samples import browse
            run_id = required(query, "id")
            return browse(self.runs.experiment_dir(run_id), run_id, query)
        if route in ("/api/run/images", "/api/analysis/images"):
            from .gallery import images
            source_id = required(query, "id")
            offline = route.startswith("/api/analysis/")
            root, _ = self.evaluation_root({"job" if offline else "run": source_id})
            return images(root, source_id, query, offline=offline)
        if route in ("/api/run/validation", "/api/analysis/validation"):
            from .validation_summary import summaries
            source_type = "job" if route.startswith("/api/analysis/") else "run"
            root, _ = self.evaluation_root({source_type: required(query, "id")})
            return summaries(root, query)
        if route in ("/api/run/classes", "/api/analysis/classes"):
            from .validation_summary import classes
            required(query, "evaluation")
            source_type = "job" if route.startswith("/api/analysis/") else "run"
            root, _ = self.evaluation_root({source_type: required(query, "id")})
            return classes(root, query)
        if route in ("/api/run/evaluations", "/api/analysis/evaluations"):
            from .comparison import evaluations
            source_type = "job" if route.startswith("/api/analysis/") else "run"
            root, _ = self.evaluation_root({source_type: required(query, "id")})
            return evaluations(root)
        if route == "/api/run":
            return self.runs.detail(required(query,"id"))
        raise FileNotFoundError("Unknown API endpoint.")

    def post(self, route, body):
        with studio_admission(self.root):
            return self._post(route, body)

    def _post(self, route, body):
        if route == PHONE_ACCESS_ROUTE:
            from . import tailnet

            return tailnet.request_phone_access(self.root, self.state, required(body, "action"))
        if route == "/api/catalog/refresh":
            return self.library.refresh_catalog()
        if route == "/api/reproduction/export":
            from .reproduction import export_run_recipe
            return export_run_recipe(self.runs, required(body, "id"), attempt_id=body.get("attemptId"),
                                     mode=body.get("mode", "replay"), weight_path=body.get("weightPath"),
                                     weights_version=body.get("weightsVersion"), hash_inputs=body.get("hashInputs", False))
        if route == "/api/reproduction/prepare":
            from .reproduction import prepare_reproduction
            return prepare_reproduction(self.library, required(body, "bundle", dict), mappings=body.get("mappings"),
                                        source_root=body.get("sourceRoot"), name=body.get("name"),
                                        seed=body.get("seed"), edits=body.get("edits"))
        if route == "/api/curriculum/phase-preview":
            return phase_snapshot_options(required(body, "baseYaml"), required(body, "edits", list))
        if route in ("/api/curricula", "/api/curriculum/retry"):
            from .curricula import PhaseRejected
            try:
                if route == "/api/curricula":
                    return self.curricula.create(required(body, "name"), required(body, "baseYaml"),
                                                 required(body, "phases", list), body.get("requestId"))
                return self.curricula.retry_phase(required(body, "id"), required(body, "phaseId"),
                                                  body.get("edits"), body.get("requestId"))
            except PhaseRejected as exc:
                raise RecipeRejected(exc.issues) from exc
        if route == "/api/studies":
            from .studies import StudyRejected
            try:
                return self.studies.create(required(body, "name"), required(body, "baseYaml"),
                                           required(body, "arms", list), required(body, "seeds", list),
                                           body.get("requestId"))
            except StudyRejected as exc:
                raise RecipeRejected(exc.issues) from exc
        if route == "/api/study/compare":
            return self.studies.compare(required(body, "studyId"), required(body, "leftArmId"),
                                        required(body, "rightArmId"), required(body, "dataset"),
                                        required(body, "metric"), body.get("step"))
        if route == "/api/analysis":
            return self.analyses.submit(body)
        if route == "/api/analysis/cancel":
            return self.analyses.cancel(required(body, "id"))
        if route == "/api/video-index/status":
            from .video_indexing import video_index_status
            return video_index_status(required(body, "yaml"), self.root)
        if route == "/api/data/alignment":
            from .alignment import inspect_pair
            return inspect_pair(body,self.root)
        if route == "/api/archives":
            return self.runs.archives.add_archive(required(body,"path"))
        if route == "/api/archives/remove":
            return {"removed":self.runs.archives.remove_archive(required(body,"id"))}
        if route == "/api/weights":
            from .weights import inspect_weights
            return inspect_weights(body,self.root)
        if route == "/api/weights/compare":
            from .weights import compare_weights
            return compare_weights(body,self.root)
        if route == "/api/queue" or route.startswith("/api/queue/"):
            return self._queue_action(route,body)
        if route == "/api/compare":
            from .comparison import compare
            left, left_offline = self.evaluation_root(body.get("left"))
            right, right_offline = self.evaluation_root(body.get("right"))
            dataset, metric = required(body, "dataset"), required(body, "metric")
            result = compare(left, right, dataset, metric, body.get("leftStep"), body.get("rightStep"),
                             left_offline=left_offline, right_offline=right_offline,
                             left_evaluation=body.get("leftEvaluation"), right_evaluation=body.get("rightEvaluation"))
            examples = result.get("regressions", []) + result.get("improvements", [])
            if examples:
                from .gallery import linked_outputs
                names = [row["sample"] for row in examples]
                for side, root, offline in (("left", left, left_offline), ("right", right, right_offline)):
                    ref = body[side]
                    source = ref if isinstance(ref, str) else ref["job" if offline else "run"]
                    links = linked_outputs(root, source, offline, dataset, result[side+"Step"], names, result.get(side+"Capture"))
                    for row in examples:
                        if row["sample"] in links:
                            row[side+"ImageUrl"] = links[row["sample"]]
            return result
        if route == "/api/fs/list":
            from .files import list_folder
            return list_folder(body,self.root)
        if route == "/api/preflight":
            return preflight(required(body,"yaml"), self.library)
        if route == "/api/recipes/import-neosr":
            from .neosr import import_toml
            if "archiveRunId" in body:
                if "toml" in body:
                    raise ValueError("Select a TOML document or one archived run.")
                record = self.runs.detail(required(body, "archiveRunId"))
                if record.get("sourceRecipeFormat") != "neosr-toml":
                    raise ValueError("The archived run has no NeoSR TOML recipe.")
                result = import_toml(record["toml"], self.library, source_root=body.get("sourceRoot"), archived=True)
                return {**result, "sourceRunId": record["id"], "sourceRecipe": record["sourceRecipe"]}
            return import_toml(required(body, "toml"), self.library, source_root=body.get("sourceRoot"))
        if route == "/api/render":
            return {"yaml":render(required(body,"yaml"), required(body,"edits",list))}
        if route == "/api/recipes":
            return self.library.save(required(body,"name"), required(body,"yaml"),body.get("revision"))
        if route == "/api/data/scan":
            from .data import scan_request
            return scan_request(body)
        if route == "/api/runs":
            text = required(body,"yaml")
            condition = body.get("inputPrecondition")
            request_id = body.get("requestId")
            if condition is not None and request_id is not None:
                previous = self.runs.prepared_start_retry(text, request_id, input_precondition=condition)
                if previous is not None:
                    return previous
            check = preflight(text,self.library)
            if not check["ok"]:
                raise RecipeRejected(check["issues"])
            return self.runs.start(launch_text(text, check["config"], self.root), request_id,
                                   input_precondition=condition)
        if route == "/api/runs/checkpoint":
            previous = body.get("previousRequestId")
            if previous is not None and not isinstance(previous, str):
                raise ValueError("previousRequestId must be a checkpoint request ID or null.")
            return self.runs.request_checkpoint(required(body, "id"), required(body, "attemptId"), previous)
        if route == "/api/runs/stop":
            return self.runs.stop(required(body,"id"))
        if route == "/api/runs/resume":
            overrides = {}
            if "batchSize" in body:
                overrides["batch_size"] = required(body, "batchSize", int)
            if "learningRates" in body:
                overrides["learning_rates"] = required(body, "learningRates", dict)
            return self.runs.resume(
                required(body, "id"), train_workers=body.get("trainWorkers"), **overrides
            )
        raise FileNotFoundError("Unknown API endpoint.")


class RecipeRejected(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("Resolve the recipe errors before starting a run.")


class Handler(BaseHTTPRequestHandler):
    server_version = "traiNNer-GUI"
    sys_version = ""
    _cache_control = "no-cache"

    def __init__(self, *args, services: Services, **kwargs):
        self.services = services
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _headers(self, code, mime, size):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", self._cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()

    def _bytes(self, data, mime, code=HTTPStatus.OK):
        self._headers(code, mime, len(data))
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, value, code=HTTPStatus.OK):
        self._bytes(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",",":")).encode("utf-8",errors="backslashreplace"),
                    "application/json; charset=utf-8", code)

    def _file(self, file, version=None):
        if not file.is_file():
            raise FileNotFoundError("File does not exist.")
        with file.open("rb") as stream:
            from .paths import artifact_version
            info = os.fstat(stream.fileno())
            if version and artifact_version(info) != version:
                raise FileExistsError("The recorded output file has been replaced.")
            self._headers(HTTPStatus.OK, mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                          info.st_size)
            if self.command != "HEAD":
                shutil.copyfileobj(stream, self.wfile, length=128*1024)

    def _authorize(self):
        port = self.server.server_address[1]
        hosts = {f"localhost:{port}",f"127.0.0.1:{port}"}
        if port == 80:
            hosts.update(("localhost","127.0.0.1"))
        host = self.headers.get("Host","").lower()
        if host not in hosts:
            raise PermissionError("This service accepts local connections only.")
        origin = self.headers.get("Origin")
        if origin and origin.lower() != "http://" + host:
            raise PermissionError("The request origin differs from the local service.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("Cross-site access is disabled.")

    def _phone_access_qr(self) -> None:
        info = self.services.get(PHONE_ACCESS_ROUTE, {})
        if info.get("state") != "ready":
            raise FileNotFoundError("Phone access is not ready.")

        from .tailnet_cli import TailnetError, _dns_name

        url = info.get("url")
        invalid_url = "Phone access reported an invalid ready URL."
        if not isinstance(url, str):
            raise RuntimeError(invalid_url)
        try:
            target = urlsplit(url)
            host = _dns_name(target.hostname)
            port = target.port
        except (TailnetError, ValueError) as exc:
            raise RuntimeError(invalid_url) from exc
        authority = host if port is None else f"{host}:{port}"
        if (
            not host.endswith(".ts.net") or port == 0
            or url not in (f"https://{authority}/", f"https://{authority}")
        ):
            raise RuntimeError(invalid_url)

        self._bytes(phone_qr_png(url), "image/png")

    def _dispatch(self):
        request = urlsplit(self.path)
        route = unquote(request.path)
        self._cache_control = "no-store" if is_phone_access_route(route) else "no-cache"
        self._authorize()
        query = {k:v[-1] for k,v in parse_qs(request.query, max_num_fields=32).items()}
        if self.command == "POST":
            if not route.startswith("/api/"):
                raise FileNotFoundError("Unknown API endpoint.")
            if self.headers.get_content_type() != "application/json":
                self._json({"error":"Use application/json."}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            length = int(self.headers.get("Content-Length", "-1"))
            limit = 4 * MAX_BODY if route == "/api/reproduction/prepare" else MAX_BODY
            if length < 0 or length > limit:
                self._json({"error":"Invalid or excessive request length."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Provide a JSON object.")
            if route == "/api/data/preview":
                from .data import preview
                path = required(body, "path")
                size = body.get("maxSize", 1536)
                if "key" in body:
                    from .data import preview_lmdb
                    mime, data = preview_lmdb(path, required(body, "key"), size)
                else:
                    mime, data = preview(path, size)
                self._bytes(data,mime)
                return
            self._json(self.services.post(route,body))
            return
        if route == PHONE_ACCESS_ROUTE + "/qr.png":
            self._phone_access_qr()
            return
        if route == "/api/analysis/artifact":
            self._file(self.services.analyses.artifact(required(query, "id"), required(query, "path")),
                       query.get("version"))
            return
        if route in ("/api/run/image", "/api/analysis/image"):
            from .gallery import image_path
            source_type = "job" if route.startswith("/api/analysis/") else "run"
            root, _ = self.services.evaluation_root({source_type: required(query, "id")})
            path = image_path(root, required(query,"path"))
            if "maxSize" in query:
                from .data import preview
                mime, data = preview(path, int(query["maxSize"]), version=query.get("version"))
                self._bytes(data,mime)
            else:
                self._file(path, query.get("version"))
            return
        if route in ("/api/run/validation-source", "/api/analysis/validation-source"):
            from .gallery import source_preview
            source_type = "job" if route.startswith("/api/analysis/") else "run"
            root, _ = self.services.evaluation_root({source_type: required(query, "id")})
            mime, data = source_preview(root, query)
            self._bytes(data,mime)
            return
        if route == "/api/run/training-sample":
            from .training_samples import artifact
            root = self.services.runs.experiment_dir(required(query, "id"))
            self._file(artifact(root, required(query, "sample"), required(query, "file")))
            return
        if route == "/api/run/artifact":
            self._file(self.services.runs.artifact(required(query, "id"), required(query, "path")))
            return
        if route.startswith("/api/"):
            self._json(self.services.get(route,query))
            return
        asset = (self.services.ui / (route.strip("/") or "index.html")).resolve()
        if not asset.is_relative_to(self.services.ui):
            raise PermissionError("Invalid asset path.")
        self._file(asset)

    def do_GET(self):
        try:
            self._dispatch()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        except RecipeRejected as exc:
            self._json({"error":str(exc),"issues":exc.issues}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except FileNotFoundError as exc:
            self._json({"error":str(exc)}, HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._json({"error":str(exc)}, HTTPStatus.FORBIDDEN)
        except (FileExistsError, UpdateDeferred) as exc:
            self._json({"error":str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, UnicodeError) as exc:
            self._json({"error":str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            LOG.exception("Unhandled request failure on %s", urlsplit(self.path).path)
            self._json({"error":"The local service encountered an error. See its log for details."},
                       HTTPStatus.INTERNAL_SERVER_ERROR)

    do_POST = do_GET
    do_HEAD = do_GET

    def log_message(self, format, *args):
        LOG.debug(format,*args)


def serve(root: Path, state: Path, port: int, desktop_fd: int | None = None) -> None:
    from .dispatch import QueueDispatcher

    def stop(signum, frame):
        raise SystemExit(0)

    # Service ownership lasts until shutdown. Admission lasts only through
    # startup, so the updater can quiesce a server that is waiting for work.
    root = root.resolve()
    with ExitStack() as service:
        with studio_admission(root) as managed:
            if managed:
                remember_state_base(state.expanduser().resolve())
            services = Services(root, state)
            root, state = services.root, services.state
            service.enter_context(exclusive_service(state))
            handler = partial(Handler, services=services)
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            except OSError as exc:
                if desktop_fd is None or exc.errno != errno.EADDRINUSE:
                    raise
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            service.enter_context(server)
            previous = signal.signal(signal.SIGTERM, stop)
            service.callback(signal.signal, signal.SIGTERM, previous)
            stamp(state, server.server_port, root, project=services.workspace.project)
            service.callback(forget, state)
            dispatch = service.enter_context(QueueDispatcher(state, lambda: services.queue))
            services.queue_wake = dispatch.wake
            if desktop_fd is None:
                from .tailnet import auto_start
                auto_start(root, state)
            if desktop_fd is not None:
                with os.fdopen(desktop_fd, "w") as pipe:
                    print(server.server_port, file=pipe)
            print(f"traiNNer-GUI  http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

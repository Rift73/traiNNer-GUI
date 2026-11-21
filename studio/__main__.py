"""Launch and use traiNNer-GUI without a browser."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .desktop import DEFAULT_PORT, desktop_url
from .paths import local_path
from .server import RecipeRejected, Services, serve
from .update_gate import remember_state_base, studio_admission
from .workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


def state_directory(root: Path = ROOT) -> Path:
    override = os.environ.get("TRAINNER_GUI_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    from .tailnet import default_state
    return default_state(root)


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="trainner-gui", description="Local traiNNer-GUI training harness")
    main.add_argument("--state-dir", type=Path, default=state_directory(),
                      help="Studio state directory. Windows projects keep their runtime state under this directory.")
    commands = main.add_subparsers(dest="command",required=True)
    server = commands.add_parser("serve",help="Serve the local application.")
    server.add_argument("--port",type=int,default=DEFAULT_PORT)
    server.add_argument("--desktop-fd", type=int, help=argparse.SUPPRESS)
    commands.add_parser("close", help="Close this local service. Managed trainers continue independently.")
    commands.add_parser("project-info", help="Verify the selected Windows project and print its UUID.")
    url = commands.add_parser("url", help="Start or reuse the local application and print its URL.")
    url.add_argument("--hold-wsl", action="store_true", help="Keep one Windows client connected while this service runs.")
    phone = commands.add_parser("tailscale", help="Show or control private phone access through Tailscale.")
    phone.add_argument("action", choices=("status", "on", "off"), nargs="?", default="status")
    for name,help_text in (("validate","Check a training recipe without initializing CUDA."),
                           ("start","Launch a training recipe as a managed run.")):
        command = commands.add_parser(name,help=help_text)
        command.add_argument("recipe")
    neo = commands.add_parser("import-neosr", help="Translate a NeoSR TOML into an editable YAML draft.")
    pick = neo.add_mutually_exclusive_group(required=True)
    pick.add_argument("toml", nargs="?")
    pick.add_argument("--archive-run", help="Import the saved NeoSR TOML from a registered experiment archive.")
    neo.add_argument("--source-root", help="Original working directory for relative training inputs.")
    neo.add_argument("--out", help="Write a new YAML file. Default prints the review and draft as JSON.")
    commands.add_parser("prepare",help="Build the local CPU tools with LLVM before opening a dataset or starting a run.")
    commands.add_parser("environment",help="Inspect the installed trainer runtime and host.")
    commands.add_parser("provenance",help="Show the recipe, source and runtime evidence from each attempt.").add_argument("id")
    reproduction = commands.add_parser("reproduce", help="Export a recorded run or prepare it on this machine.")
    repro = reproduction.add_subparsers(dest="action", required=True)
    transfer = repro.add_parser("export", help="Write the recipe bundle as JSON to standard output.")
    transfer.add_argument("id")
    transfer.add_argument("--attempt-id")
    transfer.add_argument("--mode", choices=("replay","weights-only"), default="replay")
    transfer.add_argument("--weight-path")
    transfer.add_argument("--weights-version")
    transfer.add_argument("--hash-inputs", action="store_true", help="Include portable SHA-256 identities of checkpoint inputs.")
    relocate = repro.add_parser("prepare", help="Check a bundle and return the candidate YAML and compatibility report.")
    relocate.add_argument("bundle")
    relocate.add_argument("--map", dest="mappings", action="append", nargs=2, metavar=("FROM","TO"), default=[])
    relocate.add_argument("--source-root")
    for action in ("start", "queue"):
        launch = repro.add_parser(action, help="Use a saved preparation with its observed checkpoint versions.")
        launch.add_argument("prepared", help="JSON output from reproduce prepare.")
        launch.add_argument("--request-id")
    relocate.add_argument("--name")
    relocate.add_argument("--seed", type=int)
    relocate.add_argument("--edits", help="A JSON list of recipe edits for weights-only initialization.")
    catalog = commands.add_parser("catalog",help="List installed training components and their configuration options.")
    catalog.add_argument("--refresh", action="store_true", help="Rediscover this checkout’s factories in an isolated CPU process.")
    commands.add_parser("templates",help="List the shipped training recipes.")
    commands.add_parser("runs",help="List managed runs and discovered experiments.")
    for name in ("show","stop","resume"):
        command = commands.add_parser(name)
        command.add_argument("id")
        if name == "resume":
            command.add_argument("--train-workers", type=int, help="Positive worker count for position-seeded replay.")
    queue = commands.add_parser("queue", help="Manage serial, resumable experiment submissions.")
    q = queue.add_subparsers(dest="action", required=True)
    q.add_parser("list")
    submit = q.add_parser("add", help="Save a recipe as a held entry.")
    submit.add_argument("recipe")
    submit.add_argument("--request-id")
    replay = q.add_parser("replay", help="Save a held entry from an existing run.")
    replay.add_argument("id")
    replay.add_argument("--request-id")
    for action in ("show","group","release","hold","cancel","stop"):
        q.add_parser(action).add_argument("id")
    for action in ("pause","unpause"):
        q.add_parser(action)
    study = commands.add_parser("study", help="Set up controlled experiments with matched seeds.")
    studies = study.add_subparsers(dest="action", required=True)
    studies.add_parser("list")
    studies.add_parser("show").add_argument("id")
    create = studies.add_parser("create", help="Create a held study from its JSON plan and base recipe.")
    create.add_argument("spec", help="JSON with name, arms, seeds, and optional requestId.")
    create.add_argument("base_recipe")
    compare = studies.add_parser("compare", help="Compare two study arms at matched seeds and validation iterations.")
    for key in ("id", "left", "right", "dataset", "metric"):
        compare.add_argument(key)
    compare.add_argument("--step", type=int)
    curriculum = commands.add_parser("curriculum", help="Prepare linked training phases.")
    phases = curriculum.add_subparsers(dest="action", required=True)
    phases.add_parser("list")
    phases.add_parser("show").add_argument("id")
    prepare = phases.add_parser("create", help="Create a held curriculum from a JSON plan and a base recipe.")
    prepare.add_argument("spec")
    prepare.add_argument("base_recipe")
    revise = phases.add_parser("retry", help="Prepare a new held revision of a phase.")
    revise.add_argument("id")
    revise.add_argument("phase_id")
    revise.add_argument("--edits", help="JSON file with revised recipe edits.")
    revise.add_argument("--request-id")
    storage = commands.add_parser("checkpoint-storage", help="Report observed checkpoint sizes and projected disk headroom.")
    storage.add_argument("id")
    frontier = commands.add_parser("frontier", help="Find checkpoints on the trade-off between two validation metrics.")
    frontier.add_argument("id")
    frontier.add_argument("--dataset", required=True)
    frontier.add_argument("--x", required=True)
    frontier.add_argument("--y", required=True)
    frontier.add_argument("--x-direction", choices=("higher", "lower"))
    frontier.add_argument("--y-direction", choices=("higher", "lower"))
    frontier.add_argument("--available-only", action="store_true", help="Use only weights verified against the evaluation receipt.")
    frontier.add_argument("--limit", type=int, default=50)
    selection = commands.add_parser("checkpoints", help="Rank observed validations and available numbered checkpoints.")
    selection.add_argument("id")
    selection.add_argument("--dataset")
    selection.add_argument("--metric")
    selection.add_argument("--direction", choices=("higher", "lower"), help="Declare direction when archive evidence is missing.")
    selection.add_argument("--limit", type=int, default=50)
    parameters = commands.add_parser("network-parameters", help="Inspect network shapes and fine-tuning filters before training.")
    parameters.add_argument("recipe")
    parameters.add_argument("--inspect-transfer", action="store_true",
                            help="Apply the recipe's initial generator transfer to an isolated CPU copy and report which tensors it loaded.")
    parameters.add_argument("--request-id")
    average = commands.add_parser("average-checkpoints", help="Average numbered safetensors weights from one managed run.")
    average.add_argument("id")
    average.add_argument("role", choices=("net_g", "net_g_ema", "net_ae", "net_ae_ema", "net_d"))
    average.add_argument("steps", type=int, nargs="+")
    average.add_argument("--batch-norm", choices=("requireIdentical", "fromLatest"), default="requireIdentical")
    average.add_argument("--request-id")
    convert = commands.add_parser("convert-weights", help="Save the selected .pth network as safetensors.")
    convert.add_argument("weights")
    convert.add_argument("--payload", help="Network dictionary, e.g. params_ema, when multiple are present.")
    convert.add_argument("--request-id")
    patches = commands.add_parser("extract-patches", help="Extract registered PNG pairs and a trainable patch manifest.")
    patches.add_argument("recipe")
    patches.add_argument("--crop-size", type=int, required=True, help="Square patch size in LQ pixels.")
    patches.add_argument("--stride", type=int, help="Distance in LQ pixels. Defaults to crop size.")
    patches.add_argument("--split", default="train")
    patches.add_argument("--request-id")
    split_data = commands.add_parser("split-data", help="Prepare repeatable train and validation manifests from paired data.")
    split_data.add_argument("recipe")
    split_data.add_argument("--validation-fraction", required=True, type=float)
    split_data.add_argument("--group-by", choices=("sample", "pair", "parent", "scene"),
                            help="Defaults to whole scenes for video, pairs for images, and samples for supervised data.")
    split_data.add_argument("--seed", type=int, default=0)
    split_data.add_argument("--request-id")
    leakage = commands.add_parser("leakage", help="Compare training and validation selections for identical encoded data.")
    leakage.add_argument("recipe")
    leakage.add_argument("--group-limit", type=int, default=100)
    leakage.add_argument("--request-id")
    registration = commands.add_parser("registration", help="Sample actual LQ/GT pairs and measure luma and chroma displacement.")
    registration.add_argument("recipe")
    registration.add_argument("--split", default="train")
    registration.add_argument("--limit", type=int, default=12)
    registration.add_argument("--radius", type=int, default=6)
    registration.add_argument("--seed", type=int, default=0)
    registration.add_argument("--request-id")
    evaluate = commands.add_parser("evaluate", help="Score a checkpoint on validation or test data on CPU or CUDA.")
    evaluate.add_argument("recipe")
    evaluate.add_argument("weights")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--split", help="Evaluate one named dataset section.")
    evaluate.add_argument("--save-images", action="store_true")
    evaluate.add_argument("--save-temporal-errors", action="store_true",
                          help="Measure changes in restoration error between adjacent, paired video frames.")
    evaluate.add_argument("--save-detections", action="store_true", help="Export each detection split in COCO bbox results format.")
    evaluate.add_argument("--save-detection-operating-points", action="store_true",
                          help="Report precision, recall and F1 across detector confidence thresholds during the same holdout.")
    evaluate.add_argument("--save-classifications", action="store_true",
                          help="Export labeled top-k probabilities for every classified holdout image.")
    evaluate.add_argument("--request-id")
    audit = commands.add_parser("audit", help="Inspect dataset labels and geometry using the recipe's task settings.")
    audit.add_argument("recipe")
    count = audit.add_mutually_exclusive_group()
    count.add_argument("--limit", type=int, default=100, help="Audit this many samples per split.")
    count.add_argument("--all", action="store_true", help="Audit every sample.")
    audit.add_argument("--request-id")
    jobs = commands.add_parser("analysis", help="Manage CPU prediction, export, loss analysis, and dataset preparation.")
    actions = jobs.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    actions.add_parser("submit", help="Submit a JSON job specification.").add_argument("spec")
    for action in ("show", "result", "cancel", "log"):
        actions.add_parser(action).add_argument("id")
    pack = commands.add_parser("pack", help="Prepare a portable paired LMDB dataset from original image bytes.")
    pack.add_argument("destination", help="A new directory to publish the complete dataset into.")
    pack.add_argument("--gt", nargs="+", required=True)
    pack.add_argument("--lq", nargs="+", required=True)
    pack.add_argument("--scale", type=int, required=True)
    pack.add_argument("--filename-template", default="{}")
    pack.add_argument("--gt-patch-size", type=int)
    pack.add_argument("--phase", choices=("train","val","test"), default="train")
    pack.add_argument("--request-id")
    report = commands.add_parser("loss-report", help="Measure per-term generator gradients without stepping an optimizer.")
    report.add_argument("recipe")
    report.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    report.add_argument("--batches", type=int, default=1)
    report.add_argument("--step", type=int, default=1)
    report.add_argument("--parameter", help="Observe one named generator parameter. Defaults to the output image.")
    report.add_argument("--weights", help="Use these generator weights.")
    report.add_argument("--discriminator", help="Use these discriminator weights.")
    archives = commands.add_parser("archives",help="Browse existing experiment folders read-only.")
    archive = archives.add_subparsers(dest="action",required=True)
    archive.add_parser("list")
    archive.add_parser("add").add_argument("folder")
    archive.add_parser("remove").add_argument("id")
    align = commands.add_parser("alignment",help="Measure possible spatial shifts between a GT/LQ image pair.")
    align.add_argument("gt")
    align.add_argument("lq")
    align.add_argument("--radius",type=int,default=6)
    align.add_argument("--scale",type=int)
    delta = commands.add_parser("image-difference", help="Compare two PNG images, optionally against a common reference.")
    delta.add_argument("left")
    delta.add_argument("right")
    delta.add_argument("--reference")
    delta.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    delta.add_argument("--request-id")
    ckpt = commands.add_parser("weights",help="Inspect or compare safetensors keys, shapes and dtypes.")
    ckpt.add_argument("path")
    ckpt.add_argument("--compare",metavar="OTHER")
    ckpt.add_argument("--filter",default="")
    ckpt.add_argument("--after",default="")
    ckpt.add_argument("--limit",type=int,default=100)
    timing = commands.add_parser("timing", help="Summarize actual logged training rates, data wait and memory.")
    timing.add_argument("id")
    timing.add_argument("--against", help="Compare recent rates at matching logged iterations in this run.")
    timing.add_argument("--source", help="Select a trainer log from the base run.")
    timing.add_argument("--against-source", help="Select a trainer log from the comparison run.")
    timing.add_argument("--from-step", type=int)
    timing.add_argument("--to-step", type=int)
    plot = commands.add_parser("history", help="Read logged scalars and iteration history.")
    plot.add_argument("id")
    plot.add_argument("--metric")
    plot.add_argument("--source")
    plot.add_argument("--budget", type=int, default=1000)
    reports = commands.add_parser("evaluations", help="List iterations with recorded per-image validation metrics.")
    reports.add_argument("id", help="Training run ID or job:ID for a standalone evaluation.")
    captures = commands.add_parser("training-samples", help="Browse tensors observed after feeding a training microbatch.")
    captures.add_argument("id")
    captures.add_argument("--after")
    captures.add_argument("--limit", type=int, default=50)
    gallery = commands.add_parser("images", help="List validation exports and their recorded source pairs.")
    gallery.add_argument("id")
    gallery.add_argument("--iteration", type=int)
    gallery.add_argument("--dataset")
    gallery.add_argument("--image")
    gallery.add_argument("--after")
    gallery.add_argument("--limit", type=int, default=100)
    comparison = commands.add_parser("compare", help="Compare recorded model scores on matched validation images.")
    for arg in ("left", "right", "dataset", "metric"):
        comparison.add_argument(arg, help="Training run ID or job:ID." if arg in ("left", "right") else None)
    comparison.add_argument("--left-step", type=int)
    comparison.add_argument("--right-step", type=int)
    comparison.add_argument("--left-evaluation", type=int, help="Pooled evaluation ID for a repeated training iteration.")
    comparison.add_argument("--right-evaluation", type=int)
    scan = commands.add_parser("scan",help="Inspect real image data and pair dimensions.")
    scan.add_argument("gt")
    scan.add_argument("--lq")
    scan.add_argument("--scale",type=int)
    return main


def main() -> int:
    args = parser().parse_args()
    state = args.state_dir.expanduser().resolve()
    if args.command == "serve":
        if not 0 <= args.port <= 65535:
            raise ValueError("Port must be between 0 and 65535.")
        serve(ROOT, state, args.port, args.desktop_fd)
        return 0
    with studio_admission(ROOT) as managed:
        if managed and args.command != "project-info":
            remember_state_base(state)
        if args.command != "url":
            return _command(args, state)

        from .tailnet import auto_start

        workspace = Workspace.from_environment(state)
        url = desktop_url(ROOT, state, workspace=workspace)
        auto_start(ROOT, workspace.state)
    if args.hold_wsl:
        from .desktop import hold_windows_client

        hold_windows_client(url, ROOT, workspace.state)
    else:
        print(url)
    return 0


def _command(args: argparse.Namespace, state: Path) -> int:
    if args.command in ("close", "tailscale", "project-info"):
        workspace = Workspace.from_environment(state)
        if args.command == "project-info":
            if workspace.project is None:
                raise ValueError("Select a Windows project first.")
            print(workspace.project.id)
            return 0
        if args.command == "close":
            from .desktop import stop_service

            print(json.dumps({"closed":stop_service(ROOT, workspace.state)}))
            return 0
        from .tailnet import configure, status

        if args.action == "on":
            desktop_url(ROOT, state, workspace=workspace)
        result = status(workspace.state) if args.action == "status" else configure(ROOT, workspace.state, args.action)
        print(json.dumps(result))
        return int((args.action == "on" and result["state"] != "ready") or result["state"] == "cleanup-failed")
    backend=Services(ROOT,state)
    if args.command == "prepare":
        from .native import prepare
        data = prepare()
    elif args.command == "queue":
        if args.action == "list":
            data = backend.queue.snapshot()
        elif args.action == "show":
            data = backend.queue.detail(args.id)
        elif args.action == "group":
            data = backend.queue.group(args.id)
        elif args.action == "add":
            text = local_path(args.recipe,Path.cwd()).read_text(encoding="utf-8-sig")
            data = backend.queue.submit(text,args.request_id)
        elif args.action == "replay":
            data = backend.queue.replay(args.id,args.request_id)
        elif args.action in ("pause","unpause"):
            if args.action == "unpause":
                desktop_url(ROOT, state, workspace=backend.workspace)
            data = backend.queue.set_paused(args.action == "pause")
        else:
            if args.action == "release":
                desktop_url(ROOT, state, workspace=backend.workspace)
            data = getattr(backend.queue,args.action)(args.id)
    elif args.command == "import-neosr":
        payload = {"sourceRoot": args.source_root}
        if args.archive_run:
            payload["archiveRunId"] = args.archive_run
        else:
            payload["toml"] = local_path(args.toml, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/recipes/import-neosr", payload)
        if args.out:
            out = local_path(args.out, Path.cwd())
            with out.open("x", encoding="utf-8") as draft:
                draft.write(data.pop("yaml"))
            data["path"] = str(out)
    elif args.command == "curriculum":
        if args.action == "list":
            data = backend.get("/api/curricula", {})
        elif args.action == "show":
            data = backend.get("/api/curriculum", {"id": args.id})
        elif args.action == "retry":
            edits = json.loads(local_path(args.edits, Path.cwd()).read_bytes()) if args.edits else None
            data = backend.post("/api/curriculum/retry", {"id": args.id, "phaseId": args.phase_id,
                                                        "edits": edits, "requestId": args.request_id})
        else:
            spec = json.loads(local_path(args.spec, Path.cwd()).read_bytes())
            if not isinstance(spec, dict):
                raise ValueError("Provide a curriculum plan as a JSON object.")
            text = local_path(args.base_recipe, Path.cwd()).read_text(encoding="utf-8-sig")
            data = backend.post("/api/curricula", {**spec, "baseYaml": text})
    elif args.command == "reproduce":
        if args.action == "export":
            data = backend.post("/api/reproduction/export", {"id":args.id, "attemptId":args.attempt_id,
                                "mode":args.mode, "weightPath":args.weight_path, "weightsVersion":args.weights_version,
                                "hashInputs":args.hash_inputs})
        elif args.action in ("queue", "start"):
            prepared = json.loads(local_path(args.prepared, Path.cwd()).read_bytes())
            if (not isinstance(prepared, dict) or prepared.get("ok") is not True
                    or not isinstance(prepared.get("yaml"), str) or "inputPrecondition" not in prepared):
                raise ValueError("Select a successful preparation report containing the YAML and checkpoint observations.")
            route = "/api/queue" if args.action == "queue" else "/api/runs"
            data = backend.post(route, {"yaml": prepared["yaml"], "inputPrecondition": prepared["inputPrecondition"],
                                        "requestId": args.request_id})
        else:
            bundle = json.loads(local_path(args.bundle, Path.cwd()).read_bytes())
            edits = json.loads(local_path(args.edits, Path.cwd()).read_bytes()) if args.edits else None
            data = backend.post("/api/reproduction/prepare", {"bundle":bundle, "name":args.name, "seed":args.seed,
                                "sourceRoot":args.source_root, "edits":edits,
                                "mappings":[{"from":old,"to":new} for old,new in args.mappings]})
    elif args.command == "checkpoint-storage":
        data = backend.get("/api/run/checkpoint-storage", {"id": args.id})
    elif args.command == "frontier":
        data = backend.get("/api/run/checkpoint-frontier", {"id":args.id, "dataset":args.dataset,
            "x":args.x, "y":args.y, "xDirection":args.x_direction, "yDirection":args.y_direction,
            "availableOnly":"true" if args.available_only else "false", "limit":args.limit})
    elif args.command == "checkpoints":
        data = backend.get("/api/run/checkpoint-selection", {
            "id": args.id, "dataset": args.dataset, "metric": args.metric,
            "direction": args.direction, "limit": args.limit})
    elif args.command == "registration":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind":"pair-registration", "yaml":text, "split":args.split,
                            "sampleLimit":args.limit, "radius":args.radius, "seed":args.seed, "requestId":args.request_id})
    elif args.command == "network-parameters":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind": "network-parameters", "yaml": text,
                            **({"inspectTransfer": True} if args.inspect_transfer else {}), "requestId": args.request_id})
    elif args.command == "average-checkpoints":
        data = backend.post("/api/analysis", {"kind": "average-checkpoints", "run": args.id,
                            "role": args.role, "batchNorm": args.batch_norm, "checkpoints": [{"step": step} for step in args.steps],
                            "requestId": args.request_id})
    elif args.command == "convert-weights":
        data = backend.post("/api/analysis", {"kind": "convert-weights", "weights": str(local_path(args.weights, Path.cwd())),
                            "payload": args.payload, "requestId": args.request_id})
    elif args.command == "extract-patches":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind":"extract-patches", "yaml":text, "split":args.split,
                            "cropSize":args.crop_size, "stride":args.stride if args.stride is not None else args.crop_size,
                            "requestId":args.request_id})
    elif args.command == "split-data":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind": "split-dataset", "yaml": text,
                            **({"groupBy": args.group_by} if args.group_by else {}), "seed": args.seed,
                            "validationFraction": args.validation_fraction, "requestId": args.request_id})
    elif args.command == "leakage":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind":"split-leakage", "yaml":text,
                            "groupLimit":args.group_limit, "requestId":args.request_id})
    elif args.command == "evaluate":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind":"evaluate", "yaml":text,
                            "weights":str(local_path(args.weights,Path.cwd())), "split":args.split,
                            "device":args.device, "saveImages":args.save_images, "saveDetections":args.save_detections,
                            "saveClassifications":args.save_classifications,
                            "saveTemporalErrors":args.save_temporal_errors,
                            "saveDetectionOperatingPoints":args.save_detection_operating_points,
                            "requestId":args.request_id})
    elif args.command == "audit":
        text = local_path(args.recipe, Path.cwd()).read_text(encoding="utf-8-sig")
        data = backend.post("/api/analysis", {"kind":"task-data", "yaml":text,
                            "auditLimit":None if args.all else args.limit, "requestId":args.request_id})
    elif args.command == "analysis":
        if args.action == "submit":
            body = json.loads(local_path(args.spec, Path.cwd()).read_bytes())
            if not isinstance(body, dict):
                raise ValueError("The analysis request must be a JSON object.")
            data = backend.post("/api/analysis", body)
        elif args.action == "cancel":
            data = backend.post("/api/analysis/cancel", {"id": args.id})
        elif args.action == "list":
            data = backend.get("/api/analysis", {})
        else:
            endpoint = "job" if args.action == "show" else args.action
            data = backend.get("/api/analysis/" + endpoint, {"id": args.id})
    elif args.command == "pack":
        data = backend.post("/api/analysis", {
            "kind": "pack-dataset", "destination": str(local_path(args.destination, Path.cwd())),
            "dataset": {"gtRoots": args.gt, "lqRoots": args.lq, "scale": args.scale,
                        "phase": args.phase, "filenameTemplate": args.filename_template,
                        "gtPatchSize": args.gt_patch_size}, "requestId": args.request_id,
        })
    elif args.command == "study":
        if args.action == "list":
            data = backend.get("/api/studies", {})
        elif args.action == "show":
            data = backend.get("/api/study", {"id": args.id})
        elif args.action == "create":
            spec = json.loads(local_path(args.spec, Path.cwd()).read_bytes())
            if not isinstance(spec, dict):
                raise ValueError("The study JSON must be an object.")
            base = local_path(args.base_recipe, Path.cwd()).read_text(encoding="utf-8-sig")
            data = backend.post("/api/studies", {**spec, "baseYaml": base})
        else:
            data = backend.post("/api/study/compare", {
                "studyId": args.id, "leftArmId": args.left, "rightArmId": args.right,
                "dataset": args.dataset, "metric": args.metric, "step": args.step,
            })
    elif args.command == "loss-report":
        from contextlib import redirect_stdout
        from .loss_report import loss_report
        with redirect_stdout(sys.stderr):
            data = loss_report(local_path(args.recipe,Path.cwd()), ROOT, device=args.device,
                               batches=args.batches, step=args.step, parameter=args.parameter,
                               weights=local_path(args.weights,Path.cwd()) if args.weights else None,
                               discriminator=local_path(args.discriminator,Path.cwd()) if args.discriminator else None)
    elif args.command in ("start","validate"):
        text=local_path(args.recipe,Path.cwd()).read_text(encoding="utf-8-sig")
        route="/api/runs" if args.command == "start" else "/api/preflight"
        data=backend.post(route,{"yaml":text})
        if args.command == "validate":
            data.pop("config",None)
    elif args.command == "provenance":
        data = backend.get("/api/run/provenance",{"id":args.id})
    elif args.command == "catalog" and args.refresh:
        data = backend.post("/api/catalog/refresh", {})
    elif args.command in ("templates","runs","catalog","environment"):
        data=backend.get("/api/"+args.command,{})
    elif args.command == "archives":
        if args.action == "list":
            data = backend.get("/api/archives",{})
        elif args.action == "add":
            data = backend.post("/api/archives",{"path":str(local_path(args.folder,Path.cwd()))})
        else:
            data = backend.post("/api/archives/remove",{"id":args.id})
    elif args.command == "alignment":
        data = backend.post("/api/data/alignment",{"gt":str(local_path(args.gt,Path.cwd())),
                            "lq":str(local_path(args.lq,Path.cwd())),"radius":args.radius,"scale":args.scale})
    elif args.command == "image-difference":
        body = {"kind": "image-difference", "requestId": args.request_id}
        for role in ("left", "right", "reference"):
            if value := getattr(args, role):
                body[role] = str(local_path(value, Path.cwd()))
        if args.region is not None:
            body["region"] = dict(zip(("x", "y", "width", "height"), args.region, strict=True))
        data = backend.post("/api/analysis", body)
    elif args.command == "weights":
        source = str(local_path(args.path,Path.cwd()))
        options = {"after":args.after,"limit":args.limit,"filter":args.filter}
        if args.compare:
            data = backend.post("/api/weights/compare",{**options,"left":source,"right":str(local_path(args.compare,Path.cwd()))})
        else:
            data = backend.post("/api/weights",{**options,"path":source})
    elif args.command == "timing":
        options = {"id":args.id, "source":args.source, "against":args.against, "againstSource":args.against_source,
                   "from":args.from_step, "to":args.to_step}
        data = backend.get("/api/run/timing", {k:v for k,v in options.items() if v is not None})
    elif args.command == "history":
        data = backend.get("/api/run/history", {"id": args.id, "metric": args.metric,
                          "source": args.source, "budget": args.budget})
    elif args.command == "training-samples":
        query = {"id":args.id,"limit":args.limit}
        if args.after:
            query["after"] = args.after
        data = backend.get("/api/run/training-samples", query)
    elif args.command == "images":
        query = {k:v for k,v in vars(args).items() if k in {"id","iteration","dataset","image","after","limit"} and v is not None}
        data = backend.get("/api/run/images",query)
    elif args.command == "evaluations":
        offline = args.id.startswith("job:")
        data = backend.get("/api/analysis/evaluations" if offline else "/api/run/evaluations",
                           {"id": args.id.removeprefix("job:")})
    elif args.command == "compare":
        def source(token):
            return {"job": token[4:]} if token.startswith("job:") else token
        data = backend.post("/api/compare", {"left": source(args.left), "right": source(args.right),
                            "dataset": args.dataset, "metric": args.metric,
                            "leftStep": args.left_step, "rightStep": args.right_step,
                            "leftEvaluation": args.left_evaluation, "rightEvaluation": args.right_evaluation})
    elif args.command == "show":
        data=backend.get("/api/run",{"id":args.id})
    elif args.command == "scan":
        data=backend.post("/api/data/scan",{"gt":args.gt,"lq":args.lq,"scale":args.scale})
    else:
        body = {"id": args.id}
        if args.command == "resume" and args.train_workers is not None:
            body["trainWorkers"] = args.train_workers
        data=backend.post("/api/runs/"+args.command,body)
    print(json.dumps(data,indent=2,ensure_ascii=False))
    return int((args.command == "validate" or (args.command == "reproduce" and args.action == "prepare")) and not data["ok"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        sys.exit(main())
    except RecipeRejected as exc:
        print(json.dumps({"error":str(exc),"issues":exc.issues},ensure_ascii=False),file=sys.stderr)
        sys.exit(2)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"traiNNer-GUI: {exc}",file=sys.stderr)
        sys.exit(2)

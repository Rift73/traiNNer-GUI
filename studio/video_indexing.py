"""Map the recipe's clip source selection to the shared training indexer."""

from pathlib import Path

import msgspec

from trainner_config.options import DatasetOptions

from .paths import local_path
from .yaml_edit import read_recipe


def index_inputs(config: dict, root: Path) -> dict:
    datasets = config.get("datasets")
    ds = datasets.get("train") if isinstance(datasets, dict) else None
    if not isinstance(ds, dict) or str(ds.get("type", "")).lower() != "videoclipdataset":
        raise ValueError("Select VideoClipDataset for the training dataset.")
    try:
        options = msgspec.convert({"name": "Training videos", **ds}, type=DatasetOptions)
    except msgspec.ValidationError as exc:
        raise ValueError(str(exc)) from exc
    paths = options.dataroot_gt
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("Choose a video folder or one or more episode files.")
    width, height = options.descale_size
    if min(width, height) < 1:
        raise ValueError("Set positive descale dimensions for the video index.")
    return {"roots": [str(local_path(value, root)) for value in paths],
            "settings": {"width": width, "height": height, "kernel": options.descale_kernel,
                         "border": options.descale_border, "descale_linear_light": options.descale_linear_light}}


def video_index_status(text: str, root: Path) -> dict:
    return inspect_inputs(index_inputs(read_recipe(text), root))


def inspect_inputs(inputs: dict) -> dict:
    from trainner_data.video_indexes import inspect_clips
    report = inspect_clips([Path(p) for p in inputs["roots"]], inputs["settings"], sample_limit=None)
    if report["errors"]:
        raise ValueError(" ".join(report["errors"]))
    files = [
        {"name": Path(item["path"]).name, "path": item["path"],
         "status": "invalid" if item["status"] == "incompatible" else item["status"],
         "reason": item.get("message", "")}
        for item in report["episodes"]
    ]
    return {"total": report["total"], "ready": report["ready"], "pending": report["total"] - report["ready"],
            "invalid": report["incompatible"], "files": files}


def index_episodes(inputs: dict, progress, check_cancel) -> dict:
    from traiNNer.data.video_index import prepare_clips

    def update(event):
        progress({**event, "current": event["completed"], "episode": Path(event.get("path") or "").name})

    result = prepare_clips([Path(p) for p in inputs["roots"]], inputs["settings"], update, check_cancel)
    return {"total": result["total"], "ready": result["ready"], "written": result["indexed"],
            "skipped": result["skipped"]}

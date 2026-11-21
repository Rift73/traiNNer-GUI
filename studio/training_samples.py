"""Browse completed model-input captures from a managed or imported experiment."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlencode

_ID = re.compile(r"[0-9a-f]{32}-[1-9][0-9]*")


def _base(root: Path) -> Path:
    base = root / "training-samples"
    if base.resolve() != base:
        raise ValueError("The experiment's training sample directory has moved.")
    return base


def _header(base: Path, sample: str):
    if not _ID.fullmatch(sample):
        raise FileNotFoundError("Unknown training sample.")
    folder = base / sample
    header = folder / "capture.json"
    if header.resolve() != header or not header.is_file():
        raise FileNotFoundError("This training sample is unavailable.")
    with header.open(encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("id") != sample:
        raise ValueError("The captured training sample identity has changed.")
    return folder,metadata


def artifact(root: Path, sample: str, name: str) -> Path:
    folder, meta = _header(_base(root),sample)
    images = {info["preview"] for info in meta["tensors"].values() if "preview" in info}
    if name not in (images | {"sample.safetensors"}) or Path(name).name != name:
        raise FileNotFoundError("Unknown training sample artifact.")
    path = folder / name
    if path.resolve() != path or not path.is_file():
        raise FileNotFoundError("This training sample artifact is unavailable.")
    return path


def browse(root: Path, run: str, query: dict) -> dict:
    limit = int(query.get("limit",50))
    if not 1 <= limit <= 250:
        raise ValueError("Choose a training sample page of 1 to 250 captures.")
    base = _base(root)
    if not base.is_dir():
        return {"samples":[], "next":None}
    # Attempts can revisit a step. The capture ID keeps both observations visible.
    def order(identity):
        return -int(identity.partition("-")[2]), identity
    ids = sorted((p.name for p in base.iterdir() if _ID.fullmatch(p.name) and not p.is_symlink()),key=order)
    after = query.get("after")
    if after is not None:
        if not _ID.fullmatch(after):
            raise ValueError("Invalid training sample cursor.")
        ids = [k for k in ids if order(k)>order(after)]
    rows = []
    for key in ids[:limit]:
        _,meta = _header(base,key)
        def url(name):
            return "/api/run/training-sample?" + urlencode({"id":run, "sample":key, "file":name})
        for detail in meta["tensors"].values():
            if "preview" in detail:
                detail["previewUrl"] = url(detail["preview"])
        rows.append({**meta,"rawUrl":url("sample.safetensors")})
    return {"samples":rows,"next":ids[limit-1] if len(ids)>limit else None}

"""Inspect checkpoint structure without importing the trainer or loading tensor values."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from safetensors import SafetensorError, safe_open

from .paths import local_path


def _header(path: str, root: Path):
    file = local_path(path,root)
    if file.suffix.lower() != ".safetensors":
        raise ValueError("Checkpoint structure inspection supports .safetensors weights.")
    if not file.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {file}")
    identity = file.stat()
    try:
        with safe_open(os.fspath(file),framework="numpy",device="cpu") as weights:
            entries = {}
            for name in weights.keys():
                entry = weights.get_slice(name)
                entries[name] = (tuple(entry.get_shape()),entry.get_dtype())
            metadata = weights.metadata() or {}
    except SafetensorError as exc:
        raise ValueError(f"Invalid safetensors file {file.name}: {exc}") from exc
    after = file.stat()
    if (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise FileExistsError(f"{file.name} changed during inspection. Reopen the completed checkpoint.")
    return file, after.st_size, entries, metadata


def _fingerprint(entries):
    digest = hashlib.sha256()
    for name, (shape,dtype) in sorted(entries.items()):
        digest.update(json.dumps([name,shape,dtype],separators=(",",":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _page(options):
    limit = options.get("limit",100)
    search = options.get("filter","")
    after = options.get("after","")
    if type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError("Choose a checkpoint page size from 1 to 500.")
    if not isinstance(search,str) or not isinstance(after,str):
        raise ValueError("Checkpoint name filter and page cursor must be text.")
    return limit,search.casefold(),after


def inspect_weights(request: dict, root: Path):
    limit,search,after = _page(request)
    file,size,entries,metadata = _header(request.get("path"),root)
    by_dtype = Counter()
    numel = 0
    rows = []
    matched = 0
    for name,(shape,dtype) in sorted(entries.items()):
        count = math.prod(shape)
        by_dtype[dtype] += count
        numel += count
        if search in name.casefold():
            matched += 1
            if name>after and len(rows) <= limit:
                rows.append({"name":name,"shape":shape,"dtype":dtype,"elements":count})
    # A checkpoint may include persistent buffers. Only the constructed model
    # can state which entries are trainable parameters.
    return {"path":str(file),"bytes":size,"tensorCount":len(entries),"elements":numel,
            "dtypeElements":dict(by_dtype),"layoutSha256":_fingerprint(entries),
            "metadata":{k:v[:4096] for k,v in sorted(metadata.items())[:64]},
            "metadataTruncated":len(metadata)>64 or any(len(v)>4096 for v in metadata.values()),
            "matched":matched,"tensors":rows[:limit],
            "next":rows[limit-1]["name"] if len(rows)>limit else None}


def compare_weights(request: dict, root: Path):
    limit,search,after = _page(request)
    left,_,a,_ = _header(request.get("left"),root)
    right,_,b,_ = _header(request.get("right"),root)
    counts = {"same":0,"leftOnly":0,"rightOnly":0,"shape":0,"dtype":0,"shapeAndDtype":0}
    rows = []
    for name in sorted(a.keys() | b.keys()):
        if name not in a:
            status = "rightOnly"
        elif name not in b:
            status = "leftOnly"
        else:
            shape = a[name][0] != b[name][0]
            dtype = a[name][1] != b[name][1]
            status = "shapeAndDtype" if shape and dtype else "shape" if shape else "dtype" if dtype else "same"
        counts[status] += 1
        if status != "same" and name > after and search in name.casefold() and len(rows)<=limit:
            rows.append({"name":name,"difference":status,"left":a.get(name),"right":b.get(name)})
    return {"left":str(left),"right":str(right),"counts":counts,
            "sameLayout":counts["same"] == len(a) == len(b),
            "differences":rows[:limit], "next":rows[limit-1]["name"] if len(rows)>limit else None}

"""Browse completed validation images and the inputs actually used to produce them."""

from __future__ import annotations

import base64
import heapq
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

from .data import preview, preview_lmdb
from .paths import artifact_version

_FILE = "validation-images.sqlite3"
_OLD_NAME = re.compile(r"^(.+)_([0-9]{6,})\.png$")


def explorer_executable():
    if os.environ.get("WSL_DISTRO_NAME"):
        path = Path("/mnt/c/Windows/explorer.exe")
        if path.is_file():
            return path
    return None


def reveal_image(root: Path, query: dict):
    import subprocess
    from .paths import clipboard_paths

    explorer = explorer_executable()
    if explorer is None:
        raise ValueError("Windows Explorer is unavailable on this training host.")
    if "role" in query:
        _, path, _ = _recorded_source(root, query)
    else:
        key = query.get("path")
        if not isinstance(key, str):
            raise ValueError("Select a validation image.")
        path = image_path(root, key)
        if query.get("version") and artifact_version(path.stat()) != query["version"]:
            raise FileExistsError("This image changed after validation.")
    target = clipboard_paths(path).get("windows")
    if not target:
        raise ValueError("This file has no Windows Explorer path.")
    # Explorer delegates to the desktop shell and can return 1 on success.
    result = subprocess.run([str(explorer), "/select,", target], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    if result.returncode not in (0, 1):
        raise RuntimeError("Windows Explorer could not open the image location.")
    return {"windows": target}


def _index(root):
    path = root / _FILE
    if path.is_file() and path.resolve() == path:
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)
        db.row_factory = sqlite3.Row
        return db
    return None


def _order(row):
    return -row["iteration"], row["dataset"] or "", row["image"]


def _encode(row):
    raw = json.dumps(_order(row), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _cursor(value):
    if not value:
        return None
    try:
        if not isinstance(value, str) or len(value) > 20000:
            raise ValueError
        key = json.loads(base64.b64decode(value, altchars=b"-_", validate=True))
        if (not isinstance(key,list) or len(key)!=3 or type(key[0]) is not int
                or not all(isinstance(item,str) for item in key[1:])):
            raise ValueError
        return tuple(key)
    except (ValueError, TypeError, UnicodeError) as e:
        raise ValueError("Invalid validation image page cursor.") from e


def image_path(root: Path, key: str) -> Path:
    relative = Path(key)
    if (key != relative.as_posix() or not relative.parts or relative.parts[0] != "visualization"
            or relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".png"):
        raise FileNotFoundError("Unknown validation image.")
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise FileNotFoundError("Validation image is no longer available.")
    return path


def _legacy(root, run_id, iteration, image):
    folder = root / "visualization"
    if not folder.is_dir() or folder.resolve() != folder:
        return
    for base, dirs, files in os.walk(folder):
        current = Path(base)
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (current / d).is_symlink()]
        for name in files:
            match = _OLD_NAME.fullmatch(name)
            if not match or int(match[2]) > (1<<63)-1:
                continue
            step = int(match[2])
            rel = current.relative_to(folder)
            sample = (rel / match[1]).as_posix() if rel.name != match[1] else rel.as_posix()
            if iteration is not None and iteration != step or image is not None and image != sample:
                continue
            path = current / name
            if path.is_symlink():
                continue
            key = path.relative_to(root).as_posix()
            item = {"kind":"validation", "iteration":step, "image":sample, "dataset":None,
                    "name":name, "path":key, "url":_url(run_id,key), "provenance":"filename"}
            lr = current / (match[1] + "_lr.png")
            if lr.is_file() and not lr.is_symlink():
                item["lqUrl"] = _url(run_id,lr.relative_to(root).as_posix())
            yield item


def _url(run_id, key, *, offline=False, version=None):
    prefix = "analysis" if offline else "run"
    fields = {"id":run_id, "path":key}
    if version:
        fields["version"] = version
    return f"/api/{prefix}/image?" + urlencode(fields)


def _versioned(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='image_version'").fetchone())


def images(root: Path, run_id: str, query: dict, *, offline: bool = False) -> dict:
    limit = int(query.get("limit", 100))
    if not 1 <= limit <= 500:
        raise ValueError("Choose a page of 1 to 500 images.")
    after = _cursor(query.get("after"))
    dataset, sample = query.get("dataset"), query.get("image")
    iteration = int(query["iteration"]) if query.get("iteration") is not None else None
    if iteration is not None and not (-1 if offline else 0) <= iteration < 1<<63:
        raise ValueError("Invalid validation iteration.")
    db = _index(root)
    if db is None:
        candidates = _legacy(root, run_id, iteration, sample) if dataset is None and not offline else ()
        rows = heapq.nsmallest(limit+1,
            (r for r in candidates if after is None or _order(r)>after), key=_order)
        provenance = "filename"
    else:
        filters = []
        bindings = []
        for name, val in (("dataset",dataset), ("image",sample), ("step",iteration)):
            if val is not None:
                filters.append(name+"=?")
                bindings.append(val)
        if after:
            filters.append("(step < ? OR (step = ? AND (dataset,image) > (?,?)))")
            bindings.extend((-after[0],-after[0],after[1],after[2]))
        where = " WHERE " + " AND ".join(filters) if filters else ""
        rows = []
        with closing(db):
            selected = ("SELECT * FROM images LEFT JOIN image_version USING(dataset,step,image)"
                        if _versioned(db) else "SELECT *,NULL AS version,NULL AS capture_id FROM images")
            sql = selected + where + " ORDER BY step DESC, dataset, image LIMIT ?"
            for record in db.execute(sql, (*bindings,limit+1)):
                key = record["output"]
                filters_for_source = {"id":run_id,"dataset":record["dataset"],
                                      "iteration":record["step"],"image":record["image"]}
                item = {"kind":"validation", "iteration":record["step"], "dataset":record["dataset"],
                        "image":record["image"], "name":Path(key).name, "path":key,
                        "version":record["version"],
                        "url":_url(run_id,key,offline=offline,version=record["version"]),
                        "width":record["width"],"height":record["height"],"provenance":"recorded"}
                for role, source in json.loads(record["sources"]).items():
                    if record["capture_id"]:
                        filters_for_source["capture"] = record["capture_id"]
                    item[role] = source.get("path")
                    prefix = "analysis" if offline else "run"
                    item[role+"Url"] = f"/api/{prefix}/validation-source?" + urlencode({**filters_for_source,"role":role})
                rows.append(item)
        provenance = "recorded"
    page = rows[:limit]
    for row in page:
        if row.get("version"):
            row["versionOrigin"] = "atCapture"
            continue
        # A pre-versioned gallery can identify the file visible now. It has
        # no evidence that those bytes were also present at the training step.
        try:
            current = artifact_version(image_path(root, row["path"]).stat())
        except FileNotFoundError:
            row["available"] = False
        else:
            row.update(version=current, versionOrigin="onRead",
                       url=_url(run_id, row["path"], offline=offline, version=current))
    return {"images":page, "next":_encode(rows[limit-1]) if len(rows)>limit else None,
            "provenance":provenance,"canReveal":not offline and explorer_executable() is not None}


def linked_outputs(root: Path, source: str, offline: bool, dataset: str, step: int,
                   samples: list[str], capture_id: str | None = None) -> dict[str, str]:
    """Resolve measured examples to their captured exports in one index lookup."""
    if not samples:
        return {}
    db = _index(root)
    if db is None:
        return {}
    placeholders = ",".join("?" for _ in samples)
    with closing(db):
        if _versioned(db):
            sql = ("SELECT image,output,version FROM images LEFT JOIN image_version USING(dataset,step,image) "
                   "WHERE capture_id IS ? AND dataset=? AND step=?")
            args = (capture_id, dataset, step, *samples)
        elif capture_id is None:
            sql = "SELECT image,output,NULL AS version FROM images WHERE dataset=? AND step=?"
            args = (dataset, step, *samples)
        else:
            return {}
        rows = db.execute(sql + f" AND image IN ({placeholders})", args)
        return {r["image"]: _url(source, r["output"], offline=offline,version=r["version"]) for r in rows}


def _recorded_source(root: Path, query: dict):
    role = query.get("role")
    if role not in ("lq","gt"):
        raise ValueError("Select the validation input or ground truth.")
    db = _index(root)
    if db is None:
        raise FileNotFoundError("This run did not record validation sources.")
    with closing(db):
        db.execute("BEGIN")
        if query.get("capture") is not None:
            if not _versioned(db) or db.execute(
                "SELECT 1 FROM image_version WHERE dataset=? AND step=? AND image=? AND capture_id=?",
                (query.get("dataset"),int(query.get("iteration",-1)),query.get("image"),query["capture"])
            ).fetchone() is None:
                raise FileExistsError("A different validation observation now owns this source preview.")
        record = db.execute("SELECT sources FROM images WHERE dataset=? AND step=? AND image=?",
            (query.get("dataset"), int(query.get("iteration",-1)), query.get("image"))).fetchone()
    source = json.loads(record[0]).get(role) if record else None
    if not source:
        raise FileNotFoundError("The selected validation source was not recorded.")
    storage = Path(source["database"]) / "data.mdb" if "database" in source else Path(source["path"])
    info = storage.stat()
    version = artifact_version(info)
    if (info.st_size != source["size"] or str(info.st_mtime_ns) != source["mtimeNs"]
            or (source.get("version") and source["version"] != version)):
        raise FileExistsError("This source changed after validation.")
    return source, storage, version


def source_preview(root: Path, query: dict):
    source, storage, version = _recorded_source(root, query)
    size = int(query.get("maxSize",1536))
    if "database" not in source:
        return preview(storage, size, version=version)
    result = preview_lmdb(source["database"], source["key"], size)
    if artifact_version(storage.stat()) != version:
        raise FileExistsError("This LMDB source changed while reading its preview.")
    return result

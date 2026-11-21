"""Browse local trainer folders, including the mounted Windows drives."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .native import executable
from .paths import local_path


def list_folder(request: dict, root: Path) -> dict:
    folder = local_path(request.get("path") or str(root), root).absolute()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder was not found: {folder}")
    mode = request.get("mode", "all")
    query = request.get("filter", "")
    hidden = request.get("hidden", False)
    limit = request.get("limit", 200)
    after = request.get("after") or {}
    if mode not in ("all", "folders", "files") or not isinstance(query, str) or "\x00" in query:
        raise ValueError("Provide a valid folder listing mode and name filter.")
    if type(hidden) is not bool or type(limit) is not int or not 1 <= limit <= 1024:
        raise ValueError("Specify a page size from 1 to 1024 and a boolean hidden option.")
    if not isinstance(after,dict):
        raise ValueError("Invalid page cursor.")
    kind, name = after.get("kind","folder"), after.get("name","")
    if kind not in ("folder", "file") or not isinstance(name, str) or "/" in name or "\x00" in name:
        raise ValueError("Invalid page cursor.")
    proc = subprocess.run([str(executable("files", "list")), str(folder), mode, query, str(int(hidden)),
                           kind, name, str(limit)], stdin=subprocess.DEVNULL, capture_output=True, timeout=45)
    if proc.returncode:
        raise ValueError(proc.stderr.decode(errors="replace").strip() or "Folder listing failed.")
    fields = proc.stdout.split(b"\0")
    if len(fields) < 2 or fields[-2] not in (b"end", b"more"):
        raise RuntimeError("Incomplete result from the native directory reader.")
    entries = [{"kind":"folder" if rec.startswith(b"d") else "file", "name":os.fsdecode(rec[1:])}
               for rec in fields[:-2]]
    return {"path":str(folder), "parent":str(folder.parent) if folder != folder.parent else None,
            "entries":entries, "next":entries[-1] if entries and fields[-2]==b"more" else None}

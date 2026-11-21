"""Rebuild the installed-factory catalog outside the lightweight control process."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def refresh(root: Path, directory: Path, packaged: Path, destination: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "refresh.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError("A registry refresh is already in progress.") from exc
        with tempfile.TemporaryDirectory(prefix=".refresh-", dir=directory) as tmp:
            generated = Path(tmp) / "schema.json"
            with tempfile.TemporaryFile() as diagnostics:
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "scripts.options.generate_schema", "--output", str(generated)],
                        stdin=subprocess.DEVNULL, stdout=diagnostics, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
                        cwd=root, timeout=120, check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ValueError("Registry discovery exceeded two minutes. Check your local factory imports.") from exc
                if result.returncode:
                    end = diagnostics.seek(0, os.SEEK_END)
                    diagnostics.seek(max(0, end - 4000))
                    detail = diagnostics.read().decode("utf-8", errors="replace").strip()
                    raise ValueError(f"Registry discovery failed (exit {result.returncode}). {detail}")
            schema = json.loads(generated.read_bytes())
            schema["x-trainner-refresh"] = {
                "generatedAt": datetime.now(UTC).isoformat(),
                "baseSha256": hashlib.sha256(packaged.read_bytes()).hexdigest(),
                "root": str(root),
            }
            with generated.open("w", encoding="utf-8") as stream:
                json.dump(schema, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(generated, destination)

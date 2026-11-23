"""First-use downloads for the Linux trainer's frozen model dependencies."""

import hashlib
import logging
import os
import tempfile
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
NEUFLOW_SHA256 = "76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f"

_RN50_URL = f"https://openaipublic.azureedge.net/clip/models/{RN50_SHA256}/RN50.pt"
_NEUFLOW_URL = (
    "https://raw.githubusercontent.com/neufieldrobotics/NeuFlow_v2/"
    "204b5e3744461d90303b9ff82caa7a1bb56a2ca2/neuflow_mixed.pth"
)


def rn50_cache_path() -> Path:
    return Path.home() / ".cache/clip/RN50.pt"


def neuflow_cache_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "trainner/neuflow/neuflow_mixed.pth"


def _verified_cache(path: Path, expected: str) -> bool:
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return False
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for cached weights {path}. Remove the file to download it again."
        )
    return True


def _ensure_weights(path: Path, url: str, expected: str) -> Path:
    if _verified_cache(path, expected):
        return path

    # DDP ranks and independent jobs can reach first use at the same time.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _verified_cache(path, expected):
            return path
        logging.getLogger("traiNNer").info(
            "Downloading pretrained weights from %s to %s", url, path
        )
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".part", delete=False
        ) as out:
            temp = Path(out.name)
            digest = hashlib.sha256()
            try:
                with urlopen(url, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(
                        f"SHA-256 mismatch downloading {path.name} from {url}. Retry the download."
                    )
                out.flush()
                os.replace(temp, path)
            except (HTTPException, OSError) as exc:
                raise OSError(
                    f"Could not cache {path.name} from {url} at {path}: {exc}"
                ) from exc
            finally:
                temp.unlink(missing_ok=True)
    return path


def ensure_rn50() -> Path:
    return _ensure_weights(rn50_cache_path(), _RN50_URL, RN50_SHA256)


def ensure_neuflow() -> Path:
    return _ensure_weights(neuflow_cache_path(), _NEUFLOW_URL, NEUFLOW_SHA256)

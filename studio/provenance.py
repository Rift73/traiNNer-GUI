"""Describe the exact submitted recipe, runtime and training sources of a managed attempt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile

from .native import executable
from .paths import artifact_version
from .recipes import atomic_text
from .yaml_edit import read_recipe

_PACKAGES = ("torch","torchvision","numpy","triton","safetensors","msgspec","pyvips",
             "opencv-python","opencv-python-headless","pytorch-optimizer","spandrel","Pillow")
_TUNING = ("CUDA_VISIBLE_DEVICES","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF","OMP_NUM_THREADS",
           "MKL_NUM_THREADS","TRAINNER_COMPILE_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR")


@lru_cache(maxsize=1)
def runtime():
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python":platform.python_version(),"interpreter":sys.executable,
            "platform":platform.system(),"kernel":platform.release(),"machine":platform.machine(),
            "packages":packages, "logicalCpus":len(os.sched_getaffinity(0))}


def environment(root: Path,state: Path):
    disk = shutil.disk_usage(root)
    return {**runtime(),"checkout":str(root),"state":str(state),"diskFreeBytes":disk.free,
            "diskTotalBytes":disk.total,"llvm":shutil.which("clang++"),
            "environment":{k:os.environ[k] for k in _TUNING if k in os.environ}}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path,state: Path):
    folder = state/"provenance"
    folder.mkdir(parents=True,exist_ok=True)
    cache = folder/"source-cache.v1"
    with (folder/"cache.lock").open("a") as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
        program = executable("provenance","fingerprint")
        result = subprocess.run([str(program),str(root),str(cache)],stdin=subprocess.DEVNULL,
                                capture_output=True,timeout=60)
        if result.returncode:
            raise ValueError(result.stderr.decode(errors="replace").strip() or "Could not identify training sources.")
        parts = result.stdout.split(b"\0")
        if parts[-1] or len(parts)%3!=1 or len(parts)<4:
            raise ValueError("Incomplete source fingerprint.")
        entries = {os.fsdecode(parts[i]):parts[i+2].decode("ascii") for i in range(0,len(parts)-1,3)}
        serialized = json.dumps(entries,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode()
        digest = _sha(serialized)
        path = folder/(digest+".json")
        if not path.exists():
            atomic_text(path,serialized.decode("ascii"))
        with NamedTemporaryFile(dir=folder,prefix=".cache-",delete=False) as tmp:
            try:
                tmp.write(result.stdout)
                tmp.flush()
                os.replace(tmp.name,cache)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    ref = subprocess.run(["git","-C",str(root),"rev-parse","--verify","HEAD"],stdin=subprocess.DEVNULL,
                         capture_output=True,timeout=5) if shutil.which("git") else None
    return {"sha256":digest,"files":len(entries),
            "gitHead":ref.stdout.decode().strip() if ref and ref.returncode==0 else None,
            "scope":"train.py, trainer packages, native data components and pyproject at launch"}


def capture_attempt(root: Path,state: Path,attempt: Path) -> None:
    recipe = (attempt/"options.yml").read_bytes()
    config = read_recipe(recipe.decode("utf-8-sig"))
    record = {"version":1,"recipeSha256":_sha(recipe),"environment":environment(root,state),
              "declaredSeed":config.get("manual_seed"),"inputs":{}}
    for name,value in (config.get("path") or {}).items():
        if not value or not (name.startswith("pretrain_network_") or name=="resume_state"):
            continue
        path = (root / value).resolve()
        try:
            info = path.stat()
            record["inputs"][name] = {"path":str(path),"size":info.st_size,"mtimeNs":str(info.st_mtime_ns),
                                     "fileVersion":artifact_version(info)}
        except OSError as exc:
            record["inputs"][name] = {"path":str(path),"error":str(exc)}
    try:
        record["source"] = inventory(root,state)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as e:
        # Capture is informational. Keep a visible explanation if the build
        # tools or source tree cannot be read, while allowing training to run.
        record["source"] = {"error":str(e)}
    atomic_text(attempt/"provenance.json",json.dumps(record,ensure_ascii=True)+"\n")


def read_attempt(attempt: Path):
    out = {}
    for key,name in (("launch","provenance.json"),("runtime","runtime.json")):
        p = attempt/name
        if p.is_file() and p.resolve()==p:
            out[key] = json.loads(p.read_bytes())
    return out

def write_effective(attempt: Path, opt, model, loader, initial_step: int) -> None:
    import torch

    root = Path(os.environ["TRAINNER_MANAGED_RUN_ROOT"]).resolve()
    if attempt.parent.resolve() != root/"attempts" or attempt.is_symlink():
        raise ValueError("Managed runtime record does not belong to this run.")
    def qualified(obj):
        return type(obj).__module__ + "." + type(obj).__qualname__
    precision = torch.get_float32_matmul_precision()
    device = {"type":model.device.type}
    if model.device.type == "cuda":
        props = torch.cuda.get_device_properties(model.device)
        device.update(name=props.name,computeCapability=[props.major,props.minor],memoryBytes=props.total_memory)
    record = {
        "seed":opt.manual_seed, "initialStep":initial_step, "device":device,
        "checkpointRng":opt.checkpoint_rng,
        "task":opt.task or "legacy dispatch", "modelClass":qualified(model),
        "pytorch":str(torch.__version__),"cudaBuild":torch.version.cuda,
        "deterministic":torch.are_deterministic_algorithms_enabled(),
        "deterministicWarnOnly":torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32MatmulPrecision":precision, "compileRequested":opt.use_compile,
        "ampRequested":opt.use_amp, "ampBf16Requested":opt.amp_bf16,
        "worldSize":opt.world_size, "rank":opt.rank,
        "trainingData":{"type":qualified(loader.dataset), "samples":len(loader.dataset),
                        "microbatchesPerEpoch":len(loader),"batchSize":loader.batch_size,
                        "workers":loader.num_workers,"accumulation":model.accum_iters},
        "optimizers":[qualified(o) for o in model.optimizers],
        "schedulers":[qualified(schedule) for schedule in model.schedulers],
        "schedulerBindings":model._scheduler_specs,
    }
    from trainner_config.feature_assets import feature_assets
    from trainner_config.objectives import resolve_objectives

    metrics = opt.val.metrics if opt.val and opt.val.val_enabled and opt.val.metrics_enabled else None
    assets = feature_assets(resolve_objectives(opt), metrics, Path(__file__).resolve().parents[1],
                            hub=Path(torch.hub.get_dir()))
    if assets:
        record["featureAssets"] = assets
    if (transfer := getattr(model, "transfer_g_receipt", None)) is not None:
        record["transferG"] = transfer
    atomic_text(attempt/"runtime.json",json.dumps(record,ensure_ascii=True)+"\n")

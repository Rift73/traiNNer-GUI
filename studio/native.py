"""Locate native control-plane tools, building a missing or stale binary once."""
from __future__ import annotations

import fcntl
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "native/studio"


def executable(component: str, name: str, *, source: str | None = None) -> Path:
    directory = ROOT / component
    output = directory / "build/clang" / name
    deps = [directory / (source or name + ".cpp"), directory / "Makefile", ROOT / "tool.mk"]
    latest = max(p.stat().st_mtime_ns for p in deps)
    if output.is_file() and output.stat().st_mtime_ns >= latest:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ".compile.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file() and output.stat().st_mtime_ns >= latest:
            return output
        if not shutil.which("make"):
            raise RuntimeError("GNU Make is required to build the native training tools.")
        result = subprocess.run(["make", "--no-print-directory", "-C", str(directory)],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"Could not build {name} with LLVM. " + result.stderr.decode(errors="replace"))
    return output


def prepare() -> dict:
    from .data import _inspector
    built = {"datasetInspector":str(_inspector())}
    for component, name in (("history","history"),("files","list"),("alignment","align"),
                            ("provenance","fingerprint")):
        built[component] = str(executable(component,name))
    built["eventReader"] = str(executable("tfrecord", "records.so", source="records.cpp"))
    built["weightDrift"] = str(executable("weights", "measure.so", source="measure.c"))
    built["cpuImageMetrics"] = str(executable("metrics", "metrics.so", source="metrics.c"))
    built["neosrResample"] = str(executable("resample", "resample.so", source="resample.c"))
    built["pairedPatches"] = str(executable("paired_patches", "patches.so", source="patches.c"))
    built["imageDifference"] = str(executable("image_difference", "difference.so", source="difference.c"))
    built["detectionScorer"] = str(executable("detection", "detection.so", source="detection.cpp"))
    built["splitAudit"] = str(executable("split_audit", "sha256.so", source="sha256.cpp"))
    # This library is an acceleration for the ordinary training data loader.
    directory = ROOT/"data"
    run = subprocess.run(["make","--no-print-directory","-C",str(directory),"paired-video","paired-folders","paired-crop",
                          "paired-identity","mask-histogram","task-sample-pack"],
                         stdin=subprocess.DEVNULL,capture_output=True,timeout=60)
    if run.returncode:
        raise RuntimeError("Could not prepare the native data loaders. " + run.stderr.decode(errors="replace"))
    built["pairedFolderSelector"] = str(directory/"build/clang/pair-folders")
    built["pairedCrop"] = str(directory/"build/clang/paired_crop.so")
    built["pairedVideo"] = str(directory/"build/clang/paired_video_pack.so")
    built["pairedIdentity"] = str(directory/"build/clang/paired_identity.so")
    built["maskHistogram"] = str(directory/"build/clang/mask_histogram.so")
    built["taskSamplePack"] = str(directory/"build/clang/task_sample_pack.so")
    return {"tools":built,"compiler":shutil.which("clang++")}

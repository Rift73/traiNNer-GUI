"""Sample actual training pairs and measure spatial registration in each color plane."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter
from pathlib import Path

from .alignment import inspect_pair
from .paths import local_path


def _assessment(result: dict) -> str:
    evidence = {name: observation["assessment"] for name, observation in result["evidence"].items()}
    if evidence["luma"] == "possibleShift":
        luma = result["estimate"]["luma"]
        for name in ("blueMinusLuma", "redMinusLuma"):
            if evidence[name] in ("possibleShift", "noClearShift"):
                chroma = result["estimate"][name]
                if math.hypot(luma["dx"] - chroma["dx"], luma["dy"] - chroma["dy"]) > 1:
                    return "differentPlaneShifts"
        return "possibleLumaShift"
    if any(evidence[name] == "possibleShift" for name in ("blueMinusLuma", "redMinusLuma")):
        return "possibleChromaShift"
    if "inconsistentAcrossPatches" in evidence.values():
        return "localDisagreement"
    if evidence["luma"] == "noClearShift":
        return "noClearLumaShift"
    return "inconclusive"


def pair_source_options(config: dict, split: str, root: Path) -> dict:
    datasets = config.get("datasets")
    source = datasets.get(split) if isinstance(datasets, dict) and isinstance(split, str) else None
    if not isinstance(source, dict) or str(source.get("type", "")).casefold() != "pairedimagedataset":
        raise ValueError("Registration sampling needs a PairedImageDataset split with pre-existing LQ and GT.")
    options = dict(source)
    for key in ("dataroot_gt", "dataroot_lq"):
        roots = options.get(key)
        if isinstance(roots, str):
            roots = [roots]
        if not isinstance(roots, list) or not roots:
            raise ValueError(f"Choose at least one root for {key}.")
        paths = [local_path(value, root) for value in roots]
        for path in paths:
            if not path.is_dir():
                raise ValueError(f"{key} directory is unavailable: {path}")
        options[key] = [str(path) for path in paths]
    if len(options["dataroot_gt"]) != len(options["dataroot_lq"]):
        raise ValueError("Select one LQ directory for each GT directory.")
    if options.get("meta_info") is not None:
        options["meta_info"] = str(local_path(options["meta_info"], root))
    options.update(phase="val", scale=config["scale"])
    return options


def _select_dataset(config: dict, split: str, root: Path):
    import msgspec
    from trainner_config.options import DatasetOptions
    from traiNNer.data.paired_image_dataset import PairedImageDataset

    options = pair_source_options(config, split, root)
    return PairedImageDataset(msgspec.convert(options, type=DatasetOptions, strict=True))


def audit_pairs(config: dict, root: Path, *, split: str, sample_limit: int, radius: int, seed: int,
                progress, cancel):
    if not 1 <= sample_limit <= 64 or not 2 <= radius <= 12:
        raise ValueError("Select 1–64 pairs and a search radius of 2–12 LQ pixels.")
    progress({"stage":"select","completed":0})
    dataset = _select_dataset(config, split, root)
    if not dataset.paths:
        raise ValueError("The selected split contains no paired images.")
    def priority(row):
        # The same members are selected after an unrelated change to filesystem
        # traversal order. Duplicate manifest rows carry their training weight.
        key = json.dumps([seed,row], sort_keys=True, ensure_ascii=True).encode("ascii")
        return hashlib.blake2b(key, digest_size=16).digest()

    selected = heapq.nsmallest(sample_limit, dataset.paths, key=priority)
    checks = []
    counts = Counter()
    if dataset.lmdb is not None:
        from lmdb import Error as DatabaseError
        read_errors = (OSError, ValueError, DatabaseError)
    else:
        read_errors = (OSError, ValueError)
    for index,row in enumerate(selected):
        cancel()
        progress({"stage":"registration","completed":index,"total":len(selected)})
        pair = {role:row[role+"_path"] for role in ("lq","gt")}
        if dataset.lmdb is not None:
            pair = {**{role+"Key": value for role,value in pair.items()},
                    **{role:row[role+"_db"] for role in ("lq","gt")}}
        try:
            result = inspect_pair({**pair,"scale":config["scale"],"radius":radius}, root, reader=dataset.lmdb)
        except read_errors as error:
            observation = {**pair,"assessment":"unavailable","message":str(error)}
        else:
            observation = {**pair, "assessment":_assessment(result), "estimates":result["estimate"],
                           "evidence":result["evidence"], "alphaIgnored":result["alphaIgnored"],
                           "method":result["method"], "decodedSampleBits":result["decodedSampleBits"]}
        counts[observation["assessment"]] += 1
        checks.append(observation)
    progress({"stage":"registration","completed":len(selected),"total":len(selected)})
    return {"split":split, "population":len(dataset.paths),"sampled":len(checks),"seed":seed,
            "radius":radius, "scale":config["scale"],"units":"LQ pixels","counts":dict(counts),
            "method":"Sobel gradient correlation on area-downsampled GT and original LQ over up to nine patches.",
            "pairs":checks}

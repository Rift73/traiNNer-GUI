"""Prepare scene-disjoint train/val recipes for disk PairedVideoDataset.

Root slots are one-based. Scene manifests pin the ordered frame names and
eligible centers in each selected scene. Physical identity is checked across
both roles at split time. Copies and visual similarity are outside that check.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from trainner_config.tasks import TRAINING_TASKS
from trainner_data.frame_sequences import FrameScene, FrameWindowIndex

from .dataset_pack import _publish
from .dataset_split import (
    DatasetSplitCancelled,
    _file_info,
    _validation_groups,
    render_split_recipe,
    split_dataset_options,
)
from .paths import local_path
from .recipes import task_from_config
from .video_sources import selected_frame_positions
from .yaml_edit import read_recipe

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_LIMIT = 16


def _roots(dataset: dict, key: str, root: Path) -> list[str]:
    values = dataset.get(key)
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError(f"Provide a folder or ordered folder list for {key}.")
    roots = []
    for value in values:
        path = local_path(value, root)
        if not path.is_dir() or (path / "data.mdb").is_file():
            raise ValueError(f"{key} requires prepared frame folders on disk: {path}")
        roots.append(str(path))
    return roots


def _version(info: os.stat_result) -> str:
    return f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}"


def _scene_id(scene: FrameScene) -> str:
    return json.dumps([scene.slot, *scene.key], separators=(",", ":"))


def _audit(
    db: sqlite3.Connection,
    index: FrameWindowIndex,
    tick: Callable[[str, int, int | None], None],
) -> tuple[list[str], dict[str, Any]]:
    # Store file-level observations on disk. RAM beyond the loader's index
    # scales with scenes and a bounded set of alias examples.
    db.execute("CREATE TABLE paths(path BLOB PRIMARY KEY, stamp TEXT) WITHOUT ROWID")
    db.execute("""CREATE TABLE owners(identity TEXT PRIMARY KEY, scene INTEGER,
                  role TEXT, name BLOB, shared INTEGER) WITHOUT ROWID""")
    parents = list(range(len(index.scenes)))

    def find(i: int) -> int:
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def evidence(i: int, role: str, name: str | bytes) -> dict[str, Any]:
        scene = index.scenes[i]
        return {
            "root": scene.slot,
            "numbered": scene.key[0],
            "key": scene.key[1],
            "role": role,
            "frame": os.fsdecode(name),
        }

    examples = []
    counts = Counter()
    observed = 0
    for i, scene in enumerate(index.scenes):
        for role, folder, pos, _ in selected_frame_positions(scene, index.width):
            name = scene.frames[pos]
            path = os.fsencode(os.path.join(folder, name))
            info = _file_info(path)
            if not info.st_ino:
                raise ValueError(
                    f"Filesystem supplies no physical file identity for {os.fsdecode(path)}"
                )
            stamp = _version(info)
            inserted = db.execute(
                "INSERT OR IGNORE INTO paths VALUES (?,?)", (path, stamp)
            )
            if (
                not inserted.rowcount
                and db.execute(
                    "SELECT stamp FROM paths WHERE path=?", (path,)
                ).fetchone()[0]
                != stamp
            ):
                raise ValueError(f"Source changed during split: {os.fsdecode(path)}")
            identity = f"{info.st_dev}:{info.st_ino}"
            inserted = db.execute(
                "INSERT OR IGNORE INTO owners VALUES (?,?,?,?,0)",
                (identity, i, role, os.fsencode(name)),
            )
            if not inserted.rowcount:
                owner, first_role, first_name, shared = db.execute(
                    "SELECT scene,role,name,shared FROM owners WHERE identity=?",
                    (identity,),
                ).fetchone()
                if owner != i:
                    a, b = find(i), find(owner)
                    parents[max(a, b)] = min(a, b)
                    if not shared:
                        db.execute(
                            "UPDATE owners SET shared=1 WHERE identity=?", (identity,)
                        )
                        if len(examples) < _EXAMPLE_LIMIT:
                            examples.append(
                                [
                                    evidence(owner, first_role, first_name),
                                    evidence(i, role, name),
                                ]
                            )
            observed += 1
            counts[role] += 1
            tick("identifying", observed, None)
    physical, shared = db.execute(
        "SELECT count(*),coalesce(sum(shared),0) FROM owners"
    ).fetchone()
    return [_scene_id(index.scenes[find(i)]) for i in range(len(parents))], {
        "method": "device-and-inode",
        "scope": "selected-lq-context-and-gt-centers",
        "lqFrameUsesByScene": counts["lq"],
        "gtCenterUsesByScene": counts["gt"],
        "paths": db.execute("SELECT count(*) FROM paths").fetchone()[0],
        "physicalFiles": physical,
        "sharedAcrossScenes": shared,
        "examples": examples,
        "examplesTruncated": shared > len(examples),
    }


def split_video_dataset(
    body: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    root: Path = _ROOT,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Split datasets.train from recipe yaml using seed and validationFraction.

    Atomically publish train.json, val.json, recipe.yml and split.json in a new
    directory. The train and val report counts are counts of eligible windows.
    No frames are decoded or copied. Cancel raises DatasetSplitCancelled.
    """
    recipe = body.get("yaml")
    config = read_recipe(recipe)
    task = task_from_config(config)
    datasets = config.get("datasets")
    dataset = datasets.get("train") if isinstance(datasets, dict) else None
    if (
        task != "image_restoration"
        or not isinstance(dataset, dict)
        or str(dataset.get("type", "")).lower() != "pairedvideodataset"
    ):
        raise ValueError(
            "Select an image_restoration training recipe using PairedVideoDataset."
        )
    network = config.get("network_g")
    arch = str(network.get("type", "")).lower() if isinstance(network, dict) else ""
    spec = TRAINING_TASKS[task]
    action = spec.architecture_actions.get(arch, spec.actions)["train"]
    if "sequence_pairs" not in action.inputs or config.get("high_order_degradation"):
        raise ValueError(
            "Select a generator that trains on prepared paired frame windows."
        )
    backend = dataset.get("io_backend", {})
    if not isinstance(backend, dict) or backend.get("type", "disk") != "disk":
        raise ValueError(
            "PairedVideoDataset split supports prepared disk frame folders."
        )
    if dataset.get("meta_info") is not None:
        raise ValueError(
            "PairedVideoDataset does not select from meta_info. Use video_scene_manifest or omit meta_info."
        )
    if body.get("groupBy", "scene") != "scene":
        raise ValueError(
            "PairedVideoDataset requires groupBy=scene to keep temporal windows together."
        )
    seed, fraction = body.get("seed", 0), body.get("validationFraction")
    if type(seed) is not int:
        raise ValueError("Provide an integer seed.")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("validationFraction must be strictly between zero and one.")

    root = root.absolute()
    lq = _roots(dataset, "dataroot_lq", root)
    gt = _roots(dataset, "dataroot_gt", root)
    width = dataset.get("clip_size")
    source = dataset.get("video_scene_manifest")
    source = local_path(source, root) if source is not None else None
    requested = local_path(os.fspath(destination), root)
    output = requested.parent.resolve(strict=True) / requested.name
    if requested.name in ("", ".", "..") or any(
        output.is_relative_to(Path(folder).resolve()) for folder in (*lq, *gt)
    ):
        raise ValueError("Choose a new destination outside the source roots.")
    if os.path.lexists(output):
        raise FileExistsError(f"Split destination exists: {output}")

    last = 0.0

    def check() -> None:
        if cancelled is not None and cancelled():
            raise DatasetSplitCancelled("Video split cancelled.")

    def tick(stage: str, done: int, total: int | None) -> None:
        nonlocal last
        now = monotonic()
        if done and done != total and now - last < 0.25:
            return
        last = now
        if progress is not None:
            progress({"stage": stage, "completed": done, "total": total})
        check()

    tick("selecting", 0, None)
    watch = set(lq + gt)
    if source is not None:
        watch.add(str(source))
    versions = {p: _version(os.stat(p)) for p in watch}
    index = FrameWindowIndex(lq, width, gt, scene_manifest=source, check=check)
    scenes, total = index.scenes, len(index)
    if len(scenes) < 2:
        raise ValueError(
            f"{total} eligible windows in {len(scenes)} scenes at clip_size={width}. "
            "Two nonempty partitions require at least two eligible scenes."
        )
    rounded = int(Fraction(str(fraction)) * total + Fraction(1, 2))
    target = max(1, min(total - 1, rounded))
    with (
        TemporaryDirectory(dir=output.parent, prefix=".video-split-") as temporary,
        closing(sqlite3.connect(Path(temporary) / "physical.sqlite")) as db,
    ):
        groups, identity = _audit(db, index, tick)
        sizes = Counter()
        for scene, group in zip(scenes, groups, strict=True):
            sizes[group] += len(scene.starts)
        if len(sizes) < 2:
            raise ValueError(
                f"All {len(scenes)} scenes ({total} windows) share one connected "
                f"group through {identity['sharedAcrossScenes']} shared physical files. "
                f"Two disjoint nonempty partitions are impossible. "
                f"First shared file: {identity['examples'][0]}."
            )
        tick("assigning", 0, total)
        validation = _validation_groups(sizes, seed, target)
        partitions = {"train": [], "val": []}
        for scene, group in zip(scenes, groups, strict=True):
            partitions["val" if group in validation else "train"].append(scene)
        nval = sum(sizes[g] for g in validation)
        if not 0 < nval < total:
            raise ValueError("No nonempty scene-disjoint partition could be assigned.")
        notes = [
            "Scene keys are local to ordered root slots. Only shared physical files link slots. "
            "Copied, re-encoded or neighboring footage with separate identities is outside this audit.",
            "Manifests pin frame names and eligible centers. File identity was checked at split time. "
            "Repeat the split after replacing, relinking or relocating the frame trees.",
        ]
        if (
            isinstance(config.get("val"), dict)
            and config["val"].get("val_enabled") is False
        ):
            notes.append(
                "The source recipe disables scheduled validation. Its validation setting is preserved."
            )
        if nval != target or rounded != target:
            notes.append(
                f"The requested fraction rounds to {rounded} windows. Nonempty whole groups "
                f"give {nval} validation windows for a target of {target}."
            )
        batch = dataset.get("batch_size_per_gpu")
        if type(batch) is int and batch > total - nval:
            notes.append(
                f"Train has {total - nval} windows for a per-device batch of {batch}. "
                "Check sampler enlargement and drop_last."
            )
        staged = Path(temporary) / "split"
        staged.mkdir(mode=0o700)
        manifests, hashes, configs = {}, {}, {}
        for phase, members in partitions.items():
            tick(f"writing-{phase}", 0, len(members))
            manifest = staged / f"{phase}.json"
            encoded = (
                json.dumps(index.manifest(members, check=check), indent=2) + "\n"
            ).encode("ascii")
            manifest.write_bytes(encoded)
            hashes[phase] = hashlib.sha256(encoded).hexdigest()
            replay = FrameWindowIndex(
                lq, width, gt, scene_manifest=manifest, check=check
            )
            if len(replay) != (nval if phase == "val" else total - nval) or [
                (s.slot, s.key) for s in replay.scenes
            ] != [(s.slot, s.key) for s in members]:
                raise ValueError(
                    f"{phase} scene selection failed to replay in loader order."
                )
            del replay
            name = dataset.get("name") or "Paired video"
            manifests[phase] = str(output / manifest.name)
            configs[phase] = {
                **split_dataset_options(dataset, phase),
                "phase": phase,
                "name": name if phase == "train" else f"{name} validation",
                "dataroot_lq": lq,
                "dataroot_gt": gt,
                "video_scene_manifest": manifests[phase],
            }
        text, recipe_keys = render_split_recipe(recipe, configs, notes)
        (staged / "recipe.yml").write_text(text, encoding="utf-8")
        result = {
            "algorithm": "paired-video-scene-components-sha256-v1",
            "task": task,
            "seed": seed,
            "groupBy": "scene",
            "validationFraction": fraction,
            "achievedValidationFraction": nval / total,
            "clipSize": width,
            "counts": {
                "windows": total,
                "scenes": len(scenes),
                "groups": len(sizes),
                "targetVal": target,
                "train": total - nval,
                "val": nval,
                "trainScenes": len(partitions["train"]),
                "valScenes": len(partitions["val"]),
                "trainGroups": len(sizes) - len(validation),
                "valGroups": len(validation),
            },
            "physicalIdentity": identity,
            "datasets": configs,
            "manifests": manifests,
            "manifestSha256": hashes,
            "sourceManifest": str(source) if source else None,
            "destination": str(output),
            "reportPath": str(output / "split.json"),
            "recipePath": str(output / "recipe.yml"),
            "recipeDatasetKeys": recipe_keys,
            "notes": notes,
        }
        (staged / "split.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        for i, (path, stamp) in enumerate(
            db.execute("SELECT path, stamp FROM paths"), 1
        ):
            if _version(_file_info(path)) != stamp:
                raise ValueError(f"Source changed during split: {os.fsdecode(path)}")
            tick("verifying", i, identity["paths"])
        for path, stamp in versions.items():
            if _version(os.stat(path)) != stamp:
                raise ValueError(
                    f"Source root or manifest changed during split: {path}"
                )
        tick("publishing", 0, 1)
        _publish(staged, output)
    return result

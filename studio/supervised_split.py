"""Prepare train/validation membership for the three built-in supervised disk providers."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from .dataset_pack import _publish
from .dataset_split import DatasetSplitCancelled, _validation_groups, render_split_recipe, split_dataset_options
from .paths import local_path
from .split_audit import (
    _SUPERVISED,
    _configured_split,
    _file_info,
    _Input,
    _manifest_stamps,
    _supervised_images,
    disk_file_digest,
)
from .task_data import _category_names, _class_names
from .yaml_edit import read_recipe

_ROOT = Path(__file__).resolve().parents[1]
_ALGORITHM = "supervised-membership-sha256-v1"


def _groups(
    rows: list[_Input],
    by: str,
    metadata: dict[bytes, tuple[int, str]],
    mask_roots: list[str],
    check: Callable[[], None],
    update: Callable[[str, int, int | None], None],
) -> tuple[list[str], dict[str, Any]]:
    # Merge directory groups connected by an identical encoded source. The minimum
    # logical key gives the same component ID after relocating the roots.
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def link(key: str, other: str) -> None:
        a, b = find(key), find(other)
        if a != b:
            parents[max(a, b)] = min(a, b)

    sizes = Counter(metadata[row.path][0] for row in rows)
    cached = {}
    engines: Counter[str] = Counter()
    owners = {}
    physical = {}
    keys = []
    for i, row in enumerate(rows, 1):
        update("identifying", i, len(rows))
        relative = Path(row.label_context.get("fileName", os.fsdecode(row.relative)))
        if by == "sample":
            key = f"{row.root}/{relative.as_posix()}"
        else:
            if "className" in row.label_context:
                relative = Path(*relative.parts[1:])
            key = (
                relative.parent.as_posix()
                if by == "parent"
                else (relative.parts[0] if len(relative.parts) > 1 else ".")
            )
        parents.setdefault(key, key)
        keys.append(key)
        size, stamp = metadata[row.path]
        if mask_roots:
            # A shared mask file is itself a shared source member. Independent masks
            # with identical class IDs are valid targets and do not link images.
            mask = (
                Path(mask_roots[row.root - 1]) / row.label_context["maskRelativePath"]
            )
            for version in (stamp, metadata[os.fsencode(mask)][1]):
                link(key, physical.setdefault(version, key))
        if sizes[size] == 1:
            continue
        if stamp not in cached:
            cached[stamp], engine = disk_file_digest(row.path, size, stamp, check)
            engines[engine] += 1
        identity = (size, cached[stamp])
        other = owners.setdefault(identity, key)
        link(key, other)

    return [find(key) for key in keys], {
        "method": "encoded-size-and-sha256",
        "candidateRows": sum(n for n in sizes.values() if n > 1),
        "hashedFiles": len(cached),
        "hashEngines": dict(engines),
        "duplicateRows": sum(n for n in sizes.values() if n > 1) - len(owners),
    }


def _class_quota(capacities: dict[str, int], target: int) -> dict[str, int]:
    # Retain train coverage, and cover every eligible validation class when affordable.
    eligible = sum(cap > 0 for cap in capacities.values())
    quotas = {
        name: int(cap > 0 and target >= eligible) for name, cap in capacities.items()
    }
    remaining = target - sum(quotas.values())
    if remaining:
        room = sum(capacities.values()) - sum(quotas.values())
        remainders = {}
        for name, cap in capacities.items():
            extra, remainders[name] = divmod((cap - quotas[name]) * remaining, room)
            quotas[name] += extra
        for name in sorted(quotas, key=remainders.__getitem__, reverse=True)[
            : target - sum(quotas.values())
        ]:
            quotas[name] += 1
    return quotas


def _assign(
    rows: list[_Input],
    groups: list[str],
    names: list[str] | None,
    seed: int,
    target: int,
) -> set[str]:
    sizes = Counter(groups)
    if len(sizes) < 2:
        raise ValueError(
            "At least two independent image groups are needed after grouping identical sources."
        )
    if names is None:
        return _validation_groups(sizes, seed, target)

    by_class = {name: Counter() for name in names}
    labels = defaultdict(set)
    for row, group in zip(rows, groups, strict=True):
        name = row.label_context["className"]
        if name not in by_class:
            raise ValueError(
                f"Selected class folder {name!r} is absent from task_options.class_names."
            )
        by_class[name][group] += 1
        labels[group].add(name)
    missing = [name for name, counts in by_class.items() if not counts]
    if missing:
        raise ValueError(
            f"Source has no training examples for configured classes {missing}."
        )

    if all(len(values) == 1 for values in labels.values()):
        capacities = {
            name: counts.total() - min(counts.values())
            for name, counts in by_class.items()
        }
        quotas = _class_quota(capacities, min(target, sum(capacities.values())))
        selected = set()
        for name, count in quotas.items():
            if count:
                selected.update(_validation_groups(by_class[name], seed, count))
    else:
        # Shared scenes or duplicate source images can connect several classes.
        selected = _validation_groups(sizes, seed, target)
        uncovered = {
            name for name, counts in by_class.items() if counts.keys() <= selected
        }
        for key in sorted(selected, key=lambda k: (sizes[k], k)):
            if labels[key] & uncovered:
                selected.remove(key)
                uncovered.difference_update(labels[key])
        if not selected:
            candidates = [
                key
                for key, group_labels in labels.items()
                if all(len(by_class[name]) > 1 for name in group_labels)
            ]
            if candidates:
                selected.add(min(candidates, key=lambda k: (abs(sizes[k] - target), k)))
    if not selected:
        raise ValueError(
            "Every possible validation group would remove a configured class from training."
        )
    return selected


def _write_manifest(
    path: Path, rows: list[_Input], coco: dict | None, check: Callable[[], None]
) -> str:
    digest = hashlib.sha256()
    with path.open("xb") as stream:

        def write(data: bytes) -> None:
            stream.write(data)
            digest.update(data)

        if coco is None:
            for row in rows:
                check()
                entry = {
                    "root": row.root,
                    "image": Path(os.fsdecode(row.relative)).as_posix(),
                }
                if "maskRelativePath" in row.label_context:
                    entry["mask"] = Path(
                        row.label_context["maskRelativePath"]
                    ).as_posix()
                write(
                    json.dumps(entry, ensure_ascii=True, separators=(",", ":")).encode(
                        "ascii"
                    )
                    + b"\n"
                )
        else:
            ids = {row.label_context["cocoImageId"] for row in rows}

            # Retain even unconsumed fields on images, annotations, and categories.
            def encode(value: Any) -> bytes:
                return json.dumps(
                    value, ensure_ascii=True, separators=(",", ":"), allow_nan=False
                ).encode("ascii")

            write(b"{")
            for column, (field, value) in enumerate(coco.items()):
                if column:
                    write(b",")
                write(encode(field) + b":")
                if field not in ("images", "annotations"):
                    write(encode(value))
                    continue
                foreign_key = "id" if field == "images" else "image_id"
                write(b"[")
                first = True
                for item in value:
                    check()
                    if item[foreign_key] in ids:
                        if not first:
                            write(b",")
                        write(encode(item))
                        first = False
                write(b"]")
            write(b"}\n")
    return digest.hexdigest()


def split_supervised_dataset(
    body: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    root: Path = _ROOT,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Publish membership, recipe sections and split.json under a new directory."""
    task, dataset, settings = (
        body.get(key) for key in ("task", "dataset", "task_options")
    )
    recipe_text = body.get("yaml")
    if recipe_text is not None:
        source = read_recipe(recipe_text)
        datasets = source.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("The source recipe requires datasets.train.")
        selected = (
            source.get("task"),
            datasets.get("train"),
            source.get("task_options"),
        )
        for key, expected in zip(
            ("task", "dataset", "task_options"), selected, strict=True
        ):
            if key in body and body[key] != expected:
                raise ValueError(f"The supplied {key} differs from the source recipe.")
        task, dataset, settings = selected
    if (
        not isinstance(task, str)
        or task not in _SUPERVISED
        or not isinstance(dataset, Mapping)
    ):
        raise ValueError("Choose one built-in supervised task and provide its dataset.")
    if not isinstance(settings, Mapping):
        raise ValueError("Provide the original task_options to preserve label order.")
    seed, fraction = body.get("seed"), body.get("validationFraction")
    if (
        type(seed) is not int
        or type(fraction) not in (int, float)
        or not 0 < fraction < 1
    ):
        raise ValueError(
            "Provide an integer seed and a validationFraction strictly between zero and one."
        )
    by = body.get("groupBy", "sample")
    if by == "pair":  # existing split-dataset client default
        by = "sample"
    if by not in ("sample", "parent", "scene"):
        raise ValueError("groupBy must be sample, parent, or scene.")
    classification = task == "image_classification"
    detection = task == "object_detection"
    names = None
    if detection:
        if _category_names(settings) is None:
            raise ValueError(
                "Provide task_options.categories in the original foreground label order."
            )
        if dataset.get("meta_info") is not None:
            raise ValueError("CocoDetectionDataset selects rows from annotation_file.")
    else:
        names = _class_names(settings, folders=classification)
        if names is None:
            raise ValueError(
                "Provide the original task_options.class_names in output ID order."
            )
        if task == "semantic_segmentation":
            ignore = settings.get("ignore_label", 255)
            if type(ignore) is not int or 0 <= ignore < len(names):
                raise ValueError(
                    "ignore_label must be an integer outside the configured class IDs."
                )
            crop = (
                dataset.get("gt_size")
                if dataset.get("gt_size") is not None
                else dataset.get("lq_size")
            )
            if type(crop) is not int or crop <= 0:
                raise ValueError(
                    "ImageMaskDataset training needs a positive gt_size or lq_size."
                )
    root = root.absolute()
    split = _configured_split("source", dataset, root, task, settings)
    requested = local_path(os.fspath(destination), root)
    if requested.name in ("", ".", ".."):
        raise ValueError("Provide a new destination directory.")
    output = requested.parent.resolve(strict=True) / requested.name
    if os.path.lexists(output):
        raise FileExistsError(f"Destination already exists: {output}")
    if any(
        output.is_relative_to(Path(p).resolve())
        for roots in split.roots.values()
        for p in roots
    ):
        raise ValueError("Place the destination outside the source roots.")

    next_event = next_check = 0.0

    def check() -> None:
        nonlocal next_check
        if cancelled is not None and monotonic() >= next_check:
            next_check = monotonic() + 0.1
            if cancelled():
                raise DatasetSplitCancelled("Supervised split cancelled.")

    def update(stage: str, done: int, total: int | None) -> None:
        nonlocal next_event
        check()
        if progress is not None and (done in (0, total) or monotonic() >= next_event):
            next_event = monotonic() + 0.25
            progress({"stage": stage, "completed": done, "total": total})

    source_stamp = _manifest_stamps(split)
    rows = []
    metadata = {}
    update("selecting", 0, None)
    for (row,) in _supervised_images(split, check):
        rows.append(row)
        paths = [row.path]
        if task == "semantic_segmentation":
            mask = (
                Path(split.roots["dataroot_gt"][row.root - 1])
                / row.label_context["maskRelativePath"]
            )
            paths.append(os.fsencode(mask))
        for path in paths:
            if path not in metadata:
                metadata[path] = _file_info(path)
        update("selecting", len(rows), None)
    total = len(rows)
    if total < 2:
        raise ValueError(
            f"The source has {total} selected images. Two nonempty splits require at least two."
        )
    groups, identity = _groups(
        rows, by, metadata, split.roots.get("dataroot_gt", []), check, update
    )
    target = max(
        1, min(total - 1, int(Fraction(str(fraction)) * total + Fraction(1, 2)))
    )
    update("assigning", 0, total)
    chosen = _assign(rows, groups, names if classification else None, seed, target)
    partitions = {"train": [], "val": []}
    for row, group in zip(rows, groups, strict=True):
        partitions["val" if group in chosen else "train"].append(row)
    val = len(partitions["val"])
    if not 0 < val < total:
        raise ValueError(
            "No nonempty partition is possible under the requested grouping."
        )

    key = "annotation_file" if detection else "meta_info"
    suffix = "json" if detection else "jsonl"
    configs, manifests = {}, {}
    name = dataset.get("name") or task
    for phase in partitions:
        manifest = output / f"{phase}.{suffix}"
        manifests[phase] = str(manifest)
        configs[phase] = {
            **split_dataset_options(dataset, phase),
            **split.roots,
            "phase": phase,
            "name": name if phase == "train" else f"{name} validation",
            key: str(manifest),
        }
    notes = []
    if val != target:
        notes.append(
            f"Whole groups and training coverage give {val} validation samples for a target of {target}."
        )
    batch = dataset.get("batch_size_per_gpu")
    if type(batch) is int and batch > total - val:
        notes.append(
            f"The training split contains {total - val} {'sample' if total - val == 1 else 'samples'} "
            f"for a per-device batch of {batch}. Check sampler enlargement and drop_last."
        )
    result = {
        "algorithm": _ALGORITHM,
        "task": task,
        "seed": seed,
        "groupBy": by,
        "validationFraction": fraction,
        "achievedValidationFraction": val / total,
        "destination": str(output),
        "reportPath": str(output / "split.json"),
        "task_options": dict(settings),
        "datasets": configs,
        "manifests": manifests,
        "manifestSha256": {},
        "counts": {
            "samples": total,
            "groups": len(set(groups)),
            "targetVal": target,
            "train": total - val,
            "val": val,
            "valGroups": len(chosen),
            "trainGroups": len(set(groups)) - len(chosen),
        },
        "imageIdentity": identity,
        "sourceManifest": split.manifest,
        "notes": notes,
    }
    if classification:
        counts = {
            phase: Counter(row.label_context["className"] for row in members)
            for phase, members in partitions.items()
        }
        result["classes"] = [
            {
                "id": i,
                "name": name,
                "train": counts["train"][name],
                "val": counts["val"][name],
            }
            for i, name in enumerate(names)
        ]
        absent = [name for name in names if not counts["val"][name]]
        if absent:
            notes.append(
                f"Validation has no samples for {', '.join(absent)}. Class IDs are preserved."
            )

    coco = None
    if detection:
        with Path(split.manifest).open("rb") as stream:
            coco = json.load(stream)
        extra = {k for k, v in coco.items() if isinstance(v, (dict, list))} - {
            "images",
            "categories",
            "annotations",
            "info",
            "licenses",
        }
        if extra:
            raise ValueError(
                f"Cannot partition extra COCO top-level tables: {sorted(extra)}."
            )
    with TemporaryDirectory(
        dir=output.parent, prefix=".supervised-split-"
    ) as temporary:
        staged = Path(temporary) / "split"
        staged.mkdir(mode=0o700)
        for phase, members in partitions.items():
            manifest = staged / f"{phase}.{suffix}"
            update(f"writing-{phase}", 0, len(members))
            result["manifestSha256"][phase] = _write_manifest(
                manifest, members, coco, check
            )
            replay = _configured_split(
                phase, {**configs[phase], key: str(manifest)}, root, task, settings
            )
            for i, ((observed,), expected) in enumerate(
                zip(_supervised_images(replay, check), members, strict=True), 1
            ):
                if observed != expected:
                    raise ValueError(
                        f"The {phase} membership failed to replay at row {i}."
                    )
                update(f"verifying-{phase}", i, len(members))
        for path, before in metadata.items():
            check()
            if _file_info(path) != before:
                raise ValueError(
                    f"Source changed during splitting: {os.fsdecode(path)}"
                )
        if source_stamp != _manifest_stamps(split):
            raise ValueError("Source selection manifest changed during splitting.")
        if recipe_text is not None:
            rendered, result["recipeDatasetKeys"] = render_split_recipe(recipe_text, configs, notes)
            (staged / "recipe.yml").write_text(rendered, encoding="utf-8")
            result["recipePath"] = str(output / "recipe.yml")
        (staged / "split.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        if cancelled is not None and cancelled():
            raise DatasetSplitCancelled("Supervised split cancelled.")
        _publish(staged, output)
    return result

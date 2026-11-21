"""Split one disk PairedImageDataset into replayable GT-name manifests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from trainner_data.manifests import (
    manifest_filename,
    paired_manifest_lq_path,
    paired_paths_from_meta_info_file,
)

from .dataset_pack import _publish
from .paths import local_path
from .yaml_edit import read_recipe, render

_ROOT = Path(__file__).resolve().parents[1]
_ALGORITHM = "paired-manifest-sha256-v1"


def split_dataset_options(dataset: Mapping[str, Any], phase: str) -> dict[str, Any]:
    if phase == "train":
        return dict(dataset)
    training = {"lq_size", "gt_size", "use_hflip", "use_rot", "batch_size_per_gpu",
                "accum_iter", "dataset_enlarge_ratio", "native_paired_crop"}
    return {k: v for k, v in dataset.items() if k not in training}


def render_split_recipe(text: str, configs: dict, notes: list[str]) -> tuple[str, dict[str, str]]:
    source = read_recipe(text)["datasets"]
    key, n = "val", 1
    while key in source:
        key = "val_split" if n == 1 else f"val_split{n}"
        n += 1
    used = {dataset.get("name") for dataset in source.values() if isinstance(dataset, dict)}
    base = configs["val"]["name"]
    display, n = base, 2
    while display in used:
        display = f"{base} ({n})"
        n += 1
    configs["val"]["name"] = display
    if any(k != "train" for k in source):
        notes.append("Existing evaluation datasets are preserved. This split checked only the new train and validation selections.")
    keys = {"train": "train", "val": key}
    updates = {actual: configs[phase] for phase, actual in keys.items()}
    try:
        rendered = render(text, [
            {"path": ["datasets", key], "value": value} for key, value in updates.items()
        ])
    except ValueError:
        # Other datasets can reference anchors declared inside the old training
        # dataset. The private copy must still contain their original settings.
        import yaml

        config = read_recipe(text)
        config["datasets"] = {**source, **updates}
        rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if read_recipe(rendered) != config:
            raise ValueError("Could not preserve the source recipe while splitting its training set.")
        notes.append("The generated YAML was reserialized to preserve shared settings. Copy comments from the source if needed.")
    return rendered, keys


_KEYS = ("lq", "gt")


class DatasetSplitCancelled(RuntimeError):
    """The job stopped before publishing the manifests."""


@dataclass(frozen=True, slots=True)
class _Pair:
    gt: str
    lq: str
    group: str
    stamp: bytes


def _one_root(dataset: Mapping[str, Any], key: str, root: Path) -> Path:
    value = dataset.get(key)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(
            f"Provide exactly one {key}. The loader applies a manifest to every root pair."
        )
    path = local_path(value, root).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Expected a directory for {key}: {path}")
    if path.suffix.lower() == ".lmdb" or (path / "data.mdb").is_file():
        raise ValueError(
            f"LMDB membership cannot be split with a folder meta_info file: {path}"
        )
    return path


def _file_info(path: str) -> os.stat_result:
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular file at the selected path: {path}")
    return info


def _stamp(*files: os.stat_result) -> bytes:
    # This is only an in-job change check. Manifests and assignment use relative names.
    stamps = [
        (f.st_dev, f.st_ino, f.st_size, f.st_mtime_ns, f.st_ctime_ns) for f in files
    ]
    return hashlib.sha256(repr(stamps).encode("ascii")).digest()


def paired_disk_rows(
    gt_root: Path, lq_root: Path, template: str, source_meta: Path | None
) -> list[dict[str, str]]:
    """Select one disk root pair exactly as PairedImageDataset does."""
    folders = ([str(lq_root)], [str(gt_root)])
    try:
        if source_meta is not None:
            return paired_paths_from_meta_info_file(
                folders, _KEYS, str(source_meta), template
            )
        else:
            from traiNNer.data.data_util import paired_paths_from_folder

            return paired_paths_from_folder(folders, _KEYS, template)
    except (AttributeError, IndexError, KeyError) as exc:
        raise ValueError(
            f"Invalid filename_tmpl for the paired selector: {template!r}"
        ) from exc


def _membership(
    gt_root: Path,
    lq_root: Path,
    template: str,
    source_meta: Path | None,
    grouping: str,
    tick: Callable[[str, int, int | None], None],
) -> list[_Pair]:
    tick("selecting", 0, None)
    rows = paired_disk_rows(gt_root, lq_root, template, source_meta)

    members = []
    owners: dict[tuple[int, int], str] = {}
    for i, row in enumerate(rows, 1):
        tick("selecting", i, len(rows))
        gt, lq = row["gt_path"], row["lq_path"]
        relative = Path(gt).relative_to(gt_root)
        lq_relative = Path(lq).relative_to(lq_root)
        if ".." in relative.parts or ".." in lq_relative.parts:
            raise ValueError(
                f"Manifest membership must stay under its two roots: {gt!r}, {lq!r}"
            )
        name = relative.as_posix()
        name.encode("utf-8")  # The published manifest is ordinary UTF-8 text.
        if (
            "\n" in name
            or "\r" in name
            or manifest_filename(name) != name
            or not name.strip()
        ):
            raise ValueError(
                f"GT name cannot round-trip through a trainer meta_info line: {name!r}"
            )
        # A newly added nested file could shadow a legacy flat fallback.
        projected = paired_manifest_lq_path(
            str(lq_root), name, template, flat_fallback=False
        )
        if Path(projected) != Path(lq):
            raise ValueError(
                f"GT {name!r} selects LQ {str(lq_relative)!r} in the source, but a trainer "
                f"stable meta_info entry requires {projected!r}. The loader derives the LQ "
                "extension from GT. A legacy flat fallback could later be shadowed."
            )
        gt_info, lq_info = _file_info(gt), _file_info(lq)
        for info in (gt_info, lq_info):
            owner = owners.setdefault((info.st_dev, info.st_ino), name)
            if owner != name:
                raise ValueError(
                    f"Distinct pairs {owner!r} and {name!r} share a source file. Resolve aliases before splitting."
                )
        if grouping == "parent":
            group = relative.parent.as_posix()
        elif grouping == "scene":
            group = relative.parts[0] if len(relative.parts) > 1 else "."
        else:
            group = name
        members.append(_Pair(name, str(lq_relative), group, _stamp(gt_info, lq_info)))
    return members


def _validation_groups(sizes: Counter[str], seed: int, target: int) -> set[str]:
    prefix = f"{_ALGORITHM}\0{seed}\0".encode("ascii")

    def priority(key: str) -> tuple[bytes, str]:
        return hashlib.sha256(prefix + key.encode("utf-8")).digest(), key

    total = sizes.total()
    invert = target > total - target
    if invert:
        target = total - target
    ranked = sorted(sizes, key=priority)
    selected: set[str] = set()
    used = 0
    smallest_skipped = None
    for key in ranked:
        weight = sizes[key]
        if used + weight <= target:
            selected.add(key)
            used += weight
            if used == target:
                break
        elif smallest_skipped is None or weight < sizes[smallest_skipped]:
            smallest_skipped = key
    # Optionally take the least oversized group, keeping both sides nonempty.
    if smallest_skipped is not None and len(selected) + 1 < len(sizes):
        overshoot = used + sizes[smallest_skipped] - target
        if not selected or overshoot < target - used:
            selected.add(smallest_skipped)
            used += sizes[smallest_skipped]
    if used != target:
        single = min(ranked, key=lambda key: abs(sizes[key] - target))
        if abs(sizes[single] - target) < abs(used - target):
            selected = {single}
    return set(sizes).difference(selected) if invert else selected


def split_dataset(
    body: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    root: Path = _ROOT,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Publish train.txt, val.txt and split.json under a new private destination."""
    dataset = body.get("dataset")
    if (
        body.get("task", "image_restoration") != "image_restoration"
        or not isinstance(dataset, Mapping)
        or str(dataset.get("type", "")).lower() != "pairedimagedataset"
    ):
        raise ValueError("Choose one image_restoration PairedImageDataset.")
    backend = dataset.get("io_backend", {"type": "disk"})
    if not isinstance(backend, Mapping) or backend.get("type") != "disk":
        raise ValueError(
            "Splitting supports disk folders and folder meta_info. LMDB ignores an external meta_info file."
        )
    seed, fraction = body.get("seed"), body.get("validationFraction")
    if type(seed) is not int:
        raise ValueError("Provide an integer seed.")
    if type(fraction) not in (float, int) or not 0 < fraction < 1:
        raise ValueError(
            "validationFraction must be a number strictly between zero and one."
        )
    grouping = body.get("groupBy", "pair")
    if grouping not in ("pair", "parent", "scene"):
        raise ValueError("groupBy must be pair, parent, or scene.")
    template = dataset.get("filename_tmpl", "{}")
    if not isinstance(template, str) or "\0" in template:
        raise ValueError("Provide filename_tmpl as a string without NUL characters.")
    root = root.absolute()
    gt = _one_root(dataset, "dataroot_gt", root)
    lq = _one_root(dataset, "dataroot_lq", root)
    meta = dataset.get("meta_info")
    source_meta = local_path(meta, root) if meta is not None else None
    meta_stamp = (
        _stamp(_file_info(str(source_meta))) if source_meta is not None else None
    )
    requested = local_path(os.fspath(destination), root)
    if requested.name in ("", ".", ".."):
        raise ValueError("Provide a new destination directory.")
    output = requested.parent.resolve(strict=True) / requested.name
    if output.is_relative_to(gt) or output.is_relative_to(lq):
        raise ValueError("Place the manifest destination outside both source roots.")
    if os.path.lexists(output):
        raise FileExistsError(f"Manifest destination already exists: {output}")

    last_event = 0.0

    def tick(stage: str, done: int, total: int | None) -> None:
        nonlocal last_event
        now = monotonic()
        if done not in (0, total) and now - last_event < 0.25:
            return
        last_event = now
        if progress is not None:
            progress({"stage": stage, "completed": done, "total": total})
        if cancelled is not None and cancelled():
            raise DatasetSplitCancelled("Dataset split cancelled.")

    members = _membership(gt, lq, template, source_meta, grouping, tick)
    total = len(members)
    sizes = Counter(member.group for member in members)
    if total < 2 or len(sizes) < 2:
        raise ValueError(
            f"The selector returned {total} paired rows in {len(sizes)} {grouping} groups. "
            "Two nonempty splits require at least two independently assignable groups."
        )
    # Decimal fraction from the request, rounded half-up to a row count.
    desired = Fraction(str(fraction)) * total
    rounded = int(desired + Fraction(1, 2))
    target = max(1, min(rounded, total - 1))
    tick("assigning", 0, total)
    val_groups = _validation_groups(sizes, seed, target)
    partitions: dict[str, list[_Pair]] = {"train": [], "val": []}
    for i, member in enumerate(members, 1):
        tick("assigning", i, total)
        phase = "val" if member.group in val_groups else "train"
        partitions[phase].append(member)
    del members
    ntrain, nval = len(partitions["train"]), len(partitions["val"])
    notes = []
    if rounded != target:
        notes.append(
            f"The requested fraction rounds to {rounded} validation rows. The target is {target} to keep both splits nonempty."
        )
    if nval != target:
        notes.append(
            f"Keeping {grouping} groups intact assigns {nval} of {total} rows to validation. The target was {target}."
        )
    batch = dataset.get("batch_size_per_gpu")
    if type(batch) is int and ntrain < batch:
        notes.append(
            f"The train manifest has {ntrain} rows for a configured per-device batch of {batch}. Check sampler enlargement and drop_last."
        )
    name = dataset.get("name") or "Paired images"
    manifests = {phase: str(output / f"{phase}.txt") for phase in partitions}
    configs = {}
    for phase, path in manifests.items():
        config = split_dataset_options(dataset, phase)
        config.update(
            name=name if phase == "train" else f"{name} validation",
            dataroot_gt=[str(gt)],
            dataroot_lq=[str(lq)],
            meta_info=path,
        )
        if "phase" in config:
            config["phase"] = phase
        configs[phase] = config
    result = {
        "algorithm": _ALGORITHM,
        "seed": seed,
        "groupBy": grouping,
        "validationFraction": fraction,
        "achievedValidationFraction": nval / total,
        "counts": {
            "pairs": total,
            "groups": len(sizes),
            "targetVal": target,
            "train": ntrain,
            "val": nval,
            "trainGroups": len(sizes) - len(val_groups),
            "valGroups": len(val_groups),
        },
        "sourceMetaInfo": str(source_meta) if source_meta is not None else None,
        "destination": str(output),
        "manifests": manifests,
        "datasets": configs,
        "manifestSha256": {},
        "reportPath": str(output / "split.json"),
        "notes": notes,
    }

    # Stage on the destination filesystem, consume the real trainer reader,
    # then publish the complete directory with the pack worker's no-replace rename.
    folders = ([str(lq)], [str(gt)])
    tick("writing", 0, total)
    with TemporaryDirectory(dir=output.parent, prefix=".dataset-split-") as temporary:
        staged = Path(temporary) / "split"
        staged.mkdir(mode=0o700)
        written = verified = 0
        for phase, pairs in partitions.items():
            manifest = staged / f"{phase}.txt"
            digest = hashlib.sha256()
            with manifest.open("xb") as out:
                for pair in pairs:
                    line = (pair.gt + "\n").encode("utf-8")
                    out.write(line)
                    digest.update(line)
                    written += 1
                    tick("writing", written, total)
            result["manifestSha256"][phase] = digest.hexdigest()

        tick("verifying", 0, total)
        for phase, pairs in partitions.items():
            rows = paired_paths_from_meta_info_file(
                folders, _KEYS, str(staged / f"{phase}.txt"), template
            )
            for pair, row in zip(pairs, rows, strict=True):
                gt_path, lq_path = row["gt_path"], row["lq_path"]
                if (
                    Path(gt_path) != gt / pair.gt
                    or Path(lq_path) != lq / pair.lq
                    or _stamp(_file_info(gt_path), _file_info(lq_path)) != pair.stamp
                ):
                    raise ValueError(
                        f"Selected pair changed while verifying {phase}: {pair.gt!r}"
                    )
                verified += 1
                tick("verifying", verified, total)
            del rows
        if body.get("yaml") is not None:
            rendered, result["recipeDatasetKeys"] = render_split_recipe(body["yaml"], configs, notes)
            (staged / "recipe.yml").write_text(rendered, encoding="utf-8")
            result["recipePath"] = str(output / "recipe.yml")
        (staged / "split.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        if (
            source_meta is not None
            and _stamp(_file_info(str(source_meta))) != meta_stamp
        ):
            raise ValueError(f"Source manifest changed during splitting: {source_meta}")
        if cancelled is not None and cancelled():
            raise DatasetSplitCancelled("Dataset split cancelled.")
        _publish(staged, output)
    return result

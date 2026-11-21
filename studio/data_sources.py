"""Inspect the explicitly selected records from the trainer's manifest readers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from os import path as osp
from typing import Any

from studio.data import _counts

_SAMPLE_LIMIT = 100
_ISSUE_LIMIT = 200
Size = tuple[int, int]
Issue = Callable[[str, str, str], None]


def _geometry(
    gt: Size | None,
    lq: Size | None,
    gt_path: str,
    lq_path: str,
    scale: int | None,
    patch: int | None,
    phase: str,
    issue: Issue,
) -> str:
    small = False
    if phase == "train" and patch is not None:
        if gt is not None and min(gt) < patch:
            issue(
                "error", gt_path, f"GT is smaller than the requested patch ({patch})."
            )
            small = True
        if lq is not None and scale is not None and min(lq) < patch // scale:
            issue(
                "error",
                lq_path,
                f"LQ is smaller than the required patch ({patch // scale}).",
            )
            small = True
    if gt is None or lq is None or scale is None:
        return "mismatch" if small else "unknown"
    required = (lq[0] * scale, lq[1] * scale)
    if gt == required:
        return "mismatch" if small else "exact"
    crop = phase != "train" and gt[0] >= required[0] and gt[1] >= required[1]
    condition = (
        "Validation crops excess GT at the top-left." if crop else "Scale mismatch."
    )
    issue(
        "warning" if crop else "error",
        gt_path,
        f"{condition} GT is {gt[0]} x {gt[1]}, LQ is {lq[0]} x {lq[1]}, "
        f"and the requested scale is {scale}.",
    )
    return "crop" if crop else "mismatch"


def scan_explicit(
    source: str,
    gt_paths: list[str],
    lq_paths: list[str],
    template: str,
    meta_info: str | None,
    scale: int | None,
    patch: int | None,
    phase: str,
) -> dict[str, Any]:
    import pyvips
    from trainner_data.manifests import (
        paired_paths_from_lmdb,
        paired_paths_from_meta_info_file,
    )

    is_lmdb = source == "lmdb"
    reader = None
    read_errors: tuple[type[Exception], ...] = (OSError, ValueError, pyvips.Error)
    if is_lmdb:
        try:
            from lmdb import Error as LmdbError
        except ImportError as exc:
            raise ValueError(
                "The lmdb reader is unavailable in this trainer environment."
            ) from exc
        from trainner_data.lmdb import PairedLmdbReader

        read_errors += (LmdbError,)
        reader = PairedLmdbReader({"type": "lmdb"})

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    roots: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    total_rows = 0
    issue_count = 0
    header_rows = 0

    def issue(level: str, path: str, message: str, root_id: str) -> None:
        nonlocal issue_count
        issue_count += 1
        bucket = errors if level == "error" else warnings
        if len(bucket) < _ISSUE_LIMIT:
            bucket.append(
                {"level": level, "path": path, "message": message, "rootId": root_id}
            )

    def size(path: str, key: str | None, root_id: str) -> Size | None:
        try:
            if key is None:
                if not osp.isfile(path):
                    raise FileNotFoundError(
                        "No regular image file at the selected path."
                    )
                image = pyvips.Image.new_from_file(path, access="sequential", fail=True)
            else:
                assert reader is not None
                image = pyvips.Image.new_from_buffer(
                    reader.read(path, key), "", access="sequential", fail=True
                )
            return int(image.width), int(image.height)
        except read_errors as exc:
            display = path if key is None else f"{path} [{key}]"
            issue(
                "error",
                display,
                f"Could not read the selected image header. {exc}",
                root_id,
            )
            return None

    for index, (gt_root, lq_root) in enumerate(zip(gt_paths, lq_paths, strict=True), 1):
        root_id = str(index)
        folders = ([lq_root], [gt_root])
        try:
            if is_lmdb:
                rows = paired_paths_from_lmdb(
                    folders, ("lq", "gt"), include_storage=True
                )
            else:
                assert meta_info is not None
                rows = paired_paths_from_meta_info_file(
                    folders, ("lq", "gt"), meta_info, template
                )
        except (
            OSError,
            ValueError,
            IndexError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            raise ValueError(
                f"Could not select the {source} records for root {root_id}. {exc}"
            ) from exc

        counts = _counts(len(rows), len(rows), len(rows))
        roots.append({"id": root_id, "gt": gt_root, "lq": lq_root, "counts": counts})
        total_rows += len(rows)
        seen_gt: set[str] = set()
        seen_lq: set[str] = set()
        for row_index, row in enumerate(rows, 1):
            if is_lmdb and len(samples) == _SAMPLE_LIMIT:
                # Do not copy every encoded value in a large database to Python.
                break
            gt, lq = row["gt_path"], row["lq_path"]
            if is_lmdb:
                gt_path, lq_path = row["gt_db"], row["lq_db"]
                name = gt
                gt_key, lq_key = gt, lq
            else:
                gt_path, lq_path = gt, lq
                name = osp.relpath(gt_path, gt_root)
                gt_key = lq_key = None
                for path, side, seen in (
                    (gt_path, "GT", seen_gt),
                    (lq_path, "LQ", seen_lq),
                ):
                    if path in seen:
                        issue(
                            "warning",
                            path,
                            f"The manifest selects the same {side} path more than once.",
                            root_id,
                        )
                    seen.add(path)
            gt_size = size(gt_path, gt_key, root_id)
            lq_size = size(lq_path, lq_key, root_id)
            header_rows += 1
            geometry = _geometry(
                gt_size,
                lq_size,
                gt_path,
                lq_path,
                scale,
                patch,
                phase,
                partial(issue, root_id=root_id),
            )
            if len(samples) < _SAMPLE_LIMIT:
                sample = {
                    "id": f"{root_id}:{row_index}:{name}",
                    "rootId": root_id,
                    "name": name,
                    "gt": gt_path,
                    "lq": lq_path,
                    "geometry": geometry,
                }
                if is_lmdb:
                    sample.update(gtKey=gt_key, lqKey=lq_key)
                if gt_size is not None:
                    sample.update(gtWidth=gt_size[0], gtHeight=gt_size[1])
                if lq_size is not None:
                    sample.update(lqWidth=lq_size[0], lqHeight=lq_size[1])
                samples.append(sample)

    if total_rows == 0:
        issue(
            "error",
            meta_info or gt_paths[0],
            "The selected manifests contain no pairs.",
            "1",
        )
    issues = (errors + warnings)[:_ISSUE_LIMIT]
    return {
        "source": source,
        "ok": not errors,
        "counts": _counts(total_rows, total_rows, total_rows),
        "roots": roots,
        "issues": issues,
        "samples": samples,
        "headerRowsChecked": header_rows,
        "truncated": len(samples) < total_rows or len(issues) < issue_count,
        "limits": {"samples": _SAMPLE_LIMIT, "issues": _ISSUE_LIMIT},
        "totals": {"samples": total_rows, "issues": issue_count},
    }

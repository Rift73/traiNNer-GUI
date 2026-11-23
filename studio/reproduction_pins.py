"""Bind local checkpoint observations to an exact prepared recipe without writes."""

from __future__ import annotations

import hashlib
import json
import re

from trainner_config.official_weights import NEUFLOW_SHA256, neuflow_cache_path

_SHA = re.compile(r"[0-9a-f]{64}")
_ROLE = re.compile(r"resume_state|neuflow_weights|pretrain_network_[a-z0-9_]+")


def _digest(body: dict) -> str:
    data = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def input_precondition(
    text: str, bundle_digest: str, inspected: list[dict]
) -> dict | None:
    files = {}
    for row in inspected:
        if row["kind"] == "dataset":
            continue
        if row["status"] == "downloadPending":
            files["neuflow_weights"] = {"path": row["path"], "sha256": NEUFLOW_SHA256}
            continue
        current = row["current"]
        files[row["field"][1]] = {
            "path": row["path"],
            "fileVersion": current["fileVersion"],
        }
        if row["status"] == "sha256Match":
            files[row["field"][1]]["sha256"] = current["sha256"]
    if not files:
        return None
    body = {
        "version": 1,
        "bundleDigest": bundle_digest,
        "recipeSha256": hashlib.sha256(text.encode()).hexdigest(),
        "files": files,
    }
    return {**body, "digest": _digest(body)}


def checked_files(condition: dict, text: str) -> dict:
    """Validate the caller's comparison values without touching mutable files."""
    keys = {"version", "digest", "bundleDigest", "recipeSha256", "files"}
    if (
        not isinstance(condition, dict)
        or condition.keys() != keys
        or type(condition["version"]) is not int
        or condition["version"] != 1
    ):
        raise ValueError(
            "Provide the complete version 1 inputPrecondition from preparation."
        )
    for key in ("digest", "bundleDigest", "recipeSha256"):
        if not isinstance(condition[key], str) or not _SHA.fullmatch(condition[key]):
            raise ValueError("Invalid digest in the input precondition.")
    files = condition["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 32:
        raise ValueError(
            "The prepared checkpoint file observations are missing or invalid."
        )
    for role, pin in files.items():
        if role == "neuflow_weights" and pin == {
            "path": str(neuflow_cache_path()),
            "sha256": NEUFLOW_SHA256,
        }:
            continue
        if (
            not isinstance(role, str)
            or not _ROLE.fullmatch(role)
            or not isinstance(pin, dict)
            or pin.keys()
            not in ({"path", "fileVersion"}, {"path", "fileVersion", "sha256"})
            or not isinstance(pin["path"], str)
            or not pin["path"].startswith("/")
        ):
            raise ValueError(
                "Invalid checkpoint role or path in the input precondition."
            )
        for key in pin.keys() - {"path"}:
            if not isinstance(pin[key], str) or not _SHA.fullmatch(pin[key]):
                raise ValueError(f"Invalid {key} for prepared checkpoint {role}.")
    if condition["recipeSha256"] != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError("The YAML has changed since preparation. Prepare it again.")
    if condition["digest"] != _digest(
        {k: v for k, v in condition.items() if k != "digest"}
    ):
        raise ValueError(
            "The input precondition has changed. Prepare the recipe again."
        )
    return files

"""Identify a published set of weights after its project is copied or reopened."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict
from zipfile import BadZipFile, ZipFile

from .checkpoint_association import SAVE_ID, SAVE_ID_KEY, pth_save_id

_RECEIPT = "/.data/trainner_weights.json"


class PublishedWeight(TypedDict):
    id: str
    bytes: int


def record_weights(
    state: str,
    step: int,
    weights: dict[str, PublishedWeight],
    seed: int | None,
    runtime: str,
) -> None:
    """Append to the temporary Torch archive before the trainer publishes the state."""
    with ZipFile(state, "a") as archive:
        (root,) = [
            name.removesuffix("/data.pkl")
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith("/data.pkl")
        ]
        receipt = {
            "version": 1,
            "step": step,
            "weights": weights,
            "seed": seed,
            "torch": runtime,
        }
        archive.writestr(root + _RECEIPT, json.dumps(receipt))


def matches_published_weights(state: Path, weights: list[Path]) -> bool:
    """True for a matching tagged set, False for a legacy state, or raise on mismatch."""
    try:
        opened = ZipFile(state)
    except BadZipFile:
        # Older torch.save formats have no ZIP records. The caller validates the state.
        return False
    try:
        with opened as archive:
            records = [
                item for item in archive.infolist() if item.filename.endswith(_RECEIPT)
            ]
            if not records:
                return False
            if len(records) != 1 or records[0].file_size > 16384:
                raise ValueError("Invalid checkpoint publication record.")
            receipt = json.loads(archive.read(records[0]))
    except BadZipFile as exc:
        raise ValueError("Damaged checkpoint publication record.") from exc

    if (
        not isinstance(receipt, dict)
        or receipt.get("version") != 1
        or str(receipt.get("step")) != state.stem
        or not isinstance(receipt.get("weights"), dict)
    ):
        raise ValueError("Invalid checkpoint publication record.")
    members = receipt["weights"]
    if set(members) != {path.name for path in weights}:
        raise ValueError(
            "Training state and model files describe different checkpoint sets."
        )
    for weight in weights:
        info = members[weight.name]
        if not isinstance(info, dict) or info.get("bytes") != weight.stat().st_size:
            raise ValueError(
                f"Checkpoint weight size differs from the published set: {weight.name}"
            )
        if weight.suffix == ".safetensors":
            from safetensors import SafetensorError, safe_open

            try:
                with safe_open(str(weight), framework="numpy") as tensors:
                    identity = (tensors.metadata() or {}).get(SAVE_ID_KEY)
            except SafetensorError as exc:
                raise ValueError(
                    f"Unreadable checkpoint weight: {weight.name}"
                ) from exc
        else:
            try:
                identity = pth_save_id(weight)
            except BadZipFile as exc:
                raise ValueError(
                    f"Unreadable checkpoint weight: {weight.name}"
                ) from exc
        if (
            not identity
            or not SAVE_ID.fullmatch(identity)
            or identity != info.get("id")
        ):
            raise ValueError(
                f"Checkpoint weight belongs to a different save: {weight.name}"
            )
    return True

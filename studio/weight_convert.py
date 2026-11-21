"""Convert one explicit restricted PyTorch network payload to a portable tensor file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from traiNNer.models.checkpoint_payload import read_network_checkpoint, state_dict_for_export


def _tensor_map(value) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in value.items()
    )


def convert_weights(source: Path, destination: Path, payload: str | None) -> dict:
    archive = read_network_checkpoint(source)
    if not isinstance(archive, dict):
        raise ValueError("The .pth file did not contain a weights dictionary.")
    metadata = archive.get("metadata")
    selected = payload
    if _tensor_map(archive):
        if selected is not None:
            raise ValueError("This .pth stores its weights directly. Omit the payload name.")
        weights = archive
        metadata = None
    else:
        candidates = {key: value for key, value in archive.items()
                      if isinstance(key, str) and key != "metadata" and _tensor_map(value)}
        if selected is None:
            if not candidates:
                raise ValueError("No network tensor dictionary was found in the .pth file.")
            if len(candidates) != 1:
                raise ValueError("Choose the network payload to convert: " + ", ".join(sorted(candidates)))
            selected = next(iter(candidates))
        if selected not in candidates:
            raise ValueError(f"No tensor dictionary named {selected!r}. Available: " + ", ".join(sorted(candidates)))
        weights = candidates[selected]

    for name, tensor in weights.items():
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type != "cpu":
            raise ValueError(f"{name!r} cannot be stored as a dense CPU tensor without changing its representation.")
    header = None
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("Expected a mapping for network metadata.")
        try:
            encoded = json.dumps(metadata, allow_nan=False)
            if json.loads(encoded) != metadata:
                raise ValueError("Non-JSON metadata values are present.")
            header = {"metadata": encoded}
        except (ValueError, TypeError) as exc:
            raise ValueError("The network metadata cannot be represented losslessly as JSON.") from exc
    portable = state_dict_for_export(weights, for_safetensors=True, preserve_names=True)
    save_file(portable, str(destination), metadata=header)
    return {
        "source": str(source), "payload": selected, "tensorCount": len(portable),
        "dtypes": dict(sorted(Counter(str(t.dtype) for t in portable.values()).items())),
        "tensorBytes": sum(t.numel() * t.element_size() for t in portable.values()),
        "metadataKeys": sorted(metadata or {}), "format": "safetensors",
    }

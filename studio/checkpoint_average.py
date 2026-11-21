"""Average pinned, numbered weights from one managed experiment into a new file."""

from __future__ import annotations

import ctypes
import json
import os
import re
import stat
import struct
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from safetensors import safe_open
from traiNNer.training.checkpoint_association import SAVE_ID, SAVE_ID_KEY

from .paths import artifact_version
from .native import executable
from .weight_drift import _TYPES, _header

_BLOCK_ELEMENTS = 262_144
_DISCRETE = {"BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64"}
_QUANT = {
    "activation_post_process",
    "weight_fake_quant",
    "fake_quant_enabled",
    "observer_enabled",
    "zero_point",
    "_packed_params",
    "qweight",
    "qzeros",
}
_FIXED = {
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "weight_u",
    "weight_v",
    "_u",
    "_v",
    "mask",
    "attn_mask",
    "attn_mask_0",
    "attn_mask_1",
    "attn_regions",
    "relative_position_index",
    "fold_kernel",
    "fold_bias",
    "_eval_folded_weight",
    "_eval_folded_bias",
    "fused_flag",
}
_SUPERVISED = {"image_classification", "semantic_segmentation", "object_detection"}
_DERIVATION = "trainner_checkpoint_average"


@dataclass
class _Source:
    path: Path
    stream: BinaryIO
    version: str
    header: dict
    start: int
    receipt: dict


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _metadata(stream: BinaryIO, path: Path, task: str) -> tuple[dict, dict, str | None]:
    with safe_open(
        f"/proc/self/fd/{stream.fileno()}", framework="numpy", device="cpu"
    ) as weights:
        outer = weights.metadata() or {}
    save_id = outer.pop(SAVE_ID_KEY, None)
    if save_id is not None and SAVE_ID.fullmatch(save_id) is None:
        raise ValueError(f"Invalid recorded save identity in {path.name}.")
    unknown = outer.keys() - {"metadata", "format"}
    if unknown or ("format" in outer and outer["format"] != "pt"):
        raise ValueError(
            f"{path.name} has unsupported checkpoint metadata: {sorted(outer)}."
        )
    model = json.loads(outer.get("metadata", "{}"))
    if not isinstance(model, dict) or ("task" in model and model["task"] != task):
        raise ValueError(
            f"{path.name} has incompatible model or task metadata for {task}."
        )
    if task in _SUPERVISED:
        names = model.get("class_names")
        if (
            model.get("task") != task
            or not isinstance(names, list)
            or not names
            or not all(isinstance(s, str) and s.strip() for s in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"{path.name} requires native task and ordered class_names metadata."
            )
        if task == "object_detection":
            ids = model.get("category_ids")
            if (
                not isinstance(ids, list)
                or len(ids) != len(names)
                or not all(type(i) is int for i in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(
                    f"{path.name} requires an ordered category ID for each detection class."
                )
    normalized = dict(outer)
    if "metadata" in outer:
        normalized["metadata"] = _json(model)
    return outer, normalized, save_id


def _check_pin(source: _Source) -> None:
    opened = os.fstat(source.stream.fileno())
    if (
        artifact_version(opened) != source.version
        or artifact_version(source.path.lstat()) != source.version
    ):
        raise ValueError(
            f"{source.path.name} changed since selection. Select its current version."
        )


def _inputs(
    stack: ExitStack,
    inputs: Iterable[Mapping],
    run_id: str,
    role: str,
    task: str,
    cancel: Callable[[], None] | None,
) -> tuple[list[_Source], dict, Path]:
    sources = []
    seen_files = set()
    steps = set()
    save_ids = set()
    for item in inputs:
        if cancel:
            cancel()
        path = Path(item["path"])
        match = re.fullmatch(
            re.escape(role) + r"_([1-9][0-9]{0,18})\.safetensors", path.name
        )
        models = (
            path.parent.parent if path.parent.name == "resume_models" else path.parent
        )
        experiment = models.parent
        if (
            not match
            or not path.is_absolute()
            or path.resolve(strict=True) != path
            or models.name != "models"
            or experiment.parent.name != "experiments"
            or experiment.parent.parent.name != run_id
            or experiment.parent.parent.parent.name != "runs"
        ):
            raise ValueError(
                f"Expected a canonical numbered {role} checkpoint in managed run {run_id}: {path}."
            )
        if (role.endswith("_ema") and path.parent != models) or (
            role == "net_d" and path.parent == models
        ):
            raise ValueError(
                f"{path.name} is outside the trainer's directory for {role}."
            )
        if sources and path.parent != sources[0].path.parent:
            raise ValueError(
                "Select checkpoints from the same experiment and model directory."
            )
        step = int(match[1])
        if "step" in item and item["step"] != step:
            raise ValueError(f"The supplied step does not match {path.name}.")
        stream = stack.enter_context(
            os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb")
        )
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or artifact_version(info) != item["version"]:
            raise ValueError(
                f"Select the current version of the regular checkpoint {path.name}."
            )
        inode = (info.st_dev, info.st_ino)
        if inode in seen_files or step in steps:
            raise ValueError(
                f"Each selected checkpoint must be distinct. Duplicate input at {path.name}."
            )
        seen_files.add(inode)
        steps.add(step)
        start, header = _header(stream)
        outer, normalized, save_id = _metadata(stream, path, task)
        if not sources:
            shared_metadata, baseline = outer, normalized
        else:
            if normalized != baseline:
                raise ValueError(
                    f"Model metadata differs at {path.name}, including architecture or label order."
                )
            first = sources[0].header
            if first.keys() != header.keys():
                difference = min(first.keys() ^ header.keys())
                raise ValueError(
                    f"Tensor names differ at {path.name}, including {difference!r}."
                )
            for name, spec in header.items():
                if any(spec[key] != first[name][key] for key in ("shape", "dtype")):
                    raise ValueError(
                        f"Tensor {name!r} differs in shape or dtype at {path.name}."
                    )
        receipt = {
            "path": path.relative_to(experiment).as_posix(),
            "step": step,
            "version": item["version"],
        }
        if save_id is not None:
            if save_id in save_ids:
                raise ValueError(f"Duplicate recorded save identity at {path.name}.")
            save_ids.add(save_id)
            receipt["saveId"] = save_id
        sources.append(_Source(path, stream, item["version"], header, start, receipt))
    if len(sources) < 2:
        raise ValueError("Select at least two distinct numbered checkpoints.")
    return sources, shared_metadata, experiment


def _layout(header: dict) -> tuple[list[str], set[str]]:
    # The original offsets specify the output order, including zero-length tensors.
    names = sorted(header, key=lambda name: (*header[name]["data_offsets"], name))
    averageable = set()
    for name in names:
        dtype = header[name]["dtype"]
        parts = name.split(".")
        if (
            _QUANT.intersection(parts)
            or (dtype not in _TYPES and dtype not in _DISCRETE)
            or (dtype in _DISCRETE and parts[-1] in {"weight", "bias"})
        ):
            raise ValueError(
                f"Unsupported quantization state or dtype at {name!r} ({dtype})."
            )
        if dtype in _TYPES and parts[-1] not in _FIXED:
            averageable.add(name)
    if not any(
        header[k]["data_offsets"][0] < header[k]["data_offsets"][1] for k in averageable
    ):
        raise ValueError("The checkpoint has no supported floating weights to average.")
    return names, averageable


def _read_block(
    source: _Source, size: int, cancel: Callable[[], None] | None
) -> bytearray:
    if cancel:
        cancel()
    data = bytearray(size)
    if source.stream.readinto(data) != size:
        raise ValueError(f"{source.path.name} changed during averaging.")
    return data


@cache
def _kernels():
    lib = ctypes.CDLL(str(executable("weights", "measure.so", source="measure.c")))
    arguments = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
    lib.accumulate_values.argtypes = [*arguments, ctypes.c_int]
    lib.accumulate_values.restype = ctypes.c_int
    lib.finish_average.argtypes = [*arguments, ctypes.c_uint]
    lib.finish_average.restype = ctypes.c_int
    return lib


def _accumulate(data, dtype, total, first, name):
    code, width = dtype
    address = ctypes.byref(ctypes.c_char.from_buffer(data))
    status = _kernels().accumulate_values(address, code, len(data) // width, total, first)
    if status == 1:
        raise ValueError(f"Nonfinite input values in {name!r}.")
    if status:
        raise RuntimeError("Invalid checkpoint reduction parameters.")


def _write_tensor(
    sources: list[_Source],
    output: BinaryIO,
    name: str,
    average: bool,
    cancel: Callable[[], None] | None,
    report: Callable[[int], None] | None,
    copy_from: _Source | None = None,
) -> None:
    if copy_from is not None:
        sources = [copy_from]
    spec = sources[0].header[name]
    dtype = _TYPES.get(spec["dtype"])
    for source in sources:
        source.stream.seek(source.start + source.header[name]["data_offsets"][0])
    lo, hi = spec["data_offsets"]
    width = dtype[1] if dtype else 1
    block_bytes = _BLOCK_ELEMENTS * width
    for offset in range(lo, hi, block_bytes):
        size = min(block_bytes, hi - offset)
        first = _read_block(sources[0], size, cancel)
        if average:
            assert dtype is not None
            count = size // width
            total = (ctypes.c_double * count)()
            _accumulate(first, dtype, total, True, name)
            for source in sources[1:]:
                block = _read_block(source, size, cancel)
                _accumulate(block, dtype, total, False, name)
            address = ctypes.byref(ctypes.c_char.from_buffer(first))
            if _kernels().finish_average(address, dtype[0], count, total, len(sources)):
                raise RuntimeError("Invalid checkpoint average output parameters.")
            output.write(first)
        else:
            if dtype is not None:
                _accumulate(first, dtype, None, True, name)
            for source in sources[1:]:
                if _read_block(source, size, cancel) != first:
                    raise ValueError(
                        f"Discrete or fixed model buffer {name!r} differs at {source.path.name}."
                    )
            output.write(first)
        if report:
            report(size * len(sources))


def average_checkpoints(
    inputs: Iterable[Mapping],
    destination: Path,
    *,
    run_id: str,
    role: str,
    task: str,
    batch_norm: str = "requireIdentical",
    cancel: Callable[[], None] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Publish one CPU, weights-only average. Admission supplies canonical paths and stat pins."""
    if batch_norm not in ("requireIdentical", "fromLatest"):
        raise ValueError("Choose requireIdentical or fromLatest for batchNorm.")
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("A managed run ID is required.")
    if task == "autoencoder":
        roles = {"net_ae", "net_ae_ema"}
    elif task in _SUPERVISED:
        roles = {"net_g", "net_g_ema"}
    elif task in {"image_restoration", "recurrent_restoration"}:
        roles = {"net_g", "net_g_ema", "net_d"}
    else:
        raise ValueError(f"Unsupported checkpoint task {task!r}.")
    if role not in roles:
        raise ValueError(f"Unsupported network role {role!r} for {task}.")
    destination = Path(destination)
    if (
        destination.suffix != ".safetensors"
        or not destination.is_absolute()
        or destination.parent.resolve(strict=True) != destination.parent
    ):
        raise ValueError(
            "Provide a canonical absolute .safetensors output path in an existing job directory."
        )
    if os.path.lexists(destination):
        raise FileExistsError(f"The output already exists: {destination}.")
    if sys.byteorder != "little":
        raise ValueError("Checkpoint averaging requires a little-endian host.")

    with ExitStack() as stack:
        sources, metadata, experiment = _inputs(
            stack, inputs, run_id, role, task, cancel
        )
        if destination.is_relative_to(experiment.parent.parent.parent):
            raise ValueError(
                "Choose an analysis output directory outside managed runs."
            )
        header = sources[0].header
        names, averageable = _layout(header)
        copied = set()
        latest = max(sources, key=lambda source: source.receipt["step"])
        if batch_norm == "fromLatest":
            for name in names:
                prefix, _, leaf = name.rpartition(".")
                if leaf != "num_batches_tracked" or header[name]["dtype"] != "I64" or header[name]["shape"] != []:
                    continue
                prefix = prefix + "." if prefix else ""
                mean, variance = prefix + "running_mean", prefix + "running_var"
                if (mean in header and variance in header and header[mean]["dtype"] in _TYPES
                        and header[variance]["dtype"] == header[mean]["dtype"] and len(header[mean]["shape"]) == 1
                        and header[variance]["shape"] == header[mean]["shape"]):
                    copied.update((mean, variance, name))
        manifest = {
            "schemaVersion": 1,
            "method": "uniform",
            "runId": run_id,
            "role": role,
            "task": task,
            "inputs": [s.receipt for s in sources],
        }
        if batch_norm == "fromLatest":
            manifest.update(batchNorm=batch_norm, batchNormSourceStep=latest.receipt["step"] if copied else None,
                            copiedBatchNormTensors=len(copied))
        metadata[_DERIVATION] = _json(manifest)
        encoded = _json({**header, "__metadata__": metadata}).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        if len(encoded) > 100_000_000:
            raise ValueError(
                "The combined metadata exceeds the safetensors header limit."
            )
        payload_bytes = 0
        completed = 0

        def report(read: int) -> None:
            nonlocal payload_bytes
            payload_bytes += read
            if progress is None:
                return
            progress(
                {
                    "stage": "averaging",
                    "completedTensors": completed,
                    "totalTensors": len(names),
                    "payloadBytesRead": payload_bytes,
                }
            )

        with NamedTemporaryFile(
            dir=destination.parent, prefix=".average-", suffix=".tmp"
        ) as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for name in names:
                _write_tensor(
                    sources,
                    output,
                    name,
                    name in averageable,
                    cancel,
                    report if progress is not None else None,
                    latest if name in copied else None,
                )
                completed += 1
            if progress:
                report(0)
            output.flush()
            os.fsync(output.fileno())
            with safe_open(
                f"/proc/self/fd/{output.fileno()}", framework="numpy", device="cpu"
            ):
                pass
            for source in sources:
                _check_pin(source)
            if cancel:
                cancel()
            # link(2) fails atomically if another worker created the destination.
            os.link(output.name, destination)

    # Validated safetensors ranges are contiguous. The last end is the full payload size.
    tensor_bytes = header[names[-1]]["data_offsets"][1]
    return {
        **manifest,
        "output": str(destination),
        "version": artifact_version(destination.stat()),
        "tensorCount": len(names),
        "tensorBytes": tensor_bytes,
        "averagedTensors": len(averageable),
        "identicalBufferTensors": len(names) - len(averageable) - len(copied),
    }

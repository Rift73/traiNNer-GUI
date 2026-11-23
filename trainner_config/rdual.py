"""RDUAL recipe checks and checkpoint inspection shared with Studio."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from stat import S_ISREG
from threading import Lock
from typing import TYPE_CHECKING, TypeVar

from trainner_data.file_version import artifact_version

from trainner_config.official_weights import ensure_neuflow, neuflow_cache_path

if TYPE_CHECKING:
    from torch import Tensor

    from trainner_config.options import ReduxOptions


NEUFLOW_PRETRAIN_PATH = "experiments/pretrained_models/neuflow_v2.safetensors"

# Sorted (key, shape) pairs from NeuFlowV2.state_dict(), also the layout of
# the bundled NeuFlow v2 asset. Studio can check every layer without Torch.
_NEUFLOW_LAYOUT = "83729ace7caed1c69c93e22ee1f1760e4dcd3465eead437561cbda1bef68b12e"
_FLOAT_DTYPES = {
    "F16",
    "BF16",
    "F32",
    "F64",
    "torch.float16",
    "torch.bfloat16",
    "torch.float32",
    "torch.float64",
}
_COUNTER_DTYPES = {"I64", "torch.int64"}
_BOOL_DTYPES = {"BOOL", "torch.bool"}
_CHECKPOINT_PREFIXES = ("module.", "_orig_mod.", "netG.")
_PTH_METADATA_LIMIT = 32 * 1024 * 1024
_pth_probe_lock = Lock()

_T = TypeVar("_T")
_Traits = tuple[tuple[int, ...], str]
_Inspection = tuple[bool, bool] | None


def clip_training_issues(opt: ReduxOptions) -> list[tuple[str, str]]:
    """Check recurrent model options without importing the trainer."""
    issues = []
    if opt.val is not None and (opt.val.val_enabled or opt.is_train is False):
        if opt.val.tile_size:
            issues.append(
                (
                    "val.tile_size",
                    "Recurrent clip validation requires full frames (tile_size: 0).",
                )
            )
        for phase, dataset in opt.datasets.items():
            if phase.split("_", 1)[0] not in ("val", "test"):
                continue
            if dataset.type.lower() not in ("pairedvideodataset", "singlevideodataset"):
                issues.append(
                    (
                        f"datasets.{phase}.type",
                        "Select a paired or single video frame sequence.",
                    )
                )
            if dataset.clip_size is None or dataset.clip_size < 1:
                issues.append(
                    (
                        f"datasets.{phase}.clip_size",
                        "Set a positive recurrent window length.",
                    )
                )
    for field in ("input_pixel_format", "output_pixel_format"):
        if getattr(opt, field) != "rgb":
            issues.append((field, "Recurrent clips require RGB inputs and outputs."))
    if opt.is_train is False:
        return issues

    if not opt.high_order_degradation:
        issues.append(
            (
                "high_order_degradation",
                "Enable on-the-fly degradation for training clips.",
            )
        )
    if opt.dist or (isinstance(opt.num_gpu, int) and opt.num_gpu > 1):
        issues.append(("num_gpu", "Recurrent clip training runs on one GPU."))
    if opt.otf_hf_noise_prob > 0:
        issues.append(
            (
                "otf_hf_noise_prob",
                "Set to 0 for clip training. This noise setting draws per frame.",
            )
        )
    if opt.network_g_teacher is not None:
        issues.append(
            (
                "network_g_teacher",
                "Recurrent clip training does not support a teacher generator.",
            )
        )
    if opt.dvd_prob > 0:
        issues.append(
            (
                "dvd_prob",
                "Set to 0 for clip training. The DVD response needs image context outside the crop.",
            )
        )
    if opt.train is not None:
        if opt.train.use_moa:
            issues.append(
                (
                    "train.use_moa",
                    "Recurrent clip training does not support batch augmentation.",
                )
            )
        if opt.train.vosr_target is not None:
            issues.append(
                (
                    "train.vosr_target",
                    "Recurrent clip training does not support a VOSR target.",
                )
            )
        if opt.train.distillation is not None and opt.train.distillation.enabled:
            issues.append(
                (
                    "train.distillation",
                    "Recurrent clip training does not support distillation.",
                )
            )
    return issues


def clip_training_problems(opt: ReduxOptions) -> list[str]:
    """Include objective validation for direct trainer callers."""
    from trainner_config.objectives import ObjectiveConfigError, validate_objectives

    issues = clip_training_issues(opt)
    if opt.is_train is not False and opt.train is not None:
        try:
            validate_objectives(opt)
        except ObjectiveConfigError as exc:
            issues.append((exc.path, str(exc)))
    return [f"{path}: {message}" for path, message in issues]


def _local_path(value: str, root: Path) -> Path:
    if not value.strip():
        raise ValueError("Provide a weights file path.")
    return root / Path(value).expanduser()


@dataclass(frozen=True)
class NeuflowWeightSource:
    path: Path
    official: bool = False

    @property
    def download_pending(self) -> bool:
        return self.official and not self.path.exists()


def resolve_neuflow_weights(
    opt: ReduxOptions,
    root: Path,
    resolve_path: Callable[[str, Path], Path] = _local_path,
) -> NeuflowWeightSource:
    """Select the explicit file, the legacy local file or the official cache."""
    if opt.path.neuflow_weights is not None:
        return NeuflowWeightSource(resolve_path(opt.path.neuflow_weights, root))
    local = resolve_path(NEUFLOW_PRETRAIN_PATH, root)
    if local.exists():
        return NeuflowWeightSource(local)
    return NeuflowWeightSource(neuflow_cache_path(), official=True)


def neuflow_weights_path(
    opt: ReduxOptions,
    root: Path,
    resolve_path: Callable[[str, Path], Path] = _local_path,
) -> Path:
    return resolve_neuflow_weights(opt, root, resolve_path).path


def ensure_neuflow_weights(opt: ReduxOptions, root: Path) -> Path:
    """Obtain a required external flow file at runtime."""
    source = resolve_neuflow_weights(opt, root)
    if source.official:
        return ensure_neuflow()
    if not source.path.is_file():
        raise FileNotFoundError(f"Expected a NeuFlow v2 weights file at {source.path}.")
    return source.path


def _without_common_prefix(
    weights: dict[str, _T], prefixes: tuple[str, ...]
) -> dict[str, _T]:
    for prefix in prefixes:
        if weights and all(key.startswith(prefix) for key in weights):
            weights = {key[len(prefix) :]: value for key, value in weights.items()}
    return weights


def _generator_keys(weights: dict[str, _T], ema: bool) -> dict[str, _T]:
    weights = _without_common_prefix(weights, _CHECKPOINT_PREFIXES)
    if ema and any(key.startswith("ema_model.") for key in weights):
        weights = {
            key[10:]: value
            for key, value in weights.items()
            if key.startswith("ema_model.")
        }
    return weights


def generator_weights(
    payload: object, param_key: str | None, path: Path
) -> dict[str, Tensor]:
    """Select the same generator payload for training and inspection."""
    from traiNNer.models.checkpoint_payload import unwrap_checkpoint_state_dict

    if isinstance(payload, dict):
        if param_key is not None and isinstance(payload.get(param_key), dict):
            payload = payload[param_key]
        payload, selected = unwrap_checkpoint_state_dict(payload)
        return _generator_keys(
            _tensor_dict(payload, path),
            param_key == "params_ema" or selected == "params_ema",
        )
    raise ValueError(f"{path} must contain a tensor state dict.")


def _tensor_dict(payload: object, path: Path) -> dict[str, Tensor]:
    import torch

    if (
        not isinstance(payload, dict)
        or not payload
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in payload.items()
        )
    ):
        raise ValueError(f"{path} must contain a tensor state dict.")
    return payload


def _neuflow_layout_matches(
    weights: dict[str, _T], traits: Callable[[_T], _Traits]
) -> bool:
    entries = []
    for key, tensor in sorted(weights.items()):
        shape, dtype = traits(tensor)
        expected_dtypes = (
            _COUNTER_DTYPES if key.endswith(".num_batches_tracked") else _FLOAT_DTYPES
        )
        if dtype not in expected_dtypes:
            return False
        entries.append((key, shape))
    layout = json.dumps(entries, separators=(",", ":")).encode()
    return sha256(layout).hexdigest() == _NEUFLOW_LAYOUT


def _embedded_flow(
    weights: dict[str, _T],
    traits: Callable[[_T], _Traits],
    scalar: Callable[[_T], bool],
) -> bool:
    if "flow_loaded" not in weights:
        return False
    marker = weights["flow_loaded"]
    shape, dtype = traits(marker)
    if shape != () or dtype not in _BOOL_DTYPES or not scalar(marker):
        return False
    flow = {key[5:]: value for key, value in weights.items() if key.startswith("flow.")}
    return _neuflow_layout_matches(flow, traits)


def _torch_traits(tensor: Tensor) -> _Traits:
    import torch

    dtype = (
        "" if tensor.is_meta or tensor.layout != torch.strided else str(tensor.dtype)
    )
    return tuple(tensor.shape), dtype


def has_embedded_neuflow(weights: dict[str, Tensor]) -> bool:
    return _embedded_flow(weights, _torch_traits, lambda marker: marker.item())


def complete_neuflow_state(
    weights: dict[str, Tensor], initial: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """Use the model's placeholder flow until a complete fallback can be loaded."""
    if has_embedded_neuflow(weights):
        return weights
    state = {
        key: tensor for key, tensor in weights.items()
        if key != "flow_loaded" and not key.startswith("flow.")
    }
    state.update(
        (key, tensor) for key, tensor in initial.items() if key.startswith("flow.")
    )
    state["flow_loaded"] = initial["flow_loaded"].new_zeros(())
    return state


def _external_flow(
    weights: dict[str, _T], traits: Callable[[_T], _Traits], path: Path
) -> dict[str, _T]:
    flow = _without_common_prefix(weights, (*_CHECKPOINT_PREFIXES, "flow."))
    if not _neuflow_layout_matches(flow, traits):
        raise ValueError(f"{path} does not match the NeuFlow v2 tensor layout.")
    return flow


def _torch_flow_weights(payload: object, path: Path) -> dict[str, Tensor]:
    from traiNNer.models.checkpoint_payload import unwrap_checkpoint_state_dict

    if isinstance(payload, dict):
        payload, _ = unwrap_checkpoint_state_dict(payload)
    return _external_flow(_tensor_dict(payload, path), _torch_traits, path)


def load_neuflow_weights(path: Path) -> dict[str, Tensor]:
    from traiNNer.models.checkpoint_payload import read_network_checkpoint

    return _torch_flow_weights(read_network_checkpoint(path), path)


def _inspect_safetensors(
    path: Path, param_key: str | None, external: bool, excluded: tuple[str, ...]
) -> tuple[bool, bool]:
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(path, framework="numpy") as weights:
            keys = {key: key for key in weights.keys()}

            def traits(key: str) -> _Traits:
                tensor = weights.get_slice(key)
                return tuple(tensor.get_shape()), tensor.get_dtype()

            if external:
                _external_flow(keys, traits, path)
                return False, True
            keys = _generator_keys(keys, param_key == "params_ema")
            keys = {
                key: original for key, original in keys.items() if key not in excluded
            }
            temporal = any(key.startswith("temporal.") for key in keys)
            return temporal, _embedded_flow(
                keys, traits, lambda key: weights.get_tensor(key).item()
            )
    except SafetensorError as exc:
        raise ValueError(f"Invalid checkpoint {path}: {exc}") from exc


def _inspect_pth(
    path: Path,
    param_key: str | None,
    external: bool,
    transfer: bool,
    excluded: tuple[str, ...],
) -> tuple[bool, bool] | None:
    if not zipfile.is_zipfile(path):
        with path.open("rb") as stream:
            if stream.read(1) == b"\x80":
                # torch.load supports old pickle checkpoints, but mmap does not.
                # The trainer checks these after loading the selected payload.
                return None
        raise ValueError(f"Invalid PyTorch checkpoint: {path}")
    with zipfile.ZipFile(path) as archive:
        records = [
            entry
            for entry in archive.infolist()
            if entry.filename.endswith("/data.pkl")
        ]
        if len(records) != 1:
            raise ValueError(f"{path} must contain one PyTorch metadata record.")
        if records[0].file_size > _PTH_METADATA_LIMIT:
            return None
    request = json.dumps((str(path), param_key, external, transfer, excluded))
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-m", "trainner_config.rdual", request],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Timed out inspecting checkpoint metadata: {path}") from exc
    if result.returncode:
        raise ValueError(f"Cannot inspect {path}: {result.stderr.strip()[-700:]}")
    temporal, embedded = json.loads(result.stdout)
    return temporal, embedded


def _inspect_pth_child(
    filename: str,
    param_key: str | None,
    external: bool,
    transfer: bool,
    excluded: list[str],
) -> tuple[bool, bool]:
    import torch
    from traiNNer.models.checkpoint_payload import _checkpoint_tensors

    path = Path(filename)
    # mmap keeps tensor storage on disk. Only the scalar flow flag is accessed.
    payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    if external:
        _torch_flow_weights(payload, path)
        return False, True
    if transfer:
        weights, _ = _checkpoint_tensors(payload, path, param_key)
        weights = _without_common_prefix(weights, _CHECKPOINT_PREFIXES)
    else:
        weights = generator_weights(payload, param_key, path)
    weights = {key: value for key, value in weights.items() if key not in excluded}
    return any(key.startswith("temporal.") for key in weights), has_embedded_neuflow(
        weights
    )


@lru_cache(maxsize=32)
def _inspect(
    path: Path,
    version: str,
    param_key: str | None,
    external: bool,
    transfer: bool,
    excluded: tuple[str, ...],
) -> tuple[bool, bool] | None:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        result = _inspect_safetensors(path, param_key, external, excluded)
    elif suffix == ".pth":
        try:
            result = _inspect_pth(path, param_key, external, transfer, excluded)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid PyTorch checkpoint: {path}") from exc
    else:
        raise ValueError(f"Use .safetensors or .pth weights: {path}")
    if artifact_version(path.stat()) != version:
        raise ValueError(
            f"{path} changed during preflight. Select a completed checkpoint."
        )
    return result


def _inspect_input(
    path: Path,
    param_key: str | None = None,
    *,
    external: bool = False,
    transfer: bool = False,
    excluded: tuple[str, ...] = (),
) -> _Inspection:
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        if external:
            raise ValueError(
                f"NeuFlow v2 weights are missing at {path}. "
                "Choose the trained NeuFlow v2 file under Pretrained weights."
            ) from exc
        raise
    if not S_ISREG(info.st_mode):
        raise ValueError(f"Expected a weights file: {path}")
    args = (path, artifact_version(info), param_key, external, transfer, excluded)
    if path.suffix.lower() == ".pth":
        with _pth_probe_lock:
            return _inspect(*args)
    return _inspect(*args)


def _generator_inputs(
    opt: ReduxOptions,
    root: Path,
    resolve_path: Callable[[str, Path], Path],
    resume_files: Mapping[str, str | Path] | None,
) -> tuple[list[tuple[str, str]], dict[str, Path], bool | None]:
    issues = []
    inputs = {}
    paths = opt.path
    if (
        resume_files is None
        and opt.is_train is not False
        and (paths.resume_state or opt.auto_resume)
    ):
        return issues, inputs, None

    resuming = bool(paths.resume_state or resume_files)
    transfer = paths.transfer_g if not resuming and opt.is_train else None
    raw_key = paths.param_key_g
    if resuming and raw_key in (None, "params_ema"):
        raw_key = "params"
    roles = [("pretrain_network_g", raw_key)]
    if opt.is_train is not False and opt.train is not None and opt.train.ema_decay > 0:
        roles.append(("pretrain_network_g_ema", "params_ema"))

    needs_flow = unknown = False
    for role, param_key in roles:
        source = (resume_files or {}).get(role, getattr(paths, role))
        if source is None:
            # Without separate EMA weights, EMA starts as a copy of G.
            needs_flow |= role == "pretrain_network_g"
            continue
        field = f"path.{role}"
        raw_transfer = transfer is not None and role == "pretrain_network_g"
        try:
            path = resolve_path(str(source), root)
            inputs[field] = path
            result = _inspect_input(
                path,
                param_key,
                transfer=raw_transfer,
                excluded=tuple(transfer.exclude) if raw_transfer else (),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append((field, str(exc)))
            result = None
        if result is None:
            unknown = True
            continue
        temporal, embedded = result
        needs_flow |= not embedded
        if not temporal and (resuming or (paths.strict_load_g and not raw_transfer)):
            message = (
                "Full-state resume requires the matching recurrent generator weights."
                if resuming
                else "The checkpoint has no RDUAL temporal tensors. For a DUAL Light warm start, set path.strict_load_g to false."
            )
            issues.append((field, message))
    if needs_flow:
        return issues, inputs, True
    return issues, inputs, None if unknown else False


def rdual_neuflow_required(
    opt: ReduxOptions,
    root: Path,
    *,
    resolve_path: Callable[[str, Path], Path] = _local_path,
    resume_files: Mapping[str, str | Path] | None = None,
) -> bool | None:
    """Determine fallback use without looking up the fallback asset itself.

    None means checkpoint selection or inspection could not settle the question.
    """
    _, _, required = _generator_inputs(opt, root, resolve_path, resume_files)
    return required


def preflight_rdual(
    opt: ReduxOptions,
    root: Path,
    *,
    resolve_path: Callable[[str, Path], Path] | None = None,
    resume_files: Mapping[str, str | Path] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Path]]:
    """Check clip settings and report effective generator and NeuFlow inputs.

    Supply the numbered companion paths after resolving a selected or latest
    state. Until a state is resolved, its original warmstart paths are dormant.
    A pending official download reports its future cache path in the inputs.
    Use resolve_neuflow_weights to identify official and pending sources.
    Legacy non-ZIP PTH payloads are checked only when the trainer loads them.
    """
    if str((opt.network_g or {}).get("type", "")).lower() != "rdual":
        return [], {}
    issues = clip_training_issues(opt)
    if opt.scale not in (1, 2, 4):
        issues.append(("scale", "RDUAL requires scale 1, 2 or 4."))

    resolver = resolve_path or _local_path
    weight_issues, inputs, required = _generator_inputs(
        opt, root, resolver, resume_files
    )
    issues.extend(weight_issues)
    if required:
        field = "path.neuflow_weights"
        try:
            source = resolve_neuflow_weights(opt, root, resolver)
            inputs[field] = source.path
            if not source.download_pending:
                _inspect_input(source.path, external=True)
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append((field, str(exc)))
    return issues, inputs


if __name__ == "__main__":
    import pickle

    try:
        print(json.dumps(_inspect_pth_child(*json.loads(sys.argv[1]))))
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

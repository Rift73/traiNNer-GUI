"""Reading and normalizing portable network checkpoint payloads.

Export removes parallel/EMA bookkeeping and requests the portable CPU layout.
Import unwraps one recognized container, then optionally one singleton wrapper;
common-prefix removal is a separate step so BaseModel overrides remain usable.
"""

import json
import logging
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from spandrel import StateDict


def read_network_checkpoint(
    path: str | Path, *, safetensors_device: str = "cpu"
) -> StateDict:
    """Read safetensors or a restricted PyTorch network payload."""
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return load_file(str(path), device=safetensors_device)
    if suffix == ".pth":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"Unsupported model: {path}. Use .safetensors or .pth.")


def state_dict_for_export(
    state_dict: StateDict, *, for_safetensors: bool = False, preserve_names: bool = False
) -> StateDict:
    """Build the existing portable payload without modifying the source mapping.

    Strip one module prefix per key and omit EMA bookkeeping after stripping.
    An exact format conversion preserves names and bookkeeping tensors.
    Torch archives can preserve tied storage. Safetensors needs independent
    byte ranges while retaining all public keys for a strict load.
    """
    exported = OrderedDict()
    cpu_storages: dict[int, list[str]] = {}
    for full_key, param in state_dict.items():
        key = full_key
        if not preserve_names:
            if key.startswith("module."):
                key = key[7:]
            if key in ("step", "initted"):
                continue
        value = param.to("cpu", memory_format=torch.contiguous_format)
        if for_safetensors and param.device.type == "cpu" and value.numel():
            value = value.contiguous()
            cpu_storages.setdefault(value.untyped_storage().data_ptr(), []).append(key)
        exported[key] = value
    for names in cpu_storages.values():
        if len(names) < 2:
            continue
        end = 0
        for key in sorted(names, key=lambda name: exported[name].data_ptr()):
            value = exported[key]
            start = value.data_ptr()
            if start < end:
                exported[key] = value.clone()
            else:
                end = start + value.numel() * value.element_size()
    return exported


# Wrapper precedence follows Spandrel's canonicalization convention:
# https://github.com/chaiNNer-org/spandrel/blob/ebf11bab4bc3fabccc80fcc377eaabb8cecbf8cd/libs/spandrel/spandrel/__helpers/canonicalize.py#L14
def unwrap_checkpoint_state_dict(state_dict: StateDict) -> tuple[StateDict, str | None]:
    """Select a known wrapper in priority order, then unwrap a singleton dict.

    This is intentionally not recursive. The returned key names only the known
    wrapper, never the optional singleton, preserving load diagnostics.
    """
    used_unwrap_key = None
    unwrap_keys = [
        "model_state_dict",
        "state_dict",
        "params_ema",
        "params-ema",
        "params",
        "model",
        "net",
    ]
    for unwrap_key in unwrap_keys:
        if unwrap_key in state_dict and isinstance(state_dict[unwrap_key], dict):
            state_dict = state_dict[unwrap_key]
            used_unwrap_key = unwrap_key
            break

    if len(state_dict) == 1:
        single = next(iter(state_dict.values()))
        if isinstance(single, dict):
            state_dict = single

    return state_dict, used_unwrap_key


def strip_common_prefix(
    state_dict: StateDict, prefixes: Sequence[str] = ("module.", "netG.")
) -> StateDict:
    """Strip each prefix in order only when every key shares it.

    Unchanged mappings are returned as-is; a stripped mapping is a new dict.
    Prefix ordering matters for nested names and must not be sorted or repeated.
    """
    if len(state_dict) > 0:
        for prefix in prefixes:
            if all(i.startswith(prefix) for i in state_dict.keys()):
                state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _check_checkpoint_label_map(
    checkpoint: Path,
    metadata: object,
    class_names: Sequence[str] | None,
    category_ids: Sequence[int] | None,
    *,
    expected_task: str | None,
    allow_legacy_class_order: bool,
) -> bool | None:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint {checkpoint} has invalid metadata.")
    if (
        expected_task is not None
        and "task" in metadata
        and metadata["task"] != expected_task
    ):
        raise ValueError(
            f"Checkpoint {checkpoint} declares task {metadata['task']!r}. "
            f"The recipe uses {expected_task!r}."
        )

    class_order_verified = None
    if class_names is not None:
        if "class_names" not in metadata:
            if not allow_legacy_class_order:
                raise ValueError(f"Checkpoint {checkpoint} is missing its class order.")
            class_order_verified = False
        else:
            saved_names = metadata["class_names"]
            if not isinstance(saved_names, list) or not all(
                isinstance(name, str) for name in saved_names
            ):
                raise ValueError(
                    f"Checkpoint {checkpoint} has invalid class_names metadata."
                )
            if list(class_names) != saved_names:
                raise ValueError(
                    f"Checkpoint {checkpoint} uses class_names {saved_names!r}. "
                    f"The recipe uses {list(class_names)!r}. Class order must match."
                )
            class_order_verified = True
    if category_ids is not None:
        saved_ids = metadata.get("category_ids")
        if not isinstance(saved_ids, list) or not all(
            type(category) is int for category in saved_ids
        ):
            raise ValueError(
                f"Checkpoint {checkpoint} is missing its COCO category IDs."
            )
        if list(category_ids) != saved_ids:
            raise ValueError(
                f"Checkpoint {checkpoint} uses COCO category IDs {saved_ids!r}. "
                f"The recipe uses {list(category_ids)!r}. Category order must match."
            )

    return class_order_verified


def _checkpoint_tensors(
    weights: object, checkpoint: Path, param_key: str | None
) -> tuple[StateDict, str | None]:
    if not isinstance(weights, dict):
        raise ValueError(f"Checkpoint {checkpoint} must contain a tensor state dict.")

    selected = None
    if not all(isinstance(value, torch.Tensor) for value in weights.values()):
        if param_key is not None:
            if param_key not in weights:
                raise ValueError(
                    f"Checkpoint {checkpoint} has no {param_key!r} payload."
                )
            weights = weights[param_key]
            selected = param_key
        else:
            weights, selected = unwrap_checkpoint_state_dict(weights)
    if (
        not isinstance(weights, dict)
        or not weights
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in weights.items()
        )
    ):
        raise ValueError(f"Checkpoint {checkpoint} must contain a tensor state dict.")

    return weights, selected


def _normalized_keys(
    weights: StateDict, expected: StateDict, *, strict: bool
) -> StateDict:
    if weights.keys() != expected.keys():
        normalized = strip_common_prefix(weights, ("module.", "_orig_mod.", "netG."))
        if strict or len(expected.keys() & normalized.keys()) > len(
            expected.keys() & weights.keys()
        ):
            weights = normalized
    return weights


def load_matching_network_checkpoint(
    network: torch.nn.Module,
    checkpoint: Path,
    param_key: str | None,
    *,
    class_names: Sequence[str] | None = None,
    category_ids: Sequence[int] | None = None,
    expected_task: str | None = None,
    allow_legacy_class_order: bool = False,
    strict: bool = True,
    expected_metadata: dict[str, object] | None = None,
) -> bool | None:
    """Load compatible weights and report whether an expected class order was verified."""
    validation_requested = (
        class_names is not None or category_ids is not None or expected_task is not None or bool(expected_metadata)
    )
    metadata: object = None
    if validation_requested and checkpoint.suffix.lower() == ".safetensors":
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            metadata = json.loads((file.metadata() or {}).get("metadata", "null"))
            weights = {key: file.get_tensor(key) for key in file.keys()}
    else:
        weights = read_network_checkpoint(checkpoint)
        if validation_requested and isinstance(weights, dict):
            metadata = weights.get("metadata")
            # A bare state dict may use this name for a tensor.
            if isinstance(metadata, torch.Tensor):
                metadata = None
    for key, value in (expected_metadata or {}).items():
        if not isinstance(metadata, dict) or metadata.get(key) != value:
            raise ValueError(f"Checkpoint {checkpoint} does not match the recipe's {key}. Use the original names and order.")
    class_order_verified = _check_checkpoint_label_map(
        checkpoint,
        metadata,
        class_names,
        category_ids,
        expected_task=expected_task,
        allow_legacy_class_order=allow_legacy_class_order,
    )
    weights, _ = _checkpoint_tensors(weights, checkpoint, param_key)
    expected = network.state_dict()
    weights = _normalized_keys(weights, expected, strict=strict)
    if strict:
        if weights.keys() != expected.keys():
            missing = sorted(expected.keys() - weights.keys())
            unexpected = sorted(weights.keys() - expected.keys())
            raise ValueError(
                f"Checkpoint {checkpoint} does not match {type(network).__name__}. "
                f"Missing keys ({len(missing)}): {missing[:5]}. "
                f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}."
            )
        for key, value in weights.items():
            if value.shape != expected[key].shape:
                raise ValueError(
                    f"Checkpoint {checkpoint} has {key!r} with shape {tuple(value.shape)}. "
                    f"{type(network).__name__} expects {tuple(expected[key].shape)}."
                )
    else:
        compatible = {
            key: value
            for key, value in weights.items()
            if key in expected and value.shape == expected[key].shape
        }
        if not compatible:
            raise ValueError(
                f"Checkpoint {checkpoint} has no compatible network tensors."
            )
        if len(compatible) < len(expected) or len(compatible) < len(weights):
            warnings.warn(
                f"Partially loading checkpoint {checkpoint}. "
                f"Using {len(compatible)} of {len(expected)} network tensors "
                f"from {len(weights)} checkpoint tensors.",
                stacklevel=2,
            )
        weights = compatible
    network.load_state_dict(weights, strict=strict)
    if class_order_verified is False:
        warnings.warn(
            f"Checkpoint {checkpoint} has no saved class order. "
            "Verify that task_options.class_names follows the source model's class IDs.",
            stacklevel=2,
        )
    return class_order_verified


def transfer_generator_weights(
    network: torch.nn.Module,
    checkpoint: Path,
    param_key: str | None,
    exclude: Sequence[str],
    logger: logging.Logger,
) -> dict:
    """Initialize matching parameters and persistent buffers in a new training G."""
    payload = _checkpoint_tensors(read_network_checkpoint(checkpoint), checkpoint, param_key)
    return transfer_generator_state(network, checkpoint, payload, exclude, logger)


def transfer_generator_state(
    network: torch.nn.Module, checkpoint: Path, payload: tuple[StateDict, str | None],
    exclude: Sequence[str], logger: logging.Logger,
) -> dict:
    """Transfer a decoded payload when its topology was already inspected."""
    weights, selected = payload
    target = network.state_dict(keep_vars=True)
    weights = _normalized_keys(weights, target, strict=False)
    parameters = dict(network.named_parameters(remove_duplicate=False))
    buffers = dict(network.named_buffers(remove_duplicate=False))
    excluded = set(exclude)
    if unknown := sorted(excluded - target.keys()):
        raise ValueError(
            f"path.transfer_g.exclude has unknown destination keys: {unknown!r}."
        )

    # A parameter registered under two names must stay fresh when either name is excluded.
    excluded_ids = {id(target[key]) for key in excluded}
    matched = {}
    skipped = []
    for key in sorted(weights.keys() | target.keys()):
        src, dst = weights.get(key), target.get(key)
        if key in excluded:
            reason = "excluded"
        elif key not in target:
            reason = "sourceOnly"
        elif key not in weights:
            reason = "targetOnly"
        elif id(dst) in excluded_ids:
            reason = "sharedWithExcluded"
        elif (
            not isinstance(dst, torch.Tensor)
            or (key not in parameters and key not in buffers)
            or src.is_meta
            or dst.is_meta
            or src.layout != torch.strided
            or dst.layout != torch.strided
            or src.is_quantized
            or dst.is_quantized
        ):
            reason = "unsupportedTensor"
        elif src.shape != dst.shape:
            skipped.append(
                {
                    "key": key,
                    "reason": "shapeMismatch",
                    "sourceShape": list(src.shape),
                    "targetShape": list(dst.shape),
                }
            )
            continue
        elif src.dtype != dst.dtype and not (
            src.is_floating_point() and dst.is_floating_point()
        ):
            skipped.append(
                {
                    "key": key,
                    "reason": "dtypeMismatch",
                    "sourceDtype": str(src.dtype),
                    "targetDtype": str(dst.dtype),
                }
            )
            continue
        else:
            matched[key] = src
            continue
        skipped.append({"key": key, "reason": reason})

    # Preserve native module load hooks (some restoration architectures keep
    # their own geometry buffer). Only credit the supplied entries they accepted.
    if parameters.keys() & matched.keys():
        offered = matched.copy()
        result = network.load_state_dict(offered, strict=False)
        refused = set(result.missing_keys) | set(result.unexpected_keys)
        for key in list(matched):
            if key in refused or offered.get(key) is not matched[key]:
                skipped.append({"key": key, "reason": "modelLoaderOverride"})
                del matched[key]

    skipped.sort(key=lambda item: item["key"])
    nparams = len(parameters.keys() & matched.keys())
    logger.info(
        "Initial G transfer from %s: %d matching parameter keys, %d matching buffers, %d skipped keys.",
        checkpoint,
        nparams,
        len(matched) - nparams,
        len(skipped),
    )
    for row in skipped:
        logger.info("Initial G transfer skipped %s", json.dumps(row, ensure_ascii=True))
    if not nparams:
        raise ValueError(
            f"Initial G transfer from {checkpoint} matched zero parameter keys. "
            f"{len(matched)} buffer keys matched. See the skipped keys in the training log."
        )
    return {
        "source": str(checkpoint),
        "paramKey": selected,
        "labelPolicy": "noRemap",
        "counts": {
            "source": len(weights),
            "target": len(target),
            "loaded": len(matched),
            "parameters": nparams,
            "buffers": len(matched) - nparams,
            "skipped": len(skipped),
        },
        "loaded": sorted(matched),
        "skipped": skipped,
    }

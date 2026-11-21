"""Inspect the declared network and, on request, exercise its initial weight transfer."""

from collections.abc import Callable
import json
import logging
from pathlib import Path

from trainner_config.parameter_filters import parameter_filter
from .recipes import task_from_config


def inspect_network(config: dict, output: Path, *, cancel: Callable[[], None],
                    transfer_source: Path | None = None) -> dict:
    import torch
    from trainner_config.tasks import TRAINING_TASKS
    from traiNNer.archs import build_network
    from traiNNer.tasks import load_task_provider

    task = task_from_config(config)
    provider = TRAINING_TASKS.get(task)
    if provider is None:
        raise ValueError(f"Unknown training task {task!r}.")
    role = provider.network_field
    spec = config.get(role)
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError(f"Declare the architecture and its parameters under {role}.")
    if provider.model_factory.partition(":")[0] != "traiNNer.models":
        # A runtime task can register its own architecture as well as its dataset.
        load_task_provider(provider)
    train = config.get("train") or {}
    selected = parameter_filter(train.get("optim_include_params_g"), train.get("optim_exclude_params_g")) if role == "network_g" else None
    factory_options = dict(spec) if provider.dataset_geometry == "records" else {**spec, "scale": config["scale"]}
    cancel()
    transfer = None
    loaded_ids = set()
    loaded_names = set()
    if transfer_source is None:
        try:
            with torch.device("meta"):
                network = build_network(factory_options)
        except NotImplementedError as error:
            raise ValueError(f"{spec['type']} cannot be inspected on the metadata-only device: {error}") from error
    else:
        from traiNNer.models.checkpoint_payload import _checkpoint_tensors, read_network_checkpoint, transfer_generator_state
        from trainner_config.tasks import AETHER_ARCHITECTURES
        with torch.device("cpu"):
            network = build_network(factory_options)
        paths = config["path"]
        cancel()
        payload = _checkpoint_tensors(read_network_checkpoint(transfer_source), transfer_source, paths.get("param_key_g"))
        if spec["type"].lower() in AETHER_ARCHITECTURES and network.qat_state(payload[0]) is not None:
            raise ValueError("Loading Aether QAT weights requires a strict, complete generator checkpoint.")
        receipt = transfer_generator_state(network, transfer_source, payload,
                                           paths["transfer_g"].get("exclude", []), logging.getLogger(__name__))
        del payload
        state = network.state_dict(keep_vars=True)
        loaded_names = set(receipt["loaded"])
        loaded_ids = {id(state[name]) for name in loaded_names}
        with (output.parent / "transfer-receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False)
        transfer = {"source": str(transfer_source), "counts": receipt["counts"], "labelPolicy": receipt["labelPolicy"],
                    "freshSelectedElements": 0, "freshExcludedElements": 0, "loadedSelectedElements": 0}

    parameters = trainable = matched = count = buffers = parameter_bytes = buffer_bytes = 0
    buffer_count = 0
    preview = []
    with output.open("x", encoding="utf-8") as stream:
        for kind, entries in (("parameter", network.named_parameters()), ("buffer", network.named_buffers())):
            for name, value in entries:
                cancel()
                elements = value.numel()
                size = elements * value.element_size()
                row = {"name": name, "kind": kind, "shape": list(value.shape),
                       "dtype": str(value.dtype).removeprefix("torch."), "elements": elements, "bytes": size}
                if kind == "parameter":
                    count += 1
                    parameters += elements
                    parameter_bytes += size
                    if value.requires_grad:
                        trainable += elements
                    matches = bool(value.requires_grad and (selected is None or selected(name)))
                    if matches:
                        matched += elements
                    row.update(requiresGrad=value.requires_grad, selectedByName=matches)
                    if transfer is not None:
                        loaded = id(value) in loaded_ids or name in loaded_names
                        row["initialization"] = "transferred" if loaded else "constructor"
                        if matches:
                            transfer["loadedSelectedElements" if loaded else "freshSelectedElements"] += elements
                        elif value.requires_grad and not loaded:
                            transfer["freshExcludedElements"] += elements
                else:
                    buffer_count += 1
                    buffers += elements
                    buffer_bytes += size
                if len(preview) < 40:
                    preview.append(row)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"task": task, "role": role, "architecture": spec["type"],
            "inspection": "initial_generator_transfer" if transfer is not None else "unwrapped_constructor",
            **({"transfer": transfer} if transfer is not None else {}),
            "tensorCount": count + buffer_count, "parameterTensors": count, "bufferTensors": buffer_count,
            "parameters": parameters, "trainableParameters": trainable,
            "matchedTrainableParameters": matched, "parameterBytes": parameter_bytes, "bufferElements": buffers,
            "bufferBytes": buffer_bytes, "preview": preview}

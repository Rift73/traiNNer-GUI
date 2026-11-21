"""Shared publication of portable graphs for registered headless exports."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

try:
    import onnx
except ImportError as exc:
    raise RuntimeError(
        "ONNX export requires the project's optional onnx package."
    ) from exc


def _scalar_qdq_parameters(graph) -> None:
    """ORT's quantized elementwise operators require scalar per-tensor parameters."""
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    aliases = {n.output[0]: n.input[0] for n in graph.node if n.op_type == "Identity"}
    occupied = set(initializers) | {name for n in graph.node for name in n.output}
    occupied.update(t.name for t in graph.input)
    scalars = {}
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or any(a.name == "axis" for a in node.attribute):
            continue  # A one-channel per-axis scale is still a vector.
        for i in range(1, min(3, len(node.input))):
            root = node.input[i]
            while root in aliases:
                root = aliases[root]
            tensor = initializers.get(root)
            if tensor is None or list(tensor.dims) != [1]:
                continue
            if root not in scalars:
                # Other consumers can still require the original rank-one tensor.
                name = root + "_scalar"
                while name in occupied:
                    name += "_"
                scalar = graph.initializer.add()
                scalar.CopyFrom(tensor)
                del scalar.dims[:]
                scalar.name = scalars[root] = name
                occupied.add(name)
            node.input[i] = scalars[root]


def write_portable_onnx(
    model: nn.Module,
    example: Tensor,
    destination: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    from .predict import _write_new_file

    output_specs = metadata.get("outputs")
    names = [s["name"] for s in output_specs] if output_specs else [metadata["output_name"]]
    input_spec = {"name": metadata["input_name"], "shape": metadata["input_shape"]}
    declared_outputs = output_specs or [{"name": names[0], "shape": metadata["output_shape"]}]
    dynamic_axes = {}
    for spec in [input_spec, *declared_outputs]:
        variable_dims = {i: size for i, size in enumerate(spec["shape"]) if isinstance(size, str)}
        if variable_dims:
            dynamic_axes[spec["name"]] = variable_dims
    buffer = BytesIO()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            buffer,
            dynamo=False,
            opset_version=metadata["opset"],
            input_names=[metadata["input_name"]],
            output_names=names,
            dynamic_axes=dynamic_axes or None,
            verbose=False,
        )

    graph = onnx.load_model_from_string(buffer.getvalue())
    if metadata.get("precision") == "int8_qdq":
        _scalar_qdq_parameters(graph.graph)
    if any(domain.domain not in ("", "ai.onnx") for domain in graph.opset_import):
        raise ValueError(
            "This architecture uses custom ONNX operators. Portable export requires standard ONNX operators."
        )
    if any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in graph.graph.initializer
    ):
        raise ValueError(
            "Portable export requires a single ONNX file with embedded weights."
        )
    if output_specs or metadata.get("fixed_shape"):
        # Tracing fixed gathers can leave symbolic output dimensions in the file.
        # Publish the dimensions already checked against the eager output.
        shapes = [spec["shape"] for spec in output_specs] if output_specs else [metadata["output_shape"]]
        for output, shape in zip(graph.graph.output, shapes, strict=True):
            for dim, size in zip(output.type.tensor_type.shape.dim, shape, strict=True):
                if isinstance(size, str):
                    if dim.dim_param != size:
                        raise ValueError(f"ONNX output {output.name} lost its variable {size} dimension.")
                elif dim.HasField("dim_value") and dim.dim_value != size:
                    raise ValueError("ONNX output dimensions disagree with the eager model.")
                else:
                    dim.dim_value = size
    onnx.helper.set_model_props(
        graph, {"trainner." + key: json.dumps(value) for key, value in metadata.items()}
    )
    onnx.checker.check_model(graph)
    _write_new_file(destination, lambda file: file.write(graph.SerializeToString()))
    return {**metadata, "output": str(destination)}

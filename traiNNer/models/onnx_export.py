"""Shared preparation of architecture-specific ONNX deployment graphs."""

import inspect
from collections.abc import Sequence
from typing import Any, Literal

from torch import nn


def prepare_onnx_network(
    network: nn.Module,
    input_shape: Sequence[int],
    *,
    precision: Literal["native", "tensorrt_mixed"] = "native",
    dynamic_spatial: bool = False,
) -> nn.Module:
    prepare = getattr(network, "prepare_for_onnx_export", None)
    if not callable(prepare):
        return network

    parameters = inspect.signature(prepare).parameters
    kwargs: dict[str, Any] = {}
    if "precision" in parameters:
        kwargs["precision"] = precision
    if "dynamic_spatial" in parameters:
        kwargs["dynamic_spatial"] = dynamic_spatial
    return prepare(tuple(input_shape), **kwargs)

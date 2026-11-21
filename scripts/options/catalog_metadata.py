"""Configuration arguments exposed by the same registries that construct training components."""

import ast
import inspect
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from studio.recipes import template_catalog_index
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY, TESTARCH_REGISTRY
from traiNNer.metrics import METRIC_REGISTRY, requires_reference


def _json_default(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    raise TypeError


def _literal_choices(annotation: object) -> tuple:
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    # Postponed annotations may sit beside unrelated hints unavailable at runtime.
    # Read a Literal's simple values without evaluating the factory's annotations.
    if isinstance(annotation, str) and annotation.startswith(("Literal[", "typing.Literal[")):
        try:
            expr = ast.parse(annotation, mode="eval").body
            if isinstance(expr, ast.Subscript):
                value = ast.literal_eval(expr.slice)
                return value if isinstance(value, tuple) else (value,)
        except (SyntaxError, ValueError):
            pass
    return ()


def _parameters(factory: Callable[..., Any], supplied: set[str]) -> dict[str, Any]:
    try:
        signature = inspect.signature(factory)
    except (ValueError, TypeError):
        return {"signatureAvailable": False, "parameters": []}
    options = []
    accepts_extra = False
    for key, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_extra = True
            continue
        if key in supplied or param.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL
        ):
            continue
        field: dict[str, Any] = {"name": key, "required": param.default is inspect.Parameter.empty}
        if not field["required"]:
            try:
                field["default"] = _json_default(param.default)
            except TypeError:
                # An internal Torch sentinel or callable is not a YAML value.
                pass
        if choices := _literal_choices(param.annotation):
            try:
                field["choices"] = _json_default(choices)
            except TypeError:
                # Only expose choices that can be expressed in a recipe.
                pass
        options.append(field)
    return {
        "signatureAvailable": True,
        "acceptsExtraKeywords": accepts_extra,
        "parameters": options,
    }


def _optimizer_metadata() -> list[dict]:
    from traiNNer.optimizers import AUTO_FUSED_OPTIMIZERS, OPTIMIZER_REGISTRY

    optimizers = []
    for name in sorted(OPTIMIZER_REGISTRY.keys()):
        factory = OPTIMIZER_REGISTRY.get(name)
        signature = _parameters(factory, {"params", "self", "type"})
        if factory in AUTO_FUSED_OPTIMIZERS:
            for parameter in signature["parameters"]:
                if parameter["name"] == "fused":
                    parameter["defaultDescription"] = (
                        "When fused and foreach are omitted, traiNNer enables fused "
                        "for nonempty sequences of CUDA floating-point tensors."
                    )
        spelling = getattr(factory, "__name__", name)
        optimizers.append(
            {
                "id": name,
                "displayName": spelling if spelling.casefold() == name.casefold() else name,
                "source": factory.__module__.split(".", 1)[0],
                **signature,
            }
        )
    return optimizers


def _scheduler_metadata() -> list[dict]:
    from traiNNer.schedulers.recipe import scheduler_constructor
    from trainner_config.schedules import scheduler_catalog

    schedulers = scheduler_catalog()
    for entry in schedulers:
        factory = scheduler_constructor(entry["id"])
        params = _parameters(factory, {"optimizer", "self", "scale_fn"})["parameters"]
        if entry["id"] == "ReduceLROnPlateau":
            params.append({"name": "monitor", "required": True})
            for parameter in params:
                if parameter["name"] == "mode":
                    parameter["required"] = True
                    parameter.pop("default", None)
        elif entry["id"] == "KneeLR":
            for parameter in params:
                if parameter["name"] in ("peak_lr", "total_steps"):
                    parameter["required"] = False  # supplied by the trainer when omitted
        entry["parameters"] = params
    return schedulers


def catalog_metadata() -> dict:
    arch_sources = {}
    for origin, registry in (("arch", ARCH_REGISTRY), ("spandrel", SPANDREL_REGISTRY), ("testarch", TESTARCH_REGISTRY)):
        for name in sorted(registry.keys()):
            arch_sources.setdefault(name, []).append(origin)
    return {
        "architectureRegistries": arch_sources,
        "templateSummaries": template_catalog_index(Path(__file__).resolve().parents[2]),
        "optimizers": _optimizer_metadata(),
        "schedulers": _scheduler_metadata(),
        "metrics": [
            {"id": name, "requiresReference": requires_reference(name),
             **_parameters(factory, {"img", "img2", "device"})}
            for name, factory in sorted(METRIC_REGISTRY)
        ],
    }

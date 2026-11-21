import inspect
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Union, get_type_hints

from msgspec import NODEFAULT, Meta, Struct, defstruct, field, json
from msgspec.structs import fields
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY, TESTARCH_REGISTRY
from traiNNer.data import DATASET_REGISTRY
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import LOSS_REGISTRY
from traiNNer.tasks import load_task_providers
from traiNNer.utils.redux_options import DatasetOptions, ReduxOptions, TrainOptions
from traiNNer.utils.registry import Registry
from trainner_config.objectives import add_loss_schedule_schema
from trainner_config.tasks import TRAINING_TASKS

# These factories forward **kwargs to their annotated return class, except for
# the geometry fixed by the preset. Exposing a fixed argument would generate a
# schema-valid config that fails with "multiple values for keyword argument".
PRESET_FIXED_OPTIONS = {
    "dual_light": {"embed_dim", "depths", "layer_indices", "pixel_unshuffle"},
    "dual_xs": {"embed_dim", "depths", "layer_indices"},
    "dual_s": {"embed_dim", "depths"},
    "dual_m": {"embed_dim", "depths"},
    "dual_l": {"embed_dim", "depths"},
    "dual_xl": {"embed_dim", "depths"},
    **dict.fromkeys(("aether_small", "aether_medium", "aether_large"),
                    {"embed_dim", "depths", "use_channel_attn", "use_spatial_attn"}),
    "rdual": set(),
}


def generate_registry_structs(registry: Registry | list, issues: list[dict]) -> dict[str, type[Struct]]:
    from .architecture_choices import architecture_choice_examples

    structs = {}

    for name, cls in sorted(registry, key=lambda item: item[0]):
        try:
            constructor = cls.__init__ if inspect.isclass(cls) else cls
            sig = inspect.signature(constructor)
            hints = get_type_hints(constructor, include_extras=True)
            parameters = dict(sig.parameters)
            if name in PRESET_FIXED_OPTIONS:
                forwarded = hints["return"].__init__
                parameters = {
                    **inspect.signature(forwarded).parameters,
                    **parameters,
                }
                hints = {**get_type_hints(forwarded, include_extras=True), **hints}
                for fixed in PRESET_FIXED_OPTIONS[name]:
                    parameters.pop(fixed, None)

            exclude_types = {type, Callable}

            fields = []
            closed = not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
            for k, param in parameters.items():
                if k == "self" or param.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue

                hint = hints.get(k)
                if hint is None:
                    if param.default is not inspect.Parameter.empty:
                        try:
                            json.encode(param.default)
                        except (TypeError, ValueError):
                            closed = False
                            continue
                    hint = Any

                if (
                    hasattr(hint, "__origin__") and hint.__origin__ in exclude_types
                ) or hint in exclude_types:
                    closed = False
                    continue

                default = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else NODEFAULT
                )
                if isinstance(default, (list, dict, set)):
                    default = field(default_factory=partial(deepcopy, default))
                if hint is str and (examples := architecture_choice_examples(cls, k)):
                    hint = Annotated[str, Meta(examples=list(examples))]
                fields.append((k, hint, default))

            struct = defstruct(
                name=name,
                fields=fields,
                tag=True,
                kw_only=True,
                forbid_unknown_fields=closed or name in PRESET_FIXED_OPTIONS,
            )
            structs[name] = struct
        except Exception as e:
            issues.append({"id": name, "message": str(e)})

    return structs


def create_dynamic_train_options(
    cls: type[Struct], name: str, new_val: tuple
) -> type[Struct]:
    new_fields = [
        (
            f.name,
            f.type,
            field(default_factory=f.default_factory)
            if f.default_factory != NODEFAULT
            else field(default=f.default),
        )
        for f in fields(cls)
    ]
    new_fields.append(new_val)
    new_fields = tuple(new_fields)
    return defstruct(name=name, fields=new_fields, kw_only=True)


def dataset_type_names(registry: Registry) -> tuple[str, ...]:
    """List usable spellings of concrete datasets through the YAML lookup."""
    suffix = "_traiNNer"
    candidates = {key.lower() for key in registry.keys()}
    candidates.update(
        key.removesuffix(suffix).lower()
        for key in registry.keys()
        if key.endswith(suffix)
    )

    names = []
    for name in sorted(candidates):
        try:
            factory = registry.get(name)
        except KeyError:
            continue
        if factory is not BaseDataset and not inspect.isabstract(factory):
            names.append(name)
    return tuple(names)


def build_schema() -> dict[str, Any]:
    """Generate the recipe schema from the runtime and task registries."""
    load_task_providers()
    from .catalog_metadata import catalog_metadata

    issues = []
    losses = generate_registry_structs(LOSS_REGISTRY, issues)
    architectures = generate_registry_structs(
        # Later entries win, matching build_network's registry precedence.
        list(TESTARCH_REGISTRY) + list(ARCH_REGISTRY) + list(SPANDREL_REGISTRY), issues
    )
    loss_type = Union[tuple(losses.values())]  # noqa: UP007
    arch_type = Union[tuple(architectures.values())]  # noqa: UP007

    train_options = create_dynamic_train_options(
        TrainOptions,
        "TrainOptions",
        (
            "losses",
            Annotated[
                list[loss_type],
                Meta(description="The list of loss functions to optimize."),
            ],
        ),
    )
    dataset_options = create_dynamic_train_options(
        DatasetOptions,
        "DatasetOptions",
        (
            "type",
            Annotated[
                str,
                Meta(description="The type of dataset to use.", examples=list(dataset_type_names(DATASET_REGISTRY))),
            ],
        ),
    )
    redux_options = create_dynamic_train_options(
        ReduxOptions, "ReduxOptions", ("train", train_options)
    )
    redux_options = create_dynamic_train_options(
        redux_options, "ReduxOptions", ("network_g", arch_type)
    )
    redux_options = create_dynamic_train_options(
        redux_options, "ReduxOptions", ("datasets", dict[str, dataset_options])
    )

    options_by_task = {
        name: spec.options_type
        for name, spec in sorted(TRAINING_TASKS.items())
        if spec.options_type is not None
    }
    (schema, *task_schemas), definitions = json.schema_components(
        (redux_options, *options_by_task.values())
    )
    schema["$defs"] = definitions
    task_constraints = []
    for (task, options_type), task_schema in zip(
        options_by_task.items(), task_schemas, strict=True
    ):
        options_required = any(
            option.default is NODEFAULT and option.default_factory is NODEFAULT
            for option in fields(options_type)
        )
        if options_required:
            then = {
                "properties": {"task_options": task_schema},
                "required": ["task_options"],
            }
        else:
            then = {
                "properties": {
                    "task_options": {"anyOf": [task_schema, {"type": "null"}]},
                },
            }
        task_constraints.append(
            {
                "if": {"properties": {"task": {"const": task}}, "required": ["task"]},
                "then": then,
            }
        )
    if task_constraints:
        definitions["ReduxOptions"]["allOf"] = task_constraints

    add_loss_schedule_schema(schema)
    schema["x-trainner-catalog"] = {**catalog_metadata(), "signatureIssues": issues}
    return schema


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the recipe schema from installed training factories.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parents[2] / "schemas/redux-config.schema.json"
    output.write_bytes(json.encode(build_schema()))
    print(output)

import inspect
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from importlib import import_module
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

# The verified **kwargs paths through built-in presets. Each entry records
# the next callable in the module namespace and the keyword arguments bound
# by the call instead of exposed by the wrapper.
_DRFT_CONSTRUCTOR_FIXED_OPTIONS = (
    "upscale",
    "depths",
    "num_heads",
    "unshifted_num_heads",
    "unshifted_attention_dim",
    "full_width_unshifted",
)
_DRFT_PRESET_FIXED_OPTIONS = ("embed_dim", "groups", "depth", "full_heads")
_DRFT_TEACHER_FIXED_OPTIONS = ("embed_dim", "full_heads")
_DRFT_NANO_FIXED_OPTIONS = ("dense_skip", "reconstruction")

_FACTORY_FORWARDING = {
    "traiNNer.archs.aether_arch": dict.fromkeys(
        ("aether_small", "aether_medium", "aether_large"),
        ("aether", ("embed_dim", "depths", "use_channel_attn", "use_spatial_attn")),
    ),
    "traiNNer.archs.drft_arch": {
        "_build_drft": ("DRFT", _DRFT_CONSTRUCTOR_FIXED_OPTIONS),
        "_build_drft_teacher": ("_build_drft", ()),
        "drft_nano": (
            "_build_drft",
            _DRFT_PRESET_FIXED_OPTIONS + _DRFT_NANO_FIXED_OPTIONS,
        ),
        "drft_light": ("_build_drft", ("depth",)),
        **dict.fromkeys(
            (f"drft_{size}" for size in ("micro", "xs", "s", "m", "l", "xl")),
            ("_build_drft", _DRFT_PRESET_FIXED_OPTIONS),
        ),
        "drft_nano_teacher": (
            "_build_drft_teacher",
            _DRFT_TEACHER_FIXED_OPTIONS + _DRFT_NANO_FIXED_OPTIONS,
        ),
        **dict.fromkeys(
            (
                f"drft_{size}_teacher"
                for size in ("micro", "light", "xs", "s", "m", "l", "xl")
            ),
            ("_build_drft_teacher", _DRFT_TEACHER_FIXED_OPTIONS),
        ),
    },
    "traiNNer.archs.drt_arch": {
        "_build_drt": ("DRT", (*_DRFT_CONSTRUCTOR_FIXED_OPTIONS, "mlp_ratio", "rank")),
        "DRT": ("DRFT", ()),
        **dict.fromkeys(
            (f"drt_{size}" for size in ("light", "xs", "s", "m", "l", "xl")),
            ("_build_drt", _DRFT_PRESET_FIXED_OPTIONS),
        ),
    },
    "traiNNer.archs.dual_arch": {
        "dual_light": (
            "DUAL",
            ("embed_dim", "depths", "layer_indices", "pixel_unshuffle"),
        ),
        "dual_xs": ("DUAL", ("embed_dim", "depths", "layer_indices")),
        **dict.fromkeys(
            ("dual_s", "dual_m", "dual_l", "dual_xl"), ("DUAL", ("embed_dim", "depths"))
        ),
    },
    "traiNNer.archs.gaterv3_arch": {
        "gaterv3_r": ("GateRV3", ("dim",)),
        "gaterv3_s": ("GateRV3", ("dim", "enc_blocks", "dec_blocks")),
    },
    "traiNNer.archs.rdual_arch": {"rdual": ("RDUAL", ())},
}


def _factory_parameters(
    factory: Callable,
) -> tuple[dict[str, inspect.Parameter], dict[str, Any]]:
    constructor = factory.__init__ if inspect.isclass(factory) else factory
    parameters = dict(inspect.signature(constructor).parameters)
    hints = get_type_hints(constructor, include_extras=True)

    module = getattr(factory, "__module__", "")
    name = getattr(factory, "__qualname__", "")
    forwarding = _FACTORY_FORWARDING.get(module, {}).get(name)
    if forwarding is None:
        return parameters, hints

    kwargs = next(
        (
            key
            for key, param in parameters.items()
            if param.kind == inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    owner = import_module(module)
    if kwargs is None or getattr(owner, name, None) is not factory:
        return parameters, hints

    target, fixed = forwarding
    forwarded_parameters, forwarded_hints = _factory_parameters(getattr(owner, target))
    for key in fixed:
        forwarded_parameters.pop(key, None)
    # Explicit wrapper parameters supply their own defaults and annotations.
    del parameters[kwargs]
    return forwarded_parameters | parameters, forwarded_hints | hints


def generate_registry_structs(registry: Registry | list, issues: list[dict]) -> dict[str, type[Struct]]:
    from .architecture_choices import architecture_choice_examples

    structs = {}

    for name, cls in sorted(registry, key=lambda item: item[0]):
        try:
            parameters, hints = _factory_parameters(cls)

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
                forbid_unknown_fields=closed,
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
    return defstruct(name=name, fields=new_fields, kw_only=True,
                     forbid_unknown_fields=cls.__struct_config__.forbid_unknown_fields)


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
    from .recipe_fields import add_recipe_fields

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
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
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
    catalog = catalog_metadata()
    add_recipe_fields(definitions, catalog, TRAINING_TASKS)
    schema["x-trainner-catalog"] = {**catalog, "signatureIssues": issues}
    return schema


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the recipe schema from installed training factories.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parents[2] / "schemas/redux-config.schema.json"
    output.write_bytes(json.encode(build_schema()))
    print(output)

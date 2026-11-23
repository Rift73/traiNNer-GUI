"""Describe recipe dictionaries whose keys come from the installed trainer."""

import re
from collections.abc import Iterable, Mapping

from trainner_config.tasks import TrainingTaskSpec

# Factory string arguments whose accepted spellings are implemented in code.
# Keep these as examples so an edited or extended factory can accept other values.
_PIXEL_CRITERIA = ["l1", "l2", "charbonnier"]
_LOSS_VALUES = {
    **{
        key: {"reduction": ["mean", "sum", "none"]}
        for key in ("l1loss", "mseloss", "charbonnierloss", "fftloss")
    },
    "psnrloss": {"reduction": ["mean"]},
    **{
        key: {"criterion": _PIXEL_CRITERIA}
        for key in ("colorloss", "bicubicloss", "lumaloss", "hsluvloss", "ldlloss")
    },
    "averageloss": {"criterion": ["l1", "l2"]},
    "perceptualanimeloss": {"criterion": ["l1", "charbonnier"]},
    "vosrloss": {"criterion": ["charbonnier", "l1", "perceptual"]},
    **{
        key: {"gan_type": ["vanilla", "lsgan", "wgan", "wgan_softplus", "hinge"]}
        for key in ("ganloss", "multiscaleganloss")
    },
    **{
        key: {"color_space": ["yiq", "ycbcr", "lhm"]}
        for key in ("mssimloss", "ssimloss", "structuretextureloss")
    },
    "contextualloss": {
        "distance_type": ["cosine", "l1", "l2"],
        "net": ["vgg19"],
        "calc_type": ["regular", "bilateral", "symetric"],
    },
    "neosrfdlloss": {"model": ["dinov2", "vgg", "resnet", "effnet"]},
}


def _name_pattern(name: str) -> str:
    # YAML registry lookups are case-insensitive. JSON Schema patterns also
    # have to work in JavaScript editors, which do not accept inline (?i).
    return "".join(
        f"[{c.lower()}{c.upper()}]" if c.isascii() and c.isalpha() else re.escape(c)
        for c in name
    )


def _names(names: Iterable[str]) -> dict:
    return {"pattern": "^(" + "|".join(_name_pattern(name) for name in names) + ")$"}


def _catalog_object(entries: list[dict], kind: str) -> dict:
    names = [entry.get("displayName", entry["id"]) for entry in entries]
    branches = []
    for entry in entries:
        properties = {}
        for parameter in entry.get("parameters", ()):
            prop = {}
            if "default" in parameter:
                prop["default"] = parameter["default"]
            if "choices" in parameter:
                prop["examples"] = parameter["choices"]
            if "defaultDescription" in parameter:
                prop["description"] = parameter["defaultDescription"]
            name = parameter["name"]
            if kind == "Scheduler" and name == "monitor":
                prop["$ref"] = "#/$defs/ValidationMonitor"
            if kind == "Scheduler" and name == "schedulers":
                prop.update(type="array", items={"$ref": "#/$defs/Scheduler"})
            properties[name] = prop
        required = [p["name"] for p in entry.get("parameters", ()) if p["required"]]
        then = {"properties": properties}
        if required and kind in ("Optimizer", "Scheduler"):
            then["required"] = required
        if kind == "Metric":
            for name in ("better", "dataset"):
                prop = properties.get(name, {})
                if "examples" in prop:
                    prop["enum"] = prop.pop("examples")
        branches.append(
            {
                "if": {
                    "properties": {
                        "type": {"pattern": "^" + _name_pattern(entry["id"]) + "$"}
                    },
                    "required": ["type"],
                },
                "then": then,
            }
        )
    return {
        "type": "object",
        "properties": {"type": {"type": "string", "minLength": 1, "examples": names}},
        "required": ["type"],
        "allOf": branches,
    }


def _extendable_registry(mapping: dict, union_ref: str) -> dict:
    # A linked checkout can add a factory before the next catalog refresh.
    known_names = _names(mapping)
    return {
        "properties": {"type": {"type": "string", "examples": list(mapping)}},
        "anyOf": [
            {"$ref": union_ref},
            {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "minLength": 1, "not": known_names}
                },
                "required": ["type"],
            },
        ],
    }


def add_recipe_fields(
    defs: dict, catalog: dict, tasks: Mapping[str, TrainingTaskSpec]
) -> None:
    root = defs["ReduxOptions"]["properties"]
    train = defs["TrainOptions"]["properties"]
    generator = root["network_g"]
    mapping = generator["discriminator"]["mapping"]
    loss_item = train["losses"]["items"]
    for registry in (mapping, loss_item["discriminator"]["mapping"]):
        for name, ref in registry.items():
            tag = defs[ref.rsplit("/", 1)[1]]["properties"]["type"]
            tag.clear()
            tag.update(
                type="string", pattern="^" + _name_pattern(name) + "$", examples=[name]
            )

    for name, params in _LOSS_VALUES.items():
        if name in loss_item["discriminator"]["mapping"]:
            for key, values in params.items():
                prop = defs[name]["properties"].get(key)
                if prop is not None and prop.get("type") == "string":
                    prop.setdefault("examples", values)

    # Preserve the GUI's discriminator index and expose the runtime's explicit nulls.
    generator["anyOf"].append({"type": "null"})
    train["losses"]["type"] = ["array", "null"]
    root["train"] = {
        "anyOf": [{"$ref": "#/$defs/TrainOptions"}, {"type": "null"}],
        "default": None,
    }
    root["task"]["anyOf"][0]["examples"] = sorted(tasks)

    defs["AuxiliaryNetwork"] = _extendable_registry(
        mapping, "#/$defs/ReduxOptions/properties/network_g"
    )
    for key in ("network_d", "network_g_teacher", "network_ae"):
        root[key] = {
            "$ref": "#/$defs/AuxiliaryNetwork",
            "description": root[key]["description"],
            "default": None,
        }

    rules = defs["ReduxOptions"].setdefault("allOf", [])
    for task, spec in tasks.items():
        field = spec.network_field
        other = "network_g" if field == "network_ae" else "network_ae"
        slot = {"type": "object", "required": ["type"]}
        if spec.architecture_types:
            slot["properties"] = {"type": _names(spec.architecture_types)}
        elif spec.excluded_architecture_types:
            slot["properties"] = {
                "type": {"not": _names(spec.excluded_architecture_types)}
            }
        rules.append(
            {
                "if": {"required": ["task"], "properties": {"task": {"const": task}}},
                "then": {
                    "required": [field],
                    "properties": {field: slot, other: {"type": "null"}},
                },
            }
        )

    # The built-in SR/AE trainers pass scale to each generator, decoder and teacher.
    # Discriminators and record-based providers have a different calling convention.
    without_scale = [
        name
        for name, ref in mapping.items()
        if (factory := defs[ref.rsplit("/", 1)[1]]).get("additionalProperties") is False
        and "scale" not in factory["properties"]
    ]
    spatial = {"properties": {"type": {"not": _names(without_scale)}}}
    rules.append(
        {
            "if": {
                "properties": {
                    "task": {
                        "enum": [
                            None,
                            "image_restoration",
                            "autoencoder",
                            "recurrent_restoration",
                        ]
                    }
                }
            },
            "then": {
                "properties": dict.fromkeys(
                    ("network_g", "network_g_teacher", "network_ae"), spatial
                )
            },
        }
    )
    if "autoencoder" in mapping:
        defs["autoencoder"]["properties"]["decoder_opt"] = {
            "type": "object",
            "$ref": "#/$defs/AuxiliaryNetwork",
            **spatial,
        }

    dataset = defs["DatasetOptions"]
    dataset.update(
        {
            "if": {
                "required": ["io_backend"],
                "properties": {
                    "io_backend": {
                        "required": ["type"],
                        "properties": {"type": {"const": "zip"}},
                    }
                },
            },
            "then": {
                "required": ["dataroot_gt"],
                "properties": {
                    "type": _names(["RealESRGANDataset"]),
                    "target_dataroot_gt": {"type": "null"},
                    "io_backend": {
                        "properties": {"type": {"const": "zip"}},
                        "additionalProperties": False,
                    },
                    "dataroot_gt": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                            },
                        ]
                    },
                },
            },
        }
    )

    for definition, key in (
        ("Optimizer", "optimizers"),
        ("Scheduler", "schedulers"),
        ("Metric", "metrics"),
    ):
        defs[definition] = _catalog_object(catalog[key], definition)
    for key in ("optim_g", "optim_d", "optim_ae", "scheduler"):
        definition = "Scheduler" if key == "scheduler" else "Optimizer"
        train[key]["anyOf"][0] = {"$ref": f"#/$defs/{definition}"}
    roles = ("g", "d", "ae")
    for key, value in (
        ("schedulers", {"$ref": "#/$defs/Scheduler"}),
        ("resume_previous_lrs", {"type": "number", "minimum": 0}),
    ):
        train[key]["anyOf"][0] = {
            "type": "object",
            "properties": dict.fromkeys(roles, value),
            "additionalProperties": False,
        }
    defs["TrainOptions"]["not"] = {
        "required": ["scheduler", "schedulers"],
        "properties": {key: {"type": "object"} for key in ("scheduler", "schedulers")},
    }
    defs["ValOptions"]["properties"]["metrics"]["anyOf"][0]["additionalProperties"] = {
        "$ref": "#/$defs/Metric"
    }

    distributed = root["dist_params"]["anyOf"][0]
    distributed["properties"] = {
        "backend": {"type": "string", "examples": ["nccl", "gloo"]},
        "port": {"type": ["integer", "null"], "examples": [29500]},
    }
    root["dist_params"]["description"] = (
        "Process group backend and master port used by the Slurm launcher."
    )

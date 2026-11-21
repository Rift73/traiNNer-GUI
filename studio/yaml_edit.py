"""Edit recipe fields while retaining the rest of the user's YAML verbatim."""
from __future__ import annotations

import math
from typing import Any

import yaml

Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
MAX_RECIPE_BYTES = 2 * 1024 * 1024


def read_recipe(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide a YAML training recipe.")
    if len(text.encode("utf-8")) > MAX_RECIPE_BYTES:
        raise ValueError("Recipe exceeds 2 MiB.")
    try:
        config = yaml.load(text, Loader=Loader)
    except yaml.YAMLError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(config, dict):
        raise ValueError("The recipe must be a YAML mapping.")
    # Config files travel over the JSON API. Check actual values without expanding
    # shared YAML anchors repeatedly; a cycle cannot represent a trainer option.
    # Compute expanded size with memoization. Aliases remain cheap to validate
    # but cannot expand to an unbounded JSON response or recursive value.
    sizes: dict[int, int] = {}
    active: set[int] = set()
    def size(value: Any, depth: int) -> int:
        if depth > 64:
            raise ValueError("Recipe exceeds the supported nesting depth.")
        if isinstance(value, (dict,list)):
            ident = id(value)
            if ident in active:
                raise ValueError("Recursive YAML aliases cannot be used in a recipe.")
            if ident in sizes:
                return sizes[ident]
            active.add(ident)
            total = 2
            if isinstance(value, dict):
                if any(not isinstance(k,str) for k in value):
                    raise ValueError("Recipe option names must be strings.")
                total += sum(len(k.encode("utf-8"))+4 for k in value)
            for child in (value.values() if isinstance(value,dict) else value):
                total += size(child,depth+1)+1
                if total > MAX_RECIPE_BYTES*3:
                    raise ValueError("Expanded recipe exceeds 6 MiB.")
            active.remove(ident)
            sizes[ident] = total
            return total
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Training recipes require finite numbers.")
        if value is not None and not isinstance(value, (int,float,str,bool)):
            raise ValueError("Use quoted text for non-numeric YAML scalars such as dates.")
        return len(value.encode("utf-8"))+2 if isinstance(value,str) else len(str(value))
    size(config,0)
    return config


def _flow(value: Any) -> str:
    return yaml.dump(value, Dumper=Dumper, default_flow_style=True, allow_unicode=True,
                     width=1_000_000, sort_keys=False).rstrip("\n").removesuffix("\n...")


def _child(node: yaml.Node, key: str | int) -> yaml.Node:
    if isinstance(node, yaml.SequenceNode) and type(key) is int and 0 <= key < len(node.value):
        return node.value[key]
    if isinstance(node, yaml.MappingNode) and isinstance(key, str):
        matches = [v for k, v in node.value if k.value == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Option {key!r} is repeated. Resolve it in the YAML editor.")
    raise KeyError(key)


def _assign(config: Any, path: list[str | int], value: Any) -> Any:
    # Copy just the modified branch. Two fields referencing one anchor should
    # remain independent when the user asks to edit only one of them.
    if not path:
        return value
    key, *rest = path
    if isinstance(config, dict) and isinstance(key, str):
        return {**config, key: _assign(config.get(key, {}), rest, value)}
    if isinstance(config, list) and type(key) is int and 0 <= key < len(config):
        items = config.copy()
        items[key] = _assign(items[key], rest, value)
        return items
    raise ValueError(f"Cannot set field {key!r} in this part of the recipe.")


def _without(config, path):
    key, *rest = path
    if isinstance(config, dict) and isinstance(key,str):
        if key not in config:
            return config
        out = dict(config)
        if rest:
            out[key] = _without(config[key], rest)
        else:
            del out[key]
        return out
    if isinstance(config,list) and type(key) is int:
        if key < 0 or key >= len(config):
            raise ValueError("The requested list item is out of range.")
        out = list(config)
        if rest:
            out[key] = _without(config[key], rest)
        else:
            del out[key]
        return out
    raise ValueError("Cannot remove a nested option from a scalar value.")


def _same(a,b):
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(v,b[k]) for k,v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x,y) for x,y in zip(a,b,strict=True))
    return a == b


def render(text: str, edits: list[dict[str, Any]]) -> str:
    crlf = "\r\n" in text and text.count("\r\n") == text.count("\n")
    original = read_recipe(text)
    if not isinstance(edits, list) or len(edits) > 128:
        raise ValueError("Provide a list of up to 128 field edits.")
    for edit in edits:
        if not isinstance(edit, dict) or (("value" in edit) == (edit.get("remove") is True)):
            raise ValueError("Each edit needs a path and either a value or remove=true.")
        path = edit.get("path")
        if not isinstance(path, list) or not 1 <= len(path) <= 32 or any(
            type(k) not in (int, str) or (isinstance(k, str) and not k) for k in path
        ):
            raise ValueError("Field paths are lists of option names and array indices.")
        if edit.get("remove") is True:
            from .yaml_remove import remove_text
            desired = _without(original, path)
            if _same(desired,original):
                continue
            candidate = remove_text(text,path,Loader)
            if not _same(read_recipe(candidate),desired):
                raise ValueError("This removal also affects an anchor or neighboring field. Edit it explicitly in YAML.")
            text, original = candidate, desired
            continue
        desired = _assign(original, path, edit["value"])
        if _same(desired, original):
            continue
        # Compose only for changed fields so comments, quotes and unrelated
        # hand-written options survive a form edit.
        node = yaml.compose(text, Loader=Loader)
        parent: Any = original
        wanted: Any = desired
        for offset, key in enumerate(path):
            try:
                child = _child(node, key)
            except KeyError:
                if not isinstance(node, yaml.MappingNode) or not isinstance(key, str):
                    raise ValueError("Cannot locate this field in the YAML document.") from None
                first = node.start_mark
                at = first.index - first.column
                prefix = text[at:first.index]
                if node.value and not node.flow_style and not prefix.strip():
                    patch = prefix + _flow(key) + ": " + _flow(wanted[key]) + "\n"
                    candidate = text[:at] + patch + text[at:]
                else:
                    candidate = text[:first.index] + _flow(wanted) + text[node.end_mark.index:]
                break
            node, parent, wanted = child, parent[key], wanted[key]
            if offset == len(path) - 1:
                start, end = node.start_mark.index, node.end_mark.index
                replacement = _flow(wanted)
                # A block collection or block scalar can consume the newline
                # ending the final item. Keep the next key on its own line.
                old = text[start:end]
                if old.endswith("\n") and end < len(text):
                    replacement += "\n" + " " * node.end_mark.column
                elif "\n" in old and end < len(text) and old.rsplit("\n", 1)[-1].strip() == "":
                    replacement += "\n" + old.rsplit("\n", 1)[-1]
                candidate = text[:start] + replacement + text[end:]
        if not _same(read_recipe(candidate), desired):
            raise ValueError("This edit also affects a YAML anchor or merged field. Edit it explicitly in YAML.")
        text, original = candidate, desired
    if crlf:
        return text.replace("\r\n", "\n").replace("\n", "\r\n")
    return text

"""Edit recipe fields while retaining the rest of the user's YAML verbatim."""
from __future__ import annotations

import math
import re
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


def _flow(value: Any, *, in_flow: bool = False, style: str | None = None) -> str:
    # Let the emitter apply flow-sequence quoting rules to a string splice.
    is_text = isinstance(value, str)
    wrapped = in_flow and is_text
    quote = style if is_text and style in ("'", '"') else None
    text = yaml.dump(
        [value] if wrapped else value,
        Dumper=Dumper,
        default_flow_style=True,
        default_style=quote,
        allow_unicode=True,
        width=1_000_000,
        sort_keys=False,
    ).rstrip("\n").removesuffix("\n...")
    return text[1:-1] if wrapped else text


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
    if config is None and isinstance(key, str):
        config = {}
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


def _readable_collections(text, paths, loader):
    """Expand compact collections using their original scalar tokens (including tags and quotes)."""
    from .yaml_remove import _spans

    root = _spans(text, loader)
    patches = []

    def block(node):
        raw = text[node.start:node.end]
        if not node.flow or not node.children or "\n" in raw or node.event.anchor or node.event.tag:
            return [raw]
        if node.mapping:
            rows = []
            for key, value in node.entries():
                if not isinstance(key.event, yaml.events.ScalarEvent):
                    return [raw]
                label = text[key.start:key.end] + ":"
                child = block(value)
                if len(child) == 1 and child[0] == text[value.start:value.end]:
                    rows.append(label + " " + child[0])
                else:
                    rows.append(label)
                    rows.extend("  " + line for line in child)
            return rows
        if not any(child.children for child in node.children) and len(raw) <= 80:
            return [raw]
        rows = []
        for child in node.children:
            first, *rest = block(child)
            rows.append("- " + first)
            rows.extend("  " + line for line in rest)
        return rows

    def visit(node, path, key=None):
        if paths is not None and not any(p[:len(path)] == path or path[:len(p)] == p for p in paths):
            return
        if node.flow:
            rows = block(node)
            if rows == [text[node.start:node.end]]:
                return
            start, end = node.start, node.end
            indent = node.event.start_mark.column
            prefix = ""
            if key is not None and key.event.end_mark.line == node.event.start_mark.line:
                colon = text[key.end:start]
                if colon.strip() != ":":
                    return
                start = key.end + colon.index(":") + 1
                indent = key.event.start_mark.column + 2
                prefix = "\n" + " " * indent
                # An end-of-line comment on a mapping value still belongs to its key.
                line_end = text.find("\n", end)
                line_end = len(text) if line_end < 0 else line_end
                tail = text[end:line_end].rstrip("\r")
                if tail.lstrip().startswith("#"):
                    end += len(tail)
                    prefix = tail + prefix
            elif key is not None and indent <= key.event.start_mark.column:
                extra = key.event.start_mark.column + 2 - indent
                indent += extra
                prefix = " " * extra
            patches.append((start, end, prefix + ("\n" + " " * indent).join(rows)))
        elif node.mapping:
            for entry, value in node.entries():
                visit(value, path + [getattr(entry.event, "value", None)], entry)
        else:
            for index, child in enumerate(node.children):
                visit(child, path + [index])

    visit(root, [])
    if not patches:
        return text
    pieces = []
    cursor = 0
    for start, end, replacement in patches:
        pieces.extend((text[cursor:start], replacement))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def render(text: str, edits: list[dict[str, Any]], *, format_all: bool = False) -> str:
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
        value = edit["value"]
        literal = edit.get("floatLiteral")
        if literal is not None:
            if (not isinstance(literal, str) or len(literal) > 128
                    or not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+", literal)
                    or type(value) not in (int, float)):
                raise ValueError("Provide a finite scientific-notation literal matching the numeric edit.")
            number = float(literal)
            if not math.isfinite(number) or number != value:
                raise ValueError("The scientific-notation literal does not match the numeric edit.")
            value = number
        desired = _assign(original, path, value)
        if literal is None and _same(desired, original):
            continue
        # Compose only for changed fields so comments, quotes and unrelated
        # hand-written options survive a form edit.
        node = yaml.compose(text, Loader=Loader)
        parent: Any = original
        wanted: Any = desired
        for offset, key in enumerate(path):
            if parent is None and isinstance(node, yaml.ScalarNode):
                # A bare key or explicit null can be populated like an absent optional block.
                start, end = node.start_mark.index, node.end_mark.index
                space = " " if start == end and text[start-1:start] == ":" else ""
                candidate = text[:start] + space + _flow(wanted) + text[end:]
                break
            try:
                child = _child(node, key)
            except KeyError:
                if not isinstance(node, yaml.MappingNode) or not isinstance(key, str):
                    raise ValueError("Cannot locate this field in the YAML document.") from None
                first = node.start_mark
                at = first.index - first.column
                prefix = text[at:first.index]
                if node.flow_style and text[first.index:first.index+1] == "{":
                    # Splice into the braces so sibling comments and typed literals survive.
                    insert = first.index + 1
                    pair = _flow(key, in_flow=True) + ": " + _flow(wanted[key], in_flow=True)
                    candidate = text[:insert] + pair + (", " if node.value else "") + text[insert:]
                elif node.value and not node.flow_style and not prefix.strip():
                    patch = prefix + _flow(key) + ": " + _flow(wanted[key]) + "\n"
                    candidate = text[:at] + patch + text[at:]
                else:
                    candidate = text[:first.index] + _flow(wanted) + text[node.end_mark.index:]
                break
            in_flow = bool(getattr(node, "flow_style", False))
            owner = node
            node, parent, wanted = child, parent[key], wanted[key]
            if offset == len(path) - 1:
                start, end = node.start_mark.index, node.end_mark.index
                replacement = _flow(wanted, in_flow=in_flow, style=getattr(node, "style", None))
                # A block list may align with its key. Its temporary flow spelling
                # must be indented before the source formatter can expand it.
                if isinstance(owner, yaml.MappingNode) and isinstance(node, yaml.SequenceNode) and not node.flow_style:
                    key_node = next(k for k, v in owner.value if v is node)
                    if node.start_mark.column == key_node.start_mark.column:
                        replacement = "  " + replacement
                # A block collection or block scalar can consume the newline
                # ending the final item. Keep the next key on its own line.
                old = text[start:end]
                if old.endswith("\n") and end < len(text):
                    replacement += "\n" + " " * node.end_mark.column
                elif "\n" in old and end < len(text) and old.rsplit("\n", 1)[-1].strip() == "":
                    replacement += "\n" + old.rsplit("\n", 1)[-1]
                candidate = text[:start] + replacement + text[end:]
        if literal is not None:
            # The edited path may have been inserted as part of a whole new mapping.
            # Locate that scalar in the result so identical numbers elsewhere keep their spelling.
            scalar = yaml.compose(candidate, Loader=Loader)
            for key in path:
                scalar = _child(scalar, key)
            if not isinstance(scalar, yaml.ScalarNode):
                raise ValueError("Scientific notation requires a numeric scalar.")
            candidate = (candidate[:scalar.start_mark.index] + "!!float " + literal
                         + candidate[scalar.end_mark.index:])
        if not _same(read_recipe(candidate), desired):
            raise ValueError("This edit also affects a YAML anchor or merged field. Edit it explicitly in YAML.")
        text, original = candidate, desired
    if (format_all or edits) and ("{" in text or "[" in text):
        pretty = _readable_collections(text, None if format_all else [edit["path"] for edit in edits], Loader)
        if pretty != text:
            if not _same(read_recipe(pretty), original):
                raise ValueError("Formatting would change a YAML value.")
            text = pretty
    if crlf:
        return text.replace("\r\n", "\n").replace("\n", "\r\n")
    return text

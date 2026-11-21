"""Remove a single YAML field or sequence item without serializing its neighbors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml
from yaml import events as ev


@dataclass
class Span:
    event: Any
    end: int
    children: list[Span] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.event.start_mark.index

    @property
    def mapping(self) -> bool:
        return isinstance(self.event, ev.MappingStartEvent)

    @property
    def flow(self) -> bool:
        return bool(getattr(self.event, "flow_style", False))

    def entries(self):
        if self.mapping:
            return list(zip(self.children[::2], self.children[1::2], strict=True))
        return [(child, child) for child in self.children]

    def at(self, key):
        entries = self.entries()
        if self.mapping:
            matches = [i for i,(node, _) in enumerate(entries) if getattr(node.event,"value",None) == key]
            if len(matches) != 1:
                raise ValueError(f"Cannot locate a single YAML option {key!r}.")
            return matches[0]
        if type(key) is int and 0 <= key < len(entries):
            return key
        raise ValueError("Cannot locate the requested list item.")


def _spans(text: str, loader) -> Span:
    stack: list[Span] = []
    root = None
    for event in yaml.parse(text, Loader=loader):
        if isinstance(event, (ev.StreamStartEvent,ev.StreamEndEvent,ev.DocumentStartEvent,ev.DocumentEndEvent)):
            continue
        if isinstance(event,(ev.SequenceEndEvent,ev.MappingEndEvent)):
            node = stack.pop()
            if node.flow:
                node.end = event.end_mark.index
            elif node.children:
                node.end = node.children[-1].end
            continue
        node = Span(event,event.end_mark.index)
        if stack:
            stack[-1].children.append(node)
        else:
            root = node
        if isinstance(event,(ev.SequenceStartEvent,ev.MappingStartEvent)):
            stack.append(node)
    if root is None:
        raise ValueError("Empty YAML document.")
    return root


def _line_start(text, position):
    return text.rfind("\n",0,position)+1


def _dash(text, node, parent, lower):
    pos = _line_start(text, node.start)
    # A sequence can carry an anchor on the line before its entries. Determine
    # the physical '-' from the item's own line or preceding standalone line.
    while pos >= lower:
        end = text.find("\n",pos)
        if end < 0:
            end = len(text)
        line = text[pos:end]
        stripped = line.lstrip(" ")
        if stripped == "-" or stripped.startswith(("- ", "-\t", "-\r")):
            return pos + len(line)-len(stripped)
        if pos == 0:
            break
        pos = _line_start(text,pos-1)
    raise ValueError("Cannot determine the sequence entry boundary. Edit this item in YAML.")


def remove_text(text, path, loader):
    parent = _spans(text, loader)
    for key in path[:-1]:
        parent = parent.entries()[parent.at(key)][1]
    entries = parent.entries()
    i = parent.at(path[-1])
    first, value = entries[i]
    if parent.flow:
        if len(entries) == 1:
            start, end = first.start, value.end
        elif i+1 < len(entries):
            comma = text.find(",",value.end,entries[i+1][0].start)
            if comma < 0:
                raise ValueError("Cannot locate the next flow separator.")
            start, end = first.start, comma+1
        else:
            comma = text.find(",",entries[i-1][1].end,first.start)
            if comma < 0:
                raise ValueError("Cannot locate the preceding flow separator.")
            start, end = comma, value.end
        return text[:start]+text[end:]
    if len(entries) == 1:
        # An empty mapping/list needs an explicit representation. Preserve the
        # surrounding indentation, anchor spelling and following comment lines.
        start = first.start if parent.mapping else _dash(text,first,parent,_line_start(text,parent.start))
        return text[:start]+("{}" if parent.mapping else "[]")+_after_value(text,value.end,keep_newline=True)
    if parent.mapping:
        start = first.start
    else:
        lower = entries[i-1][1].end if i else _line_start(text,parent.start)
        start = _dash(text, first, parent,lower)
    line = _line_start(text,start)
    if not text[line:start].strip():
        start = line
    return text[:start]+_after_value(text,value.end,keep_newline=text[line:start].strip() == "-")


def _after_value(text, end, keep_newline):
    if end and text[end-1] == "\n":
        return ("\n" if keep_newline else "")+text[end:]
    newline = text.find("\n",end)
    if newline < 0:
        return ""
    trailing = text[end:newline].strip()
    if trailing and not trailing.startswith("#"):
        raise ValueError("Removal would also change a neighboring YAML field.")
    return text[newline if keep_newline else newline+1:]

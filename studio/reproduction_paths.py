"""Map recorded paths without interpreting another machine's cwd or mounts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import local_path

_DRIVE = re.compile(r"([A-Za-z]):([\\/])(.*)", re.DOTALL)


def _parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    result = []
    for part in parts:
        if not part or part == ".":
            continue
        if part == "..":
            if not result:
                raise ValueError("A path cannot traverse above its source root.")
            result.pop()
        else:
            result.append(part)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SourcePath:
    kind: str
    anchor: str
    parts: tuple[str, ...]

    @property
    def key(self) -> tuple[str, ...]:
        parts = (
            tuple(value.casefold() for value in self.parts)
            if self.kind in ("drive", "unc")
            else self.parts
        )
        return (self.kind, self.anchor, *parts)

    def __str__(self) -> str:
        return self.anchor + "/".join(self.parts)


def _path_text(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\0" in value
    ):
        raise ValueError(
            "Provide an explicit path without blank padding or NUL characters."
        )
    return value


def source_path(value: str, base: SourcePath | None = None) -> SourcePath:
    """Normalize a declared path lexically, including foreign WSL/Windows roots."""
    value = _path_text(value)
    if drive := _DRIVE.fullmatch(value):
        tail = _parts(tuple(drive[3].replace("\\", "/").split("/")))
        return SourcePath("drive", drive[1].upper() + ":/", tail)
    if re.match(r"^[A-Za-z]:", value):
        raise ValueError("A drive-relative path needs an explicit drive root.")
    if value.startswith(("\\\\", "//")):
        server, sep, rest = value.replace("\\", "/")[2:].partition("/")
        share, _, tail = rest.partition("/")
        if not sep or not server or not share or share in (".", ".."):
            raise ValueError("A network or WSL path needs both its host and share.")
        if server.casefold() in ("wsl.localhost", "wsl$"):
            return SourcePath(
                "wsl",
                f"//wsl.localhost/{share.casefold()}/",
                _parts(tuple(tail.split("/"))),
            )
        return SourcePath(
            "unc",
            f"//{server.casefold()}/{share.casefold()}/",
            _parts(tuple(tail.split("/"))),
        )
    if value.startswith("/"):
        parts = _parts(tuple(value.split("/")))
        if len(parts) >= 2 and parts[0] == "mnt" and re.fullmatch(r"[a-z]", parts[1]):
            return SourcePath("drive", parts[1].upper() + ":/", parts[2:])
        return SourcePath("posix", "/", parts)
    if value == "~" or value.startswith("~/"):
        return SourcePath("home", "~/", _parts(tuple(value[2:].split("/"))))
    if value.startswith("~"):
        raise ValueError(
            "Use an explicit mapped path for archived named user home directories."
        )
    if value.startswith("\\") or base is None:
        raise ValueError(
            "Relative inputs need the original checkout path as source_root."
        )
    relative = (
        value.replace("\\", "/") if base.kind in ("drive", "unc", "wsl") else value
    )
    return SourcePath(
        base.kind, base.anchor, _parts((*base.parts, *relative.split("/")))
    )


class Relocations:
    def __init__(self, mappings: list[dict[str, str]] | None) -> None:
        if mappings is None:
            mappings = []
        if not isinstance(mappings, list) or len(mappings) > 128:
            raise ValueError("Provide up to 128 relocation mappings.")
        prefixes: dict[tuple[str, ...], Path] = {}
        for mapping in mappings:
            if not isinstance(mapping, dict) or set(mapping) != {"from", "to"}:
                raise ValueError("Each relocation needs a from prefix and a to path.")
            old = source_path(mapping["from"])
            new = local_path(_path_text(mapping["to"]), Path())
            if not new.is_absolute():
                raise ValueError(
                    "A relocation destination must be an absolute path on this host."
                )
            try:
                resolved = new.resolve()
            except (OSError, RuntimeError) as error:
                raise ValueError(
                    f"The relocation destination cannot be resolved: {new}."
                ) from error
            if old.key in prefixes and prefixes[old.key] != resolved:
                raise ValueError(f"Multiple destinations were supplied for {old}.")
            prefixes[old.key] = resolved
        self.prefixes = sorted(prefixes.items(), key=lambda row: -len(row[0]))

    def locate(self, source: SourcePath) -> tuple[Path, bool]:
        for prefix, target in self.prefixes:
            if source.key[: len(prefix)] != prefix:
                continue
            tail = source.parts[len(prefix) - 2 :]
            if any("\\" in part for part in tail):
                raise ValueError(
                    "A relocated path component contains a path separator."
                )
            candidate = target.joinpath(*tail).resolve()
            if not candidate.is_relative_to(target):
                raise ValueError(f"A relocated path escapes its destination {target}.")
            return candidate, True

        if source.kind == "home":
            raise ValueError(
                "Map the archived home prefix to an explicit path on this host."
            )
        return local_path(str(source), Path()).resolve(), False

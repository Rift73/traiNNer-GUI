"""Index the shipped Python factories without importing their training dependencies.

Unchanged modules can register on first use. New or edited project modules are
imported normally, so the existing decorator and imperative plugin API works
without having to regenerate anything during local development.
"""
from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from ._builtin_factories import BUILTINS

if TYPE_CHECKING:
    from traiNNer.utils.registry import Registry


def install(package: str, directory: Path, suffix: str, registries: dict[str, Registry]) -> None:
    shipped = BUILTINS[package.rsplit(".", 1)[-1]]
    changed: list[str] = []
    for file in sorted(directory.glob("*" + suffix)):
        entry = shipped.get(file.stem)
        module = f"{package}.{file.stem}"
        if entry is None or hashlib.sha256(file.read_bytes()).hexdigest() != entry[0]:
            changed.append(module)
            continue
        for registry, names in entry[1].items():
            for name in names.split():
                registries[registry].defer(name, module)
    for module in changed:
        import_module(module)

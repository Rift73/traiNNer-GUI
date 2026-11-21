"""Refresh the shipped factory index using the checkout's training environment.

    python -m scripts.options.index_factories                # all packages
    python -m scripts.options.index_factories --group data   # only datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path

from trainner_config._builtin_factories import BUILTINS

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "archs": ("_arch.py", {"arch": "ARCH_REGISTRY", "spandrel": "SPANDREL_REGISTRY",
                           "testarch": "TESTARCH_REGISTRY"}),
    "losses": ("_loss.py", {"loss": "LOSS_REGISTRY"}),
    "data": ("_dataset.py", {"dataset": "DATASET_REGISTRY"}),
}


def index_group(group: str) -> dict[str, tuple[str, dict[str, str]]]:
    package = import_module(f"traiNNer.{group}")
    suffix, registries = GROUPS[group]
    files = sorted((ROOT / "traiNNer" / group).glob("*" + suffix))
    names: dict[str, dict[str, list[str]]] = {p.stem: {} for p in files}
    for source in files:
        import_module(f"{package.__name__}.{source.stem}")
    for registry_id, attribute in registries.items():
        for name, factory in getattr(package, attribute):
            owner, _, module = factory.__module__.rpartition(".")
            if group == "data" and owner != package.__name__:
                # A task or installed extension owns this dataset registration.
                continue
            if module not in names:
                raise ValueError(f"Cannot index {registry_id}/{name} defined in {factory.__module__}.")
            names[module].setdefault(registry_id, []).append(name)
    return {
        file.stem: (
            hashlib.sha256(file.read_bytes()).hexdigest(),
            {reg: " ".join(sorted(keys)) for reg, keys in names[file.stem].items()},
        )
        for file in files
    }


def main(group: str | None = None) -> None:
    index = {} if group is None else dict(BUILTINS)
    for name in (GROUPS if group is None else (group,)):
        index[name] = index_group(name)

    lines = ['"""Shipped training factories. Regenerate with scripts/options/index_factories.py."""',
             "", "BUILTINS: dict[str, dict[str, tuple[str, dict[str, str]]]] = {"]
    for name, files in index.items():
        lines.append(f'    "{name}": {{')
        for module, (digest, ids) in files.items():
            lines.append(f'        "{module}": ("{digest}", {json.dumps(ids, sort_keys=True)}),')
        lines.append("    },")
    lines.extend(("}", ""))
    (ROOT / "trainner_config" / "_builtin_factories.py").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS)
    main(parser.parse_args().group)

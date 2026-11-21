"""Recipe library, training catalog, and user-owned drafts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import msgspec

from trainner_config._builtin_factories import BUILTINS

from trainner_config.tasks import TRAINING_TASKS, recurrent_architectures

from .workspace import WindowsProject
from .yaml_edit import MAX_RECIPE_BYTES, read_recipe


def safe_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A name is required.")
    if (name != name.strip() or name.startswith(".") or name.endswith(".")
            or len(name.encode("utf-8")) > 190
            or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in name)
            or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", name.split(".")[0])):
        raise ValueError("Use a single file name without path separators or reserved characters.")
    return name


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=".writing-", delete=False) as stream:
        name = stream.name
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            os.unlink(name)
            raise
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _recipe_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(MAX_RECIPE_BYTES + 1)
    if len(content) > MAX_RECIPE_BYTES:
        raise ValueError("Recipe exceeds 2 MiB.")
    return content


def task_from_config(config: dict[str, Any]) -> str:
    selected = config.get("task")
    if isinstance(selected, str) and selected not in ("", "auto"):
        return selected
    arch = config.get("network_g") or {}
    if not isinstance(arch, dict):
        return "unknown"
    if config.get("network_ae") and not arch:
        return "autoencoder"
    if str(arch.get("type", "")).lower() in recurrent_architectures:
        return "recurrent_restoration"
    return "image_restoration"


def describe_actions(actions):
    return {
        name: {"inputs": list(action.inputs), "output": action.output,
               **({"saveBest": action.snapshot_metrics is not None} if name == "validate" else {}),
               **({"optionalExports": list(action.optional_exports)} if name == "holdout" else {})}
        for name, action in actions.items()
    }


def _template_files(root: Path):
    for directory in ("options/train", "options/_templates/train"):
        for file in sorted((root / directory).rglob("*.y*ml")):
            if file.suffix in (".yaml", ".yml") and file.is_file() and not file.is_symlink():
                yield file


def _recipe_metadata(content: bytes) -> dict[str, Any]:
    cfg = read_recipe(content.decode("utf-8-sig"))
    net = cfg.get("network_g") or cfg.get("network_ae") or {}
    return {"task": task_from_config(cfg), "architecture": net.get("type") if isinstance(net, dict) else None,
            "scale": cfg.get("scale")}


def template_catalog_index(root: Path) -> dict:
    """Build the small, content-addressed summaries used by the recipe picker."""
    summaries = {}
    for file in _template_files(root):
        if file.stat().st_size <= MAX_RECIPE_BYTES:
            content = file.read_bytes()
            try:
                summaries[hashlib.sha256(content).hexdigest()] = _recipe_metadata(content)
            except ValueError:
                continue  # A locally invalid template still appears with its live parser error.
    return summaries


class RecipeLibrary:
    def __init__(self, root: Path, state_dir: Path, *, project: WindowsProject | None = None):
        self.root = root.resolve()
        self.project = project
        self.drafts = (state_dir / "recipes").resolve() if project is None else project.root / "recipes"
        self.drafts_lock = self.drafts / ".lock" if project is None else state_dir / "recipes.lock"
        self.schema_path = self.root / "schemas/redux-config.schema.json"
        self.local_schema = state_dir / "catalog" / "schema.json"
        self._schema_cache = None
        self._summaries: dict[Path, tuple[tuple[int, ...], dict[str, Any]]] = {}

    def schema(self) -> dict[str, Any]:
        base = self.schema_path.stat()
        try:
            local = self.local_schema.stat()
        except FileNotFoundError:
            local = None
        key = ((base.st_ino, base.st_size, base.st_mtime_ns),
               (local.st_ino, local.st_size, local.st_mtime_ns) if local else None)
        cached = self._schema_cache
        if cached and cached[0] == key:
            return cached[1]
        packaged = self.schema_path.read_bytes()
        schema = None
        if local:
            try:
                candidate = json.loads(self.local_schema.read_bytes())
            except FileNotFoundError:
                candidate = {}
            stamp = candidate.get("x-trainner-refresh", {})
            if stamp.get("root") == str(self.root) and stamp.get("baseSha256") == hashlib.sha256(packaged).hexdigest():
                schema = candidate
        if schema is None:
            schema = json.loads(packaged)
        self._schema_cache = (key, schema)
        return schema

    def refresh_catalog(self) -> dict:
        from .catalog_refresh import refresh

        refresh(self.root, self.local_schema.parent, self.schema_path, self.local_schema)
        return self.catalog()

    def catalog(self) -> dict[str, Any]:
        from trainner_config.schedules import scheduler_catalog

        schema = self.schema()
        defs = schema["$defs"]
        props = defs["ReduxOptions"]["properties"]
        arch_ids = props["network_g"]["discriminator"]["mapping"]
        dataset_type = defs["DatasetOptions"]["properties"]["type"]
        loss_refs = defs["TrainOptions"]["properties"]["losses"]["items"]["anyOf"]
        losses = [r["$ref"].rsplit("/", 1)[-1] for r in loss_refs if "$ref" in r]
        extensions = schema.get("x-trainner-catalog", {})
        if "architectureRegistries" in extensions:
            arch_sources = extensions["architectureRegistries"]
        else:
            arch_sources = {}
            for _, registrations in BUILTINS["archs"].values():
                for provenance, names in registrations.items():
                    for name in names.split():
                        arch_sources.setdefault(name, []).append(provenance)
        tasks = []
        by_arch: dict[str, str] = {}
        for task, spec in TRAINING_TASKS.items():
            choices = spec.recommended_architectures or spec.architecture_types or ()
            tasks.append({"id": task, "label": task.replace("_", " ").capitalize(),
                          "actions": describe_actions(spec.actions),
                          "architectureActions": {name:describe_actions(actions) for name, actions in spec.architecture_actions.items()},
                          "networkField": spec.network_field, "trainingOnly": spec.training_only,
                          "datasetGeometry": spec.dataset_geometry,
                          "architectures": list(spec.architecture_types) if spec.architecture_types else None,
                          "recommendedArchitectures": list(choices), "datasetTypes": list(spec.dataset_types),
                          "excludedArchitectures": list(spec.excluded_architecture_types),
                          "taskOptions": msgspec.json.schema(spec.options_type) if spec.options_type else None})
            for name in choices:
                by_arch[name] = task
        architecture_names = sorted(set(arch_sources).union(arch_ids, by_arch))
        return {
            "tasks": tasks,
            "architectures": [{"id": a, "task": by_arch.get(a, task_from_config(
                {"network_ae" if a == "autoencoder" else "network_g": {"type": a}})),
                                "registries": arch_sources.get(a, []),
                                "schemaRef": arch_ids.get(a)} for a in architecture_names],
            "losses": [{"id":a} for a in sorted(losses)],
            "datasets": [{"id":name} for name in dataset_type.get("examples", dataset_type.get("enum", ()))],
            "optimizers": extensions.get("optimizers", []),
            "metrics": extensions.get("metrics", []),
            "schedulers": extensions.get("schedulers") or scheduler_catalog(),
            "signatureIssues": extensions.get("signatureIssues", []),
            "source": "local" if "x-trainner-refresh" in schema else "packaged",
            "generatedAt": schema.get("x-trainner-refresh", {}).get("generatedAt"),
            "schemaUrl": "/api/schema",
        }

    def _describe(self, file: Path, key: str, known: dict | None = None) -> dict[str, Any]:
        stat = file.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        cached = self._summaries.get(file)
        if cached is None or cached[0] != identity:
            metadata: dict[str, Any] = {}
            try:
                if stat.st_size > MAX_RECIPE_BYTES:
                    raise ValueError("Recipe exceeds 2 MiB.")
                content = file.read_bytes()
                indexed = known.get(hashlib.sha256(content).hexdigest()) if known else None
                metadata = indexed if indexed is not None else _recipe_metadata(content)
            except (ValueError, OSError) as exc:
                metadata["error"] = str(exc)
            self._summaries[file] = identity, metadata
        else:
            metadata = cached[1]
        return {"path": key, "name": file.stem, **metadata}

    def templates(self) -> list[dict[str, Any]]:
        known = (self.schema().get("x-trainner-catalog", {}).get("templateSummaries")
                 if self.schema_path.is_file() else None)
        return [{**self._describe(file, file.relative_to(self.root).as_posix(), known), "family": file.parent.name}
                for file in _template_files(self.root)]

    def saved(self) -> list[dict[str, Any]]:
        if self.project is not None:
            self.project.require_directory(self.drafts)
        return [self._describe(p, "recipes/" + p.name) for p in sorted(self.drafts.glob("*.yml"))
                if p.is_file() and not p.is_symlink()]

    def _locate(self, key: str) -> Path:
        if not isinstance(key, str):
            raise ValueError("A recipe path is required.")
        for prefix, root in (("recipes/", self.drafts), ("options/train/", self.root/"options/train"),
                             ("options/_templates/train/", self.root/"options/_templates/train")):
            if key.startswith(prefix):
                requested = root / key[len(prefix):]
                if self.project is not None and prefix == "recipes/":
                    if not requested.is_relative_to(root) or ".." in requested.parts:
                        break
                    self.project.require_directory(requested.parent)
                path = requested.resolve()
                if self.project is not None and prefix == "recipes/" and path != requested:
                    break
                if path.is_relative_to(root) and path.suffix in (".yml", ".yaml"):
                    return path
        raise ValueError("Select a recipe from the library.")

    def get(self, key: str) -> dict[str, str]:
        path = self._locate(key)
        content = _recipe_bytes(path)
        return {"path": key, "name": path.stem, "yaml": content.decode("utf-8-sig"),
                "revision": hashlib.sha256(content).hexdigest()}

    def save(self, name: str, text: str, expected: str | None = None) -> dict[str, str]:
        import fcntl

        name = safe_name(name)
        read_recipe(text)
        key = "recipes/" + name + ".yml"
        self.drafts_lock.parent.mkdir(parents=True, exist_ok=True)
        with self.drafts_lock.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            path = self._locate(key)
            self.drafts.mkdir(parents=self.project is None, exist_ok=True)
            stored = text
            if expected is not None:
                previous = _recipe_bytes(path) if path.is_file() else None
                if previous is None or expected != hashlib.sha256(previous).hexdigest():
                    raise FileExistsError("The saved recipe has changed since it was opened.")
                if previous.startswith(b"\xef\xbb\xbf") and not text.startswith("\ufeff"):
                    stored = "\ufeff" + text
            content = stored.encode("utf-8")
            if len(content) > MAX_RECIPE_BYTES:
                raise ValueError("Recipe exceeds 2 MiB.")
            atomic_text(path, stored)
        return {"path": key, "name": name, "yaml": stored.removeprefix("\ufeff"),
                "revision": hashlib.sha256(content).hexdigest()}

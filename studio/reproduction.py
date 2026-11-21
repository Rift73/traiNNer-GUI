"""Export immutable run inputs and prepare a relocated recipe without dispatch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from traiNNer.training.checkpoint_association import SAVE_ID, SAVE_ID_KEY, pth_save_id

from . import provenance
from . import run_observe as observe
from . import run_process as process
from .native import ROOT as NATIVE_ROOT
from .paths import artifact_version
from .preflight import input_paths, preflight
from .recipes import RecipeLibrary, safe_name
from .reproduction_paths import Relocations, SourcePath, source_path
from .reproduction_pins import input_precondition
from .runs import ACTIVE, RunManager
from .yaml_edit import read_recipe, render

_FORMAT = "trainner-run-recipe"
_MODES = ("replay", "weights-only")
_TRAIN_OUTPUTS = {
    "experiments_root",
    "models",
    "resume_models",
    "training_states",
    "log",
    "visualization",
    "results_root",
}
_WEIGHT_NAME = re.compile(
    r"net_(g|ae)(?:_ema)?_(?:[1-9][0-9]*|latest)\.(?:pth|safetensors)"
)
_LEGACY_LOG_LIMIT = 128 * 1024
_SOURCE_MANIFEST_LIMIT = 512 * 1024
_LOG_SEED = re.compile(
    observe.LOG_PREFIX + r"Training (?:in deterministic mode )?with manual_seed=(\d+)",
    re.MULTILINE,
)
_LOG_START = re.compile(
    observe.LOG_PREFIX
    + r"(?:Start training from|Resuming training from) epoch: \d+, iter: (\d+)\.",
    re.MULTILINE,
)
_LOCAL = Relocations(None)


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(data: Any) -> str:
    try:
        text = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return _sha(text.encode("ascii"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(
            "The reproduction bundle must contain finite JSON values."
        ) from error


def _portable_numbers(value: Any) -> Any:
    # JSON readers may emit 1000000 for 1000000.0. Normalize exactly integral
    # binary64 numbers in the interoperable integer range before signing.
    if isinstance(value, dict):
        return {key: _portable_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_numbers(item) for item in value]
    if type(value) is float and value.is_integer() and abs(value) <= 2**53 - 1:
        return int(value)
    return value


def _issue(code: str, path: str, message: str, level: str = "warning") -> dict:
    return {"level": level, "code": code, "path": path, "message": message}


def _field(field: Iterable[str | int]) -> str:
    return ".".join(map(str, field))


def _seed_number(value: Any) -> int | None:
    return value if type(value) is int else None


def _argv_seed(command: dict) -> int | None:
    args = command.get("argv")
    if not isinstance(args, list) or "--manual_seed" not in args:
        return None
    try:
        return int(args[args.index("--manual_seed") + 1])
    except (IndexError, TypeError, ValueError):
        return None


def _identity(
    path: Path,
    expected_version: str | None = None,
    *,
    hash_contents: bool = False,
    expected_size: int | None = None,
) -> dict | None:
    try:
        before = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    version = artifact_version(before)
    if expected_version is not None and expected_version != version:
        raise ValueError(
            "The selected weight file changed. Select its current version."
        )
    save_id = None
    container = "unverified"
    if path.suffix == ".safetensors":
        from safetensors import SafetensorError, safe_open

        try:
            with safe_open(str(path), framework="numpy", device="cpu") as weights:
                container = "valid" if weights.keys() else "invalid"
                save_id = (weights.metadata() or {}).get(SAVE_ID_KEY)
        except SafetensorError:
            container = "invalid"
    elif path.suffix == ".pth":
        try:
            save_id = pth_save_id(path)
            container = "valid"
        except BadZipFile:
            # Legacy Torch serialization may have no ZIP metadata.
            pass
        except ValueError:
            container = "invalid"
    digest = None
    if hash_contents and (expected_size is None or expected_size == before.st_size):
        with path.open("rb") as stream:
            if artifact_version(os.fstat(stream.fileno())) != version:
                raise ValueError(f"Input changed before it could be hashed: {path}.")
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if artifact_version(os.fstat(stream.fileno())) != version:
                raise ValueError(f"Input changed while it was hashed: {path}.")
    if artifact_version(path.stat()) != version:
        raise ValueError(f"Input changed while its identity was read: {path}.")
    evidence = {
        "fileVersion": version,
        "size": before.st_size,
        "mtimeNs": str(before.st_mtime_ns),
        "container": container,
    }
    if digest is not None:
        evidence["sha256"] = digest
    if isinstance(save_id, str) and SAVE_ID.fullmatch(save_id):
        evidence["saveId"] = save_id
    elif save_id is not None:
        evidence["container"] = "invalid"
    return evidence


def _observe_input(
    value: Any, base: SourcePath | None, kind: str, hash_contents: bool
) -> tuple[str | None, dict | None]:
    try:
        original = source_path(value, base)
    except ValueError:
        if hash_contents:
            raise ValueError(
                f"Cannot locate {value!r} to hash the exported input."
            ) from None
        return None, None
    if kind == "dataset":
        return str(original), None
    try:
        path, _ = _LOCAL.locate(original)
        identity = _identity(path, hash_contents=hash_contents)
        if hash_contents and identity is None:
            raise FileNotFoundError(f"Cannot hash the missing checkpoint input {path}.")
    except (OSError, RuntimeError, ValueError):
        if hash_contents:
            raise
        identity = None
    return str(original), identity


def _kind(field: list[str | int]) -> str:
    if field[0] == "datasets":
        return "dataset"
    return "state" if field[1] == "resume_state" else "weight"


def _declared_inputs(
    config: dict, base: SourcePath | None, launch: dict, mode: str, hash_inputs: bool
) -> list[dict]:
    result = []
    recorded = _mapping(launch.get("inputs"))
    for field, value in input_paths(config):
        kind = _kind(field)
        # Fine-tune drops the source training state and all inherited student weights.
        hash_contents = (
            hash_inputs
            and kind != "dataset"
            and (mode == "replay" or field == ["path", "pretrain_network_g_teacher"])
        )
        declared, observed = _observe_input(value, base, kind, hash_contents)
        item = {"field": field, "kind": kind, "value": value, "sourcePath": declared}
        if kind != "dataset":
            if saved := _mapping(recorded.get(field[1])):
                item["launch"] = saved
            if observed is not None:
                item["observed"] = observed
        result.append(item)
    return result


def _model_role(config: dict) -> str:
    has_g, has_ae = bool(config.get("network_g")), bool(config.get("network_ae"))
    if has_g == has_ae:
        raise ValueError(
            "Select a recipe with one primary generator or autoencoder for weights-only initialization."
        )
    return "g" if has_g else "ae"


def _initialization(config: dict, reported: dict) -> str:
    paths = _mapping(config.get("path"))
    if paths.get("resume_state"):
        return "full-state"
    if type(reported.get("initialStep")) is int and reported["initialStep"] > 0:
        return "unverified"
    if any(
        value for key, value in paths.items() if key.startswith("pretrain_network_")
    ):
        return "pretrained"
    return "fresh"


def _basis(recipe: dict) -> tuple[str, str]:
    for key in ("effective", "submitted", "archived"):
        if isinstance(value := recipe.get(key), str):
            return key, value
    raise ValueError("The reproduction bundle has no saved recipe.")


def _legacy_startup(experiment: Path, recipe: Any) -> dict:
    if not isinstance(recipe, str):
        return {}
    logs, recipes = [], []
    for entry in observe.files(experiment):
        if (
            entry.name.startswith("train_")
            and entry.suffix == ".log"
            and observe.regular_file(entry)
        ):
            logs.append(entry)
        elif entry.suffix in (".yml", ".yaml") and observe.regular_file(entry):
            recipes.append(entry)
        if len(logs) > 1 or len(recipes) > 1:
            return {}
    if len(logs) != 1 or len(recipes) != 1 or observe.recipe_text(recipes[0]) != recipe:
        return {}
    log = logs[0]
    try:
        before = log.stat()
        if before.st_size > _LEGACY_LOG_LIMIT:
            return {}
        with log.open("rb", buffering=0) as stream:
            data = stream.read(_LEGACY_LOG_LIMIT + 1)
        if len(data) != before.st_size or artifact_version(before) != artifact_version(
            log.stat()
        ):
            return {}
    except OSError:
        return {}
    content = data.decode("utf-8", errors="replace")
    seeds, starts = (
        list(_LOG_SEED.finditer(content)),
        list(_LOG_START.finditer(content)),
    )
    if len(seeds) != 1 or len(starts) != 1 or seeds[0].start() > starts[0].start():
        return {}
    return {"log": log.name, "seed": int(seeds[0][1]), "initialStep": int(starts[0][1])}


def _recorded_sources(runs: RunManager, launch: dict) -> dict | None:
    recorded = _mapping(launch.get("source")).get("sha256")
    if not isinstance(recorded, str) or re.fullmatch("[0-9a-f]{64}", recorded) is None:
        return None
    path = runs.runs.parent / "provenance" / f"{recorded}.json"
    try:
        if not observe.regular_file(path):
            return None
        with path.open("rb") as stream:
            content = stream.read(_SOURCE_MANIFEST_LIMIT + 1)
        if len(content) > _SOURCE_MANIFEST_LIMIT or _sha(content) != recorded:
            return None
        files = json.loads(content)
    except (OSError, ValueError):
        return None
    return _mapping(files) or None


def export_run_recipe(
    runs: RunManager,
    run_id: str,
    *,
    attempt_id: str | None = None,
    mode: str = "replay",
    weight_path: str | None = None,
    weights_version: str | None = None,
    hash_inputs: bool = False,
) -> dict[str, Any]:
    """Return a checksum-addressed JSON bundle from the saved run evidence."""
    if type(hash_inputs) is not bool:
        raise ValueError("hash_inputs must be a boolean.")
    if mode not in _MODES:
        raise ValueError("Choose replay or weights-only reproduction.")
    if mode == "weights-only" and not isinstance(weight_path, str):
        raise ValueError("Select the primary generator or autoencoder weight artifact.")
    if mode == "replay" and (weight_path is not None or weights_version is not None):
        raise ValueError("A selected weight belongs to weights-only reproduction.")
    if weights_version is not None and not isinstance(weights_version, str):
        raise ValueError("Provide the selected weight version as text.")

    run = runs.detail(run_id)
    source = run["source"]
    origin = {
        key: run[key]
        for key in ("id", "name", "source", "status", "sourceRoot", "sourceRecipe")
        if key in run
    }
    recipe: dict[str, Any] = {"submitted": None, "effective": None, "archived": None}
    launch: dict = {}
    reported: dict = {}
    invocation: dict = {}
    legacy: dict = {}
    limits = []
    if source == "managed":
        attempt = runs.attempt_dir(run_id, attempt_id)
        if run["attemptId"] == attempt.name and run["status"] in ACTIVE:
            raise ValueError("Export an attempt after its trainer has exited.")
        result = attempt / "result.json"
        receipt = process.read_record(result) if observe.regular_file(result) else None
        origin.update(
            attemptId=attempt.name, status=(receipt or {}).get("status", "unverified")
        )
        command = attempt / "invocation.json"
        if observe.regular_file(command):
            invocation = _mapping(process.read_record(command))
        options = attempt / "options.yml"
        recipe["submitted"] = run.get("yaml")
        recipe["effective"] = (
            observe.recipe_text(options) if observe.regular_file(options) else None
        )
        records = provenance.read_attempt(attempt)
        launch, reported = (
            _mapping(records.get("launch")),
            _mapping(records.get("runtime")),
        )
        if recipe["effective"] is None:
            limits.append(
                _issue(
                    "effectiveRecipeMissing",
                    "recipe.effective",
                    "The selected attempt has no readable effective options.",
                )
            )
    else:
        if attempt_id is not None:
            raise ValueError("Imported runs do not expose managed attempt IDs.")
        recipe["archived"] = run.get("yaml")
        legacy = _legacy_startup(runs.experiment_dir(run_id), recipe["archived"])
        limits.append(
            _issue(
                "archivedRecipe",
                "recipe",
                "The archive has a saved recipe. Its exact launch options, seed overrides and attempt are unverified.",
            )
        )

    basis, text = _basis(recipe)
    config = read_recipe(text)
    recorded_sha = launch.get("recipeSha256")
    if isinstance(recorded_sha, str) and recipe["effective"] is not None:
        if _sha(text.encode("utf-8")) != recorded_sha:
            limits.append(
                _issue(
                    "effectiveRecipeChanged",
                    "recipe.effective",
                    "The saved options disagree with the launch recipe digest.",
                    "error",
                )
            )
    checkout = invocation.get("cwd") or _mapping(launch.get("environment")).get(
        "checkout"
    )
    base = source_path(checkout) if isinstance(checkout, str) else None
    origin["checkout"] = str(base) if base is not None else None

    actual = _seed_number(reported.get("seed"))
    invoked = _argv_seed(invocation)
    declared = _seed_number(config.get("manual_seed"))
    candidates = (
        ("runtime", actual),
        ("invocation", invoked),
        ("runRecord", _seed_number(run.get("seed")) if source == "managed" else None),
        ("legacyLog", _seed_number(legacy.get("seed"))),
        ("declared", declared),
    )
    seed_source, seed = next(
        ((kind, value) for kind, value in candidates if value is not None),
        ("missing", None),
    )
    if actual is not None and invoked is not None and actual != invoked:
        limits.append(
            _issue(
                "seedConflict",
                "manual_seed",
                "The runtime seed disagrees with the launch argument.",
                "error",
            )
        )

    initialization: dict[str, Any] = {
        "kind": _initialization(config, reported or legacy)
    }
    if mode == "weights-only":
        assert weight_path is not None
        role = _model_role(config)
        selected = runs.artifact(run_id, weight_path)
        match = _WEIGHT_NAME.fullmatch(selected.name)
        if match is None or match[1] != role:
            raise ValueError(f"Select a saved net_{role} or net_{role}_ema checkpoint.")
        identity = _identity(selected, weights_version, hash_contents=hash_inputs)
        if identity is None:
            raise FileNotFoundError("The selected weight file is unavailable.")
        if identity["container"] == "invalid":
            raise ValueError("Select a structurally readable weight file.")
        initialization = {
            "kind": "weights-only",
            "weight": {
                "artifact": weight_path,
                "sourcePath": str(selected),
                "role": role,
                "observed": identity,
            },
        }

    body = {
        "format": _FORMAT,
        "version": 1,
        "mode": mode,
        "origin": origin,
        "recipe": {**recipe, "basis": basis},
        "seed": {"value": seed, "evidence": seed_source},
        "initialization": initialization,
        "provenance": {
            "launch": launch or None,
            "runtime": reported or None,
            "invocation": invocation or None,
            "legacyStartup": legacy or None,
            "sourceFiles": _recorded_sources(runs, launch),
        },
        "inputs": _declared_inputs(config, base, launch, mode, hash_inputs),
        "limitations": limits,
    }
    body = _portable_numbers(body)
    return {**body, "digest": _digest(body)}


def _verify_bundle(bundle: Any) -> None:
    if (
        not isinstance(bundle, dict)
        or bundle.get("format") != _FORMAT
        or type(bundle.get("version")) is not int
        or bundle["version"] != 1
        or bundle.get("mode") not in _MODES
    ):
        raise ValueError("Provide a version 1 traiNNer run recipe bundle.")
    digest = _digest({key: value for key, value in bundle.items() if key != "digest"})
    if bundle.get("digest") != digest:
        raise ValueError("The reproduction bundle checksum does not match.")
    sections = ("origin", "recipe", "seed", "initialization", "provenance")
    if any(not isinstance(bundle.get(key), dict) for key in sections):
        raise ValueError("The reproduction bundle is missing required sections.")
    limitations = bundle.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(row, dict)
        or row.get("level") not in ("error", "warning")
        or not isinstance(row.get("path"), str)
        or not isinstance(row.get("message"), str)
        for row in limitations
    ):
        raise ValueError("The reproduction bundle has invalid limitations.")


def _source_comparison(
    library: RecipeLibrary, saved: dict, recorded_files: dict
) -> dict:
    recorded = _mapping(saved.get("source"))
    if not isinstance(expected := recorded.get("sha256"), str):
        return {
            "status": "unverified",
            "recorded": recorded or None,
            "current": None,
            "reason": "No source fingerprint was captured at launch.",
        }
    path = NATIVE_ROOT / "provenance"
    program = path / "build/clang/fingerprint"
    result: dict[str, Any] = {"recorded": recorded, "current": None}
    try:
        dependencies = (
            path / "fingerprint.cpp",
            path / "Makefile",
            NATIVE_ROOT / "tool.mk",
        )
        if not program.is_file() or program.stat().st_mtime_ns < max(
            p.stat().st_mtime_ns for p in dependencies
        ):
            raise ValueError(
                "The prepared native source fingerprint helper is unavailable or outdated."
            )
        # The existing helper reads its cache as an input and does not write it.
        cache = library.drafts.parent / "provenance/source-cache.v1"
        output = subprocess.run(
            [str(program), str(library.root), str(cache)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=25,
            check=True,
        ).stdout.split(b"\0")
        if output[-1] or len(output) < 4 or len(output) % 3 != 1:
            raise ValueError(
                "The source fingerprint helper returned incomplete evidence."
            )
        files = {
            os.fsdecode(output[i]): output[i + 2].decode("ascii")
            for i in range(0, len(output) - 1, 3)
        }
        if any(
            re.fullmatch("[0-9a-f]{64}", digest) is None for digest in files.values()
        ):
            raise ValueError("The source fingerprint helper returned invalid digests.")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {**result, "status": "unverified", "reason": str(error)}
    digest = _digest(files)
    result.update(
        status="match" if digest == expected else "different",
        current={"sha256": digest, "files": len(files)},
    )
    if recorded_files and _digest(recorded_files) == expected:
        categories = {
            "added": sorted(files.keys() - recorded_files.keys()),
            "removed": sorted(recorded_files.keys() - files.keys()),
            "changed": sorted(
                name
                for name in files.keys() & recorded_files.keys()
                if files[name] != recorded_files[name]
            ),
        }
        result["differences"] = {
            key: {"count": len(names), "paths": names[:50]}
            for key, names in categories.items()
        }
    return result


def _environment_comparison(
    library: RecipeLibrary, saved: dict, reported: dict
) -> dict:
    recorded = _mapping(saved.get("environment"))
    live = provenance.environment(library.root, library.drafts.parent)
    keys = (
        "python",
        "platform",
        "kernel",
        "machine",
        "logicalCpus",
        "packages",
        "environment",
    )
    current = {key: live[key] for key in keys if key in live}
    differences = []
    for key in keys:
        if key not in recorded:
            continue
        old, new = recorded[key], current.get(key)
        if (
            key in ("packages", "environment")
            and isinstance(old, dict)
            and isinstance(new, dict)
        ):
            differences.extend(
                {
                    "path": f"{key}.{name}",
                    "recorded": old.get(name),
                    "current": new.get(name),
                }
                for name in sorted(old.keys() | new.keys())
                if old.get(name) != new.get(name)
            )
        elif old != new:
            differences.append({"path": key, "recorded": old, "current": new})
    return {
        "status": ("different" if differences else "match")
        if recorded
        else "unverified",
        "recorded": recorded or None,
        "current": current,
        "differences": differences,
        "reportedRuntime": reported or None,
        "device": {
            "status": "unverified",
            "recorded": reported.get("device"),
            "current": None,
        },
    }


def _render(text: str, edits: list[dict]) -> str:
    for offset in range(0, len(edits), 128):
        text = render(text, edits[offset : offset + 128])
    return text


def _training_options(
    text: str, bundle: dict, edits: list[dict] | None
) -> tuple[str, dict | None]:
    mode = bundle["mode"]
    selected = None
    if edits is not None:
        if mode != "weights-only" and edits != []:
            raise ValueError(
                "Field edits require a weights-only bundle. Replay preserves the recorded configuration."
            )
        text = render(text, edits)
    config = read_recipe(text)
    paths = _mapping(config.get("path"))
    managed = [
        {"path": ["path", key], "value": None}
        for key, value in paths.items()
        if key in _TRAIN_OUTPUTS and value is not None
    ]
    if config.get("root_path") is not None:
        managed.append({"path": ["root_path"], "value": None})
    if mode == "weights-only":
        selected = _mapping(_mapping(bundle.get("initialization")).get("weight"))
        role = _model_role(config)
        if (
            role != selected.get("role")
            or not isinstance(weight := selected.get("sourcePath"), str)
            or not isinstance(selected.get("observed"), dict)
        ):
            raise ValueError(
                "The selected weight role must match the prepared generator or autoencoder."
            )
        source_path(weight)

        primary = "pretrain_network_" + role
        for key, value in paths.items():
            owned = key in (
                "resume_state",
                "ignore_resume_networks",
                "param_key_g",
                "param_key_d",
            )
            owned |= key.startswith("pretrain_network_") and key not in (
                primary,
                "pretrain_network_g_teacher",
            )
            if owned and value is not None:
                managed.append({"path": ["path", key], "value": None})
        managed.append({"path": ["path", primary], "value": weight})
        if _mapping(config.get("train")).get("restart_scheduler"):
            managed.append({"path": ["train", "restart_scheduler"], "value": False})
    return _render(text, managed), selected


def _matching_inputs(bundle: dict) -> dict[tuple[str | int, ...], dict]:
    values = bundle.get("inputs")
    if not isinstance(values, list):
        raise ValueError("The reproduction bundle is missing its input declarations.")
    result = {}
    for entry in values:
        field = _mapping(entry).get("field")
        if (
            not isinstance(field, list)
            or not field
            or any(type(part) not in (int, str) for part in field)
        ):
            raise ValueError("Invalid recorded input field in the reproduction bundle.")
        key = tuple(field)
        if key in result:
            raise ValueError("The reproduction bundle has repeated input declarations.")
        result[key] = entry
    return result


def _launch_evidence(launch: dict, observed: dict) -> str:
    """How closely the export-time observation is tied to the actual launch."""
    if not launch or not observed:
        return "unverified"
    if version := launch.get("fileVersion"):
        return (
            "fileVersionMatch"
            if version == observed.get("fileVersion")
            else "different"
        )
    if type(launch.get("size")) is int and isinstance(launch.get("mtimeNs"), str):
        return (
            "launchMetadataMatch"
            if all(launch[key] == observed.get(key) for key in ("size", "mtimeNs"))
            else "different"
        )
    return "unverified"


def _compare_input(recorded: dict, current: dict | None) -> str:
    if not current:
        return "unavailable"
    if current["container"] == "invalid":
        return "invalid"
    launch = _mapping(recorded.get("launch"))
    observed = _mapping(recorded.get("observed"))
    if "sha256" in observed:
        if observed.get("size") != current["size"]:
            return "differentSize"
        if observed["sha256"] != current.get("sha256"):
            return "sha256Mismatch"
    if launch.get("fileVersion") == current["fileVersion"]:
        return "sha256Match" if "sha256" in observed else "fileVersionMatch"
    prior = observed
    if launch and _launch_evidence(launch, observed) not in (
        "fileVersionMatch",
        "launchMetadataMatch",
    ):
        prior = {}
    size = launch.get("size", prior.get("size"))
    if type(size) is int and size != current["size"]:
        return "differentSize"
    if "sha256" in observed:
        return "sha256Match"
    if expected := prior.get("saveId"):
        return "saveIdMatch" if expected == current.get("saveId") else "saveIdMismatch"
    if version := prior.get("fileVersion"):
        if version == current["fileVersion"]:
            return "fileVersionMatch"
    if (
        launch.get("size") == current["size"]
        and launch.get("mtimeNs") == current["mtimeNs"]
    ):
        return "launchMetadataMatch"
    return "unverified"


def _resume_companions(inspected: list[dict], resume: dict, issues: list[dict]) -> None:
    fields = {tuple(row["field"]): row for row in inspected}
    for key, filename in _mapping(resume.get("files")).items():
        field = ("path", key)
        path = Path(filename).resolve()
        row = fields.get(field)
        if row is None:
            # Pin the destination companion even when the archive had no receipt for it.
            try:
                version = artifact_version(path.stat())
            except OSError as error:
                issues.append(
                    _issue("inputInspectionFailed", _field(field), str(error), "error")
                )
                continue
            inspected.append(
                {
                    "field": list(field),
                    "kind": "weight",
                    "path": str(path),
                    "status": "unverified",
                    "recorded": None,
                    "current": {"fileVersion": version},
                }
            )
            issues.append(
                _issue(
                    "resumeCompanionUnrecorded",
                    _field(field),
                    "The native full-state loader found this companion. Its source identity was unrecorded.",
                )
            )
        elif row.get("path") != str(path):
            issues.append(
                _issue(
                    "resumeCompanionMismatch",
                    _field(field),
                    f"The native loader would use {path}. Relocate the numbered state and its companion weights together.",
                    "error",
                )
            )


def prepare_reproduction(
    library: RecipeLibrary,
    bundle: dict,
    *,
    mappings: list[dict[str, str]] | None = None,
    source_root: str | None = None,
    name: str | None = None,
    seed: int | None = None,
    edits: list[dict] | None = None,
) -> dict[str, Any]:
    """Verify a bundle and run destination preflight on a new candidate YAML."""
    _verify_bundle(bundle)
    mode = bundle["mode"]
    basis, raw = _basis(_mapping(bundle.get("recipe")))
    origin = _mapping(bundle.get("origin"))
    provenance_record = _mapping(bundle.get("provenance"))
    launch, reported = (
        _mapping(provenance_record.get("launch")),
        _mapping(provenance_record.get("runtime")),
    )
    startup = reported or _mapping(provenance_record.get("legacyStartup"))
    recorded_root = origin.get("checkout")
    base = source_path(recorded_root) if isinstance(recorded_root, str) else None
    if source_root is not None:
        supplied = source_path(source_root)
        if supplied.kind == "home" or (base is not None and supplied.key != base.key):
            raise ValueError(
                "Use mappings to move a recorded checkout. source_root supplies a missing absolute original checkout."
            )
        base = supplied
    relocations = Relocations(mappings)
    text, selected = _training_options(raw, bundle, edits)
    seed_evidence = _mapping(bundle.get("seed"))
    actual_seed = _seed_number(seed_evidence.get("value"))
    if seed is not None and type(seed) is not int:
        raise ValueError("Provide an integer seed.")
    seed_recorded = seed_evidence.get("evidence") in (
        "runtime",
        "invocation",
        "runRecord",
        "legacyLog",
    )
    if (
        mode == "replay"
        and seed_recorded
        and actual_seed is not None
        and seed not in (None, actual_seed)
    ):
        raise ValueError(
            "Replay uses the recorded seed. Prepare a weights-only recipe to change a known seed."
        )
    edited_seed = None
    if mode == "weights-only" and edits:
        if any(edit.get("path") == ["manual_seed"] for edit in edits):
            edited_seed = _seed_number(read_recipe(text).get("manual_seed"))
    chosen_seed = (
        seed
        if seed is not None
        else (edited_seed if edited_seed is not None else actual_seed)
    )
    seed_origin = (
        "explicit"
        if seed is not None or edited_seed is not None
        else seed_evidence.get("evidence")
    )
    issues = [
        dict(issue)
        for issue in bundle.get("limitations", [])
        if isinstance(issue, dict)
        and (
            mode == "replay"
            or issue.get("code") not in ("seedConflict", "effectiveRecipeChanged")
        )
    ]
    if seed_evidence.get("evidence") in ("missing", "declared") or actual_seed is None:
        issues.append(
            _issue(
                "seedUnverified",
                "manual_seed",
                "The original runtime seed is unverified.",
            )
        )
    if chosen_seed is None:
        issues.append(
            _issue(
                "seedRequired",
                "manual_seed",
                "Supply an explicit seed for this recipe.",
                "error",
            )
        )
    updates = [{"path": ["manual_seed"], "value": chosen_seed}]
    if name is not None:
        updates.append({"path": ["name"], "value": safe_name(name)})
    config = read_recipe(text)
    selected_field = (
        ("path", "pretrain_network_" + selected["role"]) if selected else None
    )
    originals = _matching_inputs(bundle)
    inspected = []
    unresolved = False
    for field, value in input_paths(config):
        kind, label = _kind(field), _field(field)
        original = originals.get(tuple(field), {})
        evidence = (
            {"observed": selected["observed"]}
            if selected_field == tuple(field)
            else original
            if original.get("value") == value
            else {}
        )
        row = {
            "field": field,
            "kind": kind,
            "declared": value,
            "recorded": evidence or None,
        }
        inspected.append(row)
        try:
            old = source_path(value, base)
            local, mapped = relocations.locate(old)
        except (OSError, ValueError, RuntimeError) as error:
            unresolved = True
            row.update(status="unresolved", path=None)
            issues.append(_issue("relocationRequired", label, str(error), "error"))
            continue
        row.update(sourcePath=str(old), path=str(local), mapped=mapped)
        if str(local) != value:
            updates.append({"path": field, "value": str(local)})
        if kind == "dataset":
            row.update(status="declared")
            continue
        try:
            observed = _mapping(evidence.get("observed"))
            launch_input = _mapping(evidence.get("launch"))
            if launch_input:
                row["launchEvidence"] = _launch_evidence(launch_input, observed)
            digest = observed.get("sha256")
            if "sha256" in observed and (
                not isinstance(digest, str)
                or re.fullmatch("[0-9a-f]{64}", digest) is None
                or type(observed.get("size")) is not int
                or observed["size"] < 0
            ):
                raise ValueError(
                    "Invalid SHA-256 input declaration in the reproduction bundle."
                )
            current = _identity(
                local,
                hash_contents=digest is not None,
                expected_size=observed.get("size"),
            )
        except (OSError, ValueError) as error:
            row.update(status="unavailable")
            issues.append(_issue("inputInspectionFailed", label, str(error), "error"))
            continue
        row.update(status=_compare_input(evidence, current), current=current)
        if row.get("launchEvidence") == "different":
            issues.append(
                _issue(
                    "launchInputChanged",
                    label,
                    "The file observed at export differs from its recorded launch metadata. Its original bytes are unverified.",
                )
            )
        if row["status"] == "unavailable":
            issues.append(
                _issue(
                    "inputUnavailable",
                    label,
                    "The checkpoint file was unavailable during preparation.",
                    "error",
                )
            )
        elif row["status"] in (
            "differentSize",
            "saveIdMismatch",
            "sha256Mismatch",
            "invalid",
        ):
            issues.append(
                _issue(
                    "inputIdentityMismatch",
                    label,
                    "The destination file is invalid or disagrees with the recorded input evidence.",
                    "error",
                )
            )
        elif row["status"] in ("unverified", "launchMetadataMatch"):
            issues.append(
                _issue(
                    "inputIdentityUnverified",
                    label,
                    "This file has no matching portable save ID or local file version.",
                )
            )
    prepared = None if unresolved else _render(text, updates)
    check = preflight(prepared, library) if prepared is not None else {}
    issues.extend(check.get("issues", []))
    kind = "weights-only" if selected else _initialization(config, startup)
    resume = _mapping(_mapping(check.get("summary")).get("resume"))
    if kind == "full-state" and resume:
        _resume_companions(inspected, resume, issues)
        state_row = next(row for row in inspected if row["kind"] == "state")
        if resume.get("version") != _mapping(state_row.get("current")).get(
            "fileVersion"
        ):
            issues.append(
                _issue(
                    "stateVersionChanged",
                    "path.resume_state",
                    "The state changed during preparation. Prepare the completed save again.",
                    "error",
                )
            )
        initial = startup.get("initialStep")
        if type(initial) is int and initial != resume.get("iteration"):
            issues.append(
                _issue(
                    "resumeClockChanged",
                    "path.resume_state",
                    "The destination checkpoint does not match the attempt's recorded starting step.",
                    "error",
                )
            )
    if kind == "unverified":
        issues.append(
            _issue(
                "resumeStateUnrecorded",
                "path.resume_state",
                "The source started past zero and its initial training state is unrecorded.",
                "error",
            )
        )

    compatibility = {
        "environment": _environment_comparison(library, launch, reported),
        "source": _source_comparison(
            library, launch, _mapping(provenance_record.get("sourceFiles"))
        ),
        "config": {
            "status": (
                "modified"
                if selected
                else "unverified"
                if basis != "effective"
                or any(i.get("code") == "effectiveRecipeChanged" for i in issues)
                else "preserved"
            ),
            "basis": basis,
            "recordedDigest": _sha(raw.encode("utf-8")),
            "preparedDigest": _sha(prepared.encode("utf-8"))
            if prepared is not None
            else None,
            "preflight": check.get("summary"),
        },
        "data": {
            "status": "unverified",
            "reason": "The bundle preserves dataset paths. Data bytes and ordering require independent evidence.",
        },
    }
    for area in ("source", "environment"):
        if compatibility[area]["status"] == "different":
            issues.append(
                _issue(
                    f"{area}Changed",
                    f"compatibility.{area}",
                    f"The destination {area} differs from the run's recorded {area}.",
                )
            )
    ok = prepared is not None and not any(issue["level"] == "error" for issue in issues)
    return {
        "ok": ok,
        "inputPrecondition": input_precondition(prepared, bundle["digest"], inspected)
        if ok
        else None,
        "yaml": prepared,
        "initialization": {
            "kind": kind,
            "seed": chosen_seed,
            "seedEvidence": seed_origin,
            "optimizerState": "restore"
            if kind == "full-state"
            else ("unverified" if kind == "unverified" else "fresh"),
            "resume": resume or None,
        },
        "compatibility": compatibility,
        "inputs": inspected,
        "issues": issues,
        "bundleDigest": bundle["digest"],
    }

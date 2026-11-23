"""Select and stage whole training checkpoints without loading tensors."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from . import run_observe as observe
from .paths import artifact_version, local_path
from .workspace import WindowsProject
from .yaml_edit import read_recipe, render

if TYPE_CHECKING:
    from trainner_config.options import ReduxOptions


NUMBERED_STATE = re.compile(r"([1-9][0-9]{0,18})\.state")


def _model_names(config: dict[str, Any], iteration: int) -> dict[str, Path]:
    from trainner_config.objectives import uses_gan_loss

    if config["path"].get("ignore_resume_networks"):
        raise ValueError(
            "Clear path.ignore_resume_networks to restore the full checkpoint."
        )
    train = config["train"]
    ema = train.get("ema_decay", 0) > 0
    models = Path("models")
    raw = models / "resume_models" if ema else models
    label = "ae" if config.get("network_ae") and not config.get("network_g") else "g"
    paths = {f"pretrain_network_{label}": raw / f"net_{label}_{iteration}"}
    if ema:
        paths[f"pretrain_network_{label}_ema"] = models / f"net_{label}_ema_{iteration}"

    if label == "g" and config.get("network_d") and uses_gan_loss(train.get("losses"), train.get("gan_opt")):
        paths["pretrain_network_d"] = models / "resume_models" / f"net_d_{iteration}"
    return paths


def _numbered_weight(experiment: Path, stem: Path) -> Path:
    # Inspect the container header or archive directory without loading tensors.
    from traiNNer.training_recovery import _complete_tensor_file

    # Match the trainer's precedence if both formats exist.
    for suffix in (".pth", ".safetensors"):
        path = experiment / stem.with_suffix(suffix)
        if observe.regular_file(path):
            if not _complete_tensor_file(path):
                raise ValueError(f"Checkpoint file {path.name} is incomplete.")
            return path
    raise ValueError(
        f"Missing {stem}.pth or {stem}.safetensors in the source experiment."
    )


def _checkpoint_paths(
    experiment: Path, state: Path, config: dict[str, Any]
) -> dict[str, Path]:
    from traiNNer.training_recovery import _complete_tensor_file

    paths = {
        field: _numbered_weight(experiment, stem)
        for field, stem in _model_names(config, int(state.stem)).items()
    }
    if not observe.regular_file(state) or not _complete_tensor_file(state):
        raise ValueError(
            f"Training state {state} is missing or incomplete. Select an intact numbered state."
        )
    from traiNNer.training.portable_checkpoint import matches_published_weights

    if not matches_published_weights(state, list(paths.values())):
        state_written = _version(state)[3]
        if any(_version(path)[3] > state_written for path in paths.values()):
            raise ValueError(
                "Checkpoint models are newer than the training state. "
                "Use a completed save from the original experiment directory."
            )
    paths["resume_state"] = state
    return paths


def _with_paths(text: str, paths: dict[str, Path], config: dict[str, Any]) -> str:
    edits = [
        {"path": ["path", field], "value": str(path)} for field, path in paths.items()
    ]
    if config.get("network_g"):
        edits.append({"path": ["path", "strict_load_g"], "value": True})
    if "pretrain_network_d" in paths:
        edits.append({"path": ["path", "strict_load_d"], "value": True})
    return render(text, edits)


def latest_recipe(text: str, experiment: Path, config: dict[str, Any]) -> str:
    states = experiment / "training_states"
    candidates = (
        (int(match[1]), path)
        for path in observe.files(states)
        if (match := NUMBERED_STATE.fullmatch(path.name))
        and int(match[1]) <= config["train"]["total_iter"]
    )
    # Run the recipe checks outside the candidate loop so a compatibility
    # problem is reported directly even when there are no checkpoints.
    _model_names(config, 1)
    newest_error = None
    for _, state in sorted(candidates, reverse=True):
        try:
            paths = _checkpoint_paths(experiment, state, config)
        except ValueError as error:
            if newest_error is None:
                newest_error = error
        else:
            return _with_paths(text, paths, config)
    if newest_error is not None:
        raise newest_error
    raise ValueError(
        "No numbered training state is available. Start a fresh run, or import a complete checkpoint."
    )


def _check_source_recipe(experiment: Path, config: dict[str, Any], state: Path) -> None:
    recipe = observe.cli_recipe(experiment)
    # A later attempt in the same experiment may have changed its YAML after
    # the selected older checkpoint was saved. Its settings cannot identify it.
    if recipe is None or observe.mtime_ns(recipe) > observe.mtime_ns(state):
        return
    text = observe.recipe_text(recipe)
    if not text:
        return
    try:
        source = read_recipe(text)
    except ValueError:
        return

    if "scale" in source and source["scale"] != config.get("scale"):
        raise ValueError(
            f"Checkpoint scale is {source['scale']}. Use the source recipe's scale."
        )
    for field in ("network_g", "network_ae", "network_d"):
        original = source.get(field)
        selected = config.get(field)
        if not isinstance(original, dict) or not isinstance(selected, dict):
            continue
        if (
            str(original.get("type", "")).casefold()
            != str(selected.get("type", "")).casefold()
        ):
            raise ValueError(
                f"Checkpoint uses {field}.type {original.get('type')!r}. Use the source architecture."
            )
    source_train = source.get("train")
    if not isinstance(source_train, dict):
        return
    for field in ("optim_g", "optim_d", "optim_ae", "scheduler"):
        if field == "scheduler" and config["train"].get("restart_scheduler"):
            continue
        original = source_train.get(field)
        selected = config["train"].get(field)
        if isinstance(original, dict) and isinstance(selected, dict):
            if (
                str(original.get("type", "")).casefold()
                != str(selected.get("type", "")).casefold()
            ):
                raise ValueError(
                    f"Checkpoint uses train.{field}.type {original.get('type')!r}. Use the source configuration."
                )


def import_sources(config: dict[str, Any], root: Path) -> dict[str, Path]:
    state = local_path(config["path"]["resume_state"], root).resolve()
    match = NUMBERED_STATE.fullmatch(state.name)
    if state.parent.name != "training_states" or match is None:
        raise ValueError(
            "Select a positive numbered .state file from the source experiment's "
            "training_states directory, with its matching models directory beside it."
        )
    if int(match[1]) > config["train"]["total_iter"]:
        raise ValueError(
            "Set train.total_iter to at least the imported checkpoint iteration."
        )
    source = state.parent.parent
    _check_source_recipe(source, config, state)
    paths = _checkpoint_paths(source, state, config)
    if "pretrain_network_d" not in paths:
        discriminator = f"net_d_{match[1]}"
        if any(
            observe.regular_file(source / folder / (discriminator + suffix))
            for folder in ("models", "models/resume_models")
            for suffix in (".pth", ".safetensors")
        ):
            raise ValueError(
                "The checkpoint includes a discriminator. Use the source GAN training recipe."
            )
    return paths


def rdual_options(config: dict) -> ReduxOptions | None:
    network = config.get("network_g")
    if not isinstance(network, dict) or str(network.get("type", "")).lower() != "rdual":
        return None
    import msgspec
    from trainner_config.options import ReduxOptions

    opt = msgspec.convert(config, type=ReduxOptions, strict=True)
    opt.is_train = True
    # Studio selects the latest state by writing its explicit numbered path.
    opt.auto_resume = False
    return opt


def _rdual_import_inputs(
    opt: ReduxOptions, root: Path, checkpoint: dict[str, Path]
) -> dict[str, Path]:
    from trainner_config.rdual import preflight_rdual

    issues, inputs = preflight_rdual(opt, root, resolve_path=local_path, resume_files=checkpoint)
    if issues:
        raise ValueError("\n".join(f"{field}: {message}" for field, message in issues))
    return {field.removeprefix("path."): path.resolve() for field, path in inputs.items()}


def prepare_import(
    text: str, config: dict[str, Any], root: Path, experiment: Path
) -> tuple[str, list[dict[str, Any]]]:
    paths = import_sources(config, root)
    extras = {}
    if (opt := rdual_options(config)) is not None:
        inputs = _rdual_import_inputs(opt, root, paths)
        if "neuflow_weights" in inputs:
            from trainner_config.rdual import resolve_neuflow_weights

            # The trainer verifies the official cache at startup. Local
            # fallbacks keep their private copy in the imported experiment.
            if not resolve_neuflow_weights(opt, root, local_path).official:
                extras["neuflow_weights"] = inputs["neuflow_weights"]
    plan = _import_plan({**extras, **paths}, paths, None)
    installed = {item["option"]: experiment / item["path"] for item in plan}
    return _with_paths(text, installed, config), plan


def _import_plan(
    paths: dict[str, Path], checkpoint: dict[str, Path], pins: dict | None
) -> list[dict]:
    experiment = checkpoint["resume_state"].parent.parent if checkpoint else None
    plan = []
    for role, source in paths.items():
        relative = (
            source.relative_to(experiment)
            if role in checkpoint
            else Path("reproduction-inputs") / role / source.name
        )
        entry = {"option": role, "source": str(source), "path": str(relative)}
        pin = pins[role] if pins is not None else None
        if pin is not None and "fileVersion" not in pin:
            if entry["source"] != pin["path"]:
                raise ValueError("The official NeuFlow cache path changed. Prepare the recipe again.")
            plan.append({**entry, "downloadNeuflow": True, "sha256": pin["sha256"]})
            continue
        info = source.stat()
        entry["version"] = _stamp(info)
        if pin is not None:
            if (
                not stat.S_ISREG(info.st_mode)
                or str(source) != pin["path"]
                or artifact_version(info) != pin["fileVersion"]
            ):
                raise ValueError(
                    f"Prepared input path.{role} changed. Prepare the recipe again."
                )
            if "sha256" in pin:
                entry["sha256"] = pin["sha256"]
        plan.append(entry)
    if checkpoint:
        from traiNNer.training.portable_checkpoint import matches_published_weights

        state = checkpoint["resume_state"]
        weights = [path for role, path in checkpoint.items() if role != "resume_state"]
        if not matches_published_weights(state, weights):
            writes = {item["option"]: item["version"][3] for item in plan if item["option"] in checkpoint}
            if any(writes[role] > writes["resume_state"] for role in checkpoint):
                raise ValueError(
                    "The source models are newer than the training state. Select a completed checkpoint."
                )
    return plan


def prepared_import_plan(config: dict, root: Path, pins: dict) -> list[dict]:
    """Resolve roles from trainer inputs, including implicit full-state companions."""
    from .preflight import input_paths

    checkpoint = import_sources(config, root) if (config.get("path") or {}).get("resume_state") else {}
    opt = rdual_options(config)
    paths = {}
    for field, value in input_paths(config):
        if field[0] != "path":
            continue
        role = field[1]
        if role == "neuflow_weights":
            continue
        if opt is not None and (
            role in {"pretrain_network_g", "pretrain_network_g_ema"}
            or (checkpoint and role.startswith("pretrain_network_"))
        ):
            continue
        paths[role] = local_path(value, root).resolve()
    if opt is not None:
        paths.update(_rdual_import_inputs(opt, root, checkpoint))
        if "neuflow_weights" in pins and "neuflow_weights" not in paths:
            from trainner_config.rdual import neuflow_weights_path, rdual_neuflow_required

            # A recorded replay may pin the fallback used with an older PTH
            # whose embedded-flow value cannot be determined without loading it.
            required = rdual_neuflow_required(opt, root, resolve_path=local_path, resume_files=checkpoint)
            if required is None:
                paths["neuflow_weights"] = neuflow_weights_path(opt, root, local_path).resolve()
        flow_pin = pins.get("neuflow_weights")
        if flow_pin is not None and "fileVersion" not in flow_pin and "neuflow_weights" in paths:
            from trainner_config.official_weights import neuflow_cache_path

            if opt.path.neuflow_weights is not None:
                raise ValueError("Expected an implicit NeuFlow default. Prepare the recipe again.")
            paths["neuflow_weights"] = neuflow_cache_path()
    for role, source in checkpoint.items():
        if paths.get(role, source) != source:
            raise ValueError(
                f"The native resume file for path.{role} changed. Prepare the recipe again."
            )
        paths[role] = source
    if paths.keys() != pins.keys():
        raise ValueError(
            "The prepared checkpoint roles differ from the launch inputs. Prepare the recipe again."
        )
    # Publish the numbered state after its model files, including when YAML
    # lists resume_state first. Recovery checks their write order on disk.
    extras = {role: path for role, path in paths.items() if role not in checkpoint}
    return _import_plan({**extras, **checkpoint}, checkpoint, pins)


def import_options(
    text: str, plan: list[dict], destination: Path, *, full_state: bool
) -> str:
    paths = {item["option"]: destination / item["path"] for item in plan}
    if full_state:
        return _with_paths(text, paths, read_recipe(text))
    return render(
        text,
        [{"path": ["path", role], "value": str(path)} for role, path in paths.items()],
    )


def _require_continue_options(source: dict[str, Any], target: dict[str, Any]) -> None:
    options = (
        "optim_g",
        "optim_d",
        "optim_ae",
        "optim_include_params_g",
        "optim_exclude_params_g",
        "scheduler",
        "schedulers",
    )
    for field in ("task", "scale", "network_g", "network_ae", "network_d"):
        if source.get(field) != target.get(field):
            raise ValueError(
                f"Continue requires unchanged {field}. Use Fine-tune to change it."
            )
    for field in options:
        if source["train"].get(field) != target["train"].get(field):
            raise ValueError(
                f"Continue restores train.{field} from the source. Use Fine-tune to change it."
            )
    for field, default in (
        ("ema_decay", 0),
        ("ema_power", 10),
        ("ema_switch_iter", 0),
        ("ema_update_after_step", 0),
        ("warmup_iter", -1),
    ):
        if source["train"].get(field, default) != target["train"].get(field, default):
            raise ValueError(
                f"Continue requires unchanged train.{field}. Use Fine-tune to change it."
            )
    if target["train"].get("restart_scheduler"):
        raise ValueError(
            "Continue restores the source scheduler. Clear train.restart_scheduler."
        )


def phase_weight_roles(
    source: dict[str, Any],
    target: dict[str, Any],
    step: int,
    mode: str,
    weights: str | dict[str, str] | None,
) -> dict[str, Path]:
    """Check compatibility and resolve the model roles before any file access."""
    source_names = _model_names(source, step)
    target_names = _model_names(target, step)
    if mode == "continue":
        _require_continue_options(source, target)
        if source_names.keys() != target_names.keys():
            raise ValueError(
                "Continue requires the same active model roles. Use Fine-tune."
            )
        return source_names

    primary = "pretrain_network_g" if target.get("network_g") else "pretrain_network_ae"
    if isinstance(weights, dict):
        dataset = weights["bestValidation"]
        val = source.get("val") or {}
        selected_data = [d.get("name") for k,d in source["datasets"].items() if k.split("_")[0] == "val"]
        if not val.get("save_best") or not val.get("val_enabled") or not val.get("metrics_enabled") or dataset not in selected_data:
            raise ValueError(f"Enable val.save_best and scored validation for {dataset!r} in the preceding phase.")
        if primary not in source_names:
            raise ValueError("The validation weights and fine-tune target must use the same primary network role.")
        return {primary: source_names.get(primary + "_ema", source_names[primary])}
    selected = primary + "_ema" if weights == "ema" else primary
    if selected not in source_names:
        raise ValueError(
            f"Source phase has no numbered {weights} weights for the requested model role."
        )
    roles = {primary: source_names[selected]}
    discriminator = "pretrain_network_d"
    if (
        discriminator in source_names
        and discriminator in target_names
        and source.get("network_d") == target.get("network_d")
    ):
        roles[discriminator] = source_names[discriminator]
    return roles


def phase_checkpoint(
    experiment: Path,
    source: dict[str, Any],
    target: dict[str, Any],
    step: int,
    mode: str,
    weights: str | dict[str, str] | None,
    *,
    attempt: str | None = None,
) -> list[dict[str, Any]]:
    """Pin the source files this transition will consume."""
    roles = phase_weight_roles(source, target, step, mode, weights)
    if isinstance(weights, dict):
        from .checkpoint_selection import retained_validation_weights

        dataset, metric = weights["bestValidation"], source["val"]["save_best"]
        selected = retained_validation_weights(experiment, attempt, dataset, metric) if attempt else None
        if selected is None:
            raise ValueError(f"No retained {metric!r} weights are available for {dataset!r} in the completed attempt.")
        receipt, checkpoint = selected
        option, numbered = next(iter(roles.items()))
        expected_role = numbered.name.rsplit("_", 1)[0]
        if receipt["role"] != expected_role or not 0 <= receipt["step"] <= step:
            raise ValueError("The retained weights disagree with the completed phase's inference role or step.")
        artifact = checkpoint["artifact"]
        path = observe.artifact_path(experiment, artifact["path"])
        info = path.stat()
        if artifact_version(info) != artifact["version"]:
            raise ValueError("The retained validation weights changed while selecting the phase handoff.")
        return [{"option": option, "source": str(path), "path": artifact["path"], "version": _stamp(info),
                 "validation": receipt}]
    if mode == "continue":
        state = experiment / "training_states" / f"{step}.state"
        paths = _checkpoint_paths(experiment, state, source)
    else:
        paths = {
            option: _numbered_weight(experiment, stem) for option, stem in roles.items()
        }
    return [
        {
            "option": option,
            "source": str(path),
            "path": path.relative_to(experiment).as_posix(),
            "version": _version(path),
        }
        for option, path in paths.items()
    ]


def phase_options(
    text: str, phase: dict[str, Any], destination: Path | None = None
) -> str:
    """Render the exact source paths, or their copies in the private run."""
    handoff = phase.get("handoff")
    if handoff is None:
        return text
    paths = {
        file["option"]: (
            Path(file["source"]) if destination is None else destination / file["path"]
        )
        for file in handoff["files"]
    }
    if phase["mode"] == "continue":
        text = render(text, [{"path": ["manual_seed"], "value": handoff["seed"]}])
        return _with_paths(text, paths, read_recipe(text))
    return render(
        text,
        [
            {"path": ["path", option], "value": str(path)}
            for option, path in paths.items()
        ],
    )


def verify_phase_output(experiment: Path, text: str, target: int) -> None:
    state = experiment / "training_states" / f"{target}.state"
    _checkpoint_paths(experiment, state, read_recipe(text))


def _version(path: Path) -> list[int]:
    try:
        file = path.stat()
    except FileNotFoundError as error:
        raise ValueError(
            f"Checkpoint file disappeared: {path}. Select an intact checkpoint."
        ) from error
    return _stamp(file)


def _stamp(file: os.stat_result) -> list[int]:
    return [file.st_dev, file.st_ino, file.st_size, file.st_mtime_ns, file.st_ctime_ns]


def _copy_checkpoint(source: Path, destination: Path, entry: dict) -> None:
    # Open once. Check the descriptor supplying the bytes, even if the source
    # pathname is replaced and then restored while the copy is in progress.
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as reader:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or _stamp(before) != entry["version"]:
            raise ValueError(f"Checkpoint changed before copying: {source}.")
        with destination.open("xb") as writer:
            if "sha256" in entry:
                digest = hashlib.sha256()
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                    digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(
                        f"Copied bytes disagree with the prepared SHA-256 for {source}."
                    )
            else:
                # Keep the Linux kernel copy path for ordinary, unhashed imports.
                # A WSL mount can lack sendfile support. Both descriptors then
                # continue from their current offsets in the buffered fallback.
                try:
                    while os.sendfile(writer.fileno(), fd, None, 8 * 1024 * 1024):
                        pass
                except OSError as error:
                    if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EXDEV, errno.EOPNOTSUPP):
                        raise
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
        if _stamp(os.fstat(fd)) != entry["version"]:
            raise ValueError(f"Checkpoint changed while copying: {source}.")


def stage_import(
    plan: list[dict[str, Any]],
    control: Path,
    experiment: Path,
    *,
    project: WindowsProject | None = None,
) -> bool:
    """Copy a stable source set, then publish it on the destination filesystem."""
    print(f"Importing {len(plan)} checkpoint files.", flush=True)
    if project is not None:
        project.require_directory(experiment.parent)
    experiment.parent.mkdir(parents=project is None, exist_ok=True)
    if os.path.lexists(experiment):
        raise FileExistsError(f"Checkpoint destination already exists: {experiment}.")
    with TemporaryDirectory(prefix=".import-", dir=experiment.parent) as temporary:
        staged = Path(temporary) / "experiment"
        for entry in plan:
            if (control / "stop").exists():
                return False
            source = Path(entry["source"])
            if entry.get("downloadNeuflow"):
                from trainner_config.official_weights import ensure_neuflow, neuflow_cache_path

                if source != neuflow_cache_path():
                    raise ValueError("The official NeuFlow cache path changed. Prepare the recipe again.")
                print(f"Verifying official NeuFlow v2 weights at {source} for replay.", flush=True)
                source = ensure_neuflow().resolve(strict=True)
                entry.update(source=str(source), version=_version(source))
            destination = staged / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_checkpoint(source, destination, entry)
        for entry in plan:
            source = Path(entry["source"])
            if not observe.regular_file(source) or _version(source) != entry["version"]:
                raise ValueError(
                    f"Checkpoint changed during import: {source}. Retry after the source save finishes."
                )
        if (control / "stop").exists():
            return False
        if project is not None:
            project.require_directory(experiment.parent)
        if os.path.lexists(experiment):
            raise FileExistsError(
                f"Checkpoint destination already exists: {experiment}."
            )
        staged.rename(experiment)
    return True

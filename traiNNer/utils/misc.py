import os
import random
import re
import shutil
import sys
import time
from collections.abc import Generator, Sequence
from os import path as osp
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from rich import print
from torch.utils import deterministic

from traiNNer.utils.dist_util import master_only
from traiNNer.utils.options import struct2dict
from traiNNer.utils.redux_options import ReduxOptions


def set_random_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


class _UninitializedFillSetting(Protocol):
    # torch.utils.deterministic provides this as a module property, which
    # static checkers cannot see; the protocol names the one attribute used.
    fill_uninitialized_memory: bool


def disable_uninitialized_memory_fill() -> None:
    """Stop deterministic mode from filling new uninitialised tensors.

    With torch.use_deterministic_algorithms(True), every new uninitialised
    tensor is filled with a known value: one kernel launch per eager tensor and
    one memset per buffer of a compiled graph. The training code writes every
    tensor before reading it, so results stay deterministic without the fill.
    """
    cast(_UninitializedFillSetting, deterministic).fill_uninitialized_memory = False


def get_time_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_and_rename(path: str) -> None:
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + "_archived_" + get_time_str()
        print(f"Path already exists. Rename it to {new_name}", flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt: ReduxOptions, is_resume: bool = False) -> None:
    """Make dirs for experiments."""
    path_opt = struct2dict(opt.path)
    if not is_resume:
        if opt.is_train:
            mkdir_and_rename(path_opt.pop("experiments_root"))
        else:
            mkdir_and_rename(path_opt.pop("results_root"))
    for key, path in path_opt.items():
        if (
            key == "transfer_g"
            or ("strict_load" in key)
            or ("pretrain_network" in key)
            or ("resume_state" in key)
            or ("ignore_reusme_networks" in key)
            or ("param_key" in key)
        ):
            continue
        else:
            os.makedirs(path, exist_ok=True)


def scandir(
    dir_path: str,
    suffix: str | tuple[str] | None = None,
    recursive: bool = False,
    full_path: bool = False,
) -> Generator[str]:
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """
    root = dir_path
    ancestors: set[tuple[int, int]] = set()

    def walk(folder: str) -> Generator[str]:
        if recursive:
            info = os.stat(folder)
            identity = (info.st_dev, info.st_ino)
            if identity in ancestors:
                return
            ancestors.add(identity)
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_file():
                        name = entry.path if full_path else (
                            osp.relpath(entry.path, root) if recursive else entry.name
                        )
                        if suffix is None or name.endswith(suffix):
                            yield name
                    elif recursive and entry.is_dir():
                        yield from walk(entry.path)
        finally:
            if recursive:
                ancestors.remove(identity)

    return walk(dir_path)


def check_resume(opt: ReduxOptions, resume_iter: int) -> None:
    """Select the numbered companion weights for the state being resumed."""
    paths = opt.path
    if not paths.resume_state:
        return
    assert opt.train is not None
    assert paths.models is not None and paths.resume_models is not None

    models, resume_models = paths.models, paths.resume_models
    state_dir = osp.dirname(osp.realpath(paths.resume_state))
    source_experiment = Path(state_dir).parent
    # Resolve the experiment's ancestors, but not a training_states link into
    # another experiment: its weights belong to the target of the link.
    local_states = osp.abspath(
        paths.training_states or osp.join(osp.dirname(models), "training_states")
    )
    if osp.basename(local_states) == "training_states":
        local_dir = osp.join(osp.realpath(osp.dirname(local_states)), "training_states")
    else:
        local_dir = osp.realpath(local_states)
    external = state_dir != local_dir
    if external:
        models = osp.join(source_experiment, "models")
        resume_models = osp.join(models, "resume_models")

    extensions = ("pth", "safetensors")  # Historical precedence for local resumes.
    ema = opt.train.ema_decay > 0
    ignored = paths.ignore_resume_networks or []
    roles: dict[str, tuple[str, ...]] = {}
    if opt.network_g is not None:
        main_g = osp.join(models, f"net_g_{resume_iter}")
        raw_g = osp.join(resume_models, f"net_g_{resume_iter}")
        roles["network_g"] = (raw_g, main_g)
        if ema:
            ema_g = osp.join(models, f"net_g_ema_{resume_iter}")
            roles["network_g_ema"] = (
                (ema_g,) if opt.train.ema_mode == "neosr" else (main_g, ema_g)
            )
            if external:
                # Older split saves used models/net_g for EMA and resume_models/net_g
                # for raw weights. Without that split, models/net_g can only be raw.
                if any(osp.isfile(f"{raw_g}.{ext}") for ext in extensions):
                    roles["network_g"] = (raw_g,)
                elif "network_g" not in ignored:
                    roles["network_g_ema"] = (ema_g,)
    elif opt.network_ae is not None:
        ae_dir = resume_models if ema else models
        roles["network_ae"] = (osp.join(ae_dir, f"net_ae_{resume_iter}"),)
        if ema:
            roles["network_ae_ema"] = (osp.join(models, f"net_ae_ema_{resume_iter}"),)

    if opt.network_d is not None and "network_d" not in ignored:
        from trainner_config.objectives import uses_gan_loss

        if uses_gan_loss(opt.train.losses, opt.train.gan_opt):
            roles["network_d"] = tuple(
                osp.join(folder, f"net_d_{resume_iter}") for folder in (resume_models, models)
            )

    selected: dict[str, str] = {}
    for role, stems in roles.items():
        if role in ignored:
            continue
        if external and osp.basename(state_dir) != "training_states":
            raise ValueError(
                f"Cannot locate {role} companions for external state {paths.resume_state}. "
                "Select the state in its experiment's training_states directory, or use "
                "ignore_resume_networks and explicitly configure the weights for this role."
            )
        candidates = [f"{stem}.{ext}" for ext in extensions for stem in stems]
        matches = [path for path in candidates if osp.isfile(path)]
        if not matches:
            raise FileNotFoundError(
                f"Missing numbered {role} weights for state {paths.resume_state} "
                f"(step {resume_iter}). Looked for: {', '.join(candidates)}"
            )
        weight = matches[0]
        if external:
            targets = []
            for match in matches:
                target = Path(match).resolve(strict=True)
                if not target.is_relative_to(source_experiment):
                    raise ValueError(
                        f"Redirected {role} weights for state {paths.resume_state}: "
                        f"{match} resolves to {target} outside {source_experiment}"
                    )
                targets.append(target)
            first = targets[0]
            if any(not first.samefile(target) for target in targets[1:]):
                raise ValueError(
                    f"Ambiguous {role} weights for state {paths.resume_state}: {', '.join(matches)}"
                )
            # The loader selects its decoder by suffix. Prefer the backing file's
            # format when known, retaining the numbered alias for unlabelled files.
            if first.suffix.lower() in (".pth", ".safetensors"):
                weight = str(first)
        selected[role] = weight

    if not ignored:
        from traiNNer.training.portable_checkpoint import matches_published_weights

        matches_published_weights(Path(paths.resume_state), [Path(weight) for weight in selected.values()])

    for role, path in selected.items():
        setattr(paths, f"pretrain_{role}", path)
        print(f"Set pretrain_{role} to {path}")

    if paths.param_key_g == "params_ema":
        paths.param_key_g = "params"
        print("Set param_key_g to params")

    if paths.param_key_d == "params_ema":
        paths.param_key_d = "params"
        print("Set param_key_d to params")


def sizeof_fmt(size: float, suffix: str = "B") -> str:
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size /= 1024.0
    return f"{size:3.1f} Y{suffix}"


def free_space_gb_str(path: str = __file__) -> str:
    return f"{shutil.disk_usage(path).free / (1 << 30):,.2f} GiB"


# https://github.com/jpvanhal/inflection/blob/88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1/inflection/__init__.py#L400
def underscore(word: str) -> str:
    """
    Make an underscored, lowercase form from the expression in the string.

    Example::

        >>> underscore("DeviceType")
        'device_type'

    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::

        >>> camelize(underscore("IOError"))
        'IoError'

    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()


def loss_type_to_label(loss_type: str, network_label: str = "g") -> str:
    # label = loss_type.replace("HSLuvLoss", "HSLUVLoss")  # hack for HSLuv
    # label = underscore(label)
    label = loss_type.lower().replace("loss", "")
    return f"l_{network_label}_{label}"


def is_json_compatible(value: Any) -> bool:
    if isinstance(value, str | int | float | bool) or value is None:
        return True
    elif isinstance(value, Sequence):
        return all(is_json_compatible(item) for item in value)
    elif isinstance(value, dict):
        return all(
            isinstance(k, str) and is_json_compatible(v) for k, v in value.items()
        )
    return False


def require_triton(message: str) -> None:
    try:
        import triton  # type: ignore # noqa: F401
    except ImportError as e:
        if sys.platform == "win32":
            raise ImportError(
                f"{message} \nYou can install it with:\n\n    pip install triton-windows"
            ) from e
        else:
            raise ImportError("Error importing triton.") from e


def format_duration_min_sec(elapsed: float) -> str:
    minutes, seconds = divmod(elapsed, 60.0)
    minutes_i = int(minutes)
    prefix = f"{minutes_i:d} min " if minutes_i else ""
    return f"{prefix}{seconds:.2f} sec"

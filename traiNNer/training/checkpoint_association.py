"""Associate opt-in validation with a numbered serialization of its live weights."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4
from weakref import ReferenceType, ref
from zipfile import ZipFile

if TYPE_CHECKING:
    from torch import Tensor, nn

    from traiNNer.models.base_model import BaseModel

SAVE_ID_KEY = "trainner_save_id"
ROLES = ("net_g", "net_g_ema", "net_ae", "net_ae_ema")
SAVE_ID = re.compile(r"[0-9a-f]{32}")
_PTH_MEMBER = "trainner_save_id"
_FLAGS_ONLY = {
    ("traiNNer.archs.autoencoder_arch", "AutoEncoder.train"),
    ("traiNNer.archs.rdual_arch", "RDUAL.train"),
    ("traiNNer.archs.rsr_arch", "RSR.train"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS validation_checkpoint (
    evaluation INTEGER PRIMARY KEY REFERENCES evaluation(id) ON DELETE CASCADE,
    observation TEXT NOT NULL UNIQUE,
    role TEXT,
    weight_path TEXT,
    save_id TEXT,
    reason TEXT
)
"""


def enabled(model: BaseModel) -> bool:
    opt = model.opt
    return bool(
        opt.is_train
        and opt.rank in (None, 0)
        and opt.val
        and opt.val.record_samples
        and opt.path.experiments_root
    )


def pth_save_id(path: str | Path, *, write_id: str | None = None) -> str | None:
    """Use a separate Torch ZIP record and keep the pickled payload unchanged."""
    with ZipFile(path, mode="r" if write_id is None else "a") as archive:
        roots = [
            name.removesuffix("/data.pkl")
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith("/data.pkl")
        ]
        if len(roots) != 1:
            raise ValueError("Expected a native Torch weight archive.")
        name = f"{roots[0]}/{_PTH_MEMBER}"
        if write_id is not None:
            archive.writestr(name, write_id)
            return write_id
        records = [member for member in archive.infolist() if member.filename == name]
        if not records:
            return None
        if len(records) != 1 or records[0].file_size != 32:
            raise ValueError("Invalid embedded weight save ID.")
        identity = archive.read(records[0]).decode("ascii")
        if SAVE_ID.fullmatch(identity) is None:
            raise ValueError("Invalid embedded weight save ID.")
        return identity


@dataclass(frozen=True, slots=True)
class _Snapshot:
    module: ReferenceType[nn.Module]
    tensors: tuple[tuple, ...]

    def matches(self, other: _Snapshot | None) -> bool:
        network = self.module()
        return (
            other is not None
            and network is not None
            and network is other.module()
            and self.tensors == other.tensors
        )


@dataclass(frozen=True, slots=True)
class _Observation:
    step: int
    role: str | None
    snapshot: _Snapshot | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class NumberedSave:
    relative: str
    identity: str
    step: int
    role: str
    snapshot: _Snapshot | None


@dataclass(frozen=True, slots=True)
class _Pending:
    evaluation: int
    nonce: str
    role: str
    snapshot: _Snapshot


@dataclass(slots=True)
class _Boundary:
    step: int
    managed_attempt: str | None
    saves: dict[str, NumberedSave] = field(default_factory=dict)
    evaluations: dict[int, _Pending] = field(default_factory=dict)


def _boundary(model: BaseModel, step: int) -> _Boundary:
    attempt = os.environ.get("TRAINNER_STUDIO_ATTEMPT")
    current = getattr(model, "validation_checkpoint_boundary", None)
    if current is None or current.step != step or current.managed_attempt != attempt:
        current = _Boundary(step, attempt)
        model.validation_checkpoint_boundary = current
    return current


def _root(model: BaseModel) -> Path:
    return Path(model.opt.path.experiments_root).resolve()


def network_role(model: BaseModel, network: nn.Module) -> str | None:
    matches = []
    for name in ROLES:
        candidate = getattr(model, name, None)
        if candidate is not None and name.endswith("_ema"):
            candidate = candidate.ema_model
        if candidate is not None and network is model.get_bare_model(candidate):
            matches.append(name)
    return matches[0] if len(matches) == 1 else None


def _storage(tensor: Tensor) -> tuple:
    return (tensor.device, tensor.untyped_storage().data_ptr())


def _view(tensor: Tensor) -> tuple:
    return (
        tensor._version,  # noqa: SLF001 - Torch's mutation counter is metadata only.
        tensor.device,
        tensor.dtype,
        tensor.data_ptr(),
        tensor.shape,
        tensor.stride(),
    )


def _snapshot(
    model: BaseModel,
    network: nn.Module,
    exported_from: Mapping[str, Tensor] | None = None,
) -> tuple[_Snapshot | None, str | None]:
    from torch import Tensor, nn
    from torch._dynamo import OptimizedModule

    for part in network.modules():
        if isinstance(part, OptimizedModule):
            return None, "optimizedInferenceWeightsUnproven"
        train = type(part).train
        if type(part).eval is not nn.Module.eval or (
            train is not nn.Module.train
            and (train.__module__, train.__qualname__) not in _FLAGS_ONLY
        ):
            return None, "inferenceModeTransform"

    live = network.state_dict(keep_vars=True)
    registered = dict(network.named_parameters(remove_duplicate=False))
    if not live or registered.keys() - live.keys() or live.keys() & {"step", "initted"}:
        return None, "inferenceStateNotSerializable"
    registered.update(network.named_buffers(remove_duplicate=False))
    if any(
        not isinstance(t, Tensor) or registered.get(name) is not t
        for name, t in live.items()
    ):
        return None, "inferenceStateNotSerializable"

    try:
        switches = {
            _storage(p)
            for optimizer in model.optimizers
            if callable(getattr(optimizer, "eval", None))
            for group in optimizer.param_groups
            for p in group["params"]
        }
        if switches and any(
            t.numel() and _storage(t) in switches for t in live.values()
        ):
            # A schedule-free train/eval round trip between the validation
            # and numbered save need not reproduce identical bits.
            return None, "optimizerWeightTransform"
        if exported_from is not None and (
            live.keys() != exported_from.keys()
            or any(_view(t) != _view(exported_from[name]) for name, t in live.items())
        ):
            return None, "exportedStateDiffers"
        return _Snapshot(
            ref(network), tuple((name, id(t), *_view(t)) for name, t in live.items())
        ), None
    except (RuntimeError, NotImplementedError):
        # Inference-mode or non-strided tensors may lack version/storage data.
        return None, "inferenceStateUntracked"


@contextmanager
def observe_inference(
    model: BaseModel,
    inference: nn.Module,
    step: int,
) -> Iterator[None]:
    """Observe the exact native network selected for this validation dataset."""
    if not enabled(model) or step <= 0:
        yield
        return
    from torch.nn.parallel import DataParallel, DistributedDataParallel

    network = model.get_bare_model(inference)
    role = network_role(model, network)
    represented = (
        inference.module
        if isinstance(inference, DataParallel | DistributedDataParallel)
        else inference
    )
    if represented is not network:
        # The evaluator actually runs a wrapper whose cached weight values
        # cannot be inferred from the unwrapped module's state_dict.
        before, reason = None, "optimizedInferenceWeightsUnproven"
    elif role is None:
        before, reason = None, "inferenceRoleUnknown"
    else:
        before, reason = _snapshot(model, network)
    yield
    if before is not None:
        after, after_reason = _snapshot(model, network)
        if not before.matches(after) or network_role(model, network) != role:
            before, reason = None, after_reason or "inferenceWeightsChanged"
    model.validation_checkpoint_observation = _Observation(step, role, before, reason)


def prepare_save(
    model: BaseModel,
    network: nn.Module,
    role: str,
    step: int,
    destination: str,
    state: Mapping[str, Tensor],
) -> NumberedSave | None:
    if not enabled(model) or role not in ROLES or step <= 0:
        return None
    path = Path(destination).absolute()
    root = _root(model)
    allowed = (
        (root / "models",)
        if role.endswith("_ema")
        else (root / "models", root / "models/resume_models")
    )
    if path.parent not in allowed or path.parent.resolve() != path.parent:
        return None
    if path.name != f"{role}_{step}{path.suffix}" or path.suffix not in (
        ".pth",
        ".safetensors",
    ):
        return None
    snapshot = (
        _snapshot(model, network, state)[0] if network_role(model, network) == role else None
    )
    return NumberedSave(
        path.relative_to(root).as_posix(), uuid4().hex, step, role, snapshot
    )


def record_evaluation(
    model: BaseModel, db: sqlite3.Connection, evaluation: int, step: int
) -> None:
    """Record proof or explicit uncertainty in the evaluator's transaction."""
    db.execute(_SCHEMA)
    boundary = _boundary(model, step)
    observation = getattr(model, "validation_checkpoint_observation", None)
    model.validation_checkpoint_observation = None
    nonce = uuid4().hex
    role = None
    save = None
    reason = "inferenceWeightsUnobserved"
    if observation is not None and observation.step == step:
        role, snapshot = observation.role, observation.snapshot
        reason = observation.reason
        if role is not None and snapshot is not None:
            candidate = boundary.saves.get(role)
            if candidate is not None and snapshot.matches(candidate.snapshot):
                save = candidate
            else:
                reason = "numberedSaveUnmatched"
            boundary.evaluations[evaluation] = _Pending(
                evaluation, nonce, role, snapshot
            )
    elif step <= 0:
        reason = "stepNotNumbered"
    db.execute(
        "INSERT INTO validation_checkpoint VALUES(?,?,?,?,?,?)",
        (
            evaluation,
            nonce,
            role,
            save.relative if save else None,
            save.identity if save else None,
            reason,
        ),
    )


def publish_save(model: BaseModel, save: NumberedSave) -> None:
    """Expose this identity only after its file has been atomically installed."""
    from .validation_summary import FILE

    boundary = _boundary(model, save.step)
    boundary.saves[save.role] = save
    matches = [
        (save.relative, save.identity, pending.evaluation, pending.nonce)
        for pending in boundary.evaluations.values()
        if pending.role == save.role and pending.snapshot.matches(save.snapshot)
    ]
    if not matches:
        return
    database = _root(model) / FILE
    try:
        with (
            closing(
                sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=10)
            ) as db,
            db,
        ):
            db.executemany(
                """UPDATE validation_checkpoint SET weight_path=?, save_id=?, reason=NULL
                   WHERE evaluation=? AND observation=?""",
                matches,
            )
    except sqlite3.Error:
        # The numbered weights have already been published. Preserve them and
        # leave the evaluation's previous proof (or explicit uncertainty).
        from traiNNer.utils import get_root_logger

        get_root_logger().warning(
            "Could not record the validation association for %s.",
            save.relative,
            exc_info=True,
        )

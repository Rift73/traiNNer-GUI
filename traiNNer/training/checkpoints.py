"""Numbered checkpoint publication, recovery anchoring, and retention.

Saving individual files atomically does not make the whole checkpoint set
atomic. Recovery and retention therefore inspect the set only after save()
returns, using the pre-save signatures to reject stale or partial output.
"""

import os
import re
from logging import Logger
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from traiNNer.training_recovery import (
    CheckpointLayout,
    CheckpointRetention,
    RecoveryCheckpoint,
    TrainingRecovery,
)
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import free_space_gb_str

if TYPE_CHECKING:
    from traiNNer.models.base_model import BaseModel


class _AutoencoderCheckpointLayout(CheckpointLayout):
    def files(self, checkpoint: RecoveryCheckpoint) -> list[Path]:
        state, *weights = super().files(checkpoint)
        return [state, *(path.with_name(path.name.replace("net_g_", "net_ae_", 1)) for path in weights)]


class CheckpointRequests:
    """The rank-zero checkpoint channel for a managed attempt."""

    def __init__(
        self, control: Path, layout: CheckpointLayout, logger: Logger, handled_id: str | None
    ) -> None:
        self.control = control
        self.layout = layout
        self.logger = logger
        self._handled_id = handled_id
        self._pending_id: str | None = None
        self._result: dict[str, str | int] | None = None
        self._result_write_warned = False
        self._next_poll = 0.0

    @classmethod
    def from_environment(
        cls, model: "BaseModel", rank: int, logger: Logger
    ) -> "CheckpointRequests | None":
        if rank != 0 or not (directory := os.environ.get("TRAINNER_STUDIO_CONTROL")):
            return None

        from studio.run_process import read_record

        control = Path(directory)
        paths = model.opt.path
        options = model.opt.logger
        assert paths.training_states and paths.models and paths.resume_models and options is not None
        autoencoder = model.net_ae is not None
        layout_type = _AutoencoderCheckpointLayout if autoencoder else CheckpointLayout
        layout = layout_type(
            states=Path(paths.training_states),
            models=Path(paths.models),
            resume_models=Path(paths.resume_models),
            interval=options.save_checkpoint_freq,
            extension=options.save_checkpoint_format,
            ema=(model.net_ae_ema if autoencoder else model.net_g_ema) is not None,
            discriminator=model.net_d is not None,
        )
        try:
            previous = read_record(control / "checkpoint-result.json")
            (control / "checkpoint-ready").touch()
        except (OSError, ValueError):
            logger.warning("Could not initialize managed checkpoint control.", exc_info=True)
            return None
        handled_id = previous.get("id") if isinstance(previous, dict) else None
        return cls(control, layout, logger, handled_id if isinstance(handled_id, str) else None)

    def poll(self, *, force: bool = False) -> bool:
        if self._pending_id is not None:
            return True
        now = monotonic()
        if not force and now < self._next_poll:
            return False
        self._next_poll = now + 0.5

        # A failed receipt write must not cause the checkpoint to be saved again.
        if self._result is not None:
            self._publish_result()
            if self._result is not None:
                return False

        from studio.run_process import read_record

        try:
            request = read_record(self.control / "checkpoint-request.json")
        except (OSError, ValueError):
            self.logger.warning("Could not read the managed checkpoint request.", exc_info=True)
            return False
        request_id = request.get("id") if isinstance(request, dict) else None
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            return False
        if request_id == self._handled_id:
            return False
        self._pending_id = request_id
        return True

    def finish(self, *, iteration: int | None = None, error: str = "") -> None:
        assert self._pending_id is not None
        result: dict[str, str | int] = {"id": self._pending_id}
        if iteration is not None:
            result.update(status="saved", iteration=iteration)
        else:
            result.update(status="failed", error=" ".join(error.split())[:240])
        self._handled_id = self._pending_id
        self._pending_id = None
        self._result = result
        self._publish_result()

    def _publish_result(self) -> None:
        from studio.run_process import write_record

        assert self._result is not None
        try:
            write_record(self.control / "checkpoint-result.json", self._result)
        except OSError:
            if not self._result_write_warned:
                self.logger.warning("Could not publish the managed checkpoint result.", exc_info=True)
                self._result_write_warned = True
        else:
            self._result = None
            self._result_write_warned = False


def save_periodic_checkpoint(
    model: "BaseModel",
    epoch: int,
    current_iter: int,
    *,
    logger: Logger,
    layout: CheckpointLayout | None,
    recovery: TrainingRecovery | None,
    retention: CheckpointRetention | None,
    required: bool = False,
    requests: CheckpointRequests | None = None,
) -> bool:
    """Save one numbered set for the schedule, a request, or both.

    Filesystem failures on optional saves leave training running. Other
    exceptions still reach the training loop's failure policy.
    """
    assert model.opt.path.models is not None
    request_layout = requests.layout if requests is not None else None
    try:
        logger.info(
            "Saving models and training states to %s. Free space: %s",
            clickable_file_path(model.opt.path.models, "experiments folder"),
            free_space_gb_str(model.opt.path.models),
        )
        checkpoint_before = layout.before_save(current_iter) if layout is not None else {}
        request_before = (
            request_layout.before_save(current_iter)
            if request_layout is not None and request_layout != layout
            else checkpoint_before
        )
        model.save(epoch, current_iter)
    except Exception as error:
        if requests is not None and requests.poll(force=True):
            requests.finish(error=f"Could not write checkpoint {current_iter}. {error}")
        # Required final saves and training faults keep their usual failure policy.
        if required or not isinstance(error, OSError):
            raise
        logger.exception(
            "Checkpoint %d could not be written. Training continues from the current update. "
            "Automatic recovery retains its previous checkpoint.",
            current_iter,
        )
        return False

    complete = True
    if requests is not None and requests.poll(force=True):
        assert request_layout is not None
        # A request that arrived during the write can use this same numbered set.
        error = f"Checkpoint {current_iter} is incomplete or was not fully refreshed."
        try:
            complete = request_layout.fresh_save_complete(current_iter, request_before)
        except OSError as verification_error:
            complete = False
            error = f"Could not verify checkpoint {current_iter}. {verification_error}"
        if complete:
            requests.finish(iteration=current_iter)
        else:
            requests.finish(error=error)
            logger.warning("%s", error)

    if complete and layout is not None and current_iter % layout.interval:
        # An off-cadence final set is resumable, but it is not a periodic
        # recovery anchor and must not trigger retention of regular saves.
        complete = layout.fresh_save_complete(current_iter, checkpoint_before)
    elif complete and recovery is not None and not recovery.note_periodic_save(
        current_iter, checkpoint_before
    ):
        complete = False
        logger.warning(
            "Checkpoint %d is incomplete or not fully refreshed; "
            "automatic recovery will retain its previous checkpoint.",
            current_iter,
        )
    if retention is not None and complete:
        pruning = retention.prune_after_save(current_iter, checkpoint_before)
        if pruning.removed_files:
            logger.info(
                "Checkpoint retention: removed %d old regular sets, "
                "%d files (%.2f GiB). Keeping %d regular sets; "
                "manual checkpoints are untouched.",
                len(pruning.removed_iterations),
                len(pruning.removed_files),
                pruning.removed_bytes / 1024**3,
                retention.keep,
            )
        if pruning.error:
            logger.warning("Checkpoint retention stopped: %s", pruning.error)
        complete = not pruning.new_checkpoint_invalid
    if required and not complete:
        raise OSError(f"Final checkpoint {current_iter} is incomplete or was not fully refreshed.")
    return complete

"""Periodic checkpoint publication, recovery anchoring, and retention.

Saving individual files atomically does not make the whole checkpoint set
atomic. Recovery and retention therefore inspect the set only after save()
returns, using the pre-save signatures to reject stale or partial output.
"""

from logging import Logger
from typing import TYPE_CHECKING

from traiNNer.training_recovery import (
    CheckpointLayout,
    CheckpointRetention,
    TrainingRecovery,
)
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import free_space_gb_str

if TYPE_CHECKING:
    from traiNNer.models.base_model import BaseModel


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
) -> bool:
    """Save one scheduled set and report whether it can be reused at this boundary.

    Only filesystem failures are recoverable here. CUDA and serialization
    failures must reach the training loop's failure policy; hiding them could
    continue training from a corrupted process or publish a broken checkpoint.
    Manual interruption and final-latest saves deliberately use other paths.
    """
    assert model.opt.path.models is not None
    logger.info(
        "Saving models and training states to %s. Free space: %s",
        clickable_file_path(model.opt.path.models, "experiments folder"),
        free_space_gb_str(model.opt.path.models),
    )
    checkpoint_before = layout.before_save(current_iter) if layout is not None else {}
    try:
        model.save(epoch, current_iter)
    except OSError:
        # A completed managed run is a resumable input to a later stage.
        # A failed final state must reach the run manager as a failed run.
        if required:
            raise
        # Atomic per-file writes protect prior files. Do not publish this
        # partial set or prune the last good one.
        logger.exception(
            "Periodic checkpoint %d could not be written. Training "
            "continues; automatic recovery retains its previous "
            "checkpoint. Check disk space/permissions. The next "
            "scheduled save will try again.",
            current_iter,
        )
        return False

    complete = True
    if layout is not None and current_iter % layout.interval:
        # An off-cadence final set is resumable, but it is not a periodic
        # recovery anchor and must not trigger retention of regular saves.
        complete = layout.fresh_save_complete(current_iter, checkpoint_before)
    elif recovery is not None and not recovery.note_periodic_save(
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

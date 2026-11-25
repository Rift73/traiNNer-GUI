"""Normal training command and explicit update/lifecycle orchestration.

Keep the CPU-only recovery supervisor above all Torch imports. The main loop
owns iteration acceptance, scheduling, and shutdown; focused modules own loader
construction, checkpoint publication, and model inference/compile details.
"""

import os

# The waiting parent must not import Torch or initialize a CUDA runtime. This
# only runs for the CLI; imported train_pipeline callers remain in-process.
if __name__ == "__main__":
    from traiNNer.training_recovery import maybe_supervise_training

    maybe_supervise_training(__file__)

from torch.distributed import destroy_process_group

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from traiNNer.check.check_dependencies import check_dependencies

if __name__ == "__main__":
    check_dependencies()


import argparse
import datetime
import gc
import logging
import math
import signal
import sys
import time
import traceback
from contextlib import nullcontext
from os import path as osp
from types import FrameType
from typing import TYPE_CHECKING, Any

import torch
from rich.traceback import install
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from studio.vram_test import VramMeasurement
    from torch.utils.tensorboard.writer import SummaryWriter
    from traiNNer.models.realesrgan_model import DegradeAheadPrefetcher
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_sampler import (
    make_sampler_cursor,
    restore_sampler_cursor,
    sampler_cursor_epoch,
    sampler_geometry,
)
from traiNNer.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from traiNNer.models import build_model, degrades_ahead
from traiNNer.training.checkpoints import CheckpointRequests, save_periodic_checkpoint
from traiNNer.training.data import create_train_val_dataloader
from traiNNer.training_recovery import (
    CUDA_FAILURE_EXIT_CODE,
    NONFINITE_EXIT_CODE,
    CheckpointRetention,
    PersistentNonfiniteError,
    TrainingRecovery,
    fatal_cuda_reason,
    report_cuda_failure,
)
from traiNNer.utils import (
    AvgTimer,
    MessageLogger,
    check_resume,
    get_env_info,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
)
from traiNNer.utils.config import Config
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import disable_uninitialized_memory_fill, set_random_seed
from traiNNer.utils.options import copy_opt_file, diff_user_vs_template
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import TrainingState


def init_tb_loggers(opt: ReduxOptions) -> "SummaryWriter | None":
    # initialize wandb logger before tensorboard logger to allow proper sync
    assert opt.logger is not None
    assert opt.root_path is not None

    if (opt.logger.wandb is not None) and (opt.logger.wandb.project is not None):
        assert opt.logger.use_tb_logger, "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt.logger.use_tb_logger:
        tb_logger = init_tb_logger(
            log_dir=osp.join(opt.root_path, "tb_logger", opt.name)
        )
    return tb_logger


def cuda_memory_diagnostics() -> str:
    """Return allocator and driver memory state without allocating CUDA tensors."""
    if not torch.cuda.is_available():
        return "CUDA memory diagnostics unavailable because CUDA is not initialized."

    gib = 1024**3
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / gib
        reserved = torch.cuda.memory_reserved(device) / gib
        peak_allocated = torch.cuda.max_memory_allocated(device) / gib
        peak_reserved = torch.cuda.max_memory_reserved(device) / gib
        free, total = torch.cuda.mem_get_info(device)
        return (
            f"CUDA memory at failure on cuda:{device} "
            f"({torch.cuda.get_device_name(device)}): "
            f"{allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved, "
            f"{peak_allocated:.2f} GiB peak allocated, "
            f"{peak_reserved:.2f} GiB peak reserved; "
            f"driver reports {free / gib:.2f} GiB free of {total / gib:.2f} GiB."
        )
    except Exception as error:
        return f"CUDA memory diagnostics unavailable: {error}"


def load_resume_state(
    opt: ReduxOptions, recovery: TrainingRecovery | None = None
) -> Any | None:
    resume_state_path = None
    if recovery is not None:
        resume_state_path = recovery.choose_resume(opt)
        opt.path.resume_state = resume_state_path
    elif opt.auto_resume:
        state_path = osp.join("experiments", opt.name, "training_states")
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [
                    [int(x) for x in v.split(".state")[0].split("_")] for v in states
                ]

                resume_state_path = osp.join(
                    state_path, f"{'_'.join([str(x) for x in max(states)])}.state"
                )
                opt.path.resume_state = resume_state_path
    elif opt.path.resume_state:
        resume_state_path = opt.path.resume_state

    if resume_state_path is None:
        resume_state: TrainingState | None = None
    else:
        resume_state = torch.load(
            resume_state_path,
            map_location="cpu",
            weights_only=True,
        )
        assert resume_state is not None

        if recovery is not None:
            recovery.note_resume(resume_state_path, resume_state["iter"])

        check_resume(
            opt,
            resume_state["iter"],
        )
    return resume_state


def _configure_training_runtime(opt: ReduxOptions) -> None:
    """Configure CUDA policy and seed before constructing data or networks."""
    assert opt.manual_seed is not None
    assert opt.rank is not None
    if opt.num_gpu != 0:
        torch.cuda.set_per_process_memory_fraction(fraction=1.0)

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if opt.deterministic:
        if opt.num_gpu != 0:
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=opt.deterministic_warn_only)
        disable_uninitialized_memory_fill()
        if opt.num_gpu != 0:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif opt.num_gpu != 0:
        torch.backends.cudnn.benchmark = True
    assert opt.manual_seed is not None
    set_random_seed(opt.manual_seed + opt.rank)


def _initialize_experiment_logging(
    opt: ReduxOptions, args: argparse.Namespace, *, resuming: bool
) -> logging.Logger:
    """Create output paths before the first root-logger access.

    Changing this ordering can silently install the logger without its file
    handler. The copied YAML records this launch; resume state is loaded first.
    """
    assert opt.root_path is not None
    assert opt.logger is not None
    assert opt.path.experiments_root is not None
    assert opt.path.log is not None
    make_exp_dirs(opt, resuming)
    # mkdir for experiments and logger
    if not resuming:
        if opt.logger.use_tb_logger and opt.rank == 0:
            mkdir_and_rename(osp.join(opt.root_path, "tb_logger", opt.name))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt.path.experiments_root)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt.path.log, f"train_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="traiNNer", log_file=log_file, plain_console=opt.checkpoint_rng)
    logger.info(get_env_info(use_cuda=opt.num_gpu != 0))
    logger.debug(opt.contents)
    opt.contents = None
    diff, template_name = diff_user_vs_template(args.opt)
    if diff and template_name:
        logger.info("Diff with default config (%s):\n%s", template_name, diff)

    return logger


def _create_prefetcher(
    opt: ReduxOptions, train_loader: DataLoader, logger: logging.Logger
) -> CPUPrefetcher | CUDAPrefetcher:
    """Choose the configured prefetch implementation without changing its policy."""
    prefetch_mode = opt.datasets["train"].prefetch_mode
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info("Use %s prefetch dataloader", prefetch_mode)
        if not opt.datasets["train"].pin_memory:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'."
        )

    return prefetcher


def _open_iteration_progress(
    rank: int, iteration: int, total_iters: int, logger: logging.Logger
) -> memoryview | None:
    """Attach rank zero to the optional progress word created by Studio."""
    if rank != 0 or not (path := os.environ.get("TRAINNER_STUDIO_PROGRESS")):
        return None

    import mmap
    import stat

    try:
        if (
            sys.platform != "linux"
            or sys.byteorder != "little"
            or sys.maxsize < 2**63 - 1
        ):
            raise ValueError("the Studio progress word requires 64-bit little-endian Linux")
        if not osp.isabs(path) or not 0 <= iteration <= total_iters < 2**63:
            raise ValueError("invalid progress path or iteration range")
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != 8:
                raise ValueError("expected an existing 8-byte regular file")
            mapped = mmap.mmap(fd, 8, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        # A native aligned int64 store is visible to the reader without flushing.
        progress = memoryview(mapped).cast("q")
        progress[0] = iteration
    except (OSError, ValueError) as error:
        logger.warning("Studio iteration progress is unavailable: %s", error)
        return None
    return progress


def train_pipeline(root_path: str) -> None:
    """Run training in-process and report poisoned CUDA failures without GPU cleanup."""
    try:
        _train_pipeline(root_path)
    except Exception as error:
        if fatal_cuda_reason(error) is None:
            raise
        # Never save/copy GPU tensors or run CUDA cleanup after a poisoned-context
        # failure. Publish first: Python/C++ teardown itself can abort afterward.
        try:
            if not report_cuda_failure(error):
                print(
                    "[recovery] Automatic CUDA recovery is disabled.", file=sys.stderr
                )
        except (OSError, ValueError, TypeError) as report_error:
            print(
                f"[recovery] Cannot record CUDA failure: {report_error}",
                file=sys.stderr,
            )
        traceback.print_exception(error)
        print(
            "[recovery] Fatal CUDA failure: no emergency checkpoint saved; "
            "only the last complete recorded checkpoint may be resumed.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(CUDA_FAILURE_EXIT_CODE) from error


def _train_pipeline(root_path: str, *, probe: "VramMeasurement | None" = None) -> None:
    """Initialize, run accepted-update scheduling, then follow the exit policy.

    A microbatch is not an iteration: only accepted accumulation boundaries
    advance schedulers, log windows, regular checkpoints, and validation.
    Keep this ordering visible here instead of hiding it behind callbacks.
    """
    install()

    # Parse CLI arguments before requiring CUDA so --help stays CPU-only.
    opt, args = Config.load_config_from_file(root_path, is_train=True)

    if opt.val is not None and opt.val.save_best is not None:
        from trainner_config.metrics import best_weight_direction

        best_weight_direction(opt)

    if opt.num_gpu != 0 and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure that you have a GPU with CUDA support "
            "and that you have installed the correct CUDA-enabled version of PyTorch. "
            "You can check the installation guide at https://pytorch.org/get-started/locally/"
        )

    if opt.num_gpu == 0:
        from trainner_config.devices import validate_cpu_recipe
        validate_cpu_recipe(opt)

    if opt.checkpoint_rng:
        from trainner_config.devices import validate_checkpoint_rng
        from traiNNer.training import random_state
        from traiNNer.training_recovery import RESUME_ENV

        validate_checkpoint_rng(opt)
        if args.restart_epoch or os.environ.get(RESUME_ENV):
            raise ValueError("Checkpoint RNG replay requires the original sampler position and an explicit resume. "
                             "Set checkpoint_rng to false for an epoch restart or recovery with fresh streams.")

    opt.root_path = root_path

    assert opt.train is not None
    assert opt.logger is not None
    assert opt.manual_seed is not None
    assert opt.rank is not None
    assert opt.path.experiments_root is not None
    assert opt.path.log is not None

    if probe is not None:
        probe.configure(opt)
    recovery = TrainingRecovery.from_environment(opt) if probe is None else None
    retention = CheckpointRetention.from_options(opt) if probe is None else None
    checkpoint_layout = None
    if recovery is not None:
        checkpoint_layout = recovery.layout
    elif retention is not None:
        checkpoint_layout = retention.layout

    _configure_training_runtime(opt)
    if opt.checkpoint_rng:
        random_state.seed_legacy_numpy(opt.manual_seed)

    # load resume states if necessary
    resume_state = load_resume_state(opt, recovery)
    logger = (get_root_logger() if probe is not None else
              _initialize_experiment_logging(opt, args, resuming=resume_state is not None))

    if retention is not None:
        logger.info(
            "Checkpoint retention enabled: keep %d complete regular sets. "
            "Older regular sets will be permanently pruned after a successful save; "
            "manual checkpoints and backups are untouched.",
            retention.keep,
        )

    if opt.deterministic:
        logger.info(
            "Training in deterministic mode with manual_seed=%d (warn_only=%s). "
            "Deterministic mode has reduced training speed.",
            opt.manual_seed,
            opt.deterministic_warn_only,
        )
    else:
        logger.info(
            "Training with manual_seed=%d.",
            opt.manual_seed,
        )

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt) if probe is None else None

    # create train and validation dataloaders
    val_enabled = False
    if opt.val and probe is None:
        val_enabled = opt.val.val_enabled

    train_loader, train_sampler, val_loaders, _, total_iters = (
        create_train_val_dataloader(opt, args, val_enabled, logger)
    )

    if train_loader is None or train_sampler is None:
        raise ValueError(
            "Failed to initialize training dataloader. Make sure train dataset is defined in datasets."
        )

    paired_replay = supervised_replay = None
    if opt.checkpoint_rng:
        from traiNNer.data.paired_image_dataset import PairedImageDataset
        from trainner_data.paired_identity import capture_paired_training_identity, require_paired_training_identity
        from trainner_data.supervised_identity import (
            capture_supervised_identity, require_supervised_identity, uses_supervised_replay_identity,
        )

        if type(train_loader.dataset) is PairedImageDataset:
            paired_replay = capture_paired_training_identity(train_loader.dataset)
        elif uses_supervised_replay_identity(opt):
            supervised_replay = capture_supervised_identity(train_loader.dataset, opt)
        if resume_state is not None:
            previous = (resume_state.get("random_streams") or {}).get("paired_training")
            if paired_replay is not None:
                require_paired_training_identity(previous, paired_replay)
            elif previous is not None:
                raise ValueError("The checkpoint used a paired image dataset. Restore the original data for RNG replay.")
            previous_supervised = (resume_state.get("random_streams") or {}).get("supervised_training")
            if supervised_replay is not None:
                require_supervised_identity(previous_supervised, supervised_replay)
            elif previous_supervised is not None:
                raise ValueError("The checkpoint used a different supervised data selection. Restore the original data and worker settings.")

    if opt.fast_matmul and opt.num_gpu != 0:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.fp32_precision = "tf32"

    # create model
    model = build_model(opt)
    if probe is None:
        model.validate_validation_datasets(val_loaders)

    if torch.is_anomaly_enabled():
        logger.warning(
            "!!!! Anomaly checking is enabled. This slows down training and should be used for testing only !!!!"
        )

    geometry = sampler_geometry(train_loader, train_sampler, model.accum_iters)
    if geometry is None:
        logger.warning("This loader saves epoch-only input positions.")
    from trainner_config.devices import replay_worker_mode
    worker_augmentation = replay_worker_mode(opt)
    pending_streams = None
    rng_device = model.device if opt.num_gpu != 0 else None
    if opt.checkpoint_rng:
        if geometry is None or degrades_ahead(model):
            raise ValueError("Checkpoint RNG replay requires an ordered map dataset without degradation read-ahead.")
        # PyTorch draws a base seed even for iter(loader) with zero workers.
        # Give it a private stream so resuming halfway through an epoch cannot
        # change the generator used by the next dropout or augmentation.
        for loader in (train_loader, *val_loaders):
            loader.generator = torch.Generator().manual_seed(opt.manual_seed)
    start_sample = 0
    sampler_resume = None
    if resume_state is not None:
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
        if getattr(args, "restart_epoch", False):
            if (cursor_epoch := sampler_cursor_epoch(resume_state.get("sampler_cursor"))) is not None:
                start_epoch = cursor_epoch
            logger.warning(
                "--restart-epoch requested. Restarting the checkpoint epoch %d at its first batch.",
                start_epoch,
            )
        elif "sampler_cursor" in resume_state:
            train_dataset = opt.datasets["train"]
            sampler_resume = restore_sampler_cursor(
                resume_state["sampler_cursor"],
                geometry,
                auto_batch_size_per_gpu=(
                    train_dataset.batch_size_per_gpu
                    if train_dataset.dataset_enlarge_ratio == "auto"
                    else None
                ),
            )
            start_epoch = sampler_resume.epoch
            start_sample = sampler_resume.next_sample
        else:
            logger.warning(
                "Legacy checkpoint has no sampler cursor. Restarting epoch %d at its first batch.",
                start_epoch,
            )
        if opt.checkpoint_rng:
            pending_streams = resume_state.get("random_streams")
            if pending_streams is None or sampler_resume is None:
                raise ValueError("This checkpoint has no replayable random streams and sampler cursor. "
                                 "Set checkpoint_rng to false to resume with fresh streams.")
            if pending_streams.get("worker_augmentation") != worker_augmentation:
                raise ValueError("The checkpoint's augmentation stream differs from this loader. "
                                 "Use the same zero-worker or position-seeded paired-worker mode for replay.")
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(
            "Resuming training from epoch: %d, iter: %d.", start_epoch, current_iter
        )
        if sampler_resume is not None:
            assert geometry is not None
            if sampler_resume.saved_batch_size != geometry["batch_size"]:
                logger.warning(
                    "Continuing from checkpoint update %d with rank-local batch size "
                    "changed from %d to %d.",
                    current_iter,
                    sampler_resume.saved_batch_size,
                    geometry["batch_size"],
                )
                if sampler_resume.restarted_epoch:
                    logger.warning(
                        "Automatic dataset enlargement changed the permutation. "
                        "Restarting sampler epoch %d.",
                        start_epoch,
                    )
            logger.info(
                "Restored sampler position at epoch %d, rank-local sample %d (zero-based). "
                "%s",
                start_epoch,
                start_sample,
                "Training random streams will be restored." if opt.checkpoint_rng else
                "Random streams restart from the launch seed; worker augmentation state is fresh.",
            )
        del resume_state
    else:
        start_epoch = 0
        current_iter = 0

    if current_iter > total_iters:
        raise ValueError(f"Checkpoint iteration {current_iter} exceeds the requested total of {total_iters}.")
    if probe is not None:
        # Bound this invocation only. The recipe's schedule and phase boundaries keep their horizon.
        total_iters = current_iter + probe.iterations
    if (attempt := os.environ.get("TRAINNER_STUDIO_ATTEMPT")) and opt.rank == 0:
        from pathlib import Path
        from studio.provenance import write_effective
        write_effective(Path(attempt), opt, model, train_loader, current_iter)

    current_accum_iter = 0
    early_stopping = model.early_stopping
    early_stopped = early_stopping is not None and early_stopping.stopped
    last_validation_iter = current_iter if early_stopped else None
    initial_iter = current_iter
    last_completed_iter = current_iter
    last_saved_iter = None
    last_completed_position = (start_epoch, start_sample)

    # Optimizer/scheduler state is now initialized (or restored). Warm-up must
    # protect the first update too, without consuming a scheduler step.
    model.prepare_warmup_learning_rate(current_iter, opt.train.warmup_iter)

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetcher: CPUPrefetcher | CUDAPrefetcher | DegradeAheadPrefetcher | None = None

    # Start the steady-state loop only after all initialization is complete.
    gc.collect()
    if opt.num_gpu != 0:
        torch.cuda.empty_cache()

    logger.info("Start training from epoch: %d, iter: %d.", start_epoch, current_iter)
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.monotonic()

    interrupt_received = False
    rejected_at_stop = False

    def handle_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupt_received
        if not interrupt_received:
            logger.info("User interrupted. Preparing to save state...")
            interrupt_received = True

    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    epoch = start_epoch
    apply_gradient = False
    accepted_gradient = False
    crashed = False
    nonfinite_failure: PersistentNonfiniteError | None = None
    train_data = None
    assert model.opt.path.models is not None

    managed_run = bool(os.environ.get("TRAINNER_MANAGED_RUN_ROOT"))
    sample_recorder = None
    if opt.rank in (None, 0) and opt.logger.training_sample_steps:
        from traiNNer.training.sample_capture import TrainingSampleRecorder

        sample_recorder = TrainingSampleRecorder(opt)
    validation_drives_lr = model.validation_drives_lr
    validation_before_save = validation_drives_lr or early_stopping is not None
    validation_interval = opt.val.val_freq if val_enabled and opt.val else None
    if val_enabled and (validation_interval is None or validation_interval <= 0):
        raise ValueError("Set a positive val_freq for periodic validation.")

    def record_checkpoint_position() -> None:
        model.sampler_cursor = make_sampler_cursor(geometry, *last_completed_position)
        if opt.checkpoint_rng:
            model.checkpoint_random_streams = random_state.snapshot(opt.manual_seed, rng_device)
            if paired_replay is not None:
                model.checkpoint_random_streams["paired_training"] = paired_replay
            if supervised_replay is not None:
                model.checkpoint_random_streams["supervised_training"] = supervised_replay
            if worker_augmentation is not None:
                model.checkpoint_random_streams["worker_augmentation"] = worker_augmentation

    def validate_at(iteration: int, *, scheduled: bool) -> None:
        with random_state.preserve(opt.manual_seed, rng_device) if opt.checkpoint_rng else nullcontext():
            run_validation(iteration, scheduled=scheduled)

    def run_validation(iteration: int, *, scheduled: bool) -> None:
        assert opt.val is not None
        for val_loader in val_loaders:
            assert isinstance(val_loader.dataset, BaseDataset)
            dataset = val_loader.dataset.opt
            model.validation(val_loader, iteration, tb_logger,
                             opt.val.should_save_img(iteration, dataset), len(val_loaders) > 1)
            if scheduled and validation_drives_lr:
                model.update_validation_lr(dataset.name)
            if scheduled and early_stopping is not None and dataset.name == early_stopping.policy["monitor"]["dataset"]:
                from traiNNer.training.metric_schedule import validation_score

                metric = early_stopping.policy["monitor"]["metric"]
                score = validation_score(model.metric_results.get(metric), device=model.device,
                                         distributed=bool(opt.dist), rank=opt.rank or 0)
                if math.isfinite(score):
                    early_stopping.observe(iteration, score)
                elif opt.rank in (None, 0):
                    logger.warning("Early stopping skipped %s/%s at update %d: absent or nonfinite score.",
                                   dataset.name, metric, iteration)
        model.on_validation_boundary(iteration)

    if pending_streams is not None:
        random_state.restore(pending_streams, opt.manual_seed, rng_device)
        del pending_streams

    checkpoint_requests = CheckpointRequests.from_environment(model, opt.rank, logger)
    iteration_progress = _open_iteration_progress(
        opt.rank, current_iter, total_iters, logger
    )
    published_iter = current_iter
    try:
        epoch = start_epoch
        batch_size = geometry["batch_size"] if geometry is not None else 0
        next_sample = start_sample
        next_batch = start_sample // batch_size if batch_size else 0
        stop = interrupt_received or (early_stopped and probe is None)
        while current_iter < total_iters and not stop:
            # Set the offset before constructing the iterator or resetting prefetch.
            train_sampler.set_epoch(epoch, start_sample=next_sample)
            if prefetcher is None:
                prefetcher = _create_prefetcher(opt, train_loader, logger)
                if degrades_ahead(model):
                    from traiNNer.models.realesrgan_model import DegradeAheadPrefetcher

                    prefetcher = DegradeAheadPrefetcher(prefetcher, model)
            else:
                prefetcher.reset()
            train_data = prefetcher.next()
            if train_data is None:
                raise ValueError("The training loader yielded no full batch. Check its dataset size and batch size.")

            while train_data is not None:
                data_timer.record()
                next_batch += 1
                if geometry is not None:
                    next_sample = min(
                        next_sample + batch_size, geometry["samples_per_rank"]
                    )

                # Reserve an iteration label only at the accumulation boundary.
                # A rejected boundary rolls that label back below.
                current_accum_iter += 1

                if current_accum_iter >= model.accum_iters:
                    current_accum_iter = 0
                    current_iter += 1
                    apply_gradient = True
                else:
                    apply_gradient = False

                # training
                with model.data_feed_context():
                    model.feed_data(train_data)
                if sample_recorder is not None:
                    candidate = current_iter if apply_gradient else current_iter + 1
                    if candidate in sample_recorder.remaining:
                        sample_recorder.record(model, train_data, candidate, epoch, next_batch)
                        if not sample_recorder.remaining:
                            sample_recorder = None
                try:
                    model.optimize_parameters(
                        current_iter, current_accum_iter, apply_gradient
                    )
                except torch.cuda.OutOfMemoryError as error:
                    raise RuntimeError(
                        "CUDA ran out of memory during training. "
                        f"{cuda_memory_diagnostics()} "
                        "The original allocator error is preserved below."
                    ) from error
                if probe is None:
                    model.finalize_compile_cache()
                if model.reset_accumulation_requested:
                    current_accum_iter = 0
                final_managed_checkpoint = managed_run and current_iter == total_iters and apply_gradient
                checkpoint_due = probe is None and apply_gradient and (
                    current_iter % opt.logger.save_checkpoint_freq == 0 or final_managed_checkpoint
                )
                validation_due = apply_gradient and validation_interval and current_iter % validation_interval == 0
                log_due = apply_gradient and current_iter % opt.logger.print_freq == 0
                stopped_at_boundary = interrupt_received
                # Only rank zero polls. The model save paths have no DDP
                # collectives, and deferred update accounting is single-rank.
                checkpoint_requested = (
                    apply_gradient
                    and checkpoint_requests is not None
                    and not checkpoint_due
                    and checkpoint_requests.poll(force=stopped_at_boundary)
                )
                settled = not model.deferred_nonfinite_guard
                if not settled and apply_gradient and (
                    stopped_at_boundary or checkpoint_due or checkpoint_requested
                    or validation_due or log_due or current_iter >= total_iters
                ):
                    # Uninterrupted SR normally charges a late GPU rejection to the next
                    # update. Checkpoints and final results cannot carry that in-memory debt.
                    settled = model.confirm_training_step()
                accepted_gradient = apply_gradient and model.training_step_accepted
                if apply_gradient and not accepted_gradient:
                    current_iter -= 1
                # Commit the iteration only after the optimizer accepts it.
                # Warmup for the first update was prepared during initialization.
                if accepted_gradient:
                    model.update_learning_rate(
                        current_iter, warmup_iter=opt.train.warmup_iter
                    )
                    last_completed_iter = current_iter
                if iteration_progress is not None and apply_gradient:
                    completed_iter = (
                        current_iter
                        if settled
                        else model.confirmed_training_iteration(current_iter)
                    )
                    if completed_iter > published_iter:
                        iteration_progress[0] = completed_iter
                        published_iter = completed_iter
                if probe is not None and accepted_gradient:
                    probe.update(current_iter - initial_iter)
                # Deferred accounting can reject the logical label while still
                # training on this batch. Its one-batch window is finished
                # once the pending device outcomes have settled.
                if accepted_gradient or (
                    apply_gradient and settled and model.deferred_nonfinite_guard
                ):
                    last_completed_position = (epoch, next_sample)
                iter_timer.record()
                if accepted_gradient and current_iter == msg_logger.start_iter + 1:
                    # reset start time in msg_logger for more accurate eta_time
                    msg_logger.reset_start_time(current_iter)
                # log
                if log_due and accepted_gradient:
                    log_vars = {"epoch": epoch, "iter": current_iter}
                    log_vars.update({"lrs": model.get_current_learning_rate()})
                    log_vars.update(
                        {
                            "time": iter_timer.get_avg_time(),
                            "data_time": data_timer.get_avg_time(),
                        }
                    )
                    log_vars.update(model.get_current_log())
                    model.reset_current_log()
                    msg_logger(log_vars)

                due_validation = validation_due and accepted_gradient
                if due_validation and validation_before_save:
                    # Commit both validation consumers before publishing this update.
                    validate_at(current_iter, scheduled=True)
                    last_validation_iter = current_iter
                    if early_stopping is not None and early_stopping.stopped:
                        early_stopped = True
                        checkpoint_due = True
                        stop = True

                # save models and training states
                if accepted_gradient and (checkpoint_due or checkpoint_requested):
                    record_checkpoint_position()
                    saved = save_periodic_checkpoint(
                        model,
                        epoch,
                        current_iter,
                        logger=logger,
                        layout=checkpoint_layout if checkpoint_due else None,
                        recovery=recovery if checkpoint_due else None,
                        retention=retention if checkpoint_due else None,
                        required=final_managed_checkpoint or early_stopped,
                        requests=checkpoint_requests,
                    )
                    if saved:
                        last_saved_iter = current_iter

                if due_validation and not validation_before_save:
                    validate_at(current_iter, scheduled=True)
                    # Validation may change state that a stop checkpoint needs.
                    last_saved_iter = None
                    last_validation_iter = current_iter

                # A partial window may already have changed running model buffers.
                rejected_at_stop = interrupt_received and settled and not opt.checkpoint_rng and (
                    not accepted_gradient and (apply_gradient or model.reset_accumulation_requested)
                )
                stop_requested = interrupt_received and settled and (accepted_gradient or rejected_at_stop)
                if stop or stop_requested or current_iter >= total_iters:
                    stop = True
                    break
                data_timer.start()
                iter_timer.start()
                train_data = prefetcher.next()

            if train_data is None:
                epoch += 1
                next_batch = next_sample = 0
        # Resolve pending update failures under the same recovery policy.
        model.settle_training_updates()
    except PersistentNonfiniteError as e:
        if probe is not None:
            raise
        logger.error("%s", e)
        nonfinite_failure = e
        crashed = True
        interrupt_received = True
    except Exception as e:
        if probe is not None or fatal_cuda_reason(e) is not None:
            raise  # The outer CPU-only handler records this without saving.
        logger.exception(e)
        crashed = True
        interrupt_received = True
    finally:
        if iteration_progress is not None:
            iteration_progress.release()
        if checkpoint_requests is not None and checkpoint_requests.poll(force=True):
            checkpoint_requests.finish(error="Training ended before the checkpoint request could be saved.")
        if prefetcher is not None:
            prefetcher.close()

    if probe is not None:
        if interrupt_received or current_iter - initial_iter != probe.iterations:
            raise RuntimeError("VRAM test ended before all 20 updates completed.")
        return

    if nonfinite_failure is not None:
        logger.error(
            "Persistent NaN/Inf: not saving the failing model as a resumable "
            "checkpoint. Existing regular checkpoints are unchanged."
        )
        try:
            if recovery is not None:
                recovery.report_failure(nonfinite_failure, opt.manual_seed)
            else:
                logger.error(
                    "Automatic recovery is disabled for this launch. "
                    "Resume manually from a known-good checkpoint."
                )
        finally:
            if tb_logger is not None:
                tb_logger.close()
            if model.opt.dist:
                destroy_process_group()
        sys.exit(NONFINITE_EXIT_CODE)

    def record_outcome(reason: str) -> None:
        if attempt and opt.rank == 0:
            from studio.run_process import write_record

            outcome: dict[str, Any] = {"completionReason": reason, "completedIteration": current_iter}
            if early_stopping is not None:
                outcome["earlyStopping"] = early_stopping.state_dict()
            write_record(Path(attempt) / "outcome.json", outcome)

    if interrupt_received:
        # current_iter already denotes the last completed update during an
        # accumulation window. Decrementing it on Ctrl+C mislabels the saved
        # optimizer/model state and repeats iteration labels after each resume.
        current_iter = last_completed_iter
        if crashed:
            logger.error(
                "Training failed outside a completed update; not saving a "
                "potentially partial state. Existing checkpoints are unchanged."
            )
        # An all-ignore segmentation window ran no forward or optimizer step.
        elif rejected_at_stop and model.training_step_rejection_reason != "no_annotated_pixels":
            logger.warning("Interrupted after a rejected update. Keeping the preceding checkpoint.")
        # The source checkpoint stays intact if this launch accepted no updates.
        elif (
            current_iter > initial_iter
            and not early_stopped
            and current_iter != last_saved_iter
        ):
            epoch = last_completed_position[0]
            logger.info(
                "Saving models and training states to %s for epoch: %d, iter: %d.",
                clickable_file_path(model.opt.path.models, "experiments folder"),
                epoch,
                current_iter,
            )
            record_checkpoint_position()
            model.save(epoch, current_iter)
        if model.opt.dist:
            destroy_process_group()
        if not crashed:
            record_outcome("interrupted")
        sys.exit(1 if crashed else 0)

    if early_stopped and current_iter < total_iters:
        assert early_stopping is not None
        logger.info("Early stopping at update %d after %d non-improving observations. Best %s at update %d.",
                    current_iter, early_stopping.bad_observations, early_stopping.best, early_stopping.best_step)
    consumed_time = str(datetime.timedelta(seconds=int(time.monotonic() - start_time)))
    logger.info("End of training. Time consumed: %s", consumed_time)
    logger.info("Save the latest model.")
    model.save(
        epoch=-1,
        current_iter=-1,
    )  # -1 stands for the latest
    if val_enabled and current_iter != last_validation_iter:
        validate_at(current_iter, scheduled=False)
    if tb_logger:
        tb_logger.close()
    if model.opt.dist:
        destroy_process_group()
    record_outcome("targetReached" if current_iter == total_iters else "earlyStopping")


if __name__ == "__main__":
    managed_root = os.environ.get("TRAINNER_MANAGED_RUN_ROOT")
    if managed_root is not None:
        if not osp.isabs(managed_root) or not osp.isdir(managed_root):
            raise ValueError("TRAINNER_MANAGED_RUN_ROOT must name an existing absolute directory.")
        root_path = osp.abspath(managed_root)
    else:
        root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)

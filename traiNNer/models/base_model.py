import errno
import json
import os
import tempfile
import time
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4
from weakref import WeakKeyDictionary

import torch
from ema_pytorch import EMA
from safetensors import SafetensorError
from safetensors.torch import save_file
from spandrel import ModelLoader, StateDict
from spandrel.architectures.ESRGAN import ESRGAN
from trainner_config.metrics import metric_direction
from trainner_config.options import initial_g_transfer
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _global_optimizer_post_hooks,
    _global_optimizer_pre_hooks,
)
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

    from traiNNer.data.data_sampler import SamplerCursor

from trainner_config.schedules import SchedulerConfigError, validate_schedulers
from trainner_config.resume import resume_recipe

from traiNNer.training.checkpoint_association import SAVE_ID_KEY, pth_save_id
from traiNNer.training.checkpoint_transfer import copy_checkpoint
from traiNNer.utils._drvfs import close_drvfs_directories, is_drvfs_directory, pin_drvfs_directories
from traiNNer.training.portable_checkpoint import PublishedWeight, record_weights
from traiNNer.training.early_stopping import EarlyStopping
from traiNNer.training.metric_schedule import MetricSchedule
from traiNNer.training.schedule_free_weights import (
    checked_replay_weights, checkpoint_state, evaluation_view, native_schedule_free,
)
from traiNNer.models.checkpoint_payload import (
    load_matching_network_checkpoint,
    read_network_checkpoint,
    transfer_generator_weights,
    state_dict_for_export,
    strip_common_prefix,
    unwrap_checkpoint_state_dict,
)
from traiNNer.models.model_compile import (
    apply_compile_cache_tag,
    enable_precompile_cache,
    resolve_inductor_options,
)
from traiNNer.schedulers.recipe import build_scheduler, restart_schedulers
from traiNNer.schedulers.resume_lr import rebase_resume_rates
from traiNNer.utils import get_root_logger
from traiNNer.utils.compile_cache import configure_compile_cache_runtime
from traiNNer.utils.dist_util import master_only
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import is_json_compatible, require_triton
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed, OptimizerIdentity, TrainingState


def _checkpoint_io_failure(error: Exception) -> bool:
    """Recognize writer I/O failures, not CUDA or serialization programming errors."""
    # Torch's zip finalizer can mask the original failure with 'unexpected pos'.
    # Classify the original exception rather than treating all RuntimeErrors as I/O.
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        previous = error.__cause__ or error.__context__
        if not isinstance(previous, Exception) or id(previous) in seen:
            break
        error = previous
    if isinstance(error, OSError):
        return True
    message = str(error)
    return (
        isinstance(error, SafetensorError) and "I/O error" in message
    ) or (
        isinstance(error, RuntimeError)
        and "PytorchStreamWriter failed writing file" in message
    )


def _save_torch_checkpoint(payload: object, destination: str) -> None:
    if os.environ.get("TRAINNER_GUI_PROJECT_DIR"):
        # Coalesce the many small records in an optimizer state before crossing DrvFS.
        with open(destination, "wb", buffering=2 * 1024 * 1024) as stream:
            torch.save(payload, stream)
    else:
        torch.save(payload, destination)


def _stage_project_checkpoint(writer: Callable[[str], None], destination_fd: int) -> bool:
    """Serialize on Linux before crossing DrvFS. The Windows temp is still the publication point."""
    try:
        if is_drvfs_directory(tempfile.gettempdir()):
            return False
        local = tempfile.NamedTemporaryFile(prefix="trainner-checkpoint-", suffix=".tmp")
    except OSError:
        return False
    with local:
        try:
            writer(local.name)
        except Exception as error:
            if not _checkpoint_io_failure(error):
                raise
            # A Windows project can still save if the local temp volume is full.
            get_root_logger().warning("Linux checkpoint staging failed; writing directly to the project.", exc_info=True)
            return False
        with open(local.name, "rb", buffering=0) as completed:
            try:
                copy_checkpoint(completed.fileno(), destination_fd)
            except OSError as error:
                if error.errno not in (errno.ENOSPC, errno.EDQUOT):
                    raise
                # A dynamic WSL disk and the project may share the same physical drive.
                # Release the local copy and retry with only the Windows temporary file.
                os.ftruncate(destination_fd, 0)
                get_root_logger().warning("Insufficient space for a staged checkpoint; retrying directly.")
                return False
    return True


def _write_checkpoint_atomically(path: str, writer: Callable[[str], None]) -> int | None:
    """Publish one complete file, or leave the previous checkpoint untouched."""
    project = bool(os.environ.get("TRAINNER_GUI_PROJECT_DIR"))
    directory = os.path.dirname(path) or "."
    pinned: dict[str, int] = {}
    temporary_path = None
    size = None
    ready = False
    try:
        if project:
            pin_drvfs_directories(pinned, (path,))
        for attempt in range(3):
            try:
                if not ready:
                    fd, temporary_path = tempfile.mkstemp(
                        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
                    )
                    with os.fdopen(fd, "wb", buffering=0) as target:
                        staged = bool(pinned) and _stage_project_checkpoint(writer, target.fileno())
                        if staged:
                            os.fsync(target.fileno())
                            size = os.fstat(target.fileno()).st_size
                    if not staged:
                        writer(temporary_path)
                        if project:
                            with open(temporary_path, "rb") as checkpoint:
                                os.fsync(checkpoint.fileno())
                                size = os.fstat(checkpoint.fileno()).st_size
                    ready = True
                os.replace(temporary_path, path)
                temporary_path = None
                return size
            except Exception as error:
                if attempt == 2:
                    if _checkpoint_io_failure(error) and not isinstance(error, OSError):
                        raise OSError(f"Checkpoint write failed for {path}: {error}") from error
                    raise
                if not ready and temporary_path is not None:
                    # A serializer can fail after writing part of its output.
                    try:
                        Path(temporary_path).unlink(missing_ok=True)
                    except OSError:
                        # Keep the serialization error if Windows still owns a handle.
                        raise error
                    temporary_path = None
                get_root_logger().warning(
                    "Checkpoint %s failed for %s; retry %d/2.",
                    "rename" if ready else "write", path, attempt + 1, exc_info=True,
                )
                time.sleep(1)
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                get_root_logger().warning(
                    "Could not remove checkpoint temporary file %s.", temporary_path, exc_info=True
                )
        close_drvfs_directories(pinned)



class _FusedAdamBucket(NamedTuple):
    group: dict[str, Any]
    params: list[Tensor]
    exp_avgs: list[Tensor]
    exp_avg_sqs: list[Tensor]
    state_steps: list[Tensor]


class _FusedAdamLists(NamedTuple):
    """Per-step tensor lists of a fused Adam/AdamW, gathered once from its state."""

    state: object
    param_groups: list[dict[str, Any]]
    group_params: tuple[list[Tensor], ...]
    group_sizes: tuple[int, ...]
    buckets: list[_FusedAdamBucket]


class _DeferredUpdate(NamedTuple):
    """An applied update whose finite check is still on its way to the host."""

    candidate_iter: int
    ready: torch.cuda.Event
    # Pinned (nonfinite loss, nonfinite gradient), valid once ready completed.
    flags: Tensor
    n_samples: int
    # Log terms (value * n_samples) computed at update time: compiled outputs
    # live in CUDA Graph memory that the next iteration overwrites.
    contributions: dict[str, Tensor | float]


class _EMAPairs(NamedTuple):
    """Where the tensors that ema_pytorch lerps live, resolved once.

    Slots are (owner dict, key) so tensors replaced behind the cache, such as
    buffers swapped by Module.to(), are read live on every update.
    """

    ema_model: nn.Module
    model: nn.Module
    targets: list[tuple[dict[str, Any], str]]
    sources: list[tuple[dict[str, Any], str]]
    device: torch.device
    dtype: torch.dtype


_FUSED_ADAM_LISTS: WeakKeyDictionary[Optimizer, _FusedAdamLists] = WeakKeyDictionary()
_EMA_PAIRS: WeakKeyDictionary[EMA, _EMAPairs] = WeakKeyDictionary()


def _fused_adam_fast_path_applies(optimizer: Optimizer) -> bool:
    """Whether a step is exactly torch's fused Adam/AdamW kernel with no extras."""
    if type(optimizer) not in (Adam, AdamW):
        return False
    # A direct kernel launch must preserve Optimizer.step hook behavior.
    if (
        optimizer._optimizer_step_pre_hooks  # noqa: SLF001
        or optimizer._optimizer_step_post_hooks  # noqa: SLF001
        or _global_optimizer_pre_hooks
        or _global_optimizer_post_hooks
        or getattr(optimizer, "grad_scale", None) is not None
        or getattr(optimizer, "found_inf", None) is not None
    ):
        return False
    for group in optimizer.param_groups:
        if (
            not group["fused"]
            or group["amsgrad"]
            or group["maximize"]
            or group["differentiable"]
            or group["capturable"]
            or isinstance(group["lr"], Tensor)
            or any(isinstance(beta, Tensor) for beta in group["betas"])
        ):
            return False
    return True


def _gather_fused_adam_lists(optimizer: Optimizer) -> _FusedAdamLists | None:
    """Collect the state lists of every parameter, or None while any lacks state."""
    buckets: list[_FusedAdamBucket] = []
    for group in optimizer.param_groups:
        params = group["params"]
        states = [optimizer.state.get(param) for param in params]
        if not all(states):
            return None
        # One fused launch per (device, dtype), in parameter order, as
        # torch.optim.adam._fused_adam groups them.
        grouped: dict[tuple[torch.device, torch.dtype], _FusedAdamBucket] = {}
        for param, state in zip(params, states, strict=True):
            bucket = grouped.get((param.device, param.dtype))
            if bucket is None:
                bucket = _FusedAdamBucket(group, [], [], [], [])
                grouped[(param.device, param.dtype)] = bucket
            bucket.params.append(param)
            bucket.exp_avgs.append(state["exp_avg"])
            bucket.exp_avg_sqs.append(state["exp_avg_sq"])
            bucket.state_steps.append(state["step"])
        buckets.extend(grouped.values())
    return _FusedAdamLists(
        optimizer.state,
        optimizer.param_groups,
        tuple(group["params"] for group in optimizer.param_groups),
        tuple(len(group["params"]) for group in optimizer.param_groups),
        buckets,
    )


def _fused_adam_lists_match(lists: _FusedAdamLists, optimizer: Optimizer) -> bool:
    """load_state_dict replaces state and param_groups; add_param_group grows them."""
    param_groups = optimizer.param_groups
    return (
        lists.state is optimizer.state
        and lists.param_groups is param_groups
        and len(lists.group_params) == len(param_groups)
        and all(
            params is group["params"] and size == len(params)
            for params, size, group in zip(
                lists.group_params, lists.group_sizes, param_groups, strict=True
            )
        )
    )


def _ema_update_is_plain(ema: EMA, step: int) -> bool:
    """Whether EMA.update() at this step only lerps every tracked tensor pair."""
    return (
        ema.ema_model is not None
        and bool(ema.initted.item())
        and not ema.is_frozen
        and not ema.move_ema_to_online_device
        and ema.update_model_with_ema_every is None
        and not ema.ignore_names
        and not ema.ignore_startswith_names
        and not ema.param_or_buffer_names_no_ema
        and step % ema.update_every == 0
        and step > ema.update_after_step
    )


def _ema_slots(
    model: nn.Module, named_tensors: Iterator[tuple[str, Tensor]], attribute: str
) -> list[tuple[dict[str, Any], str]]:
    slots: list[tuple[dict[str, Any], str]] = []
    for name, _ in named_tensors:
        module_name, _, key = name.rpartition(".")
        slots.append((getattr(model.get_submodule(module_name), attribute), key))
    return slots


def _resolve_ema_pairs(ema: EMA) -> _EMAPairs | None:
    """Pair tensors in EMA.update_moving_average's order, or None if it would
    move or cast a source (allow_different_devices, coerce_dtype)."""
    ema_model = ema.ema_model
    model = ema.model
    assert ema_model is not None
    # zip() by position, parameters first, then buffers, as the library does.
    pairs = list(
        zip(
            _ema_slots(ema_model, ema.get_params_iter(ema_model), "_parameters"),
            _ema_slots(model, ema.get_params_iter(model), "_parameters"),
            strict=False,
        )
    ) + list(
        zip(
            _ema_slots(ema_model, ema.get_buffers_iter(ema_model), "_buffers"),
            _ema_slots(model, ema.get_buffers_iter(model), "_buffers"),
            strict=False,
        )
    )
    if not pairs:
        return None
    for (target_owner, target_key), (source_owner, source_key) in pairs:
        target = target_owner[target_key]
        source = source_owner[source_key]
        if target.device != source.device or target.dtype != source.dtype:
            return None
    first_owner, first_key = pairs[0][0]
    first = first_owner[first_key]
    return _EMAPairs(
        ema_model,
        model,
        [target for target, _ in pairs],
        [source for _, source in pairs],
        first.device,
        first.dtype,
    )


class BaseModel:
    """Base model."""

    def __init__(self, opt: ReduxOptions) -> None:
        self.opt = opt
        self._initial_transfer_g = initial_g_transfer(opt)
        self.transfer_g_receipt: dict | None = None
        self._resume_recipe = resume_recipe(opt) if opt.is_train else None
        self._project_weights: dict[str, PublishedWeight] = {}
        self._project_step: int | None = None
        self.device = torch.device("cuda" if opt.num_gpu != 0 else "cpu")
        self.is_train = opt.is_train
        self.schedulers: list[LRScheduler] = []
        self._step_schedules: list[tuple[int, LRScheduler]] | None = None
        self._metric_schedules: dict[int, MetricSchedule] = {}
        self.early_stopping: EarlyStopping | None = None
        self._scheduler_specs: list[dict] = []
        self.optimizers: list[Optimizer] = []
        self.optimizers_skipped: list[bool] = []
        self.optimizers_schedule_free: list[bool] = []
        self.batch_augment = None
        self.log_dict = {}
        self.l_g_gan_ema = torch.tensor(0.0, device=self.device)
        self.adaptive_d = False
        self.adaptive_d_ema_decay = 0
        self.adaptive_d_threshold = 1
        self.loss_samples = 0
        self.with_metrics = (
            opt.val is not None
            and opt.val.val_enabled
            and opt.val.metrics_enabled
            and opt.val.metrics is not None
        )
        self.use_pbar = opt.val is not None and opt.val.pbar
        self.metric_results: dict[str, Any] = {}
        self.best_metric_results: dict[str, Any] = {}
        self.best_weight_tracker = None
        if opt.is_train and opt.val and opt.val.save_best is not None:
            from traiNNer.training.best_weights import BestWeights

            self.best_weight_tracker = BestWeights(opt)
        self.first_val_completed = False
        self.model_loader = ModelLoader()
        self.net_g = None
        self.net_g_teacher: nn.Module | None = None
        self.net_g_ema: EMA | None = None
        self.net_d = None
        self.net_ae = None
        self.net_ae_ema: EMA | None = None
        self.use_amp = False
        self.use_channels_last = False
        self.memory_format = torch.preserve_format
        self.amp_dtype = torch.float16
        self.scaler_g: GradScaler | None = None
        self.scaler_d: GradScaler | None = None
        self.scaler_ae: GradScaler | None = None
        self.accum_iters: int = 1
        self.grad_clip: bool = False
        self.grad_clip_max_norm: float = 1.0
        self.nan_count = 0
        self._pending_nonfinite_rejection_count = 0
        # Finite checks resolved without waiting for the device; see
        # SRModel._deferred_nonfinite_guard_applies.
        self.deferred_nonfinite_guard = False
        self._deferred_updates: deque[_DeferredUpdate] = deque()
        self._owed_rejection_reports = 0
        self.training_step_accepted = True
        self.reset_accumulation_requested = False
        self.training_step_rejection_reason: str | None = None
        self.sampler_cursor: SamplerCursor | None = None
        self.checkpoint_random_streams: dict | None = None
        self._compile_precompile_cache_enabled = False
        self._compile_precompile_cache_signature = (0, 0)

    @abstractmethod
    def feed_data(self, data: DataFeed) -> None:
        pass

    @abstractmethod
    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        pass

    @abstractmethod
    def get_current_visuals(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def save(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        """Save networks and training state."""

    def validate_validation_datasets(self, val_loaders: Sequence[DataLoader]) -> None:
        """Check the validation dataset contract for this model."""
        if not self.with_metrics:
            return

        from traiNNer.metrics import requires_reference
        if self.opt.val is not None and self.opt.val.metrics and not any(
            requires_reference(cfg["type"]) for cfg in self.opt.val.metrics.values()
        ):
            return

        from traiNNer.data.paired_image_dataset import PairedImageDataset
        from traiNNer.data.paired_video_dataset import PairedVideoDataset

        if not any(
            isinstance(loader.dataset, PairedImageDataset | PairedVideoDataset)
            for loader in val_loaders
        ):
            raise ValueError(
                "Validation metrics are enabled, at least one validation dataset must have type PairedImageDataset or PairedVideoDataset."
            )

    def validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool = False,
        multi_val_datasets: bool = False,
    ) -> None:
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt.is_train and self.opt.val and self.opt.val.record_samples:
            self.validation_checkpoint_observation = None
        self.validation_sample_record = None
        tracker = self.best_weight_tracker
        if tracker is not None:
            tracker.observed = None
        # Providers can emit their own scores without using val.metrics. An empty
        # split must never inherit the preceding dataset's observation.
        self.metric_results = {}
        if self.opt.dist:
            self.dist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )
        else:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

        if tracker is not None and self.opt.rank in (None, 0):
            tracker.check_observed(self, dataloader.dataset.opt.name, current_iter)
        records_scores = (
            bool(self.metric_results) or self.with_metrics or tracker is not None
            or self.validation_drives_lr or self.early_stopping is not None
        )
        if records_scores and self.opt.rank in (None, 0) and (
            self.opt.is_train or (self.opt.val and self.opt.val.record_samples)
        ):
            from traiNNer.training.validation_summary import capture
            capture(self, dataloader.dataset.opt.name, current_iter)

    def begin_training_attempt(self) -> None:
        self.training_step_accepted = True
        self.reset_accumulation_requested = False
        self.training_step_rejection_reason = None

    @contextmanager
    def data_feed_context(self) -> Iterator[None]:
        """Optional isolated RNG context around data synthesis."""
        yield

    def reject_training_step(self, reason: str) -> None:
        self.training_step_accepted = False
        self.reset_accumulation_requested = True
        self.training_step_rejection_reason = reason

    def finalize_compile_cache(self) -> None:
        """Persist newly materialized compiled functions for fast restarts."""
        if not self._compile_precompile_cache_enabled:
            return

        from torch._dynamo.precompile_context import PrecompileContext

        signature = (
            len(PrecompileContext._dynamo_cache_entries),  # noqa: SLF001
            len(PrecompileContext._backend_artifacts_by_key),  # noqa: SLF001
        )
        if signature == self._compile_precompile_cache_signature or not signature[0]:
            return

        try:
            debug_info = PrecompileContext.save_to_dynamo_cache()
        except Exception as error:
            # Cache publication is an optional latency optimization. A new
            # loss, discriminator, or compiler nightly may be unserializable;
            # that must never terminate an otherwise healthy training run.
            self._compile_precompile_cache_enabled = False
            get_root_logger().warning(
                "Could not persist compiled-function artifacts; training will "
                "continue and this process will stop retrying the cache save. "
                "%s: %s",
                type(error).__name__,
                error,
            )
            return
        self._compile_precompile_cache_signature = signature
        get_root_logger().info(
            "Saved %d compiled-function cache entr%s for fast training resume.",
            len(debug_info.get("dynamo", [])),
            "y" if len(debug_info.get("dynamo", [])) == 1 else "ies",
        )

    def on_validation_boundary(self, current_iter: int) -> None:
        """Optional post-validation hook for research instrumentation."""

    @abstractmethod
    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        pass

    @abstractmethod
    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        pass

    def _initialize_best_metric_results(self, dataset_name: str) -> None:
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if dataset_name in self.best_metric_results:
            return

        # add a dataset record
        assert self.opt.val is not None
        assert self.opt.val.metrics is not None
        record = {}
        for metric, content in self.opt.val.metrics.items():
            better = metric_direction(content)
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = {"better": better, "val": init_val, "iter": -1}
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(
        self, dataset_name: str, metric: str, val: float, current_iter: int
    ) -> None:
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        elif val <= self.best_metric_results[dataset_name][metric]["val"]:
            self.best_metric_results[dataset_name][metric]["val"] = val
            self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def get_current_log(self) -> dict[str, float | torch.Tensor]:
        if self.opt.dist:
            return self.reduce_loss_dict(self.log_dict, sample_count=self.loss_samples)
        return {k: v / self.loss_samples for k, v in self.log_dict.items()}

    def reset_current_log(self) -> None:
        self.log_dict = {}
        self.loss_samples = 0

    def add_log_contributions(
        self, n_samples: int, contributions: dict[str, Tensor | float]
    ) -> None:
        """Add one accepted update's sample-weighted log terms."""
        self.loss_samples += n_samples
        for key, value in contributions.items():
            self.log_dict[key] = self.log_dict.get(key, 0) + value

    def settle_training_updates(self) -> None:
        """Wait for every update whose outcome is still on the device."""

    def confirmed_training_iteration(self, iteration: int) -> int:
        """Read the committed count after the trainer accounts for this attempt."""
        # Deferred SR counts updates provisionally. The pending outcomes and
        # resolved rejections still owed to the trainer cannot contribute yet.
        return iteration - len(self._deferred_updates) - self._owed_rejection_reports

    def confirm_training_step(self) -> bool:
        """At a published iteration, settle and charge one unpaid rejection to this attempt."""
        self.settle_training_updates()
        if self.training_step_accepted and self._owed_rejection_reports:
            self._owed_rejection_reports -= 1
            self.reject_training_step("nonfinite_late_rejection_accounting")
        return self._owed_rejection_reports == 0

    def prepare_network_for_compile(self, net: nn.Module) -> None:
        """Prime architecture caches eagerly after device/layout changes."""
        net = self.get_bare_model(net)
        prepare_for_compile = getattr(net, "prepare_for_compile", None)
        if not callable(prepare_for_compile):
            return

        train_dataset = self.opt.datasets.get("train")
        input_shape = None
        if (
            train_dataset is not None
            and train_dataset.batch_size_per_gpu is not None
            and train_dataset.lq_size is not None
        ):
            input_shape = (
                train_dataset.batch_size_per_gpu,
                3,
                train_dataset.lq_size,
                train_dataset.lq_size,
            )
        prepare_for_compile(
            input_shape=input_shape,
            input_dtype=self.amp_dtype if self.use_amp else torch.float32,
        )

    def model_to_device(
        self,
        net: nn.Module,
        compile: bool = False,
        wrap_distributed: bool = True,
    ) -> nn.Module:
        """Move, configure/compile, then optionally wrap a network for parallelism.

        Args:
            net (nn.Module)
            compile (bool): Compile the module before optional parallel wrapping.
            wrap_distributed (bool): Apply DDP/DataParallel. Frozen auxiliary
                networks should leave this disabled and run as one local copy.
        """
        assert isinstance(self.opt.num_gpu, int)
        net = net.to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

        net_name = net.__class__.__name__

        if compile:
            require_triton("torch.compile requires triton.")
            logger = get_root_logger()

            architecture_compile_options, effective_compile_options = (
                resolve_inductor_options(net, self.opt.compile_mode, logger)
            )

            # Some architectures share forward code across differently sized
            # teacher/student instances while keeping Python float attributes
            # architecture-constant. Without specialization, Dynamo can lift
            # those values into CPU scalar inputs on the second compile. The
            # compiled backward may then disagree with the forward about the
            # saved scalar device. This opt-in must remain enabled through lazy
            # tracing, so set the process-wide Dynamo policy before wrapping.
            specialize_float = getattr(net, "compile_specialize_float", False)
            if not isinstance(specialize_float, bool):
                raise ValueError("compile_specialize_float must be a bool when defined")
            if specialize_float:
                torch._dynamo.config.specialize_float = True  # noqa: SLF001
                logger.info(
                    "Network %s specializes architecture-constant Python floats during compilation.",
                    net_name,
                )

            precompile_cache = getattr(net, "compile_precompile_cache", False)
            if not isinstance(precompile_cache, bool):
                raise ValueError("compile_precompile_cache must be a bool when defined")
            if precompile_cache:
                enable_precompile_cache(
                    self.opt.compile_mode, effective_compile_options
                )
                self._compile_precompile_cache_enabled = True
                logger.info(
                    "Network %s will persist compiled-function artifacts for fast resume.",
                    net_name,
                )

            apply_compile_cache_tag(
                net,
                opt=self.opt,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
                memory_format=self.memory_format,
                architecture_options=architecture_compile_options,
                logger=logger,
            )

            compile_runtime_tag = torch.compiler.config.cache_key_tag
            if not compile_runtime_tag:
                compile_runtime_tag = (
                    f"network={net_name}:mode={self.opt.compile_mode}:"
                    f"torch={torch.__version__}"
                )
            compile_cache_path = configure_compile_cache_runtime(compile_runtime_tag)
            logger.info(
                "Network %s uses ABI-scoped compiled-function cache %s.",
                net_name,
                compile_cache_path,
            )

            logger.info(
                "Network %s will be compiled with mode %s. The first iteration may take several minutes.",
                net_name,
                self.opt.compile_mode,
            )
            compile_fullgraph = getattr(net, "compile_fullgraph", False)
            compile_dynamic = getattr(net, "compile_dynamic", None)
            if not isinstance(compile_fullgraph, bool):
                raise ValueError("compile_fullgraph must be a bool when defined")
            if compile_dynamic not in (None, True, False):
                raise ValueError("compile_dynamic must be bool or None when defined")
            self.prepare_network_for_compile(net)
            compile_kwargs: dict[str, Any] = {
                "fullgraph": compile_fullgraph,
                "dynamic": compile_dynamic,
            }
            if effective_compile_options is None:
                compile_kwargs["mode"] = self.opt.compile_mode
            else:
                # torch.compile rejects specifying both mode and options. The
                # effective options begin with the selected mode and then add
                # the architecture's measured overrides.
                compile_kwargs["options"] = effective_compile_options
            net = torch.compile(  # pyright: ignore[reportAssignmentType]
                net,
                **compile_kwargs,
            )

        if wrap_distributed and self.opt.dist:
            find_unused_parameters = self.opt.find_unused_parameters
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif wrap_distributed and self.opt.num_gpu > 1:
            net = DataParallel(net)

        return net

    def get_optimizer(
        self,
        params: ParamsT,
        opts: dict[str, Any],
    ) -> Optimizer:
        from traiNNer.optimizers import build_optimizer

        optimizer = build_optimizer(params, opts)

        if hasattr(optimizer, "train"):
            optimizer.train()  # pyright: ignore[reportAttributeAccessIssue]

        return optimizer

    def step_optimizer(
        self,
        scaler: GradScaler,
        optimizer: Optimizer,
        found_inf: Tensor | None = None,
    ) -> None:
        """Run scaler.step(optimizer).

        A fused Adam/AdamW step without a scaler, hooks or extra options is
        launched directly on tensor lists gathered once from the optimizer
        state; Optimizer.step would rebuild them in Python on every call. Any
        other case, the first step and the first step after the state changed
        (load_state_dict, add_param_group) go through the library.

        found_inf, a float tensor on the parameters' device, gates a fused
        step on the device: non-zero leaves parameters and optimizer state
        untouched, as GradScaler's found_inf does. It needs a disabled scaler.
        """
        if found_inf is not None and scaler.is_enabled():
            raise ValueError("found_inf needs a disabled GradScaler")
        if scaler.is_enabled() or not _fused_adam_fast_path_applies(optimizer):
            self._library_optimizer_step(scaler, optimizer, found_inf)
            return

        lists = _FUSED_ADAM_LISTS.get(optimizer)
        if lists is not None and _fused_adam_lists_match(lists, optimizer):
            # Gradients are rebuilt per call: zero_grad(set_to_none=True)
            # replaces them every step.
            bucket_grads = [
                [param.grad for param in bucket.params] for bucket in lists.buckets
            ]
            # The library skips parameters without a gradient.
            if all(grad is not None for grads in bucket_grads for grad in grads):
                with torch.no_grad():
                    for bucket, grads in zip(lists.buckets, bucket_grads, strict=True):
                        group = bucket.group
                        beta1, beta2 = group["betas"]
                        # As torch.optim.adam._fused_adam.
                        torch._foreach_add_(bucket.state_steps, 1)
                        fused_step = (
                            torch._fused_adamw_
                            if group["decoupled_weight_decay"]
                            else torch._fused_adam_
                        )
                        fused_step(
                            bucket.params,
                            grads,
                            bucket.exp_avgs,
                            bucket.exp_avg_sqs,
                            [],
                            bucket.state_steps,
                            amsgrad=False,
                            lr=group["lr"],
                            beta1=beta1,
                            beta2=beta2,
                            weight_decay=group["weight_decay"],
                            eps=group["eps"],
                            maximize=False,
                            grad_scale=None,
                            found_inf=found_inf,
                        )
                        if found_inf is not None:
                            # The kernel skips a found_inf step but the counter
                            # was already advanced; _fused_adam takes it back.
                            torch._foreach_sub_(
                                bucket.state_steps,
                                [found_inf] * len(bucket.state_steps),
                            )
                return

        self._library_optimizer_step(scaler, optimizer, found_inf)
        lists = _gather_fused_adam_lists(optimizer)
        if lists is None:
            _FUSED_ADAM_LISTS.pop(optimizer, None)
        else:
            _FUSED_ADAM_LISTS[optimizer] = lists

    @staticmethod
    def _library_optimizer_step(
        scaler: GradScaler, optimizer: Optimizer, found_inf: Tensor | None
    ) -> None:
        if found_inf is None:
            scaler.step(optimizer)
            return
        # Adam.step reads found_inf as a plain attribute (the way GradScaler
        # hands it over); only a fused group accepts it.
        optimizer.__dict__["found_inf"] = found_inf
        try:
            scaler.step(optimizer)
        finally:
            del optimizer.__dict__["found_inf"]

    def update_ema(self, ema: EMA) -> None:
        """Run ema.update().

        The plain update (initted, past update_after_step, nothing ignored or
        copied, sources on the targets' device and dtype) is one foreach lerp
        over tensor pairs resolved once; EMA.update would walk both models on
        every call. The result is bit-identical to EMA.update for either
        use_foreach. Every other case goes through the library.
        """
        if type(ema) is not EMA or not _ema_update_is_plain(ema, ema.step.item()):
            ema.update()
            return

        pairs = _EMA_PAIRS.get(ema)
        if (
            pairs is None
            or pairs.ema_model is not ema.ema_model
            or pairs.model is not ema.model
        ):
            pairs = _resolve_ema_pairs(ema)
            if pairs is None:
                _EMA_PAIRS.pop(ema, None)
                ema.update()
                return
            _EMA_PAIRS[ema] = pairs

        targets = [owner[key] for owner, key in pairs.targets]
        sources = [owner[key] for owner, key in pairs.sources]
        # Module.to() can move or cast a whole model behind the cache.
        if not (
            targets[0].device == sources[0].device == pairs.device
            and targets[0].dtype == sources[0].dtype == pairs.dtype
        ):
            _EMA_PAIRS.pop(ema, None)
            ema.update()
            return

        ema.step += 1
        decay = ema.get_current_decay()
        with torch.no_grad():
            torch._foreach_lerp_(targets, sources, 1.0 - decay)

    def setup_schedulers(self) -> None:
        # https://github.com/Corpsecreate/neosr/blob/a29e509dae5cd39aea94ac82d1347d2a54e1175c/neosr/models/default.py#L276

        """Set up schedulers."""
        assert self.opt.train is not None
        train = self.opt.train
        try:
            validate_schedulers(self.opt)
        except SchedulerConfigError as error:
            raise ValueError(f"{error.path}: {error}") from error
        if train.early_stopping is not None:
            self.early_stopping = EarlyStopping.from_options(self.opt)
        if train.warmup_iter > 0:
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    group.setdefault("initial_lr", group["lr"])
        if train.scheduler is None and not train.schedulers:
            return
        logger = get_root_logger()
        used = set()
        steps = []
        for index, optimizer in enumerate(self.optimizers):
            role = next((key for key in ("g","d","ae") if getattr(self,"optimizer_"+key,None) is optimizer), None)
            spec = train.scheduler if train.schedulers is None else train.schedulers.get(role)
            if spec is None:
                continue
            used.add(role)
            kind = spec["type"]
            watch = spec.get("monitor")
            path = "train.scheduler" if train.schedulers is None else f"train.schedulers.{role}"
            schedule, opts = build_scheduler(optimizer, spec, train, target=role or index, path=path)
            number = len(self.schedulers)
            self.schedulers.append(schedule)
            self._scheduler_specs.append({"target":role or index, "type":kind.lower(),
                                          "monitor":watch, "options":opts})
            if watch is not None:
                self._metric_schedules[number] = MetricSchedule(
                    schedule, index, watch["dataset"], watch["metric"]
                )
            else:
                steps.append((index,schedule))
            logger.info("Scheduler %s(%s) targets optimizer %s.",kind,opts,role or index)
        unused = set(train.schedulers or ()) - used
        if unused:
            raise ValueError("No active optimizer for schedule(s): " + ", ".join(sorted(unused)))
        if train.schedulers is not None or self._metric_schedules:
            self._step_schedules = steps

    @property
    def validation_drives_lr(self) -> bool:
        return bool(self._metric_schedules)

    def update_validation_lr(self, dataset: str) -> None:
        for watch in self._metric_schedules.values():
            if watch.dataset == dataset:
                watch.observe(self.metric_results,device=self.device,distributed=bool(self.opt.dist),
                              rank=self.opt.rank or 0,logger=get_root_logger())


    def get_bare_model(
        self, net: DataParallel | DistributedDataParallel | nn.Module
    ) -> nn.Module:
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net = net.module
        return self.unwrap_compiled_model(net)

    def unwrap_compiled_model(
        self, net: nn.Module | torch._dynamo.OptimizedModule
    ) -> nn.Module:
        if isinstance(net, torch._dynamo.OptimizedModule):  # noqa: SLF001
            return net._orig_mod  # noqa: SLF001
        return net

    @master_only
    def print_network(self, net: nn.Module) -> None:
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(x.numel() for x in net.parameters())

        logger = get_root_logger()
        logger.info("Network: %s, with parameters: %s", net_cls_str, f"{net_params:,d}")
        logger.info(net_str)

    def _set_lr(self, lr_groups_l: list[list[float]]) -> None:
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l, strict=False):
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=False):
                param_group["lr"] = lr

    def _get_init_lr(self) -> list[list[float]]:
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter: int, warmup_iter: int = -1) -> None:
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warm-up iter numbers. -1 for no warm-up.
                Default: -1.
        """
        schedules = enumerate(self.schedulers) if self._step_schedules is None else self._step_schedules
        for i, scheduler in schedules:
            if not self.optimizers_skipped[i]:
                scheduler.step()
        if self._metric_schedules:
            for watch in self._metric_schedules.values():
                if not self.optimizers_skipped[watch.optimizer_index]:
                    watch.updated = True
        # The optimizer has finished current_iter; prepare its NEXT update.
        self.prepare_warmup_learning_rate(current_iter, warmup_iter)

    def prepare_warmup_learning_rate(
        self, completed_iter: int, warmup_iter: int = -1
    ) -> None:
        """Set the next update's warm-up LR without advancing any scheduler.

        Also called once after fresh/resumed initialization, before optimization.
        Outside warm-up the restored/scheduled learning rates are untouched.
        """
        next_iter = completed_iter + 1
        if warmup_iter > 0 and 1 <= next_iter <= warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * next_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self) -> list[float]:
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(
        self,
        net: nn.Module,
        net_label: str,
        save_dir: str,
        current_iter: int,
        param_key: str,
    ) -> None:
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        parameters = None
        with ExitStack() as stack:
            for optimizer in self.optimizers:
                if not callable(getattr(optimizer, "eval", None)):
                    continue
                if parameters is None:
                    parameters = {id(p) for p in net.parameters()}
                if any(id(p) in parameters for group in optimizer.param_groups for p in group["params"]):
                    stack.enter_context(evaluation_view(optimizer))
            self._write_network_checkpoint(net, net_label, save_dir, current_iter, param_key)

    def _write_network_checkpoint(self, net: nn.Module, net_label: str, save_dir: str,
                                  current_iter: int, param_key: str) -> None:
        current_iter_str = "latest" if current_iter == -1 else str(current_iter)
        assert self.opt.logger is not None

        save_filename = (
            f"{net_label}_{current_iter_str}.{self.opt.logger.save_checkpoint_format}"
        )

        save_path = os.path.join(save_dir, save_filename)

        bare_net_ = self.get_bare_model(net)
        state = bare_net_.state_dict()
        recorded_save = None
        if (
            current_iter > 0
            and self.opt.is_train
            and self.opt.val
            and self.opt.val.record_samples
        ):
            from traiNNer.training.checkpoint_association import (
                prepare_save,
                publish_save,
            )

            recorded_save = prepare_save(
                self, bare_net_, net_label, current_iter, save_path, state
            )
        project_save = current_iter > 0 and bool(os.environ.get("TRAINNER_GUI_PROJECT_DIR"))
        identity = recorded_save.identity if recorded_save is not None else None
        if project_save and identity is None:
            identity = uuid4().hex
        new_state_dict = state_dict_for_export(
            state, for_safetensors=self.opt.logger.save_checkpoint_format == "safetensors"
        )

        metadata: dict[str, Any] | None = None
        if hasattr(bare_net_, "hyperparameters"):
            assert isinstance(bare_net_.hyperparameters, dict)
            metadata = {
                k: v for k, v in bare_net_.hyperparameters.items() if is_json_compatible(v)
            }

        def write_network(destination: str) -> None:
            assert self.opt.logger is not None
            if self.opt.logger.save_checkpoint_format == "safetensors":
                header = {"metadata": json.dumps(metadata)} if metadata else {}
                if identity is not None:
                    header[SAVE_ID_KEY] = identity
                save_file(new_state_dict, destination, metadata=header or None)
            else:
                payload = (
                    {"metadata": metadata, param_key: new_state_dict}
                    if metadata
                    else new_state_dict
                )
                _save_torch_checkpoint(payload, destination)
                if identity is not None:
                    pth_save_id(destination, write_id=identity)

        published_size = _write_checkpoint_atomically(save_path, write_network)
        if project_save:
            assert identity is not None and published_size is not None
            if self._project_step != current_iter:
                self._project_weights.clear()
                self._project_step = current_iter
            self._project_weights[save_filename] = {"id": identity, "bytes": published_size}
        if recorded_save is not None:
            publish_save(self, recorded_save)

    def _print_different_keys_loading(
        self,
        crt_net: nn.Module,
        load_net: dict[str, Any],
        file_path: str,
        strict: bool = True,
    ) -> bool:
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        if hasattr(crt_net, "map_state_dict"):
            load_net = crt_net.map_state_dict(load_net)
        crt_net_state_dict = crt_net.state_dict()
        crt_net_keys = set(crt_net_state_dict.keys())
        load_net_keys = set(load_net.keys())
        valid = True

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            current_minus_loaded = crt_net_keys - load_net_keys
            if len(current_minus_loaded) > 0:
                if strict:
                    valid = False
                logger.warning(
                    "Pretrain network is missing %d keys from current network, up to 10 shown:",
                    len(current_minus_loaded),
                )
                for v in sorted(current_minus_loaded)[:10]:
                    logger.warning("    %s", v)
            loaded_minus_current = load_net_keys - crt_net_keys
            if len(loaded_minus_current) > 0:
                if strict:
                    valid = False
                logger.warning(
                    "Current network is missing %d keys from pretrain network, up to 10 shown:",
                    len(loaded_minus_current),
                )
                for v in sorted(loaded_minus_current)[:10]:
                    logger.warning("    %s", v)

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net_state_dict[k].size() != load_net[k].size():
                    logger.warning(
                        "Size different, ignore [bold]%s[/bold]: crt_net: "
                        "%s; load_net: %s",
                        k,
                        crt_net_state_dict[k].shape,
                        load_net[k].shape,
                        extra={"markup": True},
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)

            new_common_keys = crt_net_keys & set(load_net.keys())
            if len(new_common_keys) == 0:
                logger.warning(
                    "Pretrain model %s matched %.2f%% of the keys of the currently training model. Pretrain will have no effect and the model will be trained from scratch.",
                    file_path,
                    0,
                )
            elif len(new_common_keys) == len(crt_net_keys):
                logger.info(
                    "Pretrain model %s matched %.2f%% of the keys of the currently training model. Pretrain is loaded in strict mode.",
                    file_path,
                    100,
                )
            else:
                overlap = len(new_common_keys) / len(crt_net_keys) * 100
                logger.info(
                    "Pretrain model %s matched %.2f%% of the keys of the currently training model.",
                    file_path,
                    overlap,
                )
                if overlap == 0:
                    valid = False  # even with strict false, reject 0% match as likely user error
        return valid

    def load_initial_g(
        self,
        network: nn.Module,
        *,
        class_names: Sequence[str] | None = None,
        category_ids: Sequence[int] | None = None,
        expected_task: str | None = None,
        allow_legacy_class_order: bool = False,
        strict: bool = True,
    ) -> None:
        paths = self.opt.path
        assert paths.pretrain_network_g is not None
        if self._initial_transfer_g is not None:
            logger = get_root_logger()
            if class_names is not None:
                logger.info(
                    "Initial G transfer uses tensor names and shapes. A same-sized head is copied "
                    "unless its exact keys are excluded. Class rows are not remapped."
                )
            self.transfer_g_receipt = transfer_generator_weights(
                network,
                Path(paths.pretrain_network_g),
                paths.param_key_g,
                self._initial_transfer_g.exclude,
                logger,
            )
        elif class_names is None:
            self.load_network(
                network,
                paths.pretrain_network_g,
                paths.strict_load_g,
                paths.param_key_g,
            )
        else:
            load_matching_network_checkpoint(
                network,
                Path(paths.pretrain_network_g),
                paths.param_key_g,
                class_names=class_names,
                category_ids=category_ids,
                expected_task=expected_task,
                allow_legacy_class_order=allow_legacy_class_order,
                strict=strict,
            )

    def load_network(
        self,
        net: nn.Module,
        load_path: str,
        strict: bool = True,
        param_key: str | None = "params",
    ) -> None:
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()

        # TODO refactor, messy hack to support the different ESRGAN versions
        if isinstance(net, ESRGAN):
            load_net_wrapper = self.model_loader.load_from_file(load_path)
            load_net = load_net_wrapper.model.state_dict()
            valid = self._print_different_keys_loading(net, load_net, load_path, strict)

            if not valid:
                raise ValueError(
                    f"Unable to load pretrain network due to mismatched state dict keys (see above for missing keys): {load_path}"
                )

            net.load_state_dict(load_net, strict=strict)
            logger.info(
                "Loading %s model from %s, with spandrel.",
                net.__class__.__name__,
                clickable_file_path(Path(load_path).absolute().parent, load_path),
            )
        else:
            net = self.get_bare_model(net)
            load_net = read_network_checkpoint(
                load_path, safetensors_device=str(self.device)
            )

            requested_ema = param_key == "params_ema"
            if param_key is not None:
                if param_key in load_net:
                    load_net = self.remove_common_prefix(load_net[param_key])
                if param_key not in load_net:
                    load_net, new_param_key = self.canonicalize_state_dict(load_net)
                    logger.info(
                        "Loading: %s does not exist, using %s.",
                        param_key,
                        new_param_key,
                    )
                    param_key = new_param_key
            else:
                load_net, param_key = self.canonicalize_state_dict(load_net)

            logger.info(
                "Loading %s model from %s, with param key: [bold]%s[/bold].",
                net.__class__.__name__,
                clickable_file_path(Path(load_path).absolute().parent, load_path),
                param_key,
                extra={"markup": True},
            )

            if requested_ema and load_net.keys() != net.state_dict().keys():
                # Older autoencoder snapshots serialized the entire EMA wrapper.
                # Only select its averaged model when the complete destination layout matches.
                prefix = "ema_model."
                if any(k.startswith(prefix) for k in load_net):
                    averaged = {k[len(prefix):]: v for k, v in load_net.items() if k.startswith(prefix)}
                    if averaged.keys() == net.state_dict().keys():
                        load_net = averaged

            valid = self._print_different_keys_loading(net, load_net, load_path, strict)

            if not valid:
                raise ValueError(
                    f"Unable to load pretrain network due to mismatched state dict keys (see above for missing keys): {load_path}"
                )

            net.load_state_dict(load_net, strict=strict)

    def canonicalize_state_dict(
        self, state_dict: StateDict
    ) -> tuple[StateDict, str | None]:
        """Unwrap a checkpoint and remove common architecture/parallel prefixes."""
        state_dict, used_unwrap_key = unwrap_checkpoint_state_dict(state_dict)
        # Keep the method dispatch: subclasses may customize prefix handling.
        state_dict = self.remove_common_prefix(state_dict, ["module.", "netG."])
        return state_dict, used_unwrap_key

    def remove_common_prefix(
        self, state_dict: StateDict, prefixes: Sequence[str] = ("module.", "netG.")
    ) -> StateDict:
        """Strip only prefixes shared by every key, in the supplied order."""
        return strip_common_prefix(state_dict, prefixes)

    def load_network_spandrel(
        self, net: nn.Module, load_path: str, strict: bool = True
    ) -> None:
        logger = get_root_logger()
        load_net = self.model_loader.load_from_file(load_path)
        net.load_state_dict(load_net.model.state_dict(), strict=strict)
        logger.info(
            "Loading %s model from %s, with spandrel.",
            net.__class__.__name__,
            load_path,
        )

    def _optimizer_identities(self) -> list[OptimizerIdentity]:
        """Identify live optimizer parameters independently of wrapper prefixes."""
        optimized = {
            id(parameter)
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        names: dict[int, str] = {}
        for role, wrapped in (
            ("net_g", self.net_g),
            ("net_d", self.net_d),
            ("net_ae", self.net_ae),
        ):
            if wrapped is None:
                continue
            network = wrapped
            while (bare := self.get_bare_model(network)) is not network:
                network = bare
            for name, parameter in network.named_parameters():
                parameter_id = id(parameter)
                if parameter_id in optimized:
                    qualified_name = f"{role}.{name}"
                    if parameter_id in names:
                        raise ValueError(
                            f"Optimizer parameter {qualified_name} is also registered "
                            f"as {names[parameter_id]}."
                        )
                    names[parameter_id] = qualified_name

        identities: list[OptimizerIdentity] = []
        for index, optimizer in enumerate(self.optimizers, start=1):
            cls = type(optimizer)
            groups: list[list[tuple[str, tuple[int, ...]]]] = []
            for group_index, group in enumerate(optimizer.param_groups, start=1):
                parameters = []
                for position, parameter in enumerate(group["params"], start=1):
                    name = names.get(id(parameter))
                    if name is None:
                        raise ValueError(
                            f"Optimizer {index}, group {group_index}, parameter "
                            f"{position} has no name in net_g, net_d, or net_ae."
                        )
                    parameters.append((name, tuple(parameter.shape)))
                groups.append(parameters)
            identities.append(
                {"type": f"{cls.__module__}.{cls.__qualname__}", "groups": groups}
            )
        return identities

    def _check_optimizer_identities(self, saved: list[OptimizerIdentity]) -> None:
        current = self._optimizer_identities()
        if not isinstance(saved, list) or len(saved) != len(current):
            raise ValueError("Cannot resume. Invalid number of optimizer identities.")

        for index, (checkpoint, live) in enumerate(
            zip(saved, current, strict=True), start=1
        ):
            error = f"Cannot resume optimizer {index}."
            if not isinstance(checkpoint, dict):
                raise ValueError(f"{error} Invalid identity metadata.")
            if checkpoint.get("type") != live["type"]:
                raise ValueError(
                    f"{error} Checkpoint optimizer type is {checkpoint.get('type')!r}. "
                    f"The recipe uses {live['type']!r}."
                )

            saved_groups = checkpoint.get("groups")
            if not isinstance(saved_groups, list) or len(saved_groups) != len(
                live["groups"]
            ):
                raise ValueError(f"{error} The number of parameter groups changed.")
            for group_index, (group, live_group) in enumerate(
                zip(saved_groups, live["groups"], strict=True), start=1
            ):
                if not isinstance(group, list) or len(group) != len(live_group):
                    raise ValueError(
                        f"{error} The number of parameters in group {group_index} changed."
                    )
                if group == live_group:
                    continue
                for position, (saved_parameter, parameter) in enumerate(
                    zip(group, live_group, strict=True), start=1
                ):
                    if saved_parameter != parameter:
                        raise ValueError(
                            f"{error} Group {group_index}, parameter {position} is "
                            f"{saved_parameter!r} in the checkpoint and {parameter!r} "
                            "in the current recipe. Use the original parameter "
                            "selection or initialize from model weights."
                        )

    @master_only
    def save_training_state(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        assert self.opt.path.training_states is not None

        if current_iter != -1:
            # assert self.scaler_g is not None
            # assert self.scaler_d is not None
            state: TrainingState = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "optimizer_identities": self._optimizer_identities(),
                "schedulers": [],
            }

            if self.early_stopping is not None:
                state["early_stopping"] = self.early_stopping.state_dict()
            if self.checkpoint_random_streams is not None:
                state["random_streams"] = self.checkpoint_random_streams
            if self._resume_recipe is not None:
                state["resume_recipe"] = self._resume_recipe

            if self.use_amp:
                if self.scaler_d is not None:
                    state["scaler_d"] = self.scaler_d.state_dict()
                if self.scaler_g is not None:
                    state["scaler_g"] = self.scaler_g.state_dict()
                if self.scaler_ae is not None:
                    state["scaler_ae"] = self.scaler_ae.state_dict()

            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            if self._scheduler_specs:
                state["scheduler_specs"] = self._scheduler_specs
            if self._metric_schedules:
                state["scheduler_pending"] = [index for index, watch in self._metric_schedules.items() if watch.updated]
            if self.adaptive_d:
                # A manual stop may discard a partially accumulated window.
                # Persist the controller at the same committed boundary as
                # the optimizer, not its provisional microbatch history.
                adaptive_ema = getattr(self, "_adaptive_d_window_start", None)
                if adaptive_ema is None:
                    adaptive_ema = self.l_g_gan_ema
                state["adaptive_d_loss_ema"] = adaptive_ema.detach().cpu().clone()
            if self.net_g_ema is not None:
                state["ema_step"] = self.net_g_ema.step
            elif self.net_ae_ema is not None:
                state["ema_step"] = self.net_ae_ema.step

            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.opt.path.training_states, save_filename)

            # The training loop supplies the position of the same accepted
            # update as this optimizer state, never the prefetcher's position.
            payload = (
                state
                if self.sampler_cursor is None
                else {**state, "sampler_cursor": self.sampler_cursor}
            )
            with ExitStack() as stack:
                replay = []
                for optimizer in self.optimizers:
                    if callable(getattr(optimizer, "eval", None)) and not native_schedule_free(optimizer):
                        stack.enter_context(evaluation_view(optimizer))
                    saved, originals = checkpoint_state(optimizer, exact=self.opt.checkpoint_rng)
                    if (base_lr := optimizer.defaults.get("lr")) is not None:
                        saved["recipe_lr"] = float(base_lr)
                    state["optimizers"].append(saved)
                    replay.append(originals)
                if any(replay):
                    payload["schedule_free_training_parameters"] = replay

                def write_state(destination: str) -> None:
                    _save_torch_checkpoint(payload, destination)
                    if current_iter == self._project_step and self._project_weights:
                        record_weights(destination, current_iter, self._project_weights,
                                       self.opt.manual_seed, str(torch.__version__))

                _write_checkpoint_atomically(save_path, write_state)
                self._project_weights.clear()
                self._project_step = None

    def resume_training(self, resume_state: TrainingState) -> None:
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        assert self.opt.train is not None
        if self.early_stopping is not None:
            if "early_stopping" in resume_state:
                self.early_stopping.restore(resume_state["early_stopping"], resume_state["iter"])
            else:
                get_root_logger().warning("Checkpoint has no early-stopping history. Patience starts at the next finite scheduled validation.")
        saved_ema = resume_state.get("resume_recipe", {}).get("neosr_ema")
        current_ema = (self._resume_recipe or {}).get("neosr_ema")
        if saved_ema != current_ema:
            raise ValueError(
                "Cannot resume with changed EMA mode, decay or power. Use the checkpoint recipe "
                "or initialize a new run from network weights."
            )
        if saved_ema is not None:
            step = resume_state.get("ema_step")
            if not isinstance(step, Tensor) or step.ndim != 0 or step.dtype != torch.int64 or step.item() < 0:
                raise ValueError("NeoSR EMA resume requires the saved count of successful G updates.")
            if self.opt.path.pretrain_network_g_ema is None:
                raise ValueError("NeoSR EMA resume requires its matching generator EMA weights.")

        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]

        if len(resume_optimizers) != len(self.optimizers):
            raise ValueError(
                f"Cannot resume. Checkpoint has {len(resume_optimizers)} optimizers. "
                f"The recipe has {len(self.optimizers)}."
            )
        if not self.opt.train.restart_scheduler and len(resume_schedulers) != len(self.schedulers):
            raise ValueError(
                f"Cannot resume. Checkpoint has {len(resume_schedulers)} schedulers. "
                f"The recipe has {len(self.schedulers)}."
            )

        if not self.opt.train.restart_scheduler:
            saved_specs = resume_state.get("scheduler_specs")
            if saved_specs is None:
                if self.opt.train.schedulers is not None and self.schedulers:
                    raise ValueError("Legacy checkpoint has no scheduler targets. Use the shared schedule or explicitly restart the schedulers.")
            elif len(saved_specs) != len(self._scheduler_specs):
                raise ValueError("Cannot resume. The number of scheduler declarations changed.")
            else:
                for saved,live in zip(saved_specs,self._scheduler_specs,strict=True):
                    if any(saved.get(key) != live[key] for key in ("target","type","monitor")):
                        raise ValueError("Cannot resume scheduler with a changed optimizer target, class or metric. Use restart_scheduler to start the new schedule.")
                    if saved.get("options") != live["options"]:
                        get_root_logger().warning("Restoring saved %s scheduler for %s; its configured options changed.", live["type"],live["target"])
            for index in resume_state.get("scheduler_pending", []):
                if index not in self._metric_schedules:
                    raise ValueError("Checkpoint refers to an unknown metric-driven scheduler.")
                self._metric_schedules[index].updated = True

        # Preflight every optimizer before load_state_dict changes any of them.
        if "optimizer_identities" in resume_state:
            self._check_optimizer_identities(resume_state["optimizer_identities"])
        elif resume_optimizers:
            get_root_logger().warning(
                "Legacy checkpoint has no optimizer identities. "
                "Optimizer state will be matched by group position."
            )

        for index, (saved, optimizer) in enumerate(
            zip(resume_optimizers, self.optimizers, strict=True), start=1
        ):
            saved_sizes = [len(group["params"]) for group in saved["param_groups"]]
            current_sizes = [len(group["params"]) for group in optimizer.param_groups]
            if saved_sizes != current_sizes:
                raise ValueError(
                    f"Cannot resume optimizer {index}. Checkpoint parameter group "
                    f"sizes are {saved_sizes}. The recipe uses {current_sizes}."
                )

        exact = resume_state.get("schedule_free_training_parameters")
        if exact is not None and (not isinstance(exact, list) or len(exact) != len(self.optimizers)):
            raise ValueError("Schedule-free replay has a different optimizer count.")
        restored = [checked_replay_weights(optimizer, saved, exact[i]) for i, (optimizer, saved)
                    in enumerate(zip(self.optimizers, resume_optimizers, strict=True))] if exact is not None else []
        configured_groups = [s.optimizer.param_groups for s in self.schedulers] if self.opt.train.restart_scheduler else []
        resume_optimizers, resume_schedulers = rebase_resume_rates(self, resume_state)
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
            if hasattr(self.optimizers[i], "train"):
                self.optimizers[i].train()  # pyright: ignore[reportAttributeAccessIssue]
            if restored:
                with torch.no_grad():
                    for param, original in restored[i]:
                        param.copy_(original)

        current_iter = resume_state["iter"]
        if self.opt.train.restart_scheduler:
            restart_schedulers(self.schedulers, self._scheduler_specs, configured_groups, resume_state)
            logger = get_root_logger()
            logger.info("Restarted scheduler(s) at iter %d", current_iter)
        else:
            for i, s in enumerate(resume_schedulers):
                self.schedulers[i].load_state_dict(s)

        if "scaler_g" in resume_state:
            assert self.scaler_g is not None
            self.scaler_g.load_state_dict(resume_state["scaler_g"])
        if "scaler_d" in resume_state:
            assert self.scaler_d is not None
            self.scaler_d.load_state_dict(resume_state["scaler_d"])
        if "scaler_ae" in resume_state:
            assert self.scaler_ae is not None
            self.scaler_ae.load_state_dict(resume_state["scaler_ae"])

        if "ema_step" in resume_state:
            if self.net_g_ema is not None:
                self.net_g_ema.register_buffer("step", resume_state["ema_step"])
                self.net_g_ema.register_buffer("initted", torch.tensor(True))

            elif self.net_ae_ema is not None:
                self.net_ae_ema.register_buffer("step", resume_state["ema_step"])
                self.net_ae_ema.register_buffer("initted", torch.tensor(True))

        if self.adaptive_d:
            if "adaptive_d_loss_ema" in resume_state:
                value = resume_state["adaptive_d_loss_ema"]
                if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid adaptive-D loss EMA in resume state")
                self.l_g_gan_ema.copy_(value.to(self.l_g_gan_ema))
            else:
                get_root_logger().warning(
                    "Legacy checkpoint has no adaptive-D loss EMA; restarting "
                    "its controller history. Model and optimizer states are restored."
                )

    @torch.no_grad()
    def reduce_loss_dict(
        self, loss_dict: dict[str, Any], *, sample_count: int = 1
    ) -> dict[str, Any]:
        """Combine local sums at publication, including rank-specific loss terms."""
        if not self.opt.dist:
            return loss_dict
        world = torch.distributed.get_world_size()
        windows: list[tuple[tuple[str, ...], int]] = [((), 0)] * world
        torch.distributed.all_gather_object(windows, (tuple(loss_dict), sample_count))
        keys = sorted({k for local_keys, _ in windows for k in local_keys})
        count = sum(n for _, n in windows)
        if not keys or not count:
            return {}
        totals = torch.stack([
            torch.as_tensor(loss_dict.get(k, 0), dtype=torch.float64, device=self.device).reshape(())
            for k in keys
        ])
        torch.distributed.all_reduce(totals)
        return dict(zip(keys, (totals / count).unbind(), strict=True))

    def setup_batchaug(self) -> None:
        assert self.opt.train is not None
        logger = get_root_logger()
        if self.opt.train.use_moa:
            from traiNNer.ops.batchaug import MOA_DEBUG_PATH, BatchAugment

            self.batch_augment = BatchAugment(self.opt.scale, self.opt.train)
            logger.info(
                "Mixture of augmentations (MoA) enabled with augs: %s and probs: %s",
                self.batch_augment.moa_augs,
                self.batch_augment.moa_probs,
            )
            if self.batch_augment.debug:
                logger.info(
                    "MoA debugging enabled. Augmented tiles will be saved to: %s",
                    MOA_DEBUG_PATH,
                )

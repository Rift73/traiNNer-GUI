import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

    from traiNNer.models.sr_buffer_window import BufferWindow
    from traiNNer.models.sr_qat import QATWindow
from tqdm import TqdmExperimentalWarning
from trainner_config.objectives import (
    LossSchedule,
    Objective,
    objective_schedules,
    resolve_objectives,
)

from trainner_config.tasks import AETHER_ARCHITECTURES, validate_qat
from trainner_config.parameter_filters import parameter_filter

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.archs.lqd_arch import LQD
from traiNNer.archs.mcd_arch import MCD
from traiNNer.losses import build_loss
from traiNNer.losses.vosr_loss import balance_vosr
from traiNNer.models.base_model import (
    BaseModel,
    _DeferredUpdate,
    _fused_adam_fast_path_applies,
)
from traiNNer.models.neosr_ema import NeoSREMA
from traiNNer.models.sr_distillation import (
    distillation_distance,
    distillation_feature_representation,
    distillation_scale,
    forward_with_distillation_features,
)
from traiNNer.models.sr_gan_health import GanHealthSample
from traiNNer.models.sr_inference import infer_tiled, validation_inference_context
from traiNNer.models.sr_objective_views import (
    WaveletDiscriminatorView,
    clamp_loss_output,
    neosr_lq_reference,
)
from traiNNer.models.sr_validation import run_validation
from traiNNer.training.spike_clip import SpikeClip
from traiNNer.training_recovery import (
    MAX_CONSECUTIVE_NONFINITE_ATTEMPTS,
    PersistentNonfiniteError,
)
from traiNNer.utils import get_root_logger
from traiNNer.utils.color_util import complete_output_from_input, pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.lq_condition import LQCondition, lq_discriminator_input
from traiNNer.utils.medium_condition import MediumCondition, medium_discriminator_input
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed, TrainingState


def _get_total_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    """Compute the total L2 norm of per-gradient norms for uniform dense grads.

    The per-gradient norms read each gradient in place; flattening into one
    tensor would copy every gradient (channels-last ones cannot be viewed flat).
    """
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("cannot compute a gradient norm without gradients")

    first = gradients[0]
    if all(
        not gradient.is_sparse
        and gradient.device == first.device
        and gradient.dtype == first.dtype
        for gradient in gradients
    ):
        if len(gradients) == 1:
            return torch.linalg.vector_norm(first, 2).detach()
        # torch has no public multi-tensor norm; its own gradient clipping
        # calls this one.
        return torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(gradients, 2)),  # noqa: SLF001
            2,
        ).detach()

    return get_total_norm(
        gradients,
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=None,
    ).detach()


def _zero_optimizer_gradients(optimizer: Optimizer, keep_storage: bool) -> None:
    """Clear grads; kept storage is zeroed with one foreach call.

    zero_grad(set_to_none=False) zeroes tensor by tensor unless the optimizer
    was built with foreach or fused.
    """
    if not keep_storage:
        optimizer.zero_grad(set_to_none=True)
        return
    gradients = [
        parameter.grad
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    if not gradients:
        return
    device = gradients[0].device
    # zero_grad also detaches grads that carry a graph; leave those to it.
    if all(
        gradient.device == device
        and not gradient.is_sparse
        and gradient.grad_fn is None
        and not gradient.requires_grad
        for gradient in gradients
    ):
        torch._foreach_zero_(gradients)
    else:
        optimizer.zero_grad(set_to_none=False)


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    d_conditioner: MediumCondition | LQCondition | None = None
    sed_conditioner: nn.Module | None = None
    d_input_conditioning: bool = False
    _loss_schedules: dict[str, LossSchedule] | None = None
    _spike_clippers: dict[str, SpikeClip] | None = None
    _qat: "QATWindow | None" = None
    _buffer_window: "BufferWindow | None" = None
    _filtered_g_params: tuple[nn.Parameter, ...] = ()
    _ddp_excluded_g_params: tuple[nn.Parameter, ...] = ()
    _window_log_sums: dict[str, Tensor | float] | None = None
    _window_log_samples = 0
    _gan_health_freq = 0
    _gan_health_sample: GanHealthSample | None = None
    _loss_output_clamp = "none"
    _gan_wavelet: WaveletDiscriminatorView | None = None

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        # use amp
        self.use_amp = self.opt.use_amp
        self.use_channels_last = self.opt.use_channels_last
        self.memory_format = (
            torch.channels_last
            if self.use_amp and self.use_channels_last
            else torch.preserve_format
        )
        self.amp_dtype = torch.bfloat16 if self.opt.amp_bf16 else torch.float16
        self.use_compile = self.opt.use_compile
        # Auxiliary callable only: EMA keeps ownership of its model, parameters,
        # training flags, checkpoint state and validation path.
        self._ldl_ema_forward: Callable[[Tensor], Tensor] | None = None
        self.compile_discriminator = self.opt.compile_discriminator
        self.distillation_opt = (
            opt.train.distillation
            if opt.is_train
            and opt.train is not None
            and opt.train.distillation is not None
            and opt.train.distillation.enabled
            else None
        )
        self.distillation = self.distillation_opt is not None
        self.cudagraph_stable_gradients = False
        self.cudagraph_stable_discriminator_gradients = False
        self._adaptive_d_window_start: Tensor | None = None
        self._validation_inference_net: nn.Module | None = None
        self._validation_prepared_key: tuple[int, int] | None = None

        objectives = (
            resolve_objectives(
                opt, model="clip" if getattr(self, "clip_mode", False) else "sr"
            )
            if self.is_train
            else {}
        )
        if self.is_train and opt.logger is not None and opt.logger.gan_health:
            if (
                type(self).optimize_parameters is not SRModel.optimize_parameters
                or opt.dist
                or opt.num_gpu not in (0, 1)
                or self.use_compile
                or self.compile_discriminator
            ):
                raise ValueError(
                    "logger.gan_health requires eager single-device still-image SR or OTF training."
                )
            gan = objectives.get("l_g_gan")
            if (
                opt.train is None
                or opt.train.optim_g is None
                or gan is None
                or gan.options["type"].lower() != "ganloss"
            ):
                raise ValueError("logger.gan_health requires train.optim_g and an active GANLoss.")
            self._gan_health_freq = opt.logger.print_freq
        self._loss_schedules = objective_schedules(objectives)
        self._neosr_lq_labels = frozenset(label for label, obj in objectives.items() if obj.neosr_lq)

        # define network
        assert opt.network_g is not None, "network_g must be defined"
        qat = opt.train.qat if self.is_train and opt.train else None
        if qat:
            validate_qat(opt)
        self.net_g = build_network({**opt.network_g, "scale": opt.scale})

        # load pretrained models
        if qat or (opt.path.pretrain_network_g and opt.network_g["type"].lower() in AETHER_ARCHITECTURES):
            from traiNNer.models.sr_qat import initialize_aether

            initialize_aether(self, self.net_g)
        elif self.opt.path.pretrain_network_g is not None:
            self.load_initial_g(self.net_g)

        # define teacher network
        if opt.network_g_teacher is not None:
            self.net_g_teacher = build_network(
                {**opt.network_g_teacher, "scale": opt.scale}
            )

            if self.opt.path.pretrain_network_g_teacher is None:
                raise ValueError(
                    "pretrain_network_g_teacher must be defined when using network_g_teacher"
                )
            self.load_network(
                self.net_g_teacher,
                self.opt.path.pretrain_network_g_teacher,
                True,
                None,
            )
            self.net_g_teacher.requires_grad_(False)
        elif self.distillation:
            raise ValueError("distillation.enabled requires network_g_teacher")

        self.net_g = self.model_to_device(self.net_g, compile=self.opt.use_compile)
        if qat:
            from traiNNer.models.sr_qat import QATWindow

            assert opt.train and self._resume_recipe
            self._qat = QATWindow(self.net_g, qat.schedule(opt.train.total_iter))
            self._resume_recipe["qat"]["quantize_residual"] = self.net_g.quantize_residual
        if self.net_g_teacher is not None:
            compile_teacher = self.opt.use_compile and (
                not self.distillation
                or (
                    self.distillation_opt is not None
                    and self.distillation_opt.compile_teacher
                )
            )
            self.net_g_teacher = self.model_to_device(
                self.net_g_teacher,
                compile=compile_teacher,
                wrap_distributed=False,
            )
            self.net_g_teacher.eval()

        self.lq: Tensor | None = None
        self.gt: Tensor | None = None
        self.output: Tensor | None = None
        logger = get_root_logger()

        if self.use_amp and self.device.type == "cuda":
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 was enabled for AMP but the current GPU does not support bf16. Falling back to float16 for AMP. Disable bf16 to hide this warning (amp_bf16: false)."
                )
                self.amp_dtype = torch.float16

            network_g_name = opt.network_g["type"]
            if (
                self.amp_dtype == torch.float16
                and network_g_name.lower() in ARCHS_WITHOUT_FP16
            ):
                if torch.cuda.is_bf16_supported():
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Falling back to bf16.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.amp_dtype = torch.bfloat16
                else:
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Disabling AMP.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.use_amp = False
        elif self.amp_dtype == torch.bfloat16:
            logger.warning(
                "bf16 was enabled without AMP and will have no effect. Enable AMP to use bf16 (use_amp: true)."
            )

        if self.use_amp:
            logger.info(
                "Using Automatic Mixed Precision (AMP) with fp32 and %s.",
                "bf16" if self.amp_dtype == torch.bfloat16 else "fp16",
            )

            if self.use_channels_last:
                logger.info("Using channels last memory format.")

        if self.opt.fast_matmul:
            logger.info(
                "Fast matrix multiplication and convolution operations (fast_matmul) enabled, trading precision for performance."
            )

        if self.is_train and self.opt.train:
            # Build D for any configured GAN window, including one that starts later.
            self.has_gan = "l_g_gan" in objectives

            self.net_d = None
            if self.has_gan:
                if self.opt.train.optim_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator optimizer (optim_d). Define optim_d or disable GAN loss."
                    )
                if self.opt.network_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator network (network_d). Define network_d or disable GAN loss."
                    )
                else:
                    self.net_d = build_network(self.opt.network_d)
                    if isinstance(self.net_d, MCD):
                        self.net_d.allow_rgb_warm_start = not bool(
                            self.opt.path.resume_state
                        )
                        self.net_d.allow_input_warm_start = (
                            self.net_d.allow_rgb_warm_start
                        )
                        self.d_input_conditioning = self.net_d.input_conditioning
                        self.d_conditioner = MediumCondition().to(self.device)
                        logger.info(
                            "MC-D enabled: four target band energies and two "
                            "orientations; degraded RGB input conditioning=%s. "
                            "Conditions are shared by all discriminator passes.",
                            self.d_input_conditioning,
                        )
                    elif isinstance(self.net_d, LQD):
                        self.d_conditioner = LQCondition().to(self.device)
                        logger.info(
                            "LQ-D enabled: shared detached degraded RGB only; "
                            "no HR-derived medium statistics."
                        )
                    elif self.opt.network_d["type"].lower() == "sed":
                        # Keep frozen HR features outside D's optimizer, checkpoint and phase toggles.
                        self.sed_conditioner = self.net_d.build_conditioner().to(
                            self.device, memory_format=self.memory_format
                        ).eval()
                    # load pretrained models
                    if self.opt.path.pretrain_network_d is not None:
                        self.load_network(
                            self.net_d,
                            self.opt.path.pretrain_network_d,
                            self.opt.path.strict_load_d,
                            self.opt.path.param_key_d,
                        )
                    self.net_d = self.model_to_device(
                        self.net_d, compile=self.compile_discriminator
                    )

            self.losses = {}

            self.ema_decay = 0
            self.net_g_ema: EMA | None = None

            self.optimizer_g: Optimizer | None = None
            self.optimizer_d: Optimizer | None = None
            self.optimizer_d_index: int | None = None

            self.distillation_feature_pairs: tuple[tuple[int, int, str, str], ...] = ()
            self.distillation_student_feature_count = 0
            self.distillation_teacher_feature_count = 0
            self._vosr_balance_state: tuple[Tensor, Tensor] | None = None

            self.init_training_settings(objectives)
            if self.opt.train.rollback_buffers_on_reject:
                from traiNNer.models.sr_buffer_window import BufferWindow

                if type(self).optimize_parameters is not SRModel.optimize_parameters:
                    raise ValueError("train.rollback_buffers_on_reject requires the SRModel update loop.")
                window = BufferWindow(self.net_g, self.net_d)
                if window.live:
                    self._buffer_window = window
            self.deferred_nonfinite_guard = self._deferred_nonfinite_guard_applies()

    def _deferred_nonfinite_guard_applies(self) -> bool:
        """Whether finite checks can gate the update on the device instead of
        reading it back: the plain single-optimizer case, where a skipped
        fused step undoes the whole update and nothing else waits on it."""
        optimizer = self.optimizer_g
        return (
            # Models with their own update loop keep that loop's checks.
            type(self).optimize_parameters is SRModel.optimize_parameters
            and self._loss_schedules is None
            and self._spike_clippers is None
            and self._qat is None
            and self._buffer_window is None
            and not isinstance(self.net_g_ema, NeoSREMA)
            and self.device.type == "cuda"
            and not self.opt.dist
            and optimizer is not None
            and self.optimizer_d is None
            and self.accum_iters == 1
            and self.scaler_g is not None
            and not self.scaler_g.is_enabled()
            and _fused_adam_fast_path_applies(optimizer)
            and not torch.is_anomaly_enabled()
        )

    def _resolve_deferred_updates(self, wait: bool) -> None:
        """Apply the outcome of pending updates in order: logs for a finite
        one, the rejection bookkeeping for a non-finite one. Without wait,
        stop at the first update the device has not finished."""
        while self._deferred_updates:
            update = self._deferred_updates[0]
            if wait:
                update.ready.synchronize()
            elif not update.ready.query():
                return
            self._deferred_updates.popleft()
            nonfinite_loss, nonfinite_gradient = update.flags.tolist()
            if not (nonfinite_loss or nonfinite_gradient):
                self._report_pending_nonfinite_rejections(update.candidate_iter)
                self.nan_count = 0
                self.add_log_contributions(update.n_samples, update.contributions)
                continue
            reason = (
                "nonfinite_generator_loss"
                if nonfinite_loss
                else "nonfinite_generator_gradient"
            )
            limit_reached = self._record_nonfinite_rejection(update.candidate_iter)
            # train.py already counted this update; a later call reports one
            # rejection for it so iterations and scheduler steps stay exact.
            self._owed_rejection_reports += 1
            if self.nan_count == 1 or limit_reached:
                bad = [
                    name
                    for name, value in update.contributions.items()
                    if isinstance(value, Tensor)
                    and not bool(torch.isfinite(value).all())
                ]
                get_root_logger().warning(
                    "Rejected candidate iter %d: %s; nonfinite values: %s. "
                    "The optimizer step was skipped on the device.",
                    update.candidate_iter,
                    reason,
                    ", ".join(bad) or "backward gradients",
                )
            if limit_reached:
                raise PersistentNonfiniteError(
                    reason, update.candidate_iter, self.nan_count
                )

    def settle_training_updates(self) -> None:
        self._resolve_deferred_updates(wait=True)

    def get_current_log(self) -> dict[str, float | torch.Tensor]:
        self._resolve_deferred_updates(wait=True)
        logs = super().get_current_log()
        # The shared publisher divides every sum by the global sample count.
        # Dividing by the equally normalized step count gives the per-step mean.
        for role in ("g", "d"):
            steps = logs.pop(f"_optimizer_steps_{role}", None)
            if steps is not None:
                for key in (f"scale_{role}", f"grad_norm_{role}"):
                    if key in logs:
                        logs[key] = logs[key] / steps
        if self._gan_health_sample is not None:
            logs.update(self._gan_health_sample.logs())
        if self._spike_clippers is not None:
            for role, clipper in self._spike_clippers.items():
                logs.update(clipper.logs(role, distributed=self.opt.dist))
        return logs

    def reset_current_log(self) -> None:
        super().reset_current_log()
        if self._gan_health_sample is not None:
            self._gan_health_sample = None
        if self._spike_clippers is not None:
            for clipper in self._spike_clippers.values():
                clipper.window.zero_()

    def _discriminator_condition(self, gt: Tensor, lq: Tensor) -> Tensor | None:
        if self.sed_conditioner is not None:
            return self.sed_conditioner(gt)
        if self.d_conditioner is None:
            return None
        if isinstance(self.d_conditioner, LQCondition):
            return self.d_conditioner(lq, (gt.shape[-2], gt.shape[-1]))
        return self.d_conditioner(gt, lq if self.d_input_conditioning else None)

    def _discriminate(self, image: Tensor, condition: Tensor | None) -> Tensor:
        assert self.net_d is not None
        if self.sed_conditioner is not None:
            assert condition is not None
            return self.net_d(image, condition)
        if isinstance(self.d_conditioner, LQCondition):
            return self.net_d(lq_discriminator_input(image, condition))
        return self.net_d(medium_discriminator_input(image, condition))

    def _discard_accumulated_gradients(self) -> None:
        self._window_log_sums = None
        self._window_log_samples = 0
        if self._qat is not None:
            self._qat.rollback()
        if self._buffer_window is not None:
            self._buffer_window.rollback()
        # A rejected loss may leave its entire forward graph unconsumed.
        self.output = None
        if self.optimizer_g is not None:
            self._zero_generator_optimizer(self.optimizer_g)
        if self.optimizer_d is not None:
            self._zero_discriminator_optimizer(self.optimizer_d)
        if self._adaptive_d_window_start is not None:
            self.l_g_gan_ema.copy_(self._adaptive_d_window_start)
            self._adaptive_d_window_start = None

    def _finite_on_all_ranks(self, value: Tensor) -> bool:
        """Agree before any rank can leave its peers waiting in backward or step."""
        finite = torch.isfinite(value)
        if self.opt.dist:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return bool(finite.item())

    def _reject_nonfinite_update(
        self,
        reason: str,
        candidate_iter: int,
        scaler_states: list[tuple[GradScaler, dict[str, Any]]],
        loss_dict: dict[str, Tensor | float],
    ) -> None:
        """Reject the accumulated batch before either optimizer commits."""
        limit_reached = self._record_nonfinite_rejection(candidate_iter)
        self.reject_training_step(reason)
        self.optimizers_skipped[:] = [True] * len(self.optimizers_skipped)
        bad = [
            name
            for name, value in loss_dict.items()
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all())
        ]
        for name, value in (
            ("input_lq", self.lq),
            ("target_gt", self.gt),
            ("generator_output", self.output),
        ):
            if value is not None and not bool(torch.isfinite(value).all()):
                bad.append(name)
        if self.nan_count == 1 or limit_reached:
            get_root_logger().warning(
                "Rejected candidate iter %d: %s; nonfinite values: %s. "
                "Neither optimizer nor EMA was updated.",
                candidate_iter,
                reason,
                ", ".join(bad) or (
                    "none identified on this rank" if self.opt.dist else "backward gradients"
                ),
            )
        self._discard_accumulated_gradients()
        for scaler, state in scaler_states:
            # Clear unscale_'s stateful phase. Retain real overflow backoff,
            # but do not advance a healthy scaler for a rejected G/D update.
            scaler.update()
            scale = scaler.get_scale()
            if self.opt.dist:
                # Back off on every rank if any unscale detected an overflow.
                lowest = torch.tensor(scale, device=self.device, dtype=torch.float32)
                dist.all_reduce(lowest, op=dist.ReduceOp.MIN)
                scale = lowest.item()
                if scale < state["scale"]:
                    state.update(scale=scale, _growth_tracker=0)
                scaler.load_state_dict(state)
            elif scale >= state["scale"]:
                scaler.load_state_dict(state)
        if limit_reached:
            raise PersistentNonfiniteError(reason, candidate_iter, self.nan_count)

    def init_training_settings(self, objectives: dict[str, Objective]) -> None:
        self.net_g.train()
        if self.net_g_teacher is not None:
            self.net_g_teacher.eval()
        if self.net_d is not None:
            self.net_d.train()

        train_opt = self.opt.train
        assert train_opt is not None

        logger = get_root_logger()

        self._loss_output_clamp = train_opt.loss_output_clamp
        if train_opt.gan_wavelet_start_iter is not None:
            self._gan_wavelet = WaveletDiscriminatorView(
                train_opt.gan_wavelet_start_iter
            ).to(self.device)

        enable_gradscaler = self.use_amp and not self.opt.amp_bf16

        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")
        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")

        if self.distillation:
            assert self.net_g_teacher is not None
            assert self.distillation_opt is not None
            if self.distillation_opt.output_weight < 0:
                raise ValueError("distillation output_weight must be non-negative")
            if self.distillation_opt.feature_weight < 0:
                raise ValueError("distillation feature_weight must be non-negative")
            if (
                self.distillation_opt.output_weight == 0
                and self.distillation_opt.feature_weight == 0
            ):
                raise ValueError(
                    "distillation requires a positive output_weight or feature_weight"
                )
            if self.distillation_opt.feature_pool_size < 1:
                raise ValueError("distillation feature_pool_size must be at least 1")
            if self.distillation_opt.start_iter < 0:
                raise ValueError("distillation start_iter must be non-negative")
            if self.distillation_opt.warmup_iters < 0:
                raise ValueError("distillation warmup_iters must be non-negative")
            if self.distillation_opt.feature_start_iter < 0:
                raise ValueError("distillation feature_start_iter must be non-negative")
            if self.distillation_opt.feature_warmup_iters < 0:
                raise ValueError(
                    "distillation feature_warmup_iters must be non-negative"
                )
            if (self.distillation_opt.decay_start_iter is None) != (
                self.distillation_opt.end_iter is None
            ):
                raise ValueError(
                    "distillation decay_start_iter and end_iter must be set together"
                )
            if (
                self.distillation_opt.decay_start_iter is not None
                and self.distillation_opt.end_iter is not None
                and self.distillation_opt.end_iter
                <= self.distillation_opt.decay_start_iter
            ):
                raise ValueError(
                    "distillation end_iter must be greater than decay_start_iter"
                )

            if self.distillation_opt.feature_weight == 0:
                self.distillation_feature_pairs = ()
                self.distillation_student_feature_count = 0
                self.distillation_teacher_feature_count = 0
                logger.info(
                    "Using frozen-teacher output distillation from %s.",
                    self.opt.path.pretrain_network_g_teacher,
                )
            else:
                self._setup_feature_distillation_contract()

            if any(
                parameter.requires_grad for parameter in self.net_g_teacher.parameters()
            ):
                raise RuntimeError("the distillation teacher must be fully frozen")

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.adaptive_d = train_opt.adaptive_d
        self.adaptive_d_ema_decay = train_opt.adaptive_d_ema_decay
        self.adaptive_d_threshold = train_opt.adaptive_d_threshold
        self.ema_decay = train_opt.ema_decay

        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )
            assert self.opt.network_g is not None, "network_g must be defined"

            bare_g = self.get_bare_model(self.net_g)
            path = self.opt.path.pretrain_network_g_ema
            if train_opt.ema_mode == "neosr":
                ema = NeoSREMA(bare_g, self.ema_decay, train_opt.ema_power)
                if path is not None and self.opt.network_g["type"].lower() in AETHER_ARCHITECTURES:
                    from traiNNer.models.sr_qat import initialize_aether

                    initialize_aether(self, ema.ema_model, ema=True)
                elif path is not None:
                    self.load_network(ema.ema_model, path, self.opt.path.strict_load_g, "params_ema")
            else:
                init_net_g_ema = None
                if path is not None:
                    init_net_g_ema = build_network({**self.opt.network_g, "scale": self.opt.scale})
                    if self.opt.network_g["type"].lower() in AETHER_ARCHITECTURES:
                        from traiNNer.models.sr_qat import initialize_aether

                        initialize_aether(self, init_net_g_ema, ema=True)
                    else:
                        self.load_network(init_net_g_ema, path, self.opt.path.strict_load_g, "params_ema")
                ema = EMA(
                    bare_g,
                    ema_model=init_net_g_ema,
                    beta=self.ema_decay,
                    allow_different_devices=True,
                    update_after_step=train_opt.ema_update_after_step,
                    update_every=1,
                    power=10 if train_opt.ema_power is None else train_opt.ema_power,
                    use_foreach=train_opt.ema_foreach,
                    update_model_with_ema_every=train_opt.ema_switch_iter or None,
                )
            self.net_g_ema = ema.to(device=self.device, memory_format=self.memory_format)
            self.net_g_ema.step = self.net_g_ema.step.to(device=torch.device("cpu"))
            self.net_g_ema.initted = self.net_g_ema.initted.to(device="cpu")
            if self.use_compile:
                self.prepare_network_for_compile(self.net_g)

        self.grad_clip = train_opt.grad_clip is True
        self.grad_clip_max_norm = train_opt.grad_clip_max_norm
        if self.grad_clip:
            logger.info(
                "Gradient clipping is enabled with max norm=%f.",
                self.grad_clip_max_norm,
            )

        for label, objective in objectives.items():
            self.losses[label] = build_loss(objective.options).to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

            # The VOSR teacher owns TensorRT engines and CUDA graph replays.
            if self.use_compile and label != "l_g_vosr":
                logger.info(
                    "Loss %s will be compiled. The first iteration may take several minutes.",
                    label,
                )
                self.losses[label] = torch.compile(self.losses[label])

        assert self.losses, "At least one loss must be defined."
        self._vosr_balance_state = None
        self.validate_vosr_loss()
        self.validate_vosr_target()
        self._setup_ldl_ema_forward()

        if not self.has_gan:
            # warn that discriminator network / optimizer won't be used if enabled
            if self.opt.network_d is not None:
                logger.warning(
                    "Discriminator network (network_d) is defined but GAN loss is disabled. Discriminator network will have no effect."
                )

            if train_opt.optim_d is not None:
                logger.warning(
                    "Discriminator optimizer (optim_d) is defined but GAN loss is disabled. Discriminator optimizer will have no effect."
                )

        # setup batch augmentations
        self.setup_batchaug()

        # set up optimizers and schedulers
        self.setup_optimizers()
        if train_opt.grad_clip == "spike":
            self._spike_clippers = {
                role: SpikeClip(optimizer)
                for role, optimizer in (("g", self.optimizer_g), ("d", self.optimizer_d))
                if optimizer is not None
            }
            logger.info("Using NeoSR spike gradient clipping for %s.", ", ".join(self._spike_clippers))
        self._prepare_cudagraph_gradient_accumulation()
        self.setup_schedulers()

        # logging features
        # self._setup_feature_hooks()

    def validate_vosr_loss(self) -> None:
        """The GAN term used to balance VOSR must exist and stay positive."""
        if "l_g_vosr" not in self.losses:
            return
        if "l_g_gan" not in self.losses:
            raise ValueError("vosrloss balances against the GAN loss; add a ganloss")
        gan_type = self.losses["l_g_gan"].gan_type
        if gan_type not in ("vanilla", "lsgan", "wgan_softplus"):
            raise ValueError(
                f"vosrloss balances against the GAN loss's size, which gan_type {gan_type!r} can make zero or "
                "negative; use vanilla, lsgan or wgan_softplus"
            )

    def validate_vosr_target(self) -> None:
        """Reject options that conflict with teacher-generated HR targets."""
        train_opt = self.opt.train
        if train_opt is None or train_opt.vosr_target is None:
            return
        if not self.opt.high_order_degradation:
            raise ValueError(
                "vosr_target needs on-the-fly degradations (high_order_degradation: true)"
            )
        if "l_g_vosr" in self.losses:
            raise ValueError(
                "vosr_target already makes VOSR the HR target; remove the vosrloss entry"
            )
        if self.opt.otf_hf_noise_prob > 0:
            raise ValueError(
                "otf_hf_noise adds noise to the HR, which vosr_target replaces; set otf_hf_noise_prob to 0"
            )
        if (
            self.opt.input_pixel_format != "rgb"
            or self.opt.output_pixel_format != "rgb"
        ):
            raise ValueError("vosr_target needs rgb input and output pixel formats")
        if self.opt.datasets["train"].target_dataroot_gt:
            raise ValueError(
                "vosr_target replaces the HR, so target_dataroot_gt would be ignored; remove it"
            )
        gt_size = self.opt.datasets["train"].gt_size
        if gt_size is None or gt_size > 512:
            raise ValueError(
                f"vosr_target: VOSR restores at most 512 px, the training HR patch is {gt_size} px"
            )

    def _setup_ldl_ema_forward(self) -> None:
        """Reduce DUAL launch overhead without changing EMA floating-point ops.

        LDL compares current/EMA residuals with a hard threshold. Ordinary
        Inductor fusion can change that threshold's decisions even when its
        prediction is numerically close, so use CUDA graph replay of ATen ops.
        Never freeze/fold EMA weights or force eval mode: EMA updates and the
        original train/eval transition must remain visible to every forward.
        """
        self._ldl_ema_forward = None
        if (
            "l_g_ldl" not in self.losses
            or self.net_g_ema is None
            or not self.use_compile
            or not self.opt.compile_ldl_ema
            or self.device.type != "cuda"
            or self.opt.compile_mode == "max-autotune-no-cudagraphs"
        ):
            return
        # DUAL has no stochastic or running-statistic updates in its forward.
        # Do not automatically graph-replay other architectures (or subclasses
        # with an overridden forward): warmup/capture could mutate their state.
        from traiNNer.archs.dual_arch import DUAL

        if type(self.net_g_ema.ema_model) is not DUAL:
            return
        self._ldl_ema_forward = torch.compile(
            self.net_g_ema.ema_model, backend="cudagraphs", dynamic=False
        )
        get_root_logger().info(
            "LDL EMA forward will use non-fusing CUDA graph replay. "
            "EMA updates, train/eval flags and loss arithmetic are unchanged. "
            "Set compile_ldl_ema: false to retain eager EMA execution."
        )

    def _setup_feature_distillation_contract(self) -> None:
        assert self.net_g_teacher is not None
        assert self.distillation_opt is not None
        logger = get_root_logger()
        student = self.get_bare_model(self.net_g)
        teacher = self.get_bare_model(self.net_g_teacher)
        student_contract: Callable[[], Iterable[str]] | None = getattr(
            student, "distillation_feature_names", None
        )
        teacher_contract: Callable[[], Iterable[str]] | None = getattr(
            teacher, "distillation_feature_names", None
        )
        if not callable(student_contract) or not callable(teacher_contract):
            raise TypeError(
                "feature distillation requires student and teacher architectures "
                "with distillation_feature_names() and feature-returning forwards"
            )
        student_names = tuple(student_contract())
        teacher_names = tuple(teacher_contract())
        architecture_mapping: (
            Callable[[nn.Module], Iterable[tuple[int, int, str, str]]] | None
        ) = getattr(student, "distillation_feature_pairs", None)
        if callable(architecture_mapping):
            feature_pairs = tuple(architecture_mapping(teacher))
        elif student_names == teacher_names:
            feature_pairs = tuple(
                (index, index, name, name) for index, name in enumerate(student_names)
            )
        else:
            raise ValueError(
                "student/teacher distillation feature contracts differ and "
                "the student provides no semantic mapping"
            )
        if not feature_pairs:
            raise ValueError("feature distillation requires at least one feature pair")
        for student_index, teacher_index, student_name, teacher_name in feature_pairs:
            if not (0 <= student_index < len(student_names)):
                raise ValueError("student distillation feature index is out of range")
            if not (0 <= teacher_index < len(teacher_names)):
                raise ValueError("teacher distillation feature index is out of range")
            if student_names[student_index] != student_name:
                raise ValueError("student distillation feature mapping is inconsistent")
            if teacher_names[teacher_index] != teacher_name:
                raise ValueError("teacher distillation feature mapping is inconsistent")

        self.distillation_feature_pairs = feature_pairs
        self.distillation_student_feature_count = len(student_names)
        self.distillation_teacher_feature_count = len(teacher_names)
        logger.info(
            "Frozen-teacher distillation maps %d sparse student blocks onto %d "
            "teacher blocks (%d supervised pairs, transform=%s).",
            len(student_names),
            len(teacher_names),
            len(feature_pairs),
            self.distillation_opt.feature_transform,
        )

    def _setup_feature_hooks(self) -> None:
        self._feature_entropy = None  # pyright: ignore[reportUninitializedInstanceVariable]

        def entropy_hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            with torch.no_grad():
                # Handle SPAB output which is a tuple (out, out1, sim_att)
                if isinstance(output, tuple):
                    x = output[0]
                else:
                    x = output

                if x.dim() == 3:  # (B, L, C)
                    x_avg = x.abs().mean(dim=(0, 1))
                else:  # (B, C, H, W)
                    x_avg = x.abs().mean(dim=(0, 2, 3))
                p = torch.softmax(x_avg, dim=0)
                self._feature_entropy = -(p * torch.log(p + 1e-8)).sum().item()

        # Try HAT/DAT style first
        for name, module in self.net_g.named_modules():
            if "layers.5" in name and name.endswith(
                ("blocks.5", "blocks.4", "blocks.3")
            ):
                module.register_forward_hook(entropy_hook)  # pyright: ignore[reportArgumentType]
                print(f"Entropy hook registered on: {name}")
                return

        # Try SPAN style (block_6, block_5, etc.)
        for block_name in ("block_6", "block_5", "block_4"):
            if hasattr(self.net_g, block_name):
                module = getattr(self.net_g, block_name)
                module.register_forward_hook(entropy_hook)
                print(f"Entropy hook registered on: {block_name}")
                return

        # Try RealPLKSR style - feats is a Sequential with PLKBlocks
        if hasattr(self.net_g, "feats") and isinstance(self.net_g.feats, nn.Sequential):
            for i in range(len(self.net_g.feats) - 1, -1, -1):
                if self.net_g.feats[i].__class__.__name__ == "PLKBlock":
                    self.net_g.feats[i].register_forward_hook(entropy_hook)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
                    print(f"Entropy hook registered on: feats[{i}] (PLKBlock)")
                    return

        # Try FDAT style - groups is a Sequential of SimplifiedResidualGroup
        if hasattr(self.net_g, "groups") and isinstance(
            self.net_g.groups, nn.Sequential
        ):
            # Get the last group's last block
            last_group = self.net_g.groups[-1]
            if hasattr(last_group, "blocks") and len(last_group.blocks) > 0:  # pyright: ignore[reportArgumentType]
                last_block = last_group.blocks[-1]  # pyright: ignore[reportIndexIssue]
                last_block.register_forward_hook(entropy_hook)  # pyright: ignore[reportAttributeAccessIssue]
                print(
                    "Entropy hook registered on: groups[-1].blocks[-1] (SimplifiedDATBlock)"
                )
                return

        print("WARNING: Could not find layer to register entropy hook")

    def setup_optimizers(self) -> None:
        train_opt = self.opt.train
        assert train_opt is not None
        # assert train_opt.optim_g is not None
        optim_params = []
        logger = get_root_logger()
        include_patterns = train_opt.optim_include_params_g or []
        exclude_patterns = train_opt.optim_exclude_params_g or []

        selected_for_optim = parameter_filter(include_patterns, exclude_patterns)

        if train_opt.optim_g is not None:
            filtered_trainable: list[tuple[str, nn.Parameter]] = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if selected_for_optim(k):
                        optim_params.append(v)
                    else:
                        filtered_trainable.append((k, v))
                elif "eval_" in k:
                    pass  # intentionally frozen for reparameterization, skip warning
                else:
                    logger.warning("Params %s will not be optimized.", k)

            if (include_patterns or exclude_patterns) and filtered_trainable:
                logger.info(
                    "Filtered %d trainable generator params from optimizer, up to 10 shown:",
                    len(filtered_trainable),
                )
                for name, _ in filtered_trainable[:10]:
                    logger.info("    %s", name)
            if (include_patterns or exclude_patterns) and len(optim_params) == 0:
                raise ValueError(
                    "Generator optimizer parameter filters excluded all trainable parameters."
                )

            self.optimizer_g = self.get_optimizer(optim_params, train_opt.optim_g)
            if filtered_trainable:
                self._filtered_g_params = tuple(optim_params)
                if isinstance(self.net_g, nn.parallel.DistributedDataParallel):
                    # DDP already registered reducer hooks for these gradients.
                    self._ddp_excluded_g_params = tuple(p for _, p in filtered_trainable)
                else:
                    for _, parameter in filtered_trainable:
                        parameter.requires_grad_(False)
            self.optimizers.append(self.optimizer_g)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_g["type"].upper()
            )
        else:
            logger.warning("!!! net_g will not be optimized. !!!")

        # optimizer d
        if self.net_d is not None:
            assert train_opt.optim_d is not None
            self.optimizer_d = self.get_optimizer(
                list(self.net_d.parameters()), train_opt.optim_d
            )
            self.optimizer_d_index = len(self.optimizers)
            self.optimizers.append(self.optimizer_d)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_d["type"].upper()
            )

    def prepare_warmup_learning_rate(
        self, completed_iter: int, warmup_iter: int = -1
    ) -> None:
        schedules = self._loss_schedules
        gan = schedules.get("l_g_gan") if schedules is not None else None
        if gan is None or self.optimizer_d is None or warmup_iter <= 0:
            super().prepare_warmup_learning_rate(completed_iter, warmup_iter)
            return

        # The shared warmup uses global updates. D starts its own warmup on
        # the first GAN update and retains its LR outside the GAN window.
        groups = self.optimizer_d.param_groups
        d_lrs = [group["lr"] for group in groups]
        super().prepare_warmup_learning_rate(completed_iter, warmup_iter)
        next_iter = completed_iter + 1
        next_d_update = next_iter - gan.start_iter
        d_warming_up = gan.factor(next_iter) > 0 and next_d_update <= warmup_iter
        for group, lr in zip(groups, d_lrs, strict=True):
            group["lr"] = (
                group["initial_lr"] / warmup_iter * next_d_update
                if d_warming_up
                else lr
            )

    def _record_nonfinite_rejection(self, candidate_iter: int) -> bool:
        """Record one rejected effective update and report a stalled loop."""
        self.nan_count += 1
        self._pending_nonfinite_rejection_count += 1
        limit_reached = self.nan_count >= MAX_CONSECUTIVE_NONFINITE_ATTEMPTS
        if self.nan_count % 100 == 0 or limit_reached:
            get_root_logger().warning(
                "NaN/Inf rejected attempts (count: %d), skipping update",
                self._pending_nonfinite_rejection_count,
            )
            self._pending_nonfinite_rejection_count = 0
        return limit_reached

    def _report_pending_nonfinite_rejections(self, candidate_iter: int) -> None:
        """Report all pending rejections on the next 100-iteration boundary."""
        if candidate_iter % 100 != 0 or self._pending_nonfinite_rejection_count == 0:
            return
        get_root_logger().warning(
            "NaN/Inf rejected attempts (count: %d), skipping update",
            self._pending_nonfinite_rejection_count,
        )
        self._pending_nonfinite_rejection_count = 0

    def _prepare_cudagraph_gradient_accumulation(self) -> None:
        """Allocate persistent grad buffers before the first compiled backward."""
        uses_cudagraphs = self.opt.compile_mode in {"max-autotune", "reduce-overhead"}
        self.cudagraph_stable_gradients = (
            self.use_compile and uses_cudagraphs and self.accum_iters > 1
        )
        self.cudagraph_stable_discriminator_gradients = (
            self.compile_discriminator
            and self.optimizer_d is not None
            and uses_cudagraphs
            and self.accum_iters > 1
        )
        if not (
            self.cudagraph_stable_gradients
            or self.cudagraph_stable_discriminator_gradients
        ):
            return

        optimizer_targets = (
            ("generator", self.optimizer_g, self.cudagraph_stable_gradients),
            (
                "discriminator",
                self.optimizer_d,
                self.cudagraph_stable_discriminator_gradients,
            ),
        )
        for name, optimizer, enabled in optimizer_targets:
            if optimizer is None or not enabled:
                continue
            parameter_count = 0
            element_count = 0
            seen_parameters: set[int] = set()
            with torch.no_grad():
                for parameter_group in optimizer.param_groups:
                    for parameter in parameter_group["params"]:
                        parameter_id = id(parameter)
                        if parameter_id in seen_parameters:
                            continue
                        seen_parameters.add(parameter_id)
                        if not parameter.requires_grad:
                            continue
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(
                                parameter, memory_format=torch.preserve_format
                            )
                        else:
                            parameter.grad.zero_()
                        parameter_count += 1
                        element_count += parameter.numel()

            get_root_logger().info(
                "Preallocated stable CUDA Graph gradient buffers for %d %s "
                "parameters (%d elements) across %d accumulation steps.",
                parameter_count,
                name,
                element_count,
                self.accum_iters,
            )

    def _zero_generator_optimizer(self, optimizer: Optimizer) -> None:
        """Clear generator grads without releasing CUDA Graph-owned storage."""
        _zero_optimizer_gradients(optimizer, self.cudagraph_stable_gradients)
        for parameter in self._ddp_excluded_g_params:
            if self.cudagraph_stable_gradients and parameter.grad is not None:
                parameter.grad.detach().zero_()
            else:
                parameter.grad = None

    def _zero_discriminator_optimizer(self, optimizer: Optimizer) -> None:
        """Clear discriminator grads without releasing CUDA Graph-owned storage."""
        _zero_optimizer_gradients(
            optimizer, self.cudagraph_stable_discriminator_gradients
        )

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.lq = data["lq"].to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )
        if self.is_train and self.net_g_teacher is not None and not self.distillation:
            with torch.inference_mode():
                self.gt = self.net_g_teacher(self.lq).to(
                    memory_format=self.memory_format
                )
        elif "gt" in data:
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )

        # moa
        if self.is_train and self.batch_augment and self.gt is not None:
            self.gt, self.lq = self.batch_augment(self.gt, self.lq)

    def _forward_generator_for_training(
        self, lq: Tensor, target: Tensor, current_iter: int
    ) -> Tensor:
        """Run the generator training graph.

        Specialized models can override this without duplicating the optimizer,
        AMP, EMA, discriminator, or accumulation machinery below.
        """
        del target, current_iter
        output = self.net_g(lq)
        assert isinstance(output, Tensor)
        return output

    def _generator_auxiliary_losses(self) -> dict[str, Tensor]:
        """Return already-weighted generator-only auxiliary losses."""
        return {}

    @staticmethod
    def _distillation_distance(
        prediction: Tensor,
        target: Tensor,
        loss_type: str,
        epsilon: float,
    ) -> Tensor:
        return distillation_distance(prediction, target, loss_type, epsilon)

    @staticmethod
    def _forward_with_distillation_features(
        network: nn.Module, lq: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        return forward_with_distillation_features(network, lq)

    def _distillation_scale(
        self, current_iter: int, start_iter: int, warmup_iters: int
    ) -> float:
        assert self.distillation_opt is not None
        return distillation_scale(
            self.distillation_opt, current_iter, start_iter, warmup_iters
        )

    def _distillation_feature_representation(self, feature: Tensor) -> Tensor:
        assert self.distillation_opt is not None
        return distillation_feature_representation(feature, self.distillation_opt)

    def _feature_distillation_loss(
        self,
        student_features: tuple[Tensor, ...],
        teacher_features: tuple[Tensor, ...],
    ) -> Tensor:
        assert self.distillation_opt is not None
        if len(student_features) != self.distillation_student_feature_count:
            raise ValueError(
                "runtime student feature count does not match the architecture contract"
            )
        if len(teacher_features) != len(self.distillation_feature_pairs):
            raise ValueError(
                "runtime mapped teacher feature count does not match the architecture contract"
            )

        first_student_index = self.distillation_feature_pairs[0][0]
        total = torch.zeros(
            (), device=student_features[first_student_index].device, dtype=torch.float32
        )
        for (
            student_index,
            _teacher_index,
            student_name,
            teacher_name,
        ), teacher_feature in zip(
            self.distillation_feature_pairs,
            teacher_features,
            strict=True,
        ):
            student_feature = student_features[student_index]
            if student_feature.shape[0] != teacher_feature.shape[0] or (
                student_feature.shape[-2:] != teacher_feature.shape[-2:]
            ):
                raise ValueError(
                    "distillation feature batch/spatial shape mismatch for "
                    f"{student_name!r} <- {teacher_name!r}: "
                    f"student={tuple(student_feature.shape)}, "
                    f"teacher={tuple(teacher_feature.shape)}"
                )
            student_representation = self._distillation_feature_representation(
                student_feature
            )
            teacher_representation = self._distillation_feature_representation(
                teacher_feature
            )
            if student_representation.shape != teacher_representation.shape:
                raise ValueError(
                    "distillation feature representation shape mismatch for "
                    f"{student_name!r} <- {teacher_name!r}: "
                    f"student={tuple(student_representation.shape)}, "
                    f"teacher={tuple(teacher_representation.shape)}"
                )
            total = total + self._distillation_distance(
                student_representation,
                teacher_representation,
                self.distillation_opt.feature_loss,
                self.distillation_opt.epsilon,
            )
        return total / len(self.distillation_feature_pairs)

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool, *,
        observe_objective: Callable[[Tensor, Tensor, dict[str, Tensor | float]], None] | None = None,
    ) -> None:
        # 1. Begin an attempt. With G and D, both must pass the finite checks
        #    before either optimizer steps. Eligible G-only updates are gated on device.
        self.begin_training_attempt()
        if getattr(self, "_ldl_ema_forward", None) is not None or (
            (
                self.use_compile
                or (self.compile_discriminator and self.net_d is not None)
            )
            and self.opt.compile_mode in {"max-autotune", "reduce-overhead"}
        ):
            # The trainer intentionally retains outputs and logging tensors
            # across calls. Tell CUDA Graph Trees where the new iteration
            # begins so prior-generation buffers can be reclaimed safely.
            torch.compiler.cudagraph_mark_step_begin()

        self._resolve_deferred_updates(wait=False)
        if self._owed_rejection_reports:
            # This call still trains on its batch, but takes back the
            # iteration of an earlier update found non-finite after train.py
            # counted it.
            self._owed_rejection_reports -= 1
            self.reject_training_step("nonfinite_late_rejection_accounting")

        # https://github.com/Corpsecreate/neosr/blob/2ee3e7fe5ce485e070744158d4e31b8419103db0/neosr/models/default.py#L328
        # assert self.optimizer_g is not None
        assert self.lq is not None
        assert self.gt is not None
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        candidate_iter = current_iter if apply_gradient else current_iter + 1
        multipliers = None
        if self._loss_schedules is not None:
            assert self.opt.train is not None
            if candidate_iter > self.opt.train.total_iter:
                return
            multipliers = {
                label: schedule.factor(candidate_iter)
                for label, schedule in self._loss_schedules.items()
            }
        if self._qat is not None:
            self._qat.begin(candidate_iter)
        buffer_window = self._buffer_window
        if buffer_window is not None:
            if observe_objective is not None and buffer_window.pending:
                # Inspect live microbatch state without disturbing its pending snapshot.
                from traiNNer.models.sr_buffer_window import BufferWindow

                buffer_window = BufferWindow(self.net_g, self.net_d)
            buffer_window.begin()
        gan_active = multipliers is None or (
            "l_g_gan" in self.losses and multipliers.get("l_g_gan", 1.0) > 0
        )
        wavelet = self._gan_wavelet
        if wavelet is not None and (
            not gan_active or candidate_iter < wavelet.start_iter
        ):
            wavelet = None
        skip_d_update = False
        pending_adaptive_d_ema: Tensor | None = None
        pending_vosr_balance_state: tuple[Tensor, Tensor] | None = None
        unscaled_scaler_states: list[tuple[GradScaler, dict[str, Any]]] = []
        generator_ready = False
        discriminator_ready = False
        # Rechecked per call: a resumed non-fused state, a new hook or a changed
        # group leaves the optimizer unable to take the device-side gate.
        deferred_guard = (
            self.deferred_nonfinite_guard
            and self.optimizer_g is not None
            and _fused_adam_fast_path_applies(self.optimizer_g)
        )
        # Deferred guard: (nonfinite loss, nonfinite gradient) on the device.
        nonfinite_loss: Tensor | None = None
        nonfinite_flags: Tensor | None = None
        if gan_active and self.adaptive_d and self._adaptive_d_window_start is None:
            self._adaptive_d_window_start = self.l_g_gan_ema.detach().clone()

        # Freeze D while differentiating the generator objective through it.
        if gan_active and self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples = self.gt.shape[0]
        loss_dict: dict[str, Tensor | float] = OrderedDict()
        d_condition = self._discriminator_condition(self.gt, self.lq) if gan_active else None

        lq = rgb2pixelformat_pt(
            self.lq, self.opt.input_pixel_format
        )  # lq: input_pixel_format

        # 2. Resolve the active teacher objectives before either training forward.
        teacher_output_target: Tensor | None = None
        teacher_feature_targets: tuple[Tensor, ...] | None = None
        output_distillation_scale = 0.0
        feature_distillation_scale = 0.0
        if self.distillation:
            assert self.distillation_opt is not None
            output_distillation_scale = self._distillation_scale(
                candidate_iter,
                self.distillation_opt.start_iter,
                self.distillation_opt.warmup_iters,
            )
            feature_distillation_scale = self._distillation_scale(
                candidate_iter,
                self.distillation_opt.feature_start_iter,
                self.distillation_opt.feature_warmup_iters,
            )
        output_distillation_active = (
            self.distillation
            and self.distillation_opt is not None
            and self.distillation_opt.output_weight > 0
            and output_distillation_scale > 0
        )
        feature_distillation_active = (
            self.distillation
            and self.distillation_opt is not None
            and self.distillation_opt.feature_weight > 0
            and feature_distillation_scale > 0
        )

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if output_distillation_active or feature_distillation_active:
                assert self.net_g_teacher is not None
                with torch.no_grad():
                    if feature_distillation_active:
                        teacher_raw, teacher_features = (
                            self._forward_with_distillation_features(
                                self.net_g_teacher, lq
                            )
                        )
                        if (
                            len(teacher_features)
                            != self.distillation_teacher_feature_count
                        ):
                            raise ValueError(
                                "runtime teacher feature count does not match "
                                "the architecture contract"
                            )
                        teacher_feature_targets = tuple(
                            teacher_features[teacher_index].detach()
                            for (
                                _student_index,
                                teacher_index,
                                _student_name,
                                _teacher_name,
                            ) in self.distillation_feature_pairs
                        )
                    else:
                        teacher_raw = self.net_g_teacher(lq)
                        assert isinstance(teacher_raw, Tensor)
                    if output_distillation_active:
                        teacher_output_target = pixelformat2rgb_pt(
                            teacher_raw, self.gt, self.opt.output_pixel_format
                        ).detach()

            # 3. Accumulate G loss and gradients before committing the optimizer.
            student_features: tuple[Tensor, ...] | None = None
            if self.optimizer_g is None:
                # D saves the frozen generator's image for its own backward.
                with torch.no_grad():
                    output = self.net_g(lq)
            elif feature_distillation_active:
                output, student_features = self._forward_with_distillation_features(
                    self.net_g, lq
                )
            else:
                output = self._forward_generator_for_training(lq, self.gt, current_iter)
            assert isinstance(output, Tensor)
            self.output = pixelformat2rgb_pt(output, self.gt, self.opt.output_pixel_format)
            loss_output = self.output
            if self._loss_output_clamp != "none":
                # Share a single node: the restoring clamp examines the sum of
                # gradients from all losses that consume this prediction.
                loss_output = clamp_loss_output(loss_output, self._loss_output_clamp)
            health = None
            gan_image = loss_output
            if self._gan_health_freq:
                self._gan_health_sample = None
                if (
                    gan_active
                    and apply_gradient
                    and observe_objective is None
                    and candidate_iter % self._gan_health_freq == 0
                ):
                    health = GanHealthSample()
                    gan_image = loss_output.view_as(loss_output)
            adversarial_output = gan_image if wavelet is None else wavelet(gan_image)

            if self.optimizer_g is not None:
                l_g_total = self.output.sum() * 0.0

                lq_target = None
                neosr_lq_target = None

                # Log entropy if hook captured it
                if (
                    hasattr(self, "_feature_entropy")
                    and self._feature_entropy is not None
                ):
                    loss_dict["entropy"] = torch.tensor(
                        self._feature_entropy, device=self.device
                    )

                vosr_distance: Tensor | None = None
                vosr_weight = 0.0
                raw_gan_term: Tensor | None = None
                for label, loss in self.losses.items():
                    loss_weight = loss.loss_weight
                    if multipliers is not None:
                        factor = multipliers.get(label, 1.0)
                        if factor == 0 or (label == "l_g_vosr" and not gan_active):
                            continue
                        if factor != 1.0:
                            loss_weight = loss_weight * factor
                    target = self.gt

                    if self._neosr_lq_labels and label in self._neosr_lq_labels:
                        if neosr_lq_target is None:
                            neosr_lq_target = neosr_lq_reference(self.lq, loss_output)
                        target = neosr_lq_target
                    elif loss.loss_weight < 0:
                        if lq_target is None:
                            with torch.inference_mode():
                                lq_target = torch.clamp(
                                    F.interpolate(
                                        self.lq,
                                        scale_factor=self.opt.scale,
                                        mode="bicubic",
                                        antialias=True,
                                    ),
                                    0,
                                    1,
                                )
                        target = lq_target

                    if label == "l_g_gan":
                        assert self.net_d is not None
                        fake_g_pred = self._discriminate(adversarial_output, d_condition)
                        l_g_loss = loss(fake_g_pred, True, is_disc=False)
                        if multipliers is not None:
                            raw_gan_term = l_g_loss.detach()

                        if self.adaptive_d:
                            gan_observation = l_g_loss.detach()
                            if self.opt.dist:
                                # GANLoss returns a local batch mean. All ranks
                                # must use the same EMA before choosing D's backward.
                                mean = gan_observation.double()
                                totals = torch.stack(
                                    (mean * n_samples, mean.new_tensor(n_samples))
                                )
                                dist.all_reduce(totals)
                                gan_observation = (totals[0] / totals[1]).to(self.l_g_gan_ema)
                            l_g_gan_ema = (
                                self.adaptive_d_ema_decay * self.l_g_gan_ema
                                + (1 - self.adaptive_d_ema_decay) * gan_observation
                            )

                            if (
                                l_g_gan_ema
                                > self.l_g_gan_ema * self.adaptive_d_threshold
                            ):
                                skip_d_update = True
                                assert self.optimizer_d_index is not None
                                self.optimizers_skipped[self.optimizer_d_index] = True
                                # print(current_iter, "skip_d_update")

                            pending_adaptive_d_ema = l_g_gan_ema

                    elif label == "l_g_vosr":
                        # Balance the raw distance after the GAN term is available.
                        vosr_distance = loss(self.output, self.lq)
                        vosr_weight = loss_weight
                        continue
                    elif label == "l_g_ldl":
                        assert self.net_g_ema is not None, (
                            "ema_decay must be enabled for LDL loss"
                        )
                        observer_guard = nullcontext()
                        if self._qat is not None:
                            from traiNNer.models.sr_qat import frozen_observers

                            observer_guard = frozen_observers(self.net_g_ema.ema_model)
                        with torch.inference_mode(), observer_guard:
                            ema_forward = (
                                self._ldl_ema_forward
                                if self._ldl_ema_forward is not None
                                else self.net_g_ema
                            )
                            output_ema = pixelformat2rgb_pt(
                                ema_forward(lq),
                                self.gt,
                                self.opt.output_pixel_format,
                            )
                        l_g_loss = loss(self.output, output_ema, target)
                    else:
                        l_g_loss = loss(loss_output, target)

                    if isinstance(l_g_loss, dict):
                        for sublabel, loss_val in l_g_loss.items():
                            # NaN > 0 is false; do not silently drop a broken loss.
                            weighted_loss_val = loss_val * abs(loss_weight)
                            l_g_total += weighted_loss_val
                            loss_dict[f"{label}_{sublabel}"] = weighted_loss_val
                    else:
                        weighted_l_g_loss = l_g_loss * abs(loss_weight)
                        l_g_total += weighted_l_g_loss
                        loss_dict[label] = weighted_l_g_loss

                if vosr_distance is not None:
                    if multipliers is None:
                        # Preserve the unscheduled path's existing arithmetic.
                        gan_term = cast("Tensor | None", loss_dict.get("l_g_gan"))
                        if gan_term is not None:
                            raw_gan_term = gan_term.detach() / abs(
                                self.losses["l_g_gan"].loss_weight
                            )
                    balanced = balance_vosr(
                        vosr_distance,
                        raw_gan_term,
                        self._vosr_balance_state,
                        vosr_weight,
                    )
                    if balanced is not None:
                        l_g_vosr, vosr_scale, pending_vosr_balance_state = balanced
                        l_g_total += l_g_vosr
                        loss_dict["l_g_vosr"] = l_g_vosr
                        loss_dict["vosr_scale"] = vosr_scale

                if output_distillation_active:
                    assert self.distillation_opt is not None
                    assert teacher_output_target is not None
                    output_distillation = (
                        self._distillation_distance(
                            self.output,
                            teacher_output_target,
                            self.distillation_opt.output_loss,
                            self.distillation_opt.epsilon,
                        )
                        * self.distillation_opt.output_weight
                        * output_distillation_scale
                    )
                    l_g_total += output_distillation
                    loss_dict["l_g_distill_output"] = output_distillation
                    loss_dict["distill_output_scale"] = output_distillation_scale

                if feature_distillation_active:
                    assert self.distillation_opt is not None
                    assert teacher_feature_targets is not None
                    assert student_features is not None
                    feature_distillation = (
                        self._feature_distillation_loss(
                            student_features, teacher_feature_targets
                        )
                        * self.distillation_opt.feature_weight
                        * feature_distillation_scale
                    )
                    l_g_total += feature_distillation
                    loss_dict["l_g_distill_features"] = feature_distillation
                    loss_dict["distill_feature_scale"] = feature_distillation_scale

                for label, auxiliary_loss in self._generator_auxiliary_losses().items():
                    if label in loss_dict:
                        raise RuntimeError(
                            f"duplicate generator auxiliary loss label: {label}"
                        )
                    l_g_total += auxiliary_loss
                    loss_dict[label] = auxiliary_loss

                # A separate diagnostic process can inspect exactly the objective
                # assembled above. It returns before backward, step, or EMA update.
                if observe_objective is not None:
                    try:
                        observe_objective(self.output, l_g_total, loss_dict)
                    finally:
                        if self._qat is not None:
                            self._qat.rollback()
                        if buffer_window is not None:
                            buffer_window.rollback()
                    return

                backward_g_loss = l_g_total / self.accum_iters

                if deferred_guard:
                    # Eager, in this update: the loss may live in CUDA Graph
                    # memory that the next iteration overwrites.
                    nonfinite_loss = ~torch.isfinite(backward_g_loss).all()
                elif not self._finite_on_all_ranks(backward_g_loss):
                    self._reject_nonfinite_update(
                        "nonfinite_generator_loss",
                        candidate_iter,
                        unscaled_scaler_states,
                        loss_dict,
                    )
                    return

                # add total generator loss for tensorboard tracking
                loss_dict["l_g_total"] = l_g_total

                scaled_g_loss = self.scaler_g.scale(backward_g_loss)
                if health is None:
                    scaled_g_loss.backward()
                else:
                    # These terms use the raw output and can bypass the common clamp.
                    compare_other = self._loss_output_clamp == "none" or not any(
                        k in loss_dict for k in ("l_g_vosr", "l_g_ldl", "l_g_distill_output")
                    )
                    health.backward(
                        scaled_g_loss, loss_output, gan_image, compare_other=compare_other
                    )

                if apply_gradient:
                    if self.scaler_g.is_enabled():
                        unscaled_scaler_states.append(
                            (self.scaler_g, self.scaler_g.state_dict())
                        )
                    self.scaler_g.unscale_(self.optimizer_g)
                    if self._spike_clippers is None:
                        grad_norm_g = _get_total_gradient_norm(
                            self._filtered_g_params or self.net_g.parameters()
                        )
                    else:
                        grad_norm_g, threshold_g = self._spike_clippers["g"].prepare(self.optimizer_g)

                    if nonfinite_loss is not None:
                        # A non-finite norm only clips into gradients that the
                        # gated step never applies.
                        nonfinite_flags = torch.stack(
                            (nonfinite_loss, ~torch.isfinite(grad_norm_g))
                        )
                    elif not self._finite_on_all_ranks(grad_norm_g):
                        loss_dict["grad_norm_g"] = grad_norm_g
                        self._reject_nonfinite_update(
                            "nonfinite_generator_gradient",
                            candidate_iter,
                            unscaled_scaler_states,
                            loss_dict,
                        )
                        return

                    if self.grad_clip:
                        clip_grads_with_norm_(
                            self._filtered_g_params or self.net_g.parameters(),
                            self.grad_clip_max_norm,
                            grad_norm_g,
                            foreach=None,
                        )

                    # D can still reject this batch. Defer G until both pass.
                    generator_ready = True

        # 4. Validate D loss and gradients. A rejection here discards staged G work.
        cri_gan = self.losses.get("l_g_gan") if gan_active else None
        if apply_gradient and not gan_active and self.optimizer_d_index is not None:
            self.optimizers_skipped[self.optimizer_d_index] = True

        discriminator_configured = (
            self.net_d is not None
            and cri_gan is not None
            and self.optimizer_d is not None
        )
        if discriminator_configured:
            assert self.net_d is not None
            assert self.optimizer_d is not None
            assert cri_gan is not None

            if not skip_d_update:
                for p in self.net_d.parameters():
                    p.requires_grad = True

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    # real
                    real_input = self.gt.detach()
                    if wavelet is not None:
                        real_input = wavelet(real_input)
                    real_d_pred = self._discriminate(real_input, d_condition)
                    l_d_real = cri_gan(real_d_pred, True, is_disc=True)
                    loss_dict["l_d_real"] = l_d_real
                    loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
                    # fake
                    fake_d_pred = self._discriminate(adversarial_output.detach(), d_condition)
                    l_d_fake = cri_gan(fake_d_pred, False, is_disc=True)
                    loss_dict["l_d_fake"] = l_d_fake
                    loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

                backward_d_loss = (l_d_real + l_d_fake) / self.accum_iters
                if not self._finite_on_all_ranks(backward_d_loss):
                    self._reject_nonfinite_update(
                        "nonfinite_discriminator_loss",
                        candidate_iter,
                        unscaled_scaler_states,
                        loss_dict,
                    )
                    return
                self.scaler_d.scale(backward_d_loss).backward()

                if apply_gradient:
                    if self.scaler_d.is_enabled():
                        unscaled_scaler_states.append(
                            (self.scaler_d, self.scaler_d.state_dict())
                        )
                    self.scaler_d.unscale_(self.optimizer_d)
                    if self._spike_clippers is None:
                        grad_norm_d = _get_total_gradient_norm(self.net_d.parameters())
                    else:
                        grad_norm_d, threshold_d = self._spike_clippers["d"].prepare(self.optimizer_d)
                    if not self._finite_on_all_ranks(grad_norm_d):
                        loss_dict["grad_norm_d"] = grad_norm_d
                        self._reject_nonfinite_update(
                            "nonfinite_discriminator_gradient",
                            candidate_iter,
                            unscaled_scaler_states,
                            loss_dict,
                        )
                        return

                    if self.grad_clip:
                        clip_grads_with_norm_(
                            self.net_d.parameters(),
                            self.grad_clip_max_norm,
                            grad_norm_d,
                            foreach=None,
                        )
                    discriminator_ready = True
            elif apply_gradient:
                # A final adaptive-D skip must not leak partial gradients into
                # the next effective batch.
                self._zero_discriminator_optimizer(self.optimizer_d)

        # 5. Commit the optimizer updates and publish the controller state.
        contributions: dict[str, Tensor | float] = {}
        if apply_gradient and buffer_window is not None:
            buffer_window.check_layout()
        if generator_ready:
            assert self.optimizer_g is not None
            scale_before = self.scaler_g.get_scale()
            self.step_optimizer(
                self.scaler_g,
                self.optimizer_g,
                None
                if nonfinite_flags is None
                else nonfinite_flags.any().to(torch.float32),
            )
            self.scaler_g.update()
            scale_after = self.scaler_g.get_scale()
            self.optimizers_skipped[0] = scale_after < scale_before
            if not self.optimizers_skipped[0]:
                if health is not None:
                    health.gradient_multiplier = self.accum_iters / scale_before
                    self._gan_health_sample = health
                contributions["_optimizer_steps_g"] = 1.0
                contributions["scale_g"] = scale_after
                if self._spike_clippers is None:
                    contributions["grad_norm_g"] = grad_norm_g.float()
                else:
                    self._spike_clippers["g"].commit(grad_norm_g, threshold_g)
            self._zero_generator_optimizer(self.optimizer_g)
        if discriminator_ready:
            assert self.optimizer_d is not None
            scale_before = self.scaler_d.get_scale()
            self.step_optimizer(self.scaler_d, self.optimizer_d)
            self.scaler_d.update()
            scale_after = self.scaler_d.get_scale()
            assert self.optimizer_d_index is not None
            self.optimizers_skipped[self.optimizer_d_index] = scale_after < scale_before
            if not self.optimizers_skipped[self.optimizer_d_index]:
                if health is not None:
                    health.discriminator(cri_gan, real_d_pred, fake_d_pred)
                contributions["_optimizer_steps_d"] = 1.0
                contributions["scale_d"] = scale_after
                if self._spike_clippers is None:
                    contributions["grad_norm_d"] = grad_norm_d.float()
                else:
                    self._spike_clippers["d"].commit(grad_norm_d, threshold_d)
            self._zero_discriminator_optimizer(self.optimizer_d)
        if pending_adaptive_d_ema is not None:
            self.l_g_gan_ema.copy_(pending_adaptive_d_ema)
            loss_dict["adaptive_d_loss_ema"] = pending_adaptive_d_ema
            loss_dict["adaptive_d_skip_fraction"] = float(skip_d_update)
        if pending_vosr_balance_state is not None:
            self._vosr_balance_state = pending_vosr_balance_state
        if apply_gradient:
            if self._qat is not None:
                # QAT is FP32 and requires a G optimizer. Both finite gates passed.
                assert generator_ready
                self._qat.pending = False
            if buffer_window is not None:
                buffer_window.pending = False
            self._adaptive_d_window_start = None
            if nonfinite_flags is None:
                self._report_pending_nonfinite_rejections(candidate_iter)
                self.nan_count = 0

        # 6. Publish detached logs after the finite checks and advance EMA at the boundary.
        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()  # pyright: ignore[reportAttributeAccessIssue]
            )
            contributions[key] = val * n_samples

        if self.accum_iters > 1:
            # A later microbatch can still reject the entire window.
            pending = self._window_log_sums
            if pending is not None:
                for key, value in contributions.items():
                    if key in pending:
                        pending[key] += value
                    else:
                        pending[key] = value
                contributions = pending
            n_samples += self._window_log_samples
            if not apply_gradient:
                self._window_log_sums = contributions
                self._window_log_samples = n_samples
                return
            self._window_log_sums = None
            self._window_log_samples = 0

        if nonfinite_flags is None:
            self.add_log_contributions(n_samples, contributions)
        else:
            # The flags reach the host without waiting; the next calls or a
            # settle point read them once the copy completed.
            flags = torch.empty(2, dtype=torch.bool, pin_memory=True)
            flags.copy_(nonfinite_flags, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
            self._deferred_updates.append(
                _DeferredUpdate(candidate_iter, ready, flags, n_samples, contributions)
            )

        if self.net_g_ema is not None and apply_gradient:
            if isinstance(self.net_g_ema, NeoSREMA) and not generator_ready:
                return
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.update_ema(self.net_g_ema)
                # EMA updates parameters in-place while its model stays in eval
                # mode. Force one fresh inference preparation before the next
                # validation sweep, then reuse it for every dataset and image.
                self._validation_prepared_key = None

    @property
    def validation_amp(self) -> bool:
        return self.use_amp and not (
            self.opt.is_train and self.opt.val is not None and self.opt.val.training_precision == "float32"
        )

    def infer_tiled(self, net: nn.Module, lq: torch.Tensor) -> torch.Tensor:
        """Run the configured overlapping-tile inference policy."""
        assert self.opt.val is not None
        return infer_tiled(
            net,
            lq,
            tile_size=self.opt.val.tile_size,
            tile_overlap=self.opt.val.tile_overlap,
            mode=self.opt.val.tile_mode,
            scale=self.opt.scale,
            device=self.device,
            amp_dtype=self.amp_dtype,
            use_amp=self.validation_amp,
        )

    def _select_inference_net(self, *, unwrap_compiled: bool) -> nn.Module:
        if self.net_g_ema is not None:
            assert isinstance(self.net_g_ema.ema_model, nn.Module)
            return self.net_g_ema.ema_model

        assert isinstance(self.net_g, nn.Module)
        if unwrap_compiled:
            return self.get_bare_model(self.net_g)
        return self.net_g

    @contextmanager
    def _validation_inference_context(self, current_iter: int | None) -> Iterator[None]:
        """Keep one prepared inference network for an entire dataset."""
        if self._qat is None:
            with validation_inference_context(self, current_iter):
                yield
            return

        from traiNNer.models.sr_qat import frozen_observers

        modes = [norm.training for norm in self._qat.norms]
        try:
            with validation_inference_context(self, current_iter):
                assert self._validation_inference_net is not None
                with frozen_observers(self._validation_inference_net):
                    yield
        finally:
            # The shared inference context calls train() on an online generator.
            for norm, mode in zip(self._qat.norms, modes, strict=True):
                norm.train(mode)

    def test(self) -> None:
        inference_net = self._validation_inference_net
        if inference_net is None:
            with self._validation_inference_context(None):
                self.test()
            return
        assert self.lq is not None and self.opt.val is not None
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                                     enabled=self.validation_amp):
            lq = rgb2pixelformat_pt(self.lq, self.opt.input_pixel_format)
            output = (self.infer_tiled(inference_net, lq) if self.opt.val.tile_size > 0
                      else inference_net(lq))
            if self.opt.val.complementary_planes == "input_bicubic":
                self.output = complete_output_from_input(output, self.lq, self.opt.output_pixel_format, self.opt.scale)
            else:
                reference = self.gt
                if self.opt.output_pixel_format in ("y", "uv"):
                    if reference is None:
                        raise ValueError("A target is needed for val.complementary_planes: target. Select input_bicubic to reconstruct RGB from the input alone.")
                    reference = reference.to(output.device)
                self.output = pixelformat2rgb_pt(output, reference, self.opt.output_pixel_format)

    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if self.opt.rank == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        # A persistent NaN/Inf streak still pending stops training here.
        self.settle_training_updates()
        with self._validation_inference_context(current_iter):
            self._nondist_validation(
                dataloader,
                current_iter,
                tb_logger,
                save_img,
                multi_val_datasets,
            )
            if self.best_weight_tracker is not None:
                self.best_weight_tracker.observe(self, self._validation_inference_net, dataloader.dataset.opt.name, current_iter)

    def _nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: "SummaryWriter | None",
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        run_validation(
            self, dataloader, current_iter, tb_logger, save_img, multi_val_datasets
        )

    def _log_validation_metric_values(
        self, current_iter: int, dataset_name: str, tb_logger: "SummaryWriter | None"
    ) -> None:
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric:<5}: {value:7.4f}"
            if len(self.best_metric_results) > 0:
                log_str += (
                    f"\tBest: {self.best_metric_results[dataset_name][metric]['val']:7.4f} @ "
                    f"{self.best_metric_results[dataset_name][metric]['iter']:9,} iter"
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self) -> dict[str, Tensor]:
        assert self.output is not None
        assert self.lq is not None

        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()

        if self.gt is not None:
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def resume_training(self, resume_state: TrainingState) -> None:
        saved = resume_state.get("resume_recipe", {}).get("qat")
        if saved != (self._resume_recipe or {}).get("qat"):
            raise ValueError("Full resume requires the saved Aether QAT policy and phase boundaries.")
        super().resume_training(resume_state)
        if self._qat is not None and resume_state["iter"] > 0:
            self._qat.set_phase(resume_state["iter"])

    def save(self, epoch: int, current_iter: int) -> None:
        window = self._qat or self._buffer_window
        with window.checkpoint_view() if window is not None else nullcontext():
            self._save_sr_checkpoint(epoch, current_iter)

    def _save_sr_checkpoint(self, epoch: int, current_iter: int) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None

        # Resolve pending outcomes before writing checkpoint state.
        self.settle_training_updates()

        if self.net_g_ema is not None:
            assert isinstance(self.net_g_ema.ema_model, nn.Module)
            self.save_network(
                self.net_g_ema.ema_model,
                "net_g_ema",
                self.opt.path.models,
                current_iter,
                "params_ema",
            )

            self.save_network(
                self.net_g, "net_g", self.opt.path.resume_models, current_iter, "params"
            )
        else:
            self.save_network(
                self.net_g, "net_g", self.opt.path.models, current_iter, "params"
            )

        if self.net_d is not None:
            self.save_network(
                self.net_d, "net_d", self.opt.path.resume_models, current_iter, "params"
            )

        self.save_training_state(epoch, current_iter)

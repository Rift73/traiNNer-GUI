import warnings
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_, clip_grads_with_norm_, get_total_norm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import TqdmExperimentalWarning
from trainner_config.objectives import (
    LossSchedule,
    Objective,
    objective_schedules,
    resolve_objectives,
)

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.archs.autoencoder_arch import AutoEncoder
from traiNNer.losses import build_loss
from traiNNer.models.base_model import BaseModel
from traiNNer.training.schedule_free_weights import evaluation_view, preserve_module_modes
from traiNNer.training_recovery import (
    MAX_CONSECUTIVE_NONFINITE_ATTEMPTS,
    PersistentNonfiniteError,
)
from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class AEModel(BaseModel):
    """Base SR model for single image super-resolution."""

    _loss_schedules: dict[str, LossSchedule] | None = None

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
        self.net_ae_ema: EMA | None = None
        self._validation_network: nn.Module | None = None

        objectives = resolve_objectives(opt, model="ae") if self.is_train else {}
        self._loss_schedules = objective_schedules(objectives)

        # define network
        assert opt.network_ae is not None
        self.net_ae: AutoEncoder = build_network({**opt.network_ae, "scale": opt.scale})  # pyright: ignore[reportAttributeAccessIssue]

        # load pretrained models
        if self.opt.path.pretrain_network_ae is not None:
            self.load_network(
                self.net_ae,
                self.opt.path.pretrain_network_ae,
            )
        elif self.opt.path.pretrain_network_ae_decoder is not None:
            self.load_network(
                self.net_ae.decoder,
                self.opt.path.pretrain_network_ae_decoder,
            )

        self.net_ae = self.model_to_device(self.net_ae, compile=self.use_compile)  # pyright: ignore[reportAttributeAccessIssue]

        self.gt: Tensor | None = None
        self.lq: Tensor | None = None
        self.output_gt: Tensor | None = None
        self.output_lq: Tensor | None = None
        logger = get_root_logger()

        if self.use_amp and self.device.type == "cuda":
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 was enabled for AMP but the current GPU does not support bf16. Falling back to float16 for AMP. Disable bf16 to hide this warning (amp_bf16: false)."
                )
                self.amp_dtype = torch.float16

            network_ae_name = opt.network_ae["type"]
            if (
                self.amp_dtype == torch.float16
                and network_ae_name.lower() in ARCHS_WITHOUT_FP16
            ):
                if torch.cuda.is_bf16_supported():
                    logger.warning(
                        "AMP with fp16 was enabled but network_ae [bold]%s[/bold] does not support fp16. Falling back to bf16.",
                        network_ae_name,
                        extra={"markup": True},
                    )
                    self.amp_dtype = torch.bfloat16
                else:
                    logger.warning(
                        "AMP with fp16 was enabled but network_ae [bold]%s[/bold] does not support fp16. Disabling AMP.",
                        network_ae_name,
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
            self.losses = {}

            self.ema_decay = 0
            self.ema_switch_iter = None

            self.optimizer_ae: Optimizer | None = None

            self.init_training_settings(objectives)

    def init_training_settings(self, objectives: dict[str, Objective]) -> None:
        self.net_ae.train()

        train_opt = self.opt.train
        assert train_opt is not None

        logger = get_root_logger()

        self.scaler_ae = GradScaler(enabled=self.use_amp and self.device.type == "cuda", device=self.device.type)

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.ema_decay = train_opt.ema_decay
        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )
            assert self.opt.network_ae is not None

            init_net_ae_ema: AutoEncoder | None = None

            # load pretrained model
            if self.opt.path.pretrain_network_ae_ema is not None:
                init_net_ae_ema = build_network(
                    {**self.opt.network_ae, "scale": self.opt.scale}
                )  # pyright: ignore[reportAssignmentType]
                assert init_net_ae_ema is not None
                self.load_network(
                    init_net_ae_ema,
                    self.opt.path.pretrain_network_ae_ema,
                    True,
                    "params_ema",
                )

            # define network net_ae with Exponential Moving Average (EMA)
            # net_ae_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_ae_ema = EMA(
                self.get_bare_model(self.net_ae),
                ema_model=init_net_ae_ema,
                beta=self.ema_decay,
                allow_different_devices=True,
                update_after_step=train_opt.ema_update_after_step,
                update_every=1,
                power=10 if train_opt.ema_power is None else train_opt.ema_power,
                update_model_with_ema_every=self.ema_switch_iter,
            ).to(device=self.device, memory_format=self.memory_format)  # pyright: ignore[reportCallIssue]

            assert self.net_ae_ema is not None
            self.net_ae_ema.step = self.net_ae_ema.step.to(device=torch.device("cpu"))

        self.grad_clip = train_opt.grad_clip
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

        assert self.losses, "At least one loss must be defined."

        # setup batch augmentations
        self.setup_batchaug()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self) -> None:
        train_opt = self.opt.train
        assert train_opt is not None
        assert train_opt.optim_ae is not None
        assert self.net_ae is not None
        optim_params = []
        for k, v in self.net_ae.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning("Params %s will not be optimized.", k)

        self.optimizer_ae = self.get_optimizer(optim_params, train_opt.optim_ae)
        self.optimizers.append(self.optimizer_ae)
        self.optimizers_skipped.append(False)
        self.optimizers_schedule_free.append(
            "SCHEDULEFREE" in train_opt.optim_ae["type"].upper()
        )

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.lq = data["lq"].to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )
        if "gt" in data:
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            if self.is_train and self.batch_augment is not None:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)
        else:
            self.gt = None

    def _reject_scheduled_update(self, candidate_iter: int, reason: str) -> None:
        assert self.optimizer_ae is not None
        self.output_gt = None
        self.optimizer_ae.zero_grad()
        self.optimizers_skipped[0] = True
        self.reject_training_step(reason)
        self.nan_count += 1
        limit_reached = self.nan_count >= MAX_CONSECUTIVE_NONFINITE_ATTEMPTS
        if self.nan_count == 1 or limit_reached:
            get_root_logger().warning(
                "Rejected AE candidate iter %d: %s", candidate_iter, reason
            )
        if limit_reached:
            raise PersistentNonfiniteError(reason, candidate_iter, self.nan_count)

    def optimize_parameters(
        self,
        current_iter: int,
        current_accum_iter: int,
        apply_gradient: bool,
    ) -> None:
        assert self.optimizer_ae is not None
        assert self.gt is not None
        assert self.scaler_ae is not None
        assert self.net_ae is not None

        n_samples = self.gt.shape[0]
        candidate_iter = current_iter if apply_gradient else current_iter + 1
        multipliers = None
        if self._loss_schedules is not None:
            assert self.opt.train is not None
            self.begin_training_attempt()
            if candidate_iter > self.opt.train.total_iter:
                return
            multipliers = {
                label: schedule.factor(candidate_iter)
                for label, schedule in self._loss_schedules.items()
            }
        else:
            self.loss_samples += n_samples

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            self.output_gt = self.net_ae(self.gt)
            assert isinstance(self.output_gt, Tensor)
            l_ae_total = torch.tensor(0.0, device=self.output_gt.device)
            loss_dict = OrderedDict()

            for label, loss in self.losses.items():
                weight = loss.loss_weight
                if multipliers is not None:
                    factor = multipliers.get(label, 1.0)
                    if factor == 0:
                        continue
                    if factor != 1.0:
                        weight = weight * factor
                # for output_type in ["gt", "lq"]:
                for output_type in ["gt"]:  # TODO refactor
                    l_ae_loss = (
                        loss(
                            getattr(self, f"output_{output_type}"),
                            getattr(self, output_type),
                        )
                        * weight
                    )
                    l_ae_total += l_ae_loss / self.accum_iters
                    loss_dict[f"{label}_{output_type}"] = l_ae_loss

            # add total generator loss for tensorboard tracking
            loss_dict["l_ae_total"] = l_ae_total

        if multipliers is not None and not bool(torch.isfinite(l_ae_total).item()):
            self._reject_scheduled_update(candidate_iter, "nonfinite_autoencoder_loss")
            return

        self.scaler_ae.scale(l_ae_total).backward()
        if apply_gradient:
            if multipliers is not None:
                scaler_state = (
                    self.scaler_ae.state_dict() if self.scaler_ae.is_enabled() else None
                )
                self.scaler_ae.unscale_(self.optimizer_ae)
                gradients = [
                    parameter.grad
                    for parameter in self.net_ae.parameters()
                    if parameter.grad is not None
                ]
                grad_norm = get_total_norm(gradients)
                if not bool(torch.isfinite(grad_norm).item()):
                    if scaler_state is not None:
                        self.scaler_ae.update()
                        if self.scaler_ae.get_scale() >= scaler_state["scale"]:
                            self.scaler_ae.load_state_dict(scaler_state)
                    self._reject_scheduled_update(
                        candidate_iter, "nonfinite_autoencoder_gradient"
                    )
                    return
                if self.grad_clip:
                    clip_grads_with_norm_(
                        self.net_ae.parameters(), self.grad_clip_max_norm, grad_norm
                    )
            elif self.grad_clip:
                self.scaler_ae.unscale_(self.optimizer_ae)
                clip_grad_norm_(self.net_ae.parameters(), self.grad_clip_max_norm)

            scale_before = self.scaler_ae.get_scale()
            self.scaler_ae.step(self.optimizer_ae)
            self.scaler_ae.update()
            self.optimizers_skipped[0] = self.scaler_ae.get_scale() < scale_before
            self.optimizer_ae.zero_grad()
            if multipliers is not None:
                if self.optimizers_skipped[0]:
                    self._reject_scheduled_update(
                        candidate_iter, "autoencoder_scaler_skipped"
                    )
                    return
                self.nan_count = 0

        if multipliers is not None:
            self.loss_samples += n_samples
        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        if self.net_ae_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_ae_ema.update()

    def _inference_network(self) -> nn.Module:
        trained = self.net_ae_ema.ema_model if self.net_ae_ema is not None else self.net_ae
        assert trained is not None
        return trained

    @contextmanager
    def _prepared_inference(self):
        network = self._inference_network()
        optimizer = self.optimizer_ae if self.opt.is_train and self.net_ae_ema is None else None
        with preserve_module_modes(network), evaluation_view(optimizer):
            network.eval()
            yield network

    def test(self) -> None:
        assert self.gt is not None
        context = (nullcontext(self._validation_network) if self._validation_network is not None
                   else self._prepared_inference())
        with context as trained, torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            network = self.get_bare_model(trained)
            assert isinstance(network, AutoEncoder)
            self.output_lq = network.encode(self.gt)
            self.output_gt = network.decode(self.output_lq)

    def validate_validation_datasets(self, val_loaders) -> None:
        from traiNNer.data.paired_image_dataset import PairedImageDataset
        from traiNNer.data.single_image_dataset import SingleImageDataset

        for loader in val_loaders:
            if not isinstance(loader.dataset, (PairedImageDataset, SingleImageDataset)):
                raise ValueError("Autoencoder evaluation expects paired images or single images to reconstruct.")

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
        from traiNNer.models.sr_validation import run_validation

        previous = self.is_train
        self.is_train = False
        try:
            with self._prepared_inference() as network:
                self._validation_network = network
                observation = nullcontext()
                if self.opt.is_train and self.opt.val and self.opt.val.record_samples:
                    from traiNNer.training.checkpoint_association import observe_inference

                    observation = observe_inference(self, self.get_bare_model(network), current_iter)
                with observation:
                    run_validation(self, dataloader, current_iter, tb_logger, save_img, multi_val_datasets,
                                   reconstruction=True)
                    if self.best_weight_tracker is not None:
                        self.best_weight_tracker.observe(self, network, dataloader.dataset.opt.name, current_iter)
        finally:
            self._validation_network = None
            self.is_train = previous

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
        assert self.output_lq is not None
        assert self.output_gt is not None
        assert self.gt is not None

        out_dict = OrderedDict()
        out_dict["gt"] = self.gt.detach().cpu()
        out_dict["result_lq"] = self.output_lq.detach().cpu()
        out_dict["result_gt"] = self.output_gt.detach().cpu()

        return out_dict

    def save(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None
        assert self.net_ae is not None

        if self.net_ae_ema is not None:
            self.save_network(
                self.net_ae_ema.ema_model,
                "net_ae_ema",
                self.opt.path.models,
                current_iter,
                "params_ema",
            )

            self.save_network(
                self.net_ae,
                "net_ae",
                self.opt.path.resume_models,
                current_iter,
                "params",
            )
        else:
            self.save_network(
                self.net_ae, "net_ae", self.opt.path.models, current_iter, "params"
            )

        self.save_training_state(epoch, current_iter)

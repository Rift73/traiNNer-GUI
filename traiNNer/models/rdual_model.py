"""Training of the recurrent video model RDUAL on clips with on-the-fly degradations.

A batch is B clips of T frames (gt B,T,3,H,W and shot_ids B,T). feed_data runs
the image degradation chain once per clip in clip mode: one set of blur kernels
and one draw of every setting per clip, video codecs predicting between frames,
no training pair pool. optimize_parameters trains the generator through
RDUAL.backward_clip(), whose frozen NeuFlow v2 (part of the model and its
checkpoint) computes the flow guidance of the degraded clip, with the image
loss stack per frame, averaged over the
clip, then accumulates the discriminator's gradients over every frame of the
clip. Both optimizers step once per iteration.
"""

import functools
import os
from collections import OrderedDict
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional
from torch.nn.utils import clip_grads_with_norm_
from trainner_config.objectives import ObjectiveConfigError, validate_objectives

from traiNNer.archs.rdual_arch import RDUAL
from traiNNer.data.clip_hr import hr_clip
from traiNNer.data.realesrgan_dataset import sample_degradation_kernels
from traiNNer.losses.vosr_loss import VOSRLoss, vosr_balance_state
from traiNNer.models.realesrgan_model import DegradingBatch, RealESRGANModel
from traiNNer.models.sr_model import _get_total_gradient_norm
from traiNNer.utils.lq_condition import LQCondition
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed

# Upstream NeuFlow v2 weights for an RDUAL checkpoint that has no flow.*
NEUFLOW_PRETRAIN_PATH = "experiments/pretrained_models/neuflow_v2.safetensors"

# Encoder and gradient frames per call inside backward_clip
ENCODE_CHUNK = 4
GRAD_CHUNK = 2

# vosrloss on clips: a pixel of a reuse frame is still when the largest channel
# difference between its true frame and the key frame's, averaged over the box of
# STILL_MARGIN pixels around it, is at most STILL_TOLERANCE (grain averages out in
# the box, a real change does not); a reuse frame that is still over less than
# STILL_MIN_SHARE of its area is dropped.
STILL_TOLERANCE = 2 / 255
STILL_MARGIN = 4
STILL_MIN_SHARE = 0.125

# vosrloss's plan for one clip: the loss, per frame the participating clips and
# their still masks (N,1,H,W; ones for key frames), the key frames' restorations
# (B,3,H,W) and the scale
VosrPlan = tuple[VOSRLoss, list[Tensor], list[Tensor], Tensor, Tensor]


def check_clip_training_options(opt: ReduxOptions) -> None:
    """Reject settings the clip training does not support."""
    problems = clip_training_problems(opt)
    if problems:
        raise ValueError(
            "RDUAL clip training (network_g type rdual): " + "; ".join(problems)
        )


def clip_training_problems(opt: ReduxOptions) -> list[str]:
    """The settings of opt the clip training does not support."""
    problems = []
    if opt.val is not None and (opt.val.val_enabled or opt.is_train is False):
        if opt.val.tile_size:
            problems.append("recurrent clip validation requires full frames (val.tile_size: 0)")
        for phase, dataset in opt.datasets.items():
            if phase.split("_", 1)[0] not in ("val", "test"):
                continue
            if dataset.type.lower() not in ("pairedvideodataset", "singlevideodataset"):
                problems.append(f"datasets.{phase} requires a paired or single video frame sequence")
            if dataset.clip_size is None or dataset.clip_size < 1:
                problems.append(f"datasets.{phase}.clip_size must be a positive window length")
    if opt.is_train is False:
        if opt.input_pixel_format != "rgb" or opt.output_pixel_format != "rgb":
            problems.append("recurrent evaluation requires RGB inputs and outputs")
        return problems
    if not opt.high_order_degradation:
        problems.append(
            "high_order_degradation must be true (the LR clips are made on the fly)"
        )
    if opt.input_pixel_format != "rgb" or opt.output_pixel_format != "rgb":
        problems.append("input_pixel_format and output_pixel_format must be rgb")
    if opt.dist or (isinstance(opt.num_gpu, int) and opt.num_gpu > 1):
        problems.append("training runs on one GPU")
    if opt.otf_hf_noise_prob > 0:
        problems.append("otf_hf_noise_prob must be 0 (it draws per frame)")
    if opt.network_g_teacher is not None:
        problems.append("network_g_teacher is not supported")
    if opt.dvd_prob > 0:
        problems.append(
            "dvd_prob must be 0 (the DVD response needs real image context "
            "around the patch, which clip crops do not carry)"
        )
    train = opt.train
    if train is not None:
        if train.use_moa:
            problems.append("train.use_moa (batch augmentation) is not supported")
        if train.vosr_target is not None:
            problems.append("train.vosr_target is not supported")
        if train.distillation is not None and train.distillation.enabled:
            problems.append("train.distillation is not supported")
        try:
            validate_objectives(opt)
        except ObjectiveConfigError as exc:
            problems.append(str(exc))
    return problems


class RDUALModel(RealESRGANModel):
    clip_mode = True
    check_options = staticmethod(check_clip_training_options)
    # A clip training step takes longer than the codec round trips of a clip
    # batch, so each wave of jobs needs one step.
    ahead_stride = 1
    # The jobs of a batch's clips run side by side in degradation_pool.
    pooled_feed = True

    def __init__(self, opt: ReduxOptions) -> None:
        self.check_options(opt)
        super().__init__(opt)
        self.shot_ids: Tensor | None = None
        compile = (
            functools.partial(
                torch.compile,
                mode=self.opt.compile_mode,
                fullgraph=True,
                dynamic=False,
            )
            if self.use_compile
            else None
        )
        self.step_functions = self.prepare_generator(compile)

    def prepare_generator(self, compile: Callable[[Callable], Callable] | None) -> Any:
        """Ready the generator once its pretrain is loaded; returns its step functions."""
        bare = self.get_bare_model(self.net_g)
        assert isinstance(bare, RDUAL)
        # The generator and its EMA copy exist and the pretrain is loaded: give
        # every copy whose checkpoint lacked the frozen flow the upstream weights.
        networks = [bare]
        if self.net_g_ema is not None:
            networks.append(self.net_g_ema.ema_model)
        missing = [net for net in networks if not bool(net.flow_loaded)]
        if missing:
            if not os.path.exists(NEUFLOW_PRETRAIN_PATH):
                raise FileNotFoundError(
                    "The RDUAL checkpoint has no NeuFlow v2 weights and "
                    f"{NEUFLOW_PRETRAIN_PATH} does not exist"
                )
            flow_weights = load_file(NEUFLOW_PRETRAIN_PATH, device="cpu")
            for net in missing:
                net.load_flow_weights(flow_weights)
        return bare.step_functions(compile)

    def model_to_device(
        self,
        net: nn.Module,
        compile: bool = False,
        wrap_distributed: bool = True,
    ) -> nn.Module:
        # RDUAL compiles its per-frame step functions, never the whole network.
        if isinstance(net, RDUAL):
            compile = False
        return super().model_to_device(net, compile, wrap_distributed)

    def prepare_network_for_compile(self, net: nn.Module) -> None:
        bare = self.get_bare_model(net)
        if not isinstance(bare, RDUAL):
            super().prepare_network_for_compile(net)
            return
        # Only the frame size matters; any clip length gives the same preparation.
        train_dataset = self.opt.datasets["train"]
        input_shape = None
        if (
            train_dataset.batch_size_per_gpu is not None
            and train_dataset.lq_size is not None
        ):
            input_shape = (
                train_dataset.batch_size_per_gpu,
                train_dataset.clip_lengths[0],
                3,
                train_dataset.lq_size,
                train_dataset.lq_size,
            )
        bare.prepare_for_compile(
            input_shape=input_shape,
            input_dtype=self.amp_dtype if self.use_amp else torch.float32,
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        """Clips do not use the training pair pool."""

    def feed_data(self, data: DataFeed) -> None:
        """Degrade each clip of data["gt"] (B,T,3,H,W) on its own, as one batch of T frames.

        Without "gt", each clip's HR is made on the device from its YUV window
        (gt_y, gt_u, gt_v, gt_geometry, as VideoClipDataset sends them). A batch
        that DegradeAheadPrefetcher already degraded (it carries "lq") is only
        installed.

        The clips are degraded side by side: each clip's chain runs on this
        thread up to its next video codec, then the codec round trips of all
        waiting clips run at the same time in degradation_pool, and the clips
        go on in clip order. Every random draw stays on this thread in a fixed
        order, so a seed gives the same clips whatever the threads' timing.
        With several clips the draws are taken stage by stage across the clips,
        not one whole clip after the other; with one clip they come in the
        order of image training.
        """
        if not self.is_train:
            # Validation clips already have their observed low-quality frames.
            # Keep T intact: channels_last applies to each BCHW frame, not BTCHW.
            clip = data.get("lq")
            if clip is None or clip.ndim != 5 or clip.size(2) != 3:
                raise ValueError("Recurrent validation needs a batch of RGB frame sequences.")
            self.lq = clip.to(self.device, non_blocking=True)
            target = data.get("gt")
            self.gt = target.to(self.device, non_blocking=True) if target is not None else None
            return
        super().feed_data(data)

    @torch.inference_mode()
    def test(self) -> None:
        network = self._validation_inference_net
        if network is None or self.lq is None:
            raise RuntimeError("Prepare the recurrent validation network and clip before evaluation.")
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            prediction = network(self.lq)
        # Each validation sample owns one named centre frame and its full scene
        # window. Match the loader's convention for both odd and even windows.
        self.output = prediction[:, self.lq.shape[1] // 2]
        if self.gt is not None and self.output.shape != self.gt.shape:
            raise ValueError(
                f"Centre-frame prediction {tuple(self.output.shape)} and target "
                f"{tuple(self.gt.shape)} have different dimensions."
            )

    def start_batch(self, data: DataFeed, pooled: bool = True) -> DegradingBatch:
        """Begin degrading a batch of clips: make its HR clips, then run each
        clip, with its kernels, up to its first job; that wave is started."""
        train_dataset = self.opt.datasets["train"]
        if "gt" in data:
            clips = data["gt"]
        else:
            clips = torch.stack(
                [
                    hr_clip(
                        y.to(self.device, non_blocking=True),
                        u.to(self.device, non_blocking=True),
                        v.to(self.device, non_blocking=True),
                        geometry,
                        train_dataset,
                    )
                    for y, u, v, geometry in zip(
                        data["gt_y"],
                        data["gt_u"],
                        data["gt_v"],
                        data["gt_geometry"],
                        strict=True,
                    )
                ]
            )
        frames = clips.shape[1]
        batch = DegradingBatch(pooled=pooled, carry={"shot_ids": data["shot_ids"]})
        for clip in clips:
            kernel1, kernel2, sinc_kernel = sample_degradation_kernels(train_dataset)
            # One kernel for every frame: views, copied only if a stage uploads one.
            self._add_chain(
                batch,
                self._feed_steps(
                    {
                        "gt": clip,
                        "kernel1": kernel1.expand(frames, -1, -1),
                        "kernel2": kernel2.expand(frames, -1, -1),
                        "sinc_kernel": sinc_kernel.expand(frames, -1, -1),
                    }
                ),
            )
        self._submit_wave(batch)
        return batch

    def install_batch(self, data: DataFeed) -> None:
        super().install_batch(data)
        self.shot_ids = data["shot_ids"]

    def _delivered(self, batch: DegradingBatch) -> DataFeed:
        """The clips' lq and gt (B,T,3,H,W) and the batch's shot_ids."""
        count = len(batch.chains)
        return {
            "lq": torch.stack([batch.done[i]["lq"] for i in range(count)]),
            "gt": torch.stack([batch.done[i]["gt"] for i in range(count)]),
            "shot_ids": batch.carry["shot_ids"].to(self.device, non_blocking=True),
        }

    def _autocast(self) -> AbstractContextManager:
        return torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        )

    def _frame_condition(self, t: int) -> Tensor | None:
        """Discriminator conditioning of frame t, as image training builds it."""
        assert self.lq is not None
        assert self.gt is not None
        if self.d_conditioner is None:
            return None
        lq, gt = self.lq[:, t], self.gt[:, t]
        if isinstance(self.d_conditioner, LQCondition):
            return self.d_conditioner(lq, (gt.shape[-2], gt.shape[-1]))
        if self.d_input_conditioning:
            return self.d_conditioner(gt, lq)
        return self.d_conditioner(gt)

    def _generator_frame_loss(
        self,
        frames: int,
        loss_sums: dict[str, Tensor],
        gan_terms: list[Tensor],
        vosr_plan: VosrPlan | None = None,
        vosr_shares: list[Tensor] | None = None,
        multipliers: dict[str, float] | None = None,
    ) -> Callable[[Tensor, int], Tensor]:
        """The image generator loss stack for frame t, a 1/T share of the clip's loss.

        With vosrloss, the clips that train frame t against VOSR (vosr_plan) add
        the scaled distance, and each frame's raw share goes to vosr_shares.
        """
        assert self.lq is not None
        assert self.gt is not None
        lq, gt = self.lq, self.gt

        def add(name: str, value: Tensor) -> None:
            share = value.detach() / frames
            loss_sums[name] = loss_sums[name] + share if name in loss_sums else share

        def loss(output: Tensor, t: int) -> Tensor:
            total = output.sum() * 0.0
            lq_target = None
            for label, criterion in self.losses.items():
                if label == "l_g_vosr":
                    continue
                weight = abs(criterion.loss_weight)
                if multipliers is not None:
                    factor = multipliers.get(label, 1.0)
                    if factor == 0:
                        continue
                    if factor != 1.0:
                        weight = weight * factor
                target = gt[:, t]
                if criterion.loss_weight < 0:
                    if lq_target is None:
                        with torch.no_grad():
                            lq_target = torch.clamp(
                                functional.interpolate(
                                    lq[:, t],
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
                    prediction = self.net_d(
                        self._discriminator_input(output, self._frame_condition(t))
                    )
                    value = criterion(prediction, True, is_disc=False)
                    gan_terms.append(value.detach())
                else:
                    value = criterion(output, target)
                if isinstance(value, dict):
                    for sublabel, part in value.items():
                        total = total + part * weight
                        add(f"{label}_{sublabel}", part * weight)
                else:
                    total = total + value * weight
                    add(label, value * weight)
            if vosr_plan is not None and vosr_plan[1][t].numel() > 0:
                assert vosr_shares is not None
                vosr, frame_clips, frame_masks, vosr_key, vosr_scale = vosr_plan
                clips = frame_clips[t]
                out = output[clips]
                mask = frame_masks[t]
                # Where the picture moved the target is the output itself: no pull
                target = mask * vosr_key[clips] + (1 - mask) * out.detach()
                share = vosr.distance(out, target) * (clips.numel() / output.shape[0])
                vosr_shares.append(share.detach())
                total = total + vosr_scale * share
                add("l_g_vosr", vosr_scale * share)
            return self.scaler_g.scale(total / frames / self.accum_iters)

        return loss

    @torch.no_grad()
    def _vosr_plan(self, vosr: nn.Module, multiplier: float = 1.0) -> VosrPlan:
        """Which clips train each frame against VOSR, with their still masks, the
        key frames' restorations and the scale; one host sync, the teacher once."""
        assert isinstance(vosr, VOSRLoss)
        assert self.lq is not None
        assert self.gt is not None
        assert self.shot_ids is not None
        n_samples, frames = self.gt.shape[:2]
        lq, gt = self.lq.float(), self.gt.float()
        # How well each frame survived the degradation
        gt_small = functional.interpolate(
            gt.flatten(0, 1), size=lq.shape[-2:], mode="bicubic", antialias=True
        ).view_as(lq)
        err = (lq - gt_small).abs().mean((2, 3, 4))
        # A uniformly drawn frame picks the key shot; its best frame is the key
        clips = torch.arange(n_samples, device=self.device)
        drawn = torch.randint(frames, (n_samples, 1), device=self.device)
        in_shot = self.shot_ids == self.shot_ids.gather(1, drawn)
        key = err.masked_fill(~in_shot, torch.inf).argmin(1)
        # Up to still_frames other frames of the key shot, without replacement
        count = min(vosr.still_frames, frames - 1)
        candidates = in_shot & ~functional.one_hot(key, frames).bool()
        draw = torch.rand(n_samples, frames, device=self.device)
        picked = draw.masked_fill(~candidates, -1.0).topk(count).indices
        # Still where the box mean of the true frames' difference stays small
        change = (gt[clips[:, None], picked] - gt[clips, key][:, None]).abs().amax(2)
        local = functional.avg_pool2d(
            change.flatten(0, 1)[:, None],
            2 * STILL_MARGIN + 1,
            stride=1,
            padding=STILL_MARGIN,
            count_include_pad=False,
        )
        still = (local <= STILL_TOLERANCE).float().view(*picked.shape, *local.shape[1:])
        kept = candidates.gather(1, picked) & (still.mean((2, 3, 4)) >= STILL_MIN_SHARE)
        table = torch.cat((key[:, None], picked, kept.long()), 1).cpu()
        key_table, picked_table, kept_table = table.split((1, count, count), 1)
        frame_clips: list[Tensor] = []
        frame_masks: list[Tensor] = []
        for t in range(frames):
            key_clips = (key_table[:, 0] == t).nonzero().flatten()
            reuse = ((picked_table == t) & kept_table.bool()).nonzero()
            reuse_clips = reuse[:, 0].to(self.device, non_blocking=True)
            reuse_slots = reuse[:, 1].to(self.device, non_blocking=True)
            frame_clips.append(
                torch.cat((key_clips.to(self.device, non_blocking=True), reuse_clips))
            )
            frame_masks.append(
                torch.cat(
                    (
                        still.new_ones(len(key_clips), *still.shape[2:]),
                        still[reuse_clips, reuse_slots],
                    )
                )
            )
        # Teacher once, on the key frames, before any CUDA-graph step; the clone is
        # a normal tensor, safe for autograd to save
        vosr_key = vosr.teacher.restore(self.lq[clips, key], self.opt.scale).clone()
        state = self._vosr_balance_state
        weight = vosr.loss_weight
        if multiplier != 1.0:
            weight = weight * multiplier
        vosr_scale = (
            weight * state[0] / state[1]
            if state is not None
            else torch.zeros((), device=self.device)
        )
        return vosr, frame_clips, frame_masks, vosr_key, vosr_scale

    def backward_clip(self, loss: Callable[[Tensor, int], Tensor]) -> Tensor:
        """The generator's pass over the clip, accumulating the gradients of loss."""
        assert self.lq is not None
        bare = self.get_bare_model(self.net_g)
        assert isinstance(bare, RDUAL)
        # guidance None: RDUAL's frozen NeuFlow v2 computes it from the clip
        return bare.backward_clip(
            self.lq,
            None,
            loss,
            shot_ids=self.shot_ids,
            keep=self.opt.clip_keep,
            functions=self.step_functions,
            autocast=self._autocast,
            encode_chunk=ENCODE_CHUNK,
            grad_chunk=GRAD_CHUNK,
        )

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        del current_accum_iter
        self.begin_training_attempt()
        assert self.lq is not None
        assert self.gt is not None
        assert self.shot_ids is not None
        assert self.optimizer_g is not None
        assert self.scaler_d is not None
        assert self.scaler_g is not None
        uses_cudagraphs = self.opt.compile_mode in {"max-autotune", "reduce-overhead"}

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
        gan_active = multipliers is None or (
            "l_g_gan" in self.losses and multipliers.get("l_g_gan", 1.0) > 0
        )
        skip_d_update = False
        pending_adaptive_d_ema: Tensor | None = None
        unscaled_scaler_states: list[tuple[GradScaler, dict]] = []
        generator_ready = False
        discriminator_ready = False
        if gan_active and self.adaptive_d and self._adaptive_d_window_start is None:
            self._adaptive_d_window_start = self.l_g_gan_ema.detach().clone()

        if gan_active and self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples, frames = self.gt.shape[:2]
        loss_dict: dict[str, Tensor | float] = OrderedDict()
        vosr = self.losses.get("l_g_vosr") if gan_active else None
        vosr_factor = 1.0 if multipliers is None else multipliers.get("l_g_vosr", 1.0)
        vosr_plan = (
            self._vosr_plan(vosr, vosr_factor)
            if vosr is not None and vosr_factor > 0
            else None
        )
        vosr_shares: list[Tensor] = []

        # ----------------------------- generator ----------------------------- #
        loss_sums: dict[str, Tensor] = {}
        gan_terms: list[Tensor] = []
        self.output = self.backward_clip(
            self._generator_frame_loss(
                frames, loss_sums, gan_terms, vosr_plan, vosr_shares, multipliers
            )
        )
        loss_dict.update(loss_sums)
        l_g_total = sum(loss_sums.values(), torch.zeros((), device=self.device))
        loss_dict["l_g_total"] = l_g_total / self.accum_iters
        if not bool(torch.isfinite(l_g_total).item()):
            self._reject_nonfinite_update(
                "nonfinite_generator_loss",
                candidate_iter,
                unscaled_scaler_states,
                loss_dict,
            )
            return

        if gan_terms and self.adaptive_d:
            # The controller sees the clip's mean GAN term.
            l_g_gan_ema = (
                self.adaptive_d_ema_decay * self.l_g_gan_ema
                + (1 - self.adaptive_d_ema_decay) * torch.stack(gan_terms).mean()
            )
            if l_g_gan_ema > self.l_g_gan_ema * self.adaptive_d_threshold:
                skip_d_update = True
                assert self.optimizer_d_index is not None
                self.optimizers_skipped[self.optimizer_d_index] = True
            pending_adaptive_d_ema = l_g_gan_ema

        pending_vosr_balance_state: tuple[Tensor, Tensor] | None = None
        if vosr_plan is not None:
            loss_dict["vosr_scale"] = vosr_plan[4]
            if gan_terms:
                # Running means of the clip's raw GAN term and raw VOSR distance;
                # kept only if this call is accepted, as adaptive_d's EMA
                pending_vosr_balance_state = vosr_balance_state(
                    torch.stack(vosr_shares).sum() / frames,
                    torch.stack(gan_terms).mean(),
                    self._vosr_balance_state,
                )

        if apply_gradient:
            if self.scaler_g.is_enabled():
                unscaled_scaler_states.append(
                    (self.scaler_g, self.scaler_g.state_dict())
                )
            self.scaler_g.unscale_(self.optimizer_g)
            if self._spike_clippers is None:
                grad_norm_g = _get_total_gradient_norm(self.net_g.parameters())
                loss_dict["grad_norm_g"] = grad_norm_g
            else:
                grad_norm_g, threshold_g = self._spike_clippers["g"].prepare(self.optimizer_g)
            if not bool(torch.isfinite(grad_norm_g).item()):
                self._reject_nonfinite_update(
                    "nonfinite_generator_gradient",
                    candidate_iter,
                    unscaled_scaler_states,
                    loss_dict,
                )
                return
            if self.grad_clip:
                clip_grads_with_norm_(
                    self.net_g.parameters(),
                    self.grad_clip_max_norm,
                    grad_norm_g,
                    foreach=None,
                )
            generator_ready = True

        # --------------------------- discriminator --------------------------- #
        cri_gan = self.losses.get("l_g_gan") if gan_active else None
        if apply_gradient and not gan_active and self.optimizer_d_index is not None:
            self.optimizers_skipped[self.optimizer_d_index] = True
        if (
            self.net_d is not None
            and cri_gan is not None
            and self.optimizer_d is not None
        ):
            if not skip_d_update:
                for p in self.net_d.parameters():
                    p.requires_grad = True
                d_sums: dict[str, Tensor] = {}
                for t in range(frames):
                    if self.compile_discriminator and uses_cudagraphs:
                        # each frame's discriminator pass is its own CUDA-graph step
                        torch.compiler.cudagraph_mark_step_begin()
                    condition = self._frame_condition(t)
                    real = (
                        self.gt[:, t].contiguous().to(memory_format=self.memory_format)
                    )
                    fake = (
                        self.output[:, t]
                        .contiguous()
                        .to(memory_format=self.memory_format)
                    )
                    with self._autocast():
                        real_d_pred = self.net_d(
                            self._discriminator_input(real, condition)
                        )
                        l_d_real = cri_gan(real_d_pred, True, is_disc=True)
                        fake_d_pred = self.net_d(
                            self._discriminator_input(fake, condition)
                        )
                        l_d_fake = cri_gan(fake_d_pred, False, is_disc=True)
                    for name, value in (
                        ("l_d_real", l_d_real.detach()),
                        ("l_d_fake", l_d_fake.detach()),
                        ("out_d_real", torch.mean(real_d_pred.detach())),
                        ("out_d_fake", torch.mean(fake_d_pred.detach())),
                    ):
                        share = value / frames
                        d_sums[name] = d_sums[name] + share if name in d_sums else share
                    backward_d_loss = (l_d_real + l_d_fake) / frames / self.accum_iters
                    if not bool(torch.isfinite(backward_d_loss).item()):
                        loss_dict.update(d_sums)
                        self._reject_nonfinite_update(
                            "nonfinite_discriminator_loss",
                            candidate_iter,
                            unscaled_scaler_states,
                            loss_dict,
                        )
                        return
                    self.scaler_d.scale(backward_d_loss).backward()
                loss_dict.update(d_sums)

                if apply_gradient:
                    if self.scaler_d.is_enabled():
                        unscaled_scaler_states.append(
                            (self.scaler_d, self.scaler_d.state_dict())
                        )
                    self.scaler_d.unscale_(self.optimizer_d)
                    if self._spike_clippers is None:
                        grad_norm_d = _get_total_gradient_norm(self.net_d.parameters())
                        loss_dict["grad_norm_d"] = grad_norm_d
                    else:
                        grad_norm_d, threshold_d = self._spike_clippers["d"].prepare(self.optimizer_d)
                    if not bool(torch.isfinite(grad_norm_d).item()):
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

        # ------------------------------ updates ------------------------------ #
        if generator_ready:
            scale_before = self.scaler_g.get_scale()
            self.step_optimizer(self.scaler_g, self.optimizer_g)
            self.scaler_g.update()
            scale_after = self.scaler_g.get_scale()
            loss_dict["scale_g"] = scale_after
            self.optimizers_skipped[0] = scale_after < scale_before
            if self._spike_clippers is not None and scale_after >= scale_before:
                self._spike_clippers["g"].commit(grad_norm_g, threshold_g)
            self._zero_generator_optimizer(self.optimizer_g)
        if discriminator_ready:
            assert self.optimizer_d is not None
            scale_before = self.scaler_d.get_scale()
            self.step_optimizer(self.scaler_d, self.optimizer_d)
            self.scaler_d.update()
            scale_after = self.scaler_d.get_scale()
            loss_dict["scale_d"] = scale_after
            assert self.optimizer_d_index is not None
            self.optimizers_skipped[self.optimizer_d_index] = scale_after < scale_before
            if self._spike_clippers is not None and scale_after >= scale_before:
                self._spike_clippers["d"].commit(grad_norm_d, threshold_d)
            self._zero_discriminator_optimizer(self.optimizer_d)
        if pending_adaptive_d_ema is not None:
            self.l_g_gan_ema.copy_(pending_adaptive_d_ema)
            loss_dict["adaptive_d_loss_ema"] = pending_adaptive_d_ema
            loss_dict["adaptive_d_skip_fraction"] = float(skip_d_update)
        if pending_vosr_balance_state is not None:
            self._vosr_balance_state = pending_vosr_balance_state
        if apply_gradient:
            self._adaptive_d_window_start = None
            self._report_pending_nonfinite_rejections(candidate_iter)
            self.nan_count = 0

        self.loss_samples += n_samples
        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        if self.net_g_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.update_ema(self.net_g_ema)
                self._validation_prepared_key = None

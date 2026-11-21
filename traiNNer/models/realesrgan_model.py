import os
import sys
from collections import deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from os import path as osp
from typing import TypeVar

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.data.codec_approx import (
    approximate_video,
    approximate_webp,
    can_approximate,
)
from traiNNer.data.compress_video_batch import prepare_video_batch, usable_cpus
from traiNNer.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    resize_neosr_pt,
    resize_pt,
)
from traiNNer.data.dvd_response import dvd_pair
from traiNNer.data.neosr_sample_stages import clean_resize, sample_mask, sharpen_samples
from traiNNer.data.gpu_degradations import (
    channel_shift_pt,
    chroma_shift_pt,
    chroma_subsample_pt,
    composite_rainbow_pt,
    detail_mask_neo_pt,
    interlace_pt,
    lowpass_filter_pt,
    ntsc_composite_pt,
    ordered_dither_pt,
    overshoot_pt,
    quantize_pt,
    sample_shift,
    scanline_pt,
    temporal_ghosting_pt,
)
from traiNNer.data.otf_degradations import (
    apply_dithering,
    apply_dithering_palette,
    apply_per_image,
    nlmeans_denoise_pt,
    prepare_webp_batch,
)
from traiNNer.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from traiNNer.data.still_source import device_canvas, host_canvas
from traiNNer.data.transforms import paired_random_crop, paired_random_crop_multi_gt
from traiNNer.losses.vosr_loss import VosrTeacher
from traiNNer.models.realesrgan_noise import generate_hf_noise, sample_beta_noise
from traiNNer.models.sr_model import SRModel
from traiNNer.utils import RNG, DiffJPEG, get_root_logger
from traiNNer.utils.color_util import dataset_luma_pt, dataset_luma_weights
from traiNNer.utils.fused_jpeg import fused_jpeg, fused_jpeg_fits
from traiNNer.utils.img_process_util import USMSharp, filter2d
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import MODEL_REGISTRY
from traiNNer.utils.types import DataFeed

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/otf"))
)

ANTIALIAS_MODES = {"bicubic", "bilinear"}

T = TypeVar("T")

# A CPU-only part of the degradation (a video-codec or WebP round trip): no
# random draws, no device access, no shared state, so it may run in a worker
# thread. The degradation steps of feed_data yield these and receive back
# their frames.
DegradationJob = Callable[[], Tensor]
DegradationSteps = Generator[DegradationJob, Tensor, None]


def _weighted_choice(population: Sequence[T], weights: Sequence[float] | None) -> T:
    """population[i] with probability weights[i] / sum(weights) (uniform when
    weights is None), the distribution of random.choices(population, weights)[0].

    Drawn from the project RNG, not Python's random: rich's log handler also
    draws from random, so logging changed the degradation stream.
    """
    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        p = w / w.sum()
    return population[int(RNG.get_rng().choice(len(population), p=p))]


def warn_ignored_luma_shift(opt: ReduxOptions) -> None:
    """The YUV channel shift moves chroma only; say so when a YUV shift can fire
    and the options give Y a range."""
    if (
        opt.shift_prob > 0
        and "yuv" in opt.shift_types
        and any(bound != 0 for axis in opt.shift_yuv_y for bound in axis)
    ):
        get_root_logger().warning(
            "The YUV channel shift moves chroma (U, V) only, so the LQ stays "
            "aligned with the HR: shift_yuv_y %s is ignored.",
            opt.shift_yuv_y,
        )


def _collated_any(value: bool | Tensor | None) -> bool:
    if isinstance(value, Tensor):
        return bool(value.any().item())
    return bool(value)


@dataclass(eq=False)
class DegradingBatch:
    """A training batch whose degradation is under way: its chains of
    degradation steps (one for an image batch, one per clip of a clip batch),
    the jobs each waiting chain yielded, the jobs of its wave that was started,
    what each chain left on the model when it ended, the fields the delivered
    batch carries besides, its age in DegradeAheadPrefetcher calls and whether
    its jobs run in the model's degradation_pool or on the calling thread."""

    pooled: bool
    carry: DataFeed = field(default_factory=DataFeed)
    chains: list[DegradationSteps] = field(default_factory=list)
    waiting: dict[int, DegradationJob] = field(default_factory=dict)
    running: dict[int, Future[Tensor]] = field(default_factory=dict)
    done: dict[int, DataFeed] = field(default_factory=dict)
    age: int = 0

    @property
    def finished(self) -> bool:
        return not self.running


@MODEL_REGISTRY.register(suffix="traiNNer")
class RealESRGANModel(SRModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    # Set by video subclasses: the batch is one clip in time order, every random
    # setting of the chain is drawn once per call and video codecs predict
    # between frames.
    clip_mode = False

    def __init__(self, opt: ReduxOptions) -> None:
        if opt.dvd_prob > 0 and opt.scale != 2:
            raise ValueError("dvd_prob requires scale: 2; the DVD response is fixed x2")
        super().__init__(opt)
        if not self.is_train:
            return
        self._otf_resize = resize_neosr_pt if opt.otf_resize_policy == "neosr" else resize_pt
        color = opt.datasets["train"].color
        if color and self.clip_mode:
            raise ValueError("The recurrent OTF producer requires RGB training frames.")
        self._otf_luma = dataset_luma_weights(color, device=self.device) if color else None
        # Whether a DVD hit has fallen back for a sample too small for DVD context.
        self.dvd_fallback_warned = False

        self.queue_lr: Tensor | None = None
        self.queue_gt: Tensor | None = None
        # Pool slot of each logical pool position (CPU int64).
        self.queue_order: Tensor | None = None
        self._queue_index_host: Tensor | None = None
        self.queue_ptr = 0
        self.kernel1: Tensor | None = None
        self.kernel2: Tensor | None = None
        self.sinc_kernel: Tensor | None = None

        self.jpeger = DiffJPEG(
            differentiable=False
        ).to(self.device)
        self.queue_size = opt.queue_size
        self.thicklines = None
        if self.opt.thicklines_prob > 0:
            self.thicklines = ThickLines().to(self.device)

        self.otf_debug = opt.high_order_degradations_debug
        self.otf_debug_limit = opt.high_order_degradations_debug_limit
        warn_ignored_luma_shift(opt)

        # Full distillation: VOSR's restoration of each degraded LR replaces the HR (settings checked by SRModel)
        self.vosr_target: VosrTeacher | None = None
        if (
            self.is_train
            and opt.train is not None
            and opt.train.vosr_target is not None
        ):
            target_opt = opt.train.vosr_target
            self.vosr_target = VosrTeacher(
                target_opt.engine_dir, target_opt.plugin, target_opt.lanes
            )
            get_root_logger().info(
                "VOSR target enabled: the HR target is VOSR 2.0's restoration of each degraded LR."
            )

        if self.otf_debug:
            logger = get_root_logger()
            logger.info(
                "OTF debugging enabled. LR tiles will be saved to: %s", OTF_DEBUG_PATH
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.

        The pool stays in place: queue_order maps each logical position to its
        slot. Once the pool is full, each call shuffles the logical order with
        one permutation and exchanges only the batch's slots, the first b
        logical positions, which delivers and keeps the same pairs as gathering
        the whole pool in the shuffled order would.

        Optional still LQ packing uses a device-local FP32 division table. Only
        a marked producer batch that reconstructs every FP32 bit may enter a
        byte pool. The optional equality check synchronizes on CUDA. An
        unmarked input or failed check promotes existing slots once, keeping
        the full LQ allocation for the remaining pool lifetime. During promotion
        the byte pool and the new FP32 allocation briefly coexist. GT and the
        logical permutation are independent of LQ storage.
        """

        assert self.lq is not None
        assert self.gt is not None
        b, c, h, w = self.lq.size()
        incoming = self.lq
        lut = getattr(self, "_queue_lq_lut", None) if self.queue_lr is not None else None
        eligible = getattr(self, "_queue_lq_quantized", None)
        if eligible is not None:
            # Consume the marker. Any direct, unmarked pool call also falls back.
            self._queue_lq_quantized = None
        if self.queue_lr is not None and (eligible is not None or lut is not None):
            replay_dtype = lut.dtype if lut is not None else self.queue_lr.dtype
            if incoming.dtype != replay_dtype or incoming.device != self.queue_lr.device:
                raise ValueError("Compact replay requires a consistent input LQ dtype and device.")
        if eligible and incoming.dtype == torch.float32 and (self.queue_lr is None or lut is not None):
            if lut is None:
                lut = torch.arange(256, dtype=torch.float32, device=incoming.device) / 255.0
            codes = (incoming * 255.0).round().to(torch.uint8)
            # Includes signed zero and exceptional values. Checking also covers
            # device/kernel differences in the producer's scalar division.
            if torch.equal(lut[codes.int()].view(torch.int32), incoming.view(torch.int32)):
                incoming = codes
            elif self.queue_lr is None:
                lut = None

        if lut is not None and incoming is self.lq:
            assert self.queue_lr is not None
            # Expand only initialized history, a batch at a time. Preserve slot
            # numbering during both warm-up and full-pool replacement.
            full = torch.zeros_like(self.queue_lr, dtype=lut.dtype)
            for start in range(0, self.queue_ptr, b):
                end = min(start + b, self.queue_ptr)
                full[start:end] = lut[self.queue_lr[start:end].int()]
            self.queue_lr = full
            self._queue_lq_lut = lut = None

        if self.queue_lr is None:
            assert self.queue_size % b == 0, (
                f"queue size {self.queue_size} should be divisible by batch size {b}"
            )
            self.queue_lr = torch.zeros(
                self.queue_size, c, h, w, device=self.lq.device, dtype=incoming.dtype
            )
            self._queue_lq_lut = lut
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(
                self.queue_size, c, h, w, device=self.gt.device, dtype=self.gt.dtype
            )
            self.queue_ptr = 0
            # Filling writes the slots in order, so the logical order starts as
            # the slot order.
            self.queue_order = torch.arange(self.queue_size)
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle the logical order
            assert self.queue_lr is not None
            assert self.queue_gt is not None
            assert self.queue_order is not None
            idx = torch.randperm(self.queue_size)
            self.queue_order = self.queue_order[idx]
            # slots of the first b logical positions
            picked = self.queue_order[0:b]
            device_picked = picked
            if self.lq.is_cuda:
                # Retain the pinned source alongside the async index upload.
                self._queue_index_host = picked.pin_memory()
                device_picked = self._queue_index_host.to(
                    device=self.lq.device, non_blocking=True
                )
            # get first b samples
            lq_dequeue = self.queue_lr.index_select(0, device_picked)
            gt_dequeue = self.queue_gt.index_select(0, device_picked)
            # update the queue: the batch takes the slots it was dealt from
            self.queue_lr.index_copy_(0, device_picked, incoming)
            self.queue_gt.index_copy_(0, device_picked, self.gt)
            self.lq = lq_dequeue if lut is None else lut[lq_dequeue.int()]
            self.gt = gt_dequeue
        else:
            assert self.queue_lr is not None
            assert self.queue_gt is not None

            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                incoming.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    def _sample_beta_noise(
        self,
        channels: int,
        height: int,
        width: int,
        shape_a: Tensor,
        shape_b: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return sample_beta_noise(
            channels, height, width, shape_a, shape_b, dtype, device
        )

    def _generate_hf_noise(self, img: Tensor) -> Tensor:
        return generate_hf_noise(
            img, self.opt, RNG.get_rng(), self._sample_beta_noise
        )

    def _apply_hf_noise(self) -> None:
        if self.opt.otf_hf_noise_prob <= 0:
            return

        assert self.gt is not None

        if RNG.get_rng().uniform() >= self.opt.otf_hf_noise_prob:
            return

        noise = self._generate_hf_noise(self.gt)
        noise = torch.max(-self.gt, torch.min(1 - self.gt, noise))
        self.gt = self.gt + noise

    def _select_active_degradations(self) -> set[str]:
        """Select which custom degradations are active for this iteration.

        Uses the combo system if configured, otherwise all degradations
        are eligible (backward-compatible default behavior).

        Returns:
            Set of active degradation names. A degradation's probability
            check is only reached if its name is in this set.
        """
        combos = self.opt.otf_degradation_combos
        if combos is None:
            # No combo system configured — all degradations eligible
            return {
                "ntsc", "rainbow", "ghosting", "interlace", "scanline",
                "lowpass", "overshoot", "shift", "subsampling", "dithering",
                "compress", "hf_noise", "nlmeans",
            }

        active: set[str] = set()

        # Always add global degradations
        if self.opt.otf_global_degradations is not None:
            active.update(self.opt.otf_global_degradations)

        # Two-step selection: first check if any combo fires, then pick which
        no_combo_prob = self.opt.otf_no_combo_weight
        if RNG.get_rng().uniform() >= no_combo_prob:
            # A combo fires — pick which one using relative weights (uniform if None)
            selected = _weighted_choice(
                combos, self.opt.otf_degradation_combo_weights
            )
            active.update(selected)

        return active

    def _stage_noise(self, source, gaussian_probability, sigma_range, poisson_range, gray_probability, probability):
        gaussian = RNG.get_rng().uniform() < gaussian_probability
        if probability == 0:
            return source.clamp(0, 1)
        if gaussian:
            noisy = random_add_gaussian_noise_pt(
                source, sigma_range=sigma_range, gray_prob=gray_probability, shared=self.clip_mode,
                clip=True, rounds=False,
            )
        else:
            noisy = random_add_poisson_noise_pt(
                source, scale_range=poisson_range, gray_prob=gray_probability, shared=self.clip_mode,
                clip=True, rounds=False,
            )
        if probability == 1:
            return noisy
        return torch.where(sample_mask(source, probability), noisy, source.clamp(0, 1))

    def _apply_compression(
        self, out: Tensor, stage: int = 1
    ) -> Generator[DegradationJob, Tensor, Tensor]:
        """Apply a randomly chosen compression algorithm; returns the result.

        JPEG: GPU DiffJPEG, as one fused Triton kernel when fused_jpeg_fits.
        WebP: Pillow method=0 via parallel ThreadPool (1.8x faster than cv2),
        or the same pixels from webp_reconstruct outside clip mode.
        Video codecs: TorchCodec/PyAV batch encoding (9-48x faster than subprocess).
        The video codec's encode/decode round trip (prepare_video_batch) and
        the WebP round trip (prepare_webp_batch) are yielded and their frames
        are received back; JPEG runs inline.
        With compress_video_approx_prob (video) and compress_webp_approx (WebP)
        an eligible call runs the GPU approximation of codec_approx at once
        instead and yields a no-op job, so the number of yields is unchanged.
        """
        algos = (
            self.opt.compress_algorithms
            if stage == 1
            else self.opt.compress_algorithms2
        )
        probs = (
            self.opt.compress_algorithm_probs
            if stage == 1
            else self.opt.compress_algorithm_probs2
        )
        algo = _weighted_choice(algos, probs)

        if algo == "jpeg":
            settings = self.opt.neosr_samples
            probability = (settings.jpeg_prob if stage == 1 else settings.jpeg_prob2) if settings else 1.0
            if probability == 0:
                return out
            original = out if probability < 1 else None
            jpeg_range = self.opt.compress_jpeg_range if stage == 1 else self.opt.compress_jpeg_range2
            if self.clip_mode:
                jpeg_p = out.new_zeros(1).uniform_(*jpeg_range).repeat(out.size(0))
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(out, 0, 1)
            if fused_jpeg_fits(out, jpeg_p):
                out = fused_jpeg(out, jpeg_p)
            else:
                out = self.jpeger(out, quality=jpeg_p)
            if original is not None:
                out = torch.where(sample_mask(original, probability), out, original)
        elif algo == "webp":
            quality_range = (
                self.opt.compress_webp_range
                if stage == 1
                else self.opt.compress_webp_range2
            )
            quality = int(RNG.get_rng().integers(*quality_range))
            out = torch.clamp(out, 0, 1)
            if self.opt.compress_webp_approx and can_approximate(out, "webp"):
                # Computed now on the current stream; the no-op job keeps the
                # chain's number of yields.
                out = approximate_webp(out, quality)
                yield lambda: None
            else:
                round_trip, finish = prepare_webp_batch(
                    out, quality, direct=not self.clip_mode
                )
                out = finish((yield round_trip))
        elif algo in ("h264", "hevc", "mpeg2", "mpeg4", "vp9"):
            range_attr = f"compress_{algo}_range{'2' if stage == 2 else ''}"
            quality_range = getattr(self.opt, range_attr)
            quality = int(RNG.get_rng().integers(*quality_range))
            video_sampling = (
                self.opt.compress_video_sampling
                if stage == 1
                else self.opt.compress_video_sampling2
            )
            sampling_weights = (
                self.opt.compress_video_sampling_weights
                if stage == 1
                else self.opt.compress_video_sampling_weights2
            )
            # Select one sampling format using weights (or uniform if None)
            selected_sampling = [_weighted_choice(video_sampling, sampling_weights)]
            out = torch.clamp(out, 0, 1)
            lead_in = (
                int(RNG.get_rng().integers(0, min(8, out.size(0) - 1), endpoint=True))
                if self.clip_mode
                else 0
            )
            approx_prob = self.opt.compress_video_approx_prob
            approximate = (
                approx_prob > 0
                and can_approximate(out, algo, self.clip_mode, lead_in)
                and (
                    approx_prob >= 1
                    or RNG.get_codec_route_rng().uniform() < approx_prob
                )
            )
            if approximate:
                # Computed now on the current stream; the no-op job keeps the
                # chain's number of yields.
                out = approximate_video(
                    out, algo, quality, selected_sampling, self.clip_mode, lead_in
                )
                yield lambda: None
            else:
                round_trip, finish = prepare_video_batch(
                    out,
                    algo,
                    quality,
                    selected_sampling,
                    temporal=self.clip_mode,
                    lead_in=lead_in,
                    # A clip batch runs one round trip per clip side by side.
                    jobs=(
                        self.opt.datasets["train"].batch_size_per_gpu or 1
                        if self.clip_mode
                        else 1
                    ),
                )
                out = finish((yield round_trip))
        return out

    def _apply_shift(self, out: Tensor) -> Tensor:
        """Apply channel shift degradation (GPU, no apply_per_image)."""
        rng = RNG.get_rng()
        shift_type = rng.choice(self.opt.shift_types)

        if shift_type == "rgb":
            amounts = [
                self.opt.shift_rgb_r,
                self.opt.shift_rgb_g,
                self.opt.shift_rgb_b,
            ]
        elif shift_type == "yuv":
            # Chroma only: the LQ stays registered with the HR. Cb then Cr, x
            # then y; in clip mode the one draw serves every frame of the clip.
            _, _, h, w = out.shape
            percent = self.opt.shift_percent
            cb_offset = sample_shift(self.opt.shift_yuv_u, h, w, percent)
            cr_offset = sample_shift(self.opt.shift_yuv_v, h, w, percent)
            return chroma_shift_pt(out, cb_offset, cr_offset)
        elif shift_type == "cmyk":
            amounts = [
                self.opt.shift_cmyk_c,
                self.opt.shift_cmyk_m,
                self.opt.shift_cmyk_y,
                self.opt.shift_cmyk_k,
            ]
        else:
            return out

        return channel_shift_pt(out, shift_type, amounts, self.opt.shift_percent)

    def _apply_subsampling(self, out: Tensor) -> Tensor:
        """Apply chroma subsampling degradation (GPU, no apply_per_image)."""
        rng = RNG.get_rng()
        down_alg = rng.choice(self.opt.subsampling_down_algorithms)
        up_alg = rng.choice(self.opt.subsampling_up_algorithms)
        fmt = rng.choice(self.opt.subsampling_formats)
        ycbcr = rng.choice(self.opt.subsampling_ycbcr_type)
        blur_sigma = None
        if self.opt.subsampling_blur_range is not None:
            blur_sigma = float(rng.uniform(*self.opt.subsampling_blur_range))

        return chroma_subsample_pt(out, down_alg, up_alg, fmt, blur_sigma, ycbcr)

    def _apply_dithering(self, out: Tensor) -> Tensor:
        """Apply dithering degradation to the LQ tensor.

        Two modes:
        - Palette mode: content-adaptive indexed color (like GIF/PNG8) via median cut.
          The median cut is CPU (Pillow, per image); the dithering uses the exact
          CUDA kernel when it builds, else CPU (chainner_ext PaletteQuantization).
        - Uniform mode: per-channel quantization levels.
          GPU path for quantize/ordered dither; error diffusion/riemersma use the
          exact CUDA kernel when it builds, else CPU (chainner_ext).
        """
        rng = RNG.get_rng()
        dtype = rng.choice(self.opt.dithering_types).lower()

        if self.opt.dithering_palette_mode > 0 and dtype != "order" and rng.uniform() < self.opt.dithering_palette_mode:
            # Palette mode: extract content-adaptive palette, dither to those colors
            num_colors = int(rng.integers(*self.opt.dithering_palette_size))
            map_size = int(rng.choice(self.opt.dithering_map_size))
            history = int(rng.integers(*self.opt.dithering_history_range))
            decay_ratio = float(rng.uniform(*self.opt.dithering_ratio_range))
            from traiNNer.data.dither_cuda import (
                PALETTE_DITHERING_TYPES,
                dither_cuda_available,
                palette_dither_cuda,
            )

            # Same median cut either way; the GPU path dithers the whole batch
            # bit-identically to the per-image CPU path.
            if (
                out.is_cuda
                and dtype in PALETTE_DITHERING_TYPES
                and dither_cuda_available()
            ):
                return palette_dither_cuda(out, dtype, num_colors, history, decay_ratio)
            return apply_per_image(
                out,
                lambda img: apply_dithering_palette(
                    img, dtype, num_colors, map_size, history, decay_ratio
                ),
            )

        # Uniform quantization mode
        quantize_ch = int(rng.integers(*self.opt.dithering_quantize_range))

        if dtype == "quantize":
            return quantize_pt(out, quantize_ch)
        elif dtype == "order":
            map_size = int(rng.choice(self.opt.dithering_map_size))
            return ordered_dither_pt(out, quantize_ch, map_size)
        else:
            # Error diffusion / riemersma — exact CUDA kernel (bit-identical to
            # chainner_ext) when it builds, else the per-image CPU path. map_size
            # is drawn on both paths so the RNG stream does not depend on the path.
            map_size = int(rng.choice(self.opt.dithering_map_size))
            history = int(rng.integers(*self.opt.dithering_history_range))
            decay_ratio = float(rng.uniform(*self.opt.dithering_ratio_range))
            from traiNNer.data.dither_cuda import (
                DIFFUSION_ALGORITHMS,
                dither_cuda_available,
                error_diffusion_dither_cuda,
                riemersma_dither_cuda,
            )

            # Riemersma is one serial Hilbert walk per image-channel, so a single
            # GPU thread per plane only beats a CPU core from batch 2 upward
            # (measured: 3x faster at batch 8, 0.6x at batch 1). Same bits either way.
            use_kernel = out.is_cuda and (
                dtype in DIFFUSION_ALGORITHMS
                or (dtype == "riemersma" and out.shape[0] >= 2)
            )
            if use_kernel and dither_cuda_available():
                if dtype == "riemersma":
                    return riemersma_dither_cuda(out, quantize_ch, history, decay_ratio)
                return error_diffusion_dither_cuda(out, quantize_ch, dtype)
            return apply_per_image(
                out,
                lambda img: apply_dithering(
                    img, dtype, quantize_ch, map_size, history, decay_ratio
                ),
            )

    def _apply_ghosting(self, out: Tensor) -> Tensor:
        """Apply temporal ghosting degradation (GPU)."""
        rng = RNG.get_rng()
        sx = int(rng.integers(*self.opt.ghosting_shift_x))
        sy = int(rng.integers(*self.opt.ghosting_shift_y))
        opacity = float(rng.uniform(*self.opt.ghosting_opacity))
        return temporal_ghosting_pt(out, sx, sy, opacity)

    def _apply_interlace(self, out: Tensor) -> Tensor:
        """Apply interlace combing degradation (GPU)."""
        rng = RNG.get_rng()
        shift = int(rng.integers(*self.opt.interlace_field_shift))
        field = rng.choice(self.opt.interlace_dominant_field)
        return interlace_pt(out, shift, field)

    def _apply_lowpass(self, out: Tensor) -> Tensor:
        """Apply Butterworth lowpass filter degradation (GPU).

        Optionally preserves detail/edges using a mask computed from HQ.
        """
        rng = RNG.get_rng()
        cutoff = float(rng.uniform(*self.opt.lowpass_cutoff))
        order = int(rng.integers(*self.opt.lowpass_order))

        filtered = lowpass_filter_pt(out, cutoff, order)

        if self.opt.lowpass_detail_mask:
            mask = detail_mask_neo_pt(
                self.gt, lines_brz=self.opt.lowpass_mask_lines_brz
            )
            # Resize mask to match LQ if sizes differ
            if mask.shape[2:] != out.shape[2:]:
                mask = F.interpolate(
                    mask, size=out.shape[2:], mode="bilinear", align_corners=False
                )
            filtered = mask * out + (1.0 - mask) * filtered

        return filtered

    def _apply_ntsc(self, out: Tensor) -> Tensor:
        """Apply full NTSC composite simulation (GPU)."""
        _NTSC_PRESETS = {
            "broadcast": (False, 4.2, 500, 0.03, 0.0, 0.0),
            "vhs_sp": (True, 3.0, 500, 0.06, 0.03, 0.8),
            "vhs_ep": (True, 1.6, 300, 0.10, 0.06, 0.5),
        }

        rng = RNG.get_rng()
        preset = rng.choice(self.opt.ntsc_preset)
        enable_vhs, vlbw, cubw, n, ln, er = _NTSC_PRESETS.get(
            preset, _NTSC_PRESETS["broadcast"]
        )

        # Override with user config or preset defaults
        if self.opt.ntsc_enable_vhs:
            enable_vhs = True

        noise_val = float(rng.uniform(*self.opt.ntsc_noise)) if self.opt.ntsc_noise != [n, n] else n
        luma_noise_val = float(rng.uniform(*self.opt.ntsc_luma_noise)) if self.opt.ntsc_luma_noise != [ln, ln] else ln
        ghost_amp = float(rng.uniform(*self.opt.ntsc_ghost_amplitude))
        ghost_del = float(rng.uniform(*self.opt.ntsc_ghost_delay_us))
        ghost_ph = float(rng.uniform(*self.opt.ntsc_ghost_phase))
        jitter_val = float(rng.uniform(*self.opt.ntsc_jitter))
        ringing_val = float(rng.uniform(*self.opt.ntsc_edge_ringing)) if self.opt.ntsc_edge_ringing != [er, er] else er
        vlbw_val = float(rng.uniform(*self.opt.ntsc_vhs_luma_bw)) if self.opt.ntsc_vhs_luma_bw != [vlbw, vlbw] else vlbw
        cubw_val = float(rng.uniform(*self.opt.ntsc_color_under_bw)) if self.opt.ntsc_color_under_bw != [cubw, cubw] else cubw
        trail_val = float(rng.uniform(*self.opt.ntsc_tape_trailing))
        intensity_val = float(rng.uniform(*self.opt.ntsc_intensity))

        return ntsc_composite_pt(
            out,
            noise=noise_val,
            luma_noise=luma_noise_val,
            ghost_amplitude=ghost_amp,
            ghost_delay_us=ghost_del,
            ghost_phase=ghost_ph,
            jitter=jitter_val,
            edge_ringing=ringing_val,
            vhs_luma_bw=vlbw_val,
            color_under_bw=cubw_val,
            tape_trailing=trail_val,
            intensity=intensity_val,
            comb_mode=self.opt.ntsc_comb_mode,
            enable_vhs=enable_vhs,
        )

    def _apply_overshoot(self, out: Tensor) -> Tensor:
        """Apply edge overshoot/undershoot degradation (GPU)."""
        rng = RNG.get_rng()
        amount = float(rng.uniform(*self.opt.overshoot_amount))
        cutoff = float(rng.uniform(*self.opt.overshoot_cutoff))
        order = int(rng.integers(*self.opt.overshoot_order))
        return overshoot_pt(out, amount, cutoff, order)

    def _apply_rainbow(self, out: Tensor) -> Tensor:
        """Apply composite rainbow artifact (GPU)."""
        rng = RNG.get_rng()
        freq = float(rng.uniform(*self.opt.rainbow_subcarrier_freq))
        bw = float(rng.uniform(*self.opt.rainbow_chroma_bandwidth))
        intensity = float(rng.uniform(*self.opt.rainbow_intensity))
        phase_offset = float(rng.uniform(0, 2 * 3.141592653589793))
        return composite_rainbow_pt(
            out, freq, bw, intensity, self.opt.rainbow_phase_alternation, phase_offset
        )

    def _apply_scanline(self, out: Tensor) -> Tensor:
        """Apply CRT scanline darkening (GPU)."""
        rng = RNG.get_rng()
        strength = float(rng.uniform(*self.opt.scanline_strength))
        return scanline_pt(out, strength, self.opt.scanline_even_lines)

    # Training steps the jobs of a wave get when DegradeAheadPrefetcher degrades
    # ahead: the codec and WebP round trips of an image batch take longer than
    # an image training step, so each wave gets three.
    ahead_stride = 3
    # feed_data runs an image batch's jobs on the calling thread: its one chain
    # has nothing to run beside them.
    pooled_feed = False
    # Runs degradation jobs in worker threads; made on first use
    degradation_pool: ThreadPoolExecutor | None = None

    @torch.no_grad()
    def feed_data(self, data: DataFeed) -> None:
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.

        In training, a batch DegradeAheadPrefetcher delivers (it carries "lq")
        is installed as it is; any other batch is started, run to its end and
        installed here, its degradation jobs in degradation_pool when
        pooled_feed is set and on this thread otherwise.
        """
        if not self.is_train:
            # Paired or validation data: nothing is degraded or yielded.
            next(self._feed_steps(data), None)
            return
        if "lq" not in data:
            data = self.finish_batch(self.start_batch(data, pooled=self.pooled_feed))
        self.install_batch(data)

    def start_batch(self, data: DataFeed, pooled: bool = True) -> DegradingBatch:
        """Begin degrading a training batch: its chain runs to its first job,
        and that wave of jobs is started."""
        if "gt_orientation" in data:
            # Source bytes: their canvases on the current stream, which the
            # prefetcher made wait for the upload, before the chain reads gt.
            source = data["gt"].to(self.device, non_blocking=True)
            flags = torch.as_tensor(data["gt_orientation"]).to(
                self.device, non_blocking=True
            )
            if source.is_cuda:
                gt = device_canvas(source, flags, self.memory_format)
            else:
                gt = torch.stack(
                    [host_canvas(s, int(f)) for s, f in zip(source, flags, strict=True)]
                )
            data = {**data, "gt": gt}
            del data["gt_orientation"]
        batch = DegradingBatch(pooled=pooled)
        self._add_chain(batch, self._feed_steps(data))
        self._submit_wave(batch)
        return batch

    def advance_batch(self, batch: DegradingBatch) -> None:
        """Take the results of the batch's started wave (waiting for them if
        needed), run each chain in order to its next job and start that wave.
        Nothing to do once the batch is finished."""
        wave = sorted(batch.running)
        results = [batch.running[index].result() for index in wave]
        batch.running = {}
        for index, result in zip(wave, results, strict=True):
            self._step_chain(batch, index, result)
        self._submit_wave(batch)

    def finish_batch(self, batch: DegradingBatch) -> DataFeed:
        """Advance the batch to its end; the batch for install_batch."""
        while not batch.finished:
            self.advance_batch(batch)
        return self._delivered(batch)

    def install_batch(self, data: DataFeed) -> None:
        """Make a finished batch the one the next training step uses."""
        self.lq = data["lq"]
        self.gt = data["gt"]

    def _delivered(self, batch: DegradingBatch) -> DataFeed:
        """The chain's batch, its LQ in the generator's memory format, as
        SRModel.feed_data delivers a paired LQ; GT keeps the chain's layout."""
        data = batch.done[0]
        return {**data, "lq": data["lq"].to(memory_format=self.memory_format)}

    def _add_chain(self, batch: DegradingBatch, chain: DegradationSteps) -> None:
        """Add a chain to the batch and run it to its first job (or its end)."""
        batch.chains.append(chain)
        self._step_chain(batch, len(batch.chains) - 1, None)

    def _step_chain(
        self, batch: DegradingBatch, index: int, frames: Tensor | None
    ) -> None:
        """Run chain index to its next job, or to its end, where what it left on
        the model is that chain's until another chain runs."""
        try:
            batch.waiting[index] = batch.chains[index].send(frames)
        except StopIteration:
            assert self.lq is not None
            assert self.gt is not None
            batch.done[index] = {
                "lq": self.lq,
                "gt": self.gt,
            }

    def _submit_wave(self, batch: DegradingBatch) -> None:
        """Start the jobs the chains of the batch are waiting for."""
        waiting, batch.waiting = batch.waiting, {}
        for index in sorted(waiting):
            if batch.pooled:
                pool = self._degradation_pool(len(batch.chains))
                batch.running[index] = pool.submit(waiting[index])
            else:
                future: Future[Tensor] = Future()
                future.set_result(waiting[index]())
                batch.running[index] = future

    def _degradation_pool(self, chains: int) -> ThreadPoolExecutor:
        if self.degradation_pool is None:
            # Degrading ahead, every batch younger than two strides can have a
            # wave running: one job per chain.
            waves = 2 * max(self.ahead_stride, 1)
            self.degradation_pool = ThreadPoolExecutor(
                max_workers=min(waves * chains, usable_cpus()),
                thread_name_prefix="degradation",
            )
        return self.degradation_pool

    def _dvd_fits(self, data: DataFeed, gt_size: int) -> bool:
        """Whether every sample of the batch can give a DVD patch with real
        context; the first time one cannot, a warning names it."""
        samples = data.get("dvd")
        if not isinstance(samples, list):
            raise ValueError(
                "dvd_prob needs the DVD geometry of RealESRGANDataset image samples"
            )
        for sample, path in zip(samples, data["gt_path"], strict=True):
            geometry = sample.geometry
            if not geometry.has_context(gt_size):
                if not self.dvd_fallback_warned:
                    self.dvd_fallback_warned = True
                    get_root_logger().warning(
                        "DVD needs a %dx%d patch with real image context; %s is "
                        "%dx%d after preprocessing. Batches with such images take "
                        "the ordinary degradation chain.",
                        gt_size,
                        gt_size,
                        path,
                        geometry.width,
                        geometry.height,
                    )
                return False
        return True

    @torch.no_grad()
    def _feed_steps(self, data: DataFeed) -> DegradationSteps:
        """feed_data as steps: yields each degradation job (video-codec and
        WebP round trips) and receives its frames.

        The random draws happen between yields. Each suspended chain keeps
        its kernels and GT in locals until it sets self.lq and self.gt.
        """
        if self.is_train:
            assert (
                "gt" in data
                and "kernel1" in data
                and "kernel2" in data
                and "sinc_kernel" in data
            )
            center_crop_height = _collated_any(data.get("otf_padded_height"))
            center_crop_width = _collated_any(data.get("otf_padded_width"))
            # training data synthesis
            gt_source = data["gt"]
            if gt_source.device != self.device:
                gt_source = gt_source.to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )
            target_gt = data.get("target_gt")
            if target_gt is not None and target_gt.device != self.device:
                target_gt = target_gt.to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )

            gt = target_gt if target_gt is not None else gt_source
            self.gt = gt

            # A kernel whose stage has probability 0 is never applied and stays
            # where it is.
            kernel1 = data["kernel1"]
            if self.opt.blur_prob > 0 and kernel1.device != self.device:
                kernel1 = kernel1.to(
                    self.device,
                    non_blocking=True,
                )
            self.kernel1 = kernel1

            kernel2 = data["kernel2"]
            if self.opt.blur_prob2 > 0 and kernel2.device != self.device:
                kernel2 = kernel2.to(
                    self.device,
                    non_blocking=True,
                )
            self.kernel2 = kernel2

            sinc_kernel = data["sinc_kernel"]
            if (
                self.opt.datasets["train"].final_sinc_prob > 0
                and sinc_kernel.device != self.device
            ):
                sinc_kernel = sinc_kernel.to(
                    self.device,
                    non_blocking=True,
                )
            self.sinc_kernel = sinc_kernel

            ori_h, ori_w = gt_source.size()[2:4]

            gt_size = self.opt.datasets["train"].gt_size
            assert gt_size is not None
            use_dvd = (
                self.opt.dvd_prob > 0
                and RNG.get_rng().uniform() < self.opt.dvd_prob
                and self._dvd_fits(data, gt_size)
            )
            settings = self.opt.neosr_samples
            if use_dvd:
                # The DVD pair replaces the chain and the final crop below: no
                # USM, random resize, codec, noise or post-crop artifacts on top.
                active = set()
            elif settings and settings.clean_prob == 1:
                active = set()
                clean = clean_resize(gt_source, (ori_h // self.opt.scale, ori_w // self.opt.scale), settings)
                self.lq = (clean * 255).round().clamp_(0, 255).div_(255)
            else:
                # ----------------------- The first degradation process ----------------------- #
                if self.opt.lq_usm:
                    usm_sharpener = USMSharp(
                        RNG.get_rng().integers(
                            *self.opt.lq_usm_radius_range, dtype=int, endpoint=True
                        )
                    ).to(
                        self.device,
                        memory_format=self.memory_format,
                        non_blocking=True,
                    )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765
                    out = usm_sharpener(gt_source)
                else:
                    out = gt_source

                # thick lines
                if RNG.get_rng().uniform() < self.opt.thicklines_prob:
                    assert self.thicklines is not None
                    out = self.thicklines(out)

                # ── Select which custom degradations are active this iteration ──
                active = self._select_active_degradations()

                # ── NLMeans denoise source (before all degradations) ──
                if (
                    "nlmeans" in active
                    and self.opt.otf_hf_noise_denoise_lq
                    and self.opt.otf_hf_noise_prob > 0
                ):
                    out = nlmeans_denoise_pt(
                        out, h=self.opt.otf_hf_noise_denoise_strength
                    )

                # ── Source-level artifacts (before blur1) ──
                if "ntsc" in active and RNG.get_rng().uniform() < self.opt.ntsc_prob:
                    out = self._apply_ntsc(out)
                if (
                    "rainbow" in active
                    and RNG.get_rng().uniform() < self.opt.rainbow_prob
                ):
                    out = self._apply_rainbow(out)
                if (
                    "ghosting" in active
                    and RNG.get_rng().uniform() < self.opt.ghosting_prob
                ):
                    out = self._apply_ghosting(out)
                if (
                    "interlace" in active
                    and RNG.get_rng().uniform() < self.opt.interlace_prob
                ):
                    out = self._apply_interlace(out)
                if (
                    "scanline" in active
                    and RNG.get_rng().uniform() < self.opt.scanline_prob
                ):
                    out = self._apply_scanline(out)

                # blur
                if RNG.get_rng().uniform() < self.opt.blur_prob:
                    out = filter2d(out, kernel1)

                # ── Mastering artifacts (after blur1, before shift) ──
                if (
                    "lowpass" in active
                    and RNG.get_rng().uniform() < self.opt.lowpass_prob
                ):
                    out = self._apply_lowpass(out)
                if (
                    "overshoot" in active
                    and RNG.get_rng().uniform() < self.opt.overshoot_prob
                ):
                    out = self._apply_overshoot(out)

                # channel shift (before resize1)
                if "shift" in active and RNG.get_rng().uniform() < self.opt.shift_prob:
                    out = self._apply_shift(out)

                # chroma subsampling (before resize1)
                if (
                    "subsampling" in active
                    and RNG.get_rng().uniform() < self.opt.subsampling_prob
                ):
                    out = self._apply_subsampling(out)

                # random resize
                updown_type = _weighted_choice(
                    ("up", "down", "keep"), self.opt.resize_prob
                )
                if updown_type == "up":
                    scale = RNG.get_rng().uniform(1, self.opt.resize_range[1])
                elif updown_type == "down":
                    scale = RNG.get_rng().uniform(self.opt.resize_range[0], 1)
                else:
                    scale = 1

                if scale != 1:
                    assert len(self.opt.resize_mode_list) == len(
                        self.opt.resize_mode_prob
                    ), "resize_mode_list and resize_mode_prob must be the same length"
                    mode = _weighted_choice(
                        self.opt.resize_mode_list, self.opt.resize_mode_prob
                    )

                    out = self._otf_resize(
                        out,
                        scale_factor=scale,
                        mode=mode,
                    )

                out = self._stage_noise(out, self.opt.gaussian_noise_prob, self.opt.noise_range,
                                        self.opt.poisson_scale_range, self.opt.gray_noise_prob,
                                        settings.noise_prob if settings else 1.0)
                # compression (stage 1: JPEG, WebP, or video codec)
                if (
                    "compress" in active
                    and RNG.get_rng().uniform() < self.opt.compress_prob
                ):
                    out = yield from self._apply_compression(out, stage=1)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if RNG.get_rng().uniform() < self.opt.blur_prob2:
                    out = filter2d(out, kernel2)
                # random resize
                updown_type = _weighted_choice(
                    ("up", "down", "keep"), self.opt.resize_prob2
                )
                if updown_type == "up":
                    scale = RNG.get_rng().uniform(1, self.opt.resize_range2[1])
                elif updown_type == "down":
                    scale = RNG.get_rng().uniform(self.opt.resize_range2[0], 1)
                else:
                    scale = 1

                if scale != 1:
                    assert len(self.opt.resize_mode_list2) == len(
                        self.opt.resize_mode_prob2
                    ), "resize_mode_list2 and resize_mode_prob2 must be the same length"
                    mode = _weighted_choice(
                        self.opt.resize_mode_list2, self.opt.resize_mode_prob2
                    )
                    out = self._otf_resize(
                        out,
                        size=(
                            int(ori_h / self.opt.scale * scale),
                            int(ori_w / self.opt.scale * scale),
                        ),
                        mode=mode,
                    )
                out = self._stage_noise(out, self.opt.gaussian_noise_prob2, self.opt.noise_range2,
                                        self.opt.poisson_scale_range2, self.opt.gray_noise_prob2,
                                        settings.noise_prob2 if settings else 1.0)

                # Compression + the final sinc filter
                # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
                # as one operation.
                # We consider two orders:
                #   1. [resize back + sinc filter] + compression
                #   2. compression + [resize back + sinc filter]
                # Empirically, we find other combinations (sinc + compression + Resize) will introduce twisted lines.

                assert len(self.opt.resize_mode_list3) == len(
                    self.opt.resize_mode_prob3
                ), "resize_mode_list3 and resize_mode_prob3 must be the same length"

                mode = _weighted_choice(
                    self.opt.resize_mode_list3, self.opt.resize_mode_prob3
                )
                if RNG.get_rng().uniform() < 0.5:
                    # resize back + the final sinc filter
                    out = self._otf_resize(
                        out,
                        size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                        mode=mode,
                    )
                    if self.opt.datasets["train"].final_sinc_prob > 0:
                        out = filter2d(out, sinc_kernel)
                    else:
                        # The final kernel is the unit pulse: filtering only returns
                        # the image in standard layout, which later stages' results
                        # depend on.
                        out = out.contiguous()
                    if settings and settings.sharpen_prob:
                        out = sharpen_samples(out, settings)
                    # compression (stage 2)
                    if (
                        "compress" in active
                        and RNG.get_rng().uniform() < self.opt.compress_prob2
                    ):
                        out = yield from self._apply_compression(out, stage=2)
                else:
                    # compression (stage 2)
                    if (
                        "compress" in active
                        and RNG.get_rng().uniform() < self.opt.compress_prob2
                    ):
                        out = yield from self._apply_compression(out, stage=2)
                    # resize back + the final sinc filter
                    out = self._otf_resize(
                        out,
                        size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                        mode=mode,
                    )
                    if self.opt.datasets["train"].final_sinc_prob > 0:
                        out = filter2d(out, sinc_kernel)
                    else:
                        # The final kernel is the unit pulse: filtering only returns
                        # the image in standard layout, which later stages' results
                        # depend on.
                        out = out.contiguous()

                    if settings and settings.sharpen_prob:
                        out = sharpen_samples(out, settings)

                if settings and settings.clean_prob:
                    mask = sample_mask(out, settings.clean_prob)
                    clean = clean_resize(gt_source, out.shape[-2:], settings)
                    out = torch.where(mask, clean, out)
                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # No job follows: the attributes are this call's from here on.
            self.gt = gt
            # Center away the synthetic OTF work-canvas border. The larger source
            # crop was already sampled randomly by RealESRGANDataset.
            if use_dvd:
                self.lq, self.gt = dvd_pair(
                    data["dvd"],
                    gt_source,
                    target_gt,
                    gt_size,
                    self.opt.dvd_strength,
                    center_height=center_crop_height,
                    center_width=center_crop_width,
                )
            elif target_gt is not None:
                cropped_gts, self.lq = paired_random_crop_multi_gt(
                    [self.gt, gt_source],
                    self.lq,
                    gt_size,
                    self.opt.scale,
                    center_height=center_crop_height,
                    center_width=center_crop_width,
                )
                self.gt = cropped_gts[0]
            else:
                self.gt, self.lq = paired_random_crop(
                    self.gt,
                    self.lq,
                    gt_size,
                    self.opt.scale,
                    center_height=center_crop_height,
                    center_width=center_crop_width,
                )
            if "hf_noise" in active:
                self._apply_hf_noise()

            # dithering (last degradation, applied to LQ only)
            dithered = "dithering" in active and RNG.get_rng().uniform() < self.opt.dithering_prob
            if dithered:
                self.lq = self._apply_dithering(self.lq)

            # The ordinary and clean producers rounded LQ immediately before
            # the crop. DVD replaces that source; dithering runs after the crop.
            if self.opt.queue_lq_uint8 and not self.clip_mode:
                self._queue_lq_quantized = not use_dvd and not dithered
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            if self.vosr_target is not None:
                # After the pair pool: VOSR restores exactly the LR this step trains on. clone() leaves
                # inference mode, so losses and discriminator penalties can use the target normally.
                self.gt = self.vosr_target.restore(self.lq, self.opt.scale).clone(
                    memory_format=self.memory_format
                )

            if self._otf_luma is not None:
                # JPEG, chroma shifts and an optional RGB teacher consume the original
                # colour signal. Only the model-facing pair is one-channel.
                self.gt = dataset_luma_pt(self.gt, *self._otf_luma)
                self.lq = dataset_luma_pt(self.lq, *self._otf_luma).mul_(255).round_().clamp_(0, 255).div_(255)

            i = 1
            if self.otf_debug:
                os.makedirs(OTF_DEBUG_PATH, exist_ok=True)
                while os.path.exists(rf"{OTF_DEBUG_PATH}/{i:06d}_otf_lq.png"):
                    i += 1

                if i <= self.otf_debug_limit or self.otf_debug_limit == 0:
                    torchvision.utils.save_image(
                        self.lq,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_lq.png"),
                        padding=0,
                    )

                    torchvision.utils.save_image(
                        self.gt,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_gt.png"),
                        padding=0,
                    )

            # moa
            if self.is_train and self.batch_augment:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)
        else:
            # for paired training or validation
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


class ThickLines(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, original: Tensor) -> Tensor:
        # Calculate the padding needed for the convolution
        pad = self.kernel_size // 2

        # Apply a max pooling operation with the same kernel size to both images
        min_original = -F.max_pool2d(-original, self.kernel_size, padding=pad, stride=1)
        avg_original = original * 3 / 4 + min_original * 1 / 4

        return avg_original

    def blend(self, original: Tensor, usm: Tensor) -> Tensor:
        input = torch.cat((original, usm), dim=1)
        return self.conv(input)  # pyright: ignore[reportCallIssue]


class DegradeAheadPrefetcher:
    """The trainer's prefetcher, delivering training batches the model has
    already degraded, so the degradation jobs of a batch run in worker threads
    while the training steps before it run.

    Each next() takes the next raw batch; every batch in flight, oldest first,
    advances by one wave when its age in calls is a multiple of the model's
    ahead_stride; the new batch starts at age 0. 2 x ahead_stride batches stay
    in flight besides the one delivered, the oldest, so every raw batch is
    delivered once, in order, ahead_stride x 2 calls after it started.

    A batch's first stage is drawn that many iterations before its training
    step, between the later stages of the batches ahead of it: the random
    draws of different batches are interleaved, not taken one batch after the
    other, so a seed gives other batches than feeding them one by one does.
    """

    def __init__(
        self, prefetcher: CPUPrefetcher | CUDAPrefetcher, model: RealESRGANModel
    ) -> None:
        self.prefetcher = prefetcher
        self.model = model
        self.in_flight: deque[DegradingBatch] = deque()

    def next(self) -> DataFeed | None:
        stride = self.model.ahead_stride
        with self.model.data_feed_context():
            while True:
                data = self.prefetcher.next()
                for batch in self.in_flight:
                    batch.age += 1
                    if batch.age % max(stride, 1) == 0:
                        self.model.advance_batch(batch)
                if data is not None:
                    self.in_flight.append(self.model.start_batch(data))
                if data is None or len(self.in_flight) > 2 * stride:
                    break
            if not self.in_flight:
                return None
            return self.model.finish_batch(self.in_flight.popleft())

    def drop(self) -> None:
        """Forget the batches in flight, freeing their device memory, without
        waiting for their jobs; next() goes on with the next raw batch."""
        self.in_flight.clear()

    def close(self) -> None:
        self.drop()
        self.prefetcher.close()

    def reset(self) -> None:
        self.drop()
        self.prefetcher.reset()

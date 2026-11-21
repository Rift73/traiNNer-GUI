"""Paired batch augmentation for Corpsecreate/neosr at 3fe738f43d66."""

from math import sqrt

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from trainner_config.neosr_batchaug import NEOSR_BATCH_ORDER, paired_batchaug_slots

from traiNNer.utils import RNG


def _axis_box(length: int, ratio: float, rng: Generator) -> tuple[int, int]:
    extent = max(1, int(length * ratio))
    start = int(rng.integers(length)) - extent // 2
    return max(0, start), min(length, start + extent)


def _resize(img: Tensor, height: int, width: int, antialias: bool) -> Tensor:
    return F.interpolate(
        img, (height, width), mode="bicubic", antialias=antialias
    ).clamp(0, 1)


class NeoSRBatchAugment:
    def __init__(
        self, modes: list[str], weights: list[float], multi_prob: float, antialias: bool
    ) -> None:
        slots = paired_batchaug_slots(modes, weights, multi_prob, antialias)
        active = [(name, w) for name, w in slots if w > 0]
        self.names = [name for name, _ in active]
        self.weights = np.array([w for _, w in active], dtype=np.float64)
        if active:
            self.weights /= self.weights.sum()
        self.multi_prob = multi_prob
        self.antialias = antialias

    def _choose(self, rng: Generator) -> tuple[str, ...]:
        count = len(self.names)
        if count > 1 and rng.random() < self.multi_prob:
            # With >2 slots the upper bound is exclusive in the pinned source.
            draws = rng.integers(2, count) if count > 2 else 2
            selected = set(
                rng.choice(self.names, size=draws, replace=False, p=self.weights)
            )
        else:
            selected = {rng.choice(self.names, p=self.weights)}
        return tuple(name for name in NEOSR_BATCH_ORDER if name in selected)

    @torch.no_grad()
    def __call__(
        self, gt: Tensor, lq: Tensor, scale: int
    ) -> tuple[str, Tensor, Tensor]:
        if not self.names:
            return "none", gt, lq
        rng = RNG.get_rng()
        modes = self._choose(rng)
        if not modes:
            return "none", gt, lq
        if gt.shape[0] == 1 and modes != ("cutblur",):
            raise ValueError(
                "Selected NeoSR batch augmentations need batch >1 except cutblur."
            )
        if gt.ndim != 4 or lq.ndim != 4 or gt.shape[:2] != lq.shape[:2]:
            raise ValueError(
                "NeoSR paired augmentation expects matching N and C in NCHW batches."
            )
        h, w = lq.shape[-2:]
        if scale < 1 or gt.shape[-2:] != (h * scale, w * scale):
            raise ValueError(
                f"GT {tuple(gt.shape)} must be {scale}x LQ {tuple(lq.shape)}."
            )

        gt, lq = gt.clone(), lq.clone()
        for mode in modes:
            if mode == "mixup":
                lam = rng.uniform(0.4, 0.6)
                partner = torch.randperm(gt.shape[0], device=gt.device)
                gt = lam * gt + (1 - lam) * gt[partner]
                lq = lam * lq + (1 - lam) * lq[partner]
                continue

            if mode == "cutmix":
                ratio = sqrt(1.0 - rng.uniform(0, 0.9))
            elif mode == "resizemix":
                ratio = rng.uniform(0.2, 0.9)
            else:  # cutblur
                ratio = rng.uniform(0.2, 0.7)
            top, bottom = _axis_box(h, ratio, rng)
            left, right = _axis_box(w, ratio, rng)
            lq_box = lq[..., top:bottom, left:right]
            gt_box = gt[..., top * scale : bottom * scale, left * scale : right * scale]
            if mode == "cutblur":
                patch = gt_box
                if scale > 1:
                    patch = _resize(patch, bottom - top, right - left, True)
                lq_box.copy_(patch)
            else:
                partner = torch.randperm(gt.shape[0], device=gt.device)
                if mode == "cutmix":
                    gt_box.copy_(gt_box[partner])
                    lq_box.copy_(lq_box[partner])
                else:  # resizemix
                    gt_box.copy_(
                        _resize(gt[partner], *gt_box.shape[-2:], self.antialias)
                    )
                    lq_box.copy_(
                        _resize(lq[partner], *lq_box.shape[-2:], self.antialias)
                    )

        return "+".join(modes), gt, lq

"""Tiled inference and validation-only network state transitions.

SRModel keeps ownership of the networks and preparation keys. These helpers
perform inference work without introducing another state holder or module wrapper.
"""

from __future__ import annotations

# This module implements SRModel internals without owning another copy of state.
# ruff: noqa: SLF001
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from traiNNer.training.schedule_free_weights import evaluation_view, preserve_module_modes

if TYPE_CHECKING:
    from traiNNer.models.sr_model import SRModel


def _context_spans(length: int, size: int, halo: int) -> Iterator[tuple[int, int, int]]:
    """Yield a window origin and the disjoint interval it owns."""
    first = 0
    origin = 0
    while origin + size < length:
        end = origin + size - halo
        yield origin, first, end
        first = end
        origin = min(origin + size - 2 * halo, length - size)
    yield origin, first, length


def _infer_context(net: nn.Module, lq: Tensor, size: int, halo: int, scale: int) -> Tensor:
    if 2 * halo >= size:
        raise ValueError("Context tiling needs a tile larger than twice its halo.")
    height, width = lq.shape[-2:]
    output = None
    columns = list(_context_spans(width, size, halo))
    for y, top, bottom in _context_spans(height, size, halo):
        for x, left, right in columns:
            patch = net(lq[..., y:y + size, x:x + size])
            if output is None:
                output = torch.empty((*patch.shape[:2], height * scale, width * scale),
                                     dtype=patch.dtype, device="cpu")
            core = patch[..., (top-y)*scale:(bottom-y)*scale, (left-x)*scale:(right-x)*scale]
            output[..., top*scale:bottom*scale, left*scale:right*scale].copy_(core)
            del core, patch
    assert output is not None
    return output


def infer_tiled(
    net: nn.Module,
    lq: Tensor,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    mode: str = "blend",
) -> Tensor:
    """Use the configured tile geometry without changing the network's mode."""
    b, _, h, w = lq.shape
    if not 0 <= tile_overlap < tile_size or scale < 1:
        raise ValueError("Tiled inference requires a positive scale and 0 <= overlap < tile size.")
    assert b == 1, "Only batch size 1 is supported for tiled inference"
    if mode == "context":
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            return _infer_context(net, lq, tile_size, tile_overlap, scale)
    if mode != "blend":
        raise ValueError(f"Unknown tile mode {mode!r}.")

    if h <= tile_size and w <= tile_size:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            return net(lq)

    pad_h = (tile_size - (h % tile_size)) % tile_size if h > tile_size else 0
    pad_w = (tile_size - (w % tile_size)) % tile_size if w > tile_size else 0

    lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, h_pad, w_pad = lq.shape

    output = None
    weight_map = torch.zeros((1, 1, h_pad * scale, w_pad * scale), device=lq.device,
                             dtype=torch.float32)
    hr_tile = tile_size * scale
    # Pixel centres have positive weight even at exposed image borders and
    # between disjoint tiles. Their normalized sum preserves constant images.
    position = torch.arange(hr_tile, device=lq.device, dtype=torch.float32) + 0.5
    axis = torch.minimum(position, hr_tile - position) * (2 / hr_tile)
    weight = torch.outer(axis, axis)

    stride = tile_size - tile_overlap
    tiles_y = max(1, (h_pad - tile_overlap + stride - 1) // stride)
    tiles_x = max(1, (w_pad - tile_overlap + stride - 1) // stride)

    for y in range(tiles_y):
        for x in range(tiles_x):
            in_y0 = y * stride
            in_x0 = x * stride
            in_y1 = min(in_y0 + tile_size, h_pad)
            in_x1 = min(in_x0 + tile_size, w_pad)

            lq_patch = lq[:, :, in_y0:in_y1, in_x0:in_x1]

            ph, pw = lq_patch.shape[-2:]
            pad_bottom = max(tile_size - ph, 0) if ph < tile_size else 0
            pad_right = max(tile_size - pw, 0) if pw < tile_size else 0

            if pad_bottom > 0 or pad_right > 0:
                pad_bottom = min(pad_bottom, ph - 1)
                pad_right = min(pad_right, pw - 1)
                lq_patch = torch.nn.functional.pad(
                    lq_patch,
                    (0, pad_right, 0, pad_bottom),
                    mode="reflect",
                )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out_patch = net(lq_patch)
            if output is None:
                output = weight_map.new_zeros((b, out_patch.shape[1], h_pad * scale, w_pad * scale))
            out_patch = out_patch[:, :, : ph * scale, : pw * scale]
            w_patch = weight[: ph * scale, : pw * scale]

            out_y0 = in_y0 * scale
            out_x0 = in_x0 * scale
            out_y1 = out_y0 + ph * scale
            out_x1 = out_x0 + pw * scale

            output[:, :, out_y0:out_y1, out_x0:out_x1] += out_patch * w_patch
            weight_map[:, :, out_y0:out_y1, out_x0:out_x1] += w_patch

    assert output is not None
    out_final = output.div_(weight_map)
    return out_final[:, :, : h * scale, : w * scale]


@contextmanager
def validation_inference_context(model: SRModel, current_iter: int | None) -> Iterator[None]:
    """Prepare inference once and keep it stable for a validation dataset.

    Some architectures build reversible eval-only caches from their training
    parameters. Re-entering ``eval()`` for every image rebuilds those caches;
    doing it once per weight identity keeps validation exact and removes that
    repeated work.
    """
    if model._validation_inference_net is not None:
        raise RuntimeError("validation inference contexts cannot be nested")

    previous_is_train = model.is_train
    inference_net = model._select_inference_net(
        unwrap_compiled=previous_is_train and model.use_compile
    )
    was_training = inference_net.training
    prepared_key = (id(inference_net), current_iter) if current_iter is not None else None

    optimizer = getattr(model, "optimizer_g", None) if model.net_g_ema is None else None
    restore = model.net_g_ema is None and model.opt.is_train
    modes = preserve_module_modes(inference_net) if restore else nullcontext()
    with modes, evaluation_view(optimizer):
        model.is_train = False
        try:
            # A newly updated EMA can have an eval-mode cache from the previous step.
            if inference_net.training or prepared_key is None or model._validation_prepared_key != prepared_key:
                inference_net.eval()
            model._validation_prepared_key = prepared_key
            model._validation_inference_net = inference_net
            if current_iter is not None and model.opt.is_train and model.opt.val and model.opt.val.record_samples:
                from traiNNer.training.checkpoint_association import observe_inference

                with observe_inference(model, inference_net, current_iter):
                    yield
            else:
                yield
        finally:
            model._validation_inference_net = None
            model.is_train = previous_is_train
            if restore and was_training:
                model._validation_prepared_key = None

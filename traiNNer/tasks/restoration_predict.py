"""CPU deployment for reconstruction and scene-local temporal restoration."""
from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from trainner_config.rdual import (
    complete_neuflow_state,
    ensure_neuflow_weights,
    load_neuflow_weights,
)
from trainner_config.tasks import recurrent_architectures, resolve_task, task_name
from trainner_data.frame_sequences import FrameWindowIndex

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import TEMPORAL_ARCHS
from traiNNer.models.checkpoint_payload import (
    _checkpoint_tensors,
    load_matching_network_checkpoint,
    read_network_checkpoint,
    strip_common_prefix,
)
from traiNNer.utils.img_util import tensor2img


def sequence_model(opt) -> bool:
    architecture = (opt.network_g or {}).get("type", "").lower()
    return architecture in TEMPORAL_ARCHS or architecture in recurrent_architectures


def _window_width(opt, network: nn.Module, requested: int | None) -> int:
    fixed = getattr(network, "num_frames", None) or getattr(network, "clip_size", None)
    if fixed is not None:
        if requested is not None and requested != fixed:
            raise ValueError(f"This network expects exactly {fixed} frames per window.")
        return fixed
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError("clipSize must be a positive integer.")
        return requested
    sizes = {ds.clip_size for key, ds in opt.datasets.items()
             if key.partition("_")[0] in ("val","test") and ds.clip_size}
    if not sizes and opt.datasets.get("train") is not None and opt.datasets["train"].clip_size:
        sizes.add(opt.datasets["train"].clip_size)
    if len(sizes) != 1:
        raise ValueError("Choose clipSize for this variable-window network or supply one evaluation clip size.")
    return sizes.pop()


class Reconstruction(nn.Module):
    """The same encoder -> decoder used during autoencoder validation."""

    def __init__(self, network: nn.Module, scale: int):
        super().__init__()
        self.network = network
        self.scale = scale

    def forward(self, image: Tensor) -> Tensor:
        if not torch.jit.is_tracing() and any(size % self.scale for size in image.shape[-2:]):
            raise ValueError(f"Reconstruction height and width must be divisible by {self.scale}.")
        return self.network(image).float().clamp(0, 1)


@torch.inference_mode()
def predict_sequence(opt, checkpoint: Path, source: Path, destination: Path, clip_size: int | None):
    # These readers/writers are shared with ordinary trainer image deployment.
    from .predict import _prediction_paths, _read_image, _write_new_file

    if not source.is_dir():
        raise ValueError("Select a folder of chronological frames for temporal prediction.")
    if opt.input_pixel_format != "rgb" or opt.output_pixel_format != "rgb":
        raise ValueError("Frame sequence prediction requires RGB inputs and outputs.")
    if opt.val and opt.val.tile_size:
        raise ValueError("Frame sequence prediction requires full frames (val.tile_size: 0).")
    opt.is_train = False
    opt.task = task_name(opt)
    selection = resolve_task(opt)
    assert selection is not None and opt.network_g is not None
    _, architecture, _ = selection
    network = build_network({**opt.network_g, "scale": opt.scale}).to(device="cpu", dtype=torch.float32)
    uses_neuflow = hasattr(network, "flow_loaded")
    if uses_neuflow:
        # Older checkpoints can omit the frozen optical flow.
        weights, _ = _checkpoint_tensors(
            read_network_checkpoint(checkpoint), checkpoint, opt.path.param_key_g
        )
        weights = strip_common_prefix(weights, ("module.", "_orig_mod.", "netG."))
        state = complete_neuflow_state(weights, network.state_dict())
        network.load_state_dict(state, strict=True)
    else:
        load_matching_network_checkpoint(network, checkpoint, opt.path.param_key_g)
    network.eval()

    width = _window_width(opt, network, clip_size)
    index = FrameWindowIndex([str(source)], width)
    if not index:
        raise ValueError(f"No scene in {source} contains {width} frames. Use numbered PNG/JPEG frames.")
    pairs = _prediction_paths(source, destination, list(index.centers()))
    if uses_neuflow and not bool(network.flow_loaded):
        path = ensure_neuflow_weights(opt, Path.cwd())
        network.load_flow_weights(load_neuflow_weights(path))

    # Consecutive windows overlap. Retain at most one window's decoded inputs;
    # the learned state still starts at the same center-specific validation boundary.
    @lru_cache(maxsize=width)
    def decode(path: str):
        return _read_image(Path(path), segmentation=False)[0]

    for i, (center, target) in enumerate(pairs):
        paths, _ = index.window(i)
        frames = [decode(path) for path in paths]
        if any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError(f"All frames around {center} must have the same dimensions.")
        image = torch.stack(frames).unsqueeze(0)
        result = network(image)
        if architecture in recurrent_architectures:
            if not isinstance(result, Tensor) or result.ndim != 5 or result.shape[:2] != (1,width):
                raise ValueError(f"{architecture} returned an invalid temporal output.")
            result = result[:,width // 2]
        h, w = image.shape[-2:]
        if (not isinstance(result, Tensor) or result.shape != (1,3,h*opt.scale,w*opt.scale)
                or not result.is_floating_point() or not torch.isfinite(result).all()):
            raise ValueError(f"{architecture} returned an invalid RGB center frame for {center}.")
        png = Image.fromarray(tensor2img(result, to_bgr=False))
        _write_new_file(target, partial(png.save, format="PNG"))
    return {"task": opt.task, "architecture": architecture, "output_kind": "rgb",
            "clipSize": width, "scenes": len(index.scenes), "inputFrames": index.frame_count,
            "centerFrames": len(pairs), "boundaryFramesOmitted": index.frame_count - len(pairs),
            "outputs": [str(path) for _, path in pairs]}

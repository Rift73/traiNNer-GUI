"""Fixed-window RGB centre-frame export for temporal restoration."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode
from trainner_config.options import ReduxOptions
from trainner_config.tasks import require_task_action, resolve_task, task_name

from traiNNer.archs import build_network
from traiNNer.models.checkpoint_payload import load_matching_network_checkpoint


class _CenterFrame(nn.Module):
    def __init__(self, network: nn.Module, center: int) -> None:
        super().__init__()
        self.network = network
        self.center = center

    def forward(self, frames: Tensor) -> Tensor:
        return self.network(frames)[:, self.center]


class _FlowDownsample(torch.autograd.Function):
    """Lower the frozen flow input resize to standard ONNX Resize-18."""

    @staticmethod
    def forward(ctx, images: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(
            images, size=size, mode="bilinear", align_corners=False, antialias=True
        )

    @staticmethod
    def symbolic(g, images, size):
        from torch.onnx.symbolic_helper import _optional_input_placeholder_tensor

        omitted = _optional_input_placeholder_tensor(g)
        sizes = g.op("Constant", value_t=torch.tensor(size, dtype=torch.int64))
        # Use the actual rounded H/W. PyTorch centers the widened triangle at
        # half pixels and renormalizes the in-bounds weights at each edge.
        return g.op(
            "Resize", images, omitted, omitted, sizes, axes_i=[2, 3],
            mode_s="linear", coordinate_transformation_mode_s="half_pixel",
            antialias_i=1, exclude_outside_i=1,
        )


class _FixedFlowTrace(TorchFunctionMode):
    """Standard ONNX spellings for the native RSR guidance trace only."""

    def __init__(self, frames: Tensor, shape: torch.Size) -> None:
        super().__init__()
        self.frames = frames
        self.shape = shape

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func == Tensor.shape.__get__ and args[0] is self.frames:
            # The legacy tracer tensorizes sizes. Guidance rounds the fixed H/W
            # in Python to select the NeuFlow grid and pair batching.
            return self.shape
        if (
            func is F.interpolate
            and kwargs
            and kwargs.get("antialias")
            and kwargs.get("mode") == "bilinear"
            and kwargs.get("align_corners") is False
            and kwargs.get("size") is not None
        ):
            return _FlowDownsample.apply(args[0], kwargs["size"])
        if func is torch.grid_sampler_2d:
            # NeuFlow's 4-D bilinear lookup uses the same modes and corner flag.
            return torch.grid_sampler(*args, **(kwargs or {}))
        if func is torch.hypot:
            x, y = args
            return (x.square() + y.square()).sqrt()
        return func(*args, **(kwargs or {}))


class _AlignedCenterFrame(_CenterFrame):
    def __init__(self, network: nn.Module, center: int, shape: torch.Size) -> None:
        super().__init__(network, center)
        self.shape = shape

    def forward(self, frames: Tensor) -> Tensor:
        if torch.onnx.is_in_onnx_export():
            with _FixedFlowTrace(frames, self.shape):
                guidance = self.network.guidance(frames)
            return self.network(frames, guidance=guidance)[:, self.center]
        return super().forward(frames)


def export_sequence_onnx(
    opt: ReduxOptions,
    checkpoint: Path,
    destination: Path,
    *,
    height: int,
    width: int,
    opset: int,
    clip_size: int | None = None,
) -> dict[str, Any]:
    require_task_action(opt, "export")
    architecture = str((opt.network_g or {})["type"]).lower()
    if architecture == "rsr_compact" and (type(clip_size) is not int or clip_size < 1):
        raise ValueError(
            "RSR ONNX export requires an explicit positive clip_size (--clip-size)."
        )
    resolve_task(opt)
    from .onnx_export import write_portable_onnx
    from .restoration_predict import _window_width

    assert opt.network_g is not None
    temporal_read = opt.network_g.get("temporal_read", "block")
    aligned = architecture == "rsr_compact" and temporal_read == "aligned"
    if aligned:
        from traiNNer.archs.rsr_arch import flow_working_factor

        if opset < 16:
            raise ValueError("Flow-aligned RSR needs ONNX opset 16 or later for GridSample.")
        if opset < 18 and clip_size != 1 and flow_working_factor(height, width) > 1:
            raise ValueError(
                "Flow-aligned RSR with adjacent frames and a shorter input side of at least "
                "512 pixels requires ONNX opset 18 or later for antialiased flow downsampling."
            )
    network = build_network({**opt.network_g, "scale": opt.scale}).to(device="cpu", dtype=torch.float32)
    clip_size = _window_width(opt, network, clip_size)
    load_matching_network_checkpoint(network, checkpoint, opt.path.param_key_g)

    center_index = clip_size // 2
    example = torch.zeros(
        (1, clip_size, 3, height, width), dtype=torch.float32, device="cpu"
    )
    if aligned:
        model = _AlignedCenterFrame(network, center_index, example.shape)
    elif architecture == "rsr_compact":
        model = _CenterFrame(network, center_index)
    else:
        model = network
    model.eval()
    expected = (1, 3, height * opt.scale, width * opt.scale)
    with torch.inference_mode():
        result = model(example)
    if result.shape != expected or not torch.isfinite(result).all():
        raise ValueError(
            f"{architecture} returned an invalid RGB centre frame. Expected {expected}."
        )

    metadata = {
        "task": task_name(opt),
        "architecture": architecture,
        "output_kind": "rgb",
        "fixed_shape": True,
        "clip_size": clip_size,
        "center_index": center_index,
        "input_name": "input",
        "input_layout": "NTCHW",
        "input_shape": list(example.shape),
        "input_dtype": "float32",
        "input_color": "rgb_0_1",
        "output_name": "rgb",
        "output_layout": "NCHW",
        "output_shape": list(result.shape),
        "output_dtype": "float32",
        "output_color": "rgb_unclamped",
        "opset": opset,
    }
    if architecture == "rsr_compact":
        metadata["temporal_read"] = temporal_read
    return write_portable_onnx(model, example, destination, metadata)

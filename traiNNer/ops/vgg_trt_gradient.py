"""Build a fixed-shape TensorRT input-gradient engine from the loaded VGG19 weights."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    import tensorrt as trt

    from traiNNer.losses.perceptual_fp16_loss import VGG

_ENGINES: WeakValueDictionary[int, InputGradientEngine] = WeakValueDictionary()
_TAPS = ("conv1_2", "conv2_2", "conv3_4", "conv4_4", "conv5_4")


def supported(vgg: VGG, x: Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.float32
        and x.ndim == 4
        and not torch.is_autocast_enabled("cuda")
        and tuple(vgg.stages) == _TAPS
        and not vgg.antialiased
        and not any(parameter.requires_grad for parameter in vgg.parameters())
        and x.shape[2] % 16 == 0
        and x.shape[3] % 16 == 0
        and x.shape[2] >= 16
        and x.shape[3] >= 16
    )


class InputGradientEngine:
    def __init__(self, trt: ModuleType, vgg: VGG, x: Tensor) -> None:
        self.shape = (x.device, tuple(x.shape))
        self.dtype = torch.float16
        dtype = trt.float16
        self.logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(self.logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()
        config.max_aux_streams = 0
        config.builder_optimization_level = 5
        if torch.backends.cudnn.conv.fp32_precision == "ieee":
            config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        batch, _, height, width = x.shape
        weights = []  # TensorRT reads the host tensors during the build.
        self.mask_names = []
        self.tap_names = []
        stages = list(vgg.stages.items())

        def transpose(tensor: trt.ITensor, order: tuple[int, ...]) -> trt.ITensor:
            node = network.add_shuffle(tensor)
            node.first_transpose = order
            return node.get_output(0)

        def masked(
            tensor: trt.ITensor, name: str, shape: tuple[int, ...]
        ) -> trt.ITensor:
            mask = network.add_input(name, trt.bool, shape)
            cast = network.add_cast(mask, dtype).get_output(0)
            return network.add_elementwise(
                tensor, cast, trt.ElementWiseOperation.PROD
            ).get_output(0)

        for i, (tap, stage) in enumerate(stages):
            end = list(stage.named_children())[-1][0]
            self.mask_names.extend(
                f"relu_{name}"
                for name, layer in stage.named_children()
                if isinstance(layer, nn.Conv2d) and name != end
            )
            if i + 1 < len(stages):
                self.mask_names.append(f"pool_{tap}")
            self.tap_names.append(f"dy_{tap}")

        gradient = None
        for i in reversed(range(len(stages))):
            tap, stage = stages[i]
            layers = [
                (name, layer)
                for name, layer in stage.named_children()
                if isinstance(layer, nn.Conv2d)
            ]
            h, w = height // 2**i, width // 2**i
            channels = layers[-1][1].out_channels
            shape = (batch, h, w, channels)
            direct = network.add_input(f"dy_{tap}", dtype, shape)
            if gradient is None:
                gradient = direct
            else:
                resize = network.add_resize(transpose(gradient, (0, 3, 1, 2)))
                resize.shape = (batch, channels, h, w)
                resize.resize_mode = trt.InterpolationMode.NEAREST
                expanded = transpose(resize.get_output(0), (0, 2, 3, 1))
                gradient = masked(expanded, f"pool_{tap}", shape)
                gradient = network.add_elementwise(
                    gradient, direct, trt.ElementWiseOperation.SUM
                ).get_output(0)
            for name, layer in reversed(layers):
                if name != layers[-1][0]:
                    gradient = masked(
                        gradient, f"relu_{name}", (batch, h, w, layer.out_channels)
                    )
                weight = (
                    layer.weight.detach()
                    .to(device="cpu", dtype=self.dtype)
                    .contiguous()
                )
                weights.append(weight)
                operation = network.add_deconvolution_nd(
                    transpose(gradient, (0, 3, 1, 2)),
                    layer.in_channels,
                    layer.kernel_size,
                    trt.Weights(dtype, weight.data_ptr(), weight.numel()),
                    trt.Weights(dtype),
                )
                operation.padding_nd = (
                    (0, 0) if layer.padding_mode == "replicate" else layer.padding
                )
                operation.name = f"vgg19_input_gradient_{name}"
                gradient = transpose(operation.get_output(0), (0, 2, 3, 1))
        gradient.name = "padded_gradient"
        network.mark_output(gradient)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(
                "TensorRT could not build the VGG19 input-gradient engine."
            )
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError(
                "TensorRT could not load the VGG19 input-gradient engine."
            )
        self.contexts = {}
        self.worker: torch.cuda.Stream | None = None
        _ENGINES[id(self)] = self

    def run(self, x: Tensor, masks: list[Tensor], gradients: list[Tensor]) -> Tensor:
        stream = torch.cuda.current_stream(x.device)
        if stream.cuda_stream:
            return self._execute(x, masks, gradients, stream)
        # TensorRT synchronizes enqueues on CUDA's legacy default stream.
        # The events preserve ordering without blocking the autograd worker.
        if self.worker is None:
            self.worker = torch.cuda.Stream(device=x.device)
        worker = self.worker
        worker.wait_stream(stream)
        with torch.cuda.stream(worker):
            result = self._execute(x, masks, gradients, worker)
            for tensor in (*masks, *gradients):
                tensor.record_stream(worker)
        stream.wait_stream(worker)
        result.record_stream(stream)
        return result

    def _execute(
        self,
        x: Tensor,
        masks: list[Tensor],
        gradients: list[Tensor],
        stream: torch.cuda.Stream,
    ) -> Tensor:
        handle = stream.cuda_stream
        context = self.contexts.get(handle)
        if context is None:
            context = self.engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    "TensorRT could not allocate the VGG19 execution context."
                )
            self.contexts[handle] = context
        bindings = {}
        for name, tensor in zip(
            self.mask_names + self.tap_names, [*masks, *gradients], strict=True
        ):
            bindings[name] = tensor.permute(0, 2, 3, 1).contiguous()
        result = torch.empty(
            tuple(self.engine.get_tensor_shape("padded_gradient")),
            device=x.device,
            dtype=self.dtype,
        )
        bindings["padded_gradient"] = result
        for name, tensor in bindings.items():
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected the VGG19 binding {name}.")
        if not context.execute_async_v3(handle):
            raise RuntimeError("TensorRT VGG19 input-gradient execution failed.")
        return result.permute(0, 3, 1, 2)


@torch.library.custom_op("trainner_vgg::input_gradient", mutates_args=())
def _input_gradient(
    x: Tensor, masks: list[Tensor], gradients: list[Tensor], key: int
) -> Tensor:
    return _ENGINES[key].run(x, masks, gradients)


@_input_gradient.register_fake
def _input_gradient_fake(
    x: Tensor, masks: list[Tensor], gradients: list[Tensor], key: int
) -> Tensor:
    b, c, h, w = x.shape
    return torch.empty(
        (b, c, h + 2, w + 2),
        device=x.device,
        dtype=gradients[0].dtype,
        memory_format=torch.channels_last,
    )


def _pool_winners(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    tiles = x.reshape(b, c, h // 2, 2, w // 2, 2)
    a, r = tiles[:, :, :, 0, :, 0], tiles[:, :, :, 0, :, 1]
    d, e = tiles[:, :, :, 1, :, 0], tiles[:, :, :, 1, :, 1]
    maximum = torch.maximum(torch.maximum(a, r), torch.maximum(d, e))
    valid = maximum > 0
    first, second, third = a == maximum, r == maximum, d == maximum
    top = torch.stack((valid & first, valid & ~first & second), dim=-1)
    bottom = torch.stack(
        (valid & ~first & ~second & third, valid & ~first & ~second & ~third), dim=-1
    )
    return (
        torch.stack((top, bottom), dim=3)
        .reshape_as(x)
        .contiguous(memory_format=torch.channels_last)
    )


class _Features(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: Tensor, vgg: VGG, engine: InputGradientEngine
    ) -> tuple[Tensor, ...]:
        h = (x - vgg.mean) / vgg.std
        masks, taps = [], []
        for tap, stage in vgg.stages.items():
            end = list(stage.named_children())[-1][0]
            for name, layer in stage.named_children():
                h = layer(h)
                if isinstance(layer, nn.Conv2d) and name != end:
                    masks.append(h > 0)
            taps.append(h)
            if tap != _TAPS[-1]:
                masks.append(_pool_winners(h))
        ctx.save_for_backward(x, vgg.std, *masks)
        ctx.engine = engine
        return tuple(taps)

    @staticmethod
    def backward(ctx: Any, *gradients: Tensor) -> tuple[Tensor, None, None]:
        x, std, *masks = ctx.saved_tensors
        # Normalize all five tap derivatives together. GradScaler and recipe weights
        # can change their magnitude without changing which FP16 range we need.
        peak = torch.stack([g.abs().amax() for g in gradients]).amax()
        scale = peak.clamp_min(1e-30).reciprocal()
        packed = [(g * scale).half() for g in gradients]
        padded = _input_gradient(x, masks, packed, id(ctx.engine)).float()
        dx = torch.ops.aten.replication_pad2d_backward.default(padded, x, [1, 1, 1, 1])
        return dx / (std * scale), None, None


@torch.compiler.disable
def build(vgg: VGG, x: Tensor) -> InputGradientEngine:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt_fp16_backward requires the TensorRT Python runtime."
        ) from exc
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up the VGG19 TensorRT loss before external CUDA graph capture."
        )
    with torch.cuda.device(x.device):
        return InputGradientEngine(trt, vgg, x)


def extract(vgg: VGG, x: Tensor, engine: InputGradientEngine) -> dict[str, Tensor]:
    return dict(zip(_TAPS, _Features.apply(x, vgg, engine), strict=True))

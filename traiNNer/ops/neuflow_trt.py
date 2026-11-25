"""TensorRT FP32 inference for the frozen NeuFlow v2 training guide."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from traiNNer.archs.rdual_arch import NeuFlowV2, _indexed_flow_pairs_per_call

if TYPE_CHECKING:
    import tensorrt as trt


class _IndexedFlow(nn.Module):
    def __init__(self, flow: NeuFlowV2) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, frames: Tensor, sources: Tensor) -> Tensor:
        return self.flow.forward_indexed(frames, sources)


def _export_flow(flow: NeuFlowV2, height: int, width: int) -> bytes:
    # Export a CPU copy. Its traced coordinate grids must never enter the training model.
    with torch.device("meta"):
        replica = NeuFlowV2()
    replica.load_state_dict(
        {name: tensor.detach().cpu() for name, tensor in flow.state_dict().items()},
        assign=True,
    )
    replica.eval()
    example = torch.zeros(3, 3, height, width)
    indices = torch.tensor([1, 0, 2, 1], dtype=torch.int32)
    result = io.BytesIO()
    torch.onnx.export(
        _IndexedFlow(replica).eval(),
        (example, indices),
        result,
        dynamo=False,
        opset_version=18,
        input_names=["frames", "sources"],
        output_names=["flow"],
        dynamic_axes={
            "frames": {0: "unique_frames"},
            "sources": {0: "directions"},
            "flow": {0: "directions"},
        },
    )
    return result.getvalue()


class FrozenFlowRunner:
    """Own one crop-specific engine and execution stream per training worker."""

    def __init__(self, flow: NeuFlowV2) -> None:
        try:
            import onnx  # noqa: F401
            import tensorrt
        except ImportError as exc:
            raise RuntimeError(
                "train.rdual_flow_tensorrt requires the TensorRT and ONNX Python packages."
            ) from exc
        self.trt = tensorrt
        self.flow = flow
        self.logger = tensorrt.Logger(tensorrt.Logger.WARNING)
        self.engine: trt.ICudaEngine | None = None
        self.context: trt.IExecutionContext | None = None
        self.runtime: trt.Runtime | None = None
        self.stream: torch.cuda.Stream | None = None
        self.key: tuple[torch.device, int, int] | None = None
        self.counts: tuple[int, int] | None = None
        flow.register_load_state_dict_post_hook(self._weights_loaded)

    def _weights_loaded(self, _module: nn.Module, _keys: Any) -> None:
        self._clear()

    def _clear(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        self.context = None
        self.engine = None
        self.runtime = None
        self.stream = None
        self.key = None
        self.counts = None

    def _build(self, device: torch.device, height: int, width: int) -> None:
        self._clear()
        trt = self.trt
        with torch.cuda.device(device):
            builder = trt.Builder(self.logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(_export_flow(self.flow, height, width)):
                errors = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise RuntimeError(f"TensorRT could not import NeuFlow v2.\n{errors}")
            config = builder.create_builder_config()
            config.builder_optimization_level = 5
            config.clear_flag(trt.BuilderFlag.TF32)
            config.max_aux_streams = 0
            cells = (height // 8) * (width // 8)
            # A single pair can exceed the usual correlation budget at large crops.
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, max(384 << 20, 3 * 8 * cells * cells)
            )
            profile = builder.create_optimization_profile()
            max_images = 2 * _indexed_flow_pairs_per_call(height, width)
            typical_frames = min(max_images, max_images // 2 + 2)
            profile.set_shape(
                "frames",
                (2, 3, height, width),
                (typical_frames, 3, height, width),
                (max_images, 3, height, width),
            )
            profile.set_shape("sources", (2,), (max_images,), (max_images,))
            config.add_optimization_profile(profile)
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError(
                    "TensorRT could not build the NeuFlow v2 training engine."
                )
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError(
                    "TensorRT could not load the NeuFlow v2 training engine."
                )
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    "TensorRT could not create the NeuFlow v2 execution context."
                )
            self.stream = torch.cuda.Stream(device=device)
            self.runtime, self.engine, self.context = runtime, engine, context
            self.key = (device, height, width)

    def __call__(self, frames: Tensor, sources: Tensor) -> Tensor:
        unique, _, height, width = frames.shape
        directions = sources.numel()
        if self.key != (frames.device, height, width):
            self._build(frames.device, height, width)
        context, worker = self.context, self.stream
        assert context is not None and worker is not None
        caller = torch.cuda.current_stream(frames.device)
        worker.wait_stream(caller)
        with torch.cuda.stream(worker):
            if self.counts != (unique, directions):
                for name, tensor in (("frames", frames), ("sources", sources)):
                    if not context.set_input_shape(name, tuple(tensor.shape)):
                        raise RuntimeError(
                            f"Unsupported NeuFlow v2 {name} shape {tuple(tensor.shape)}."
                        )
                self.counts = (unique, directions)
            output = torch.empty(
                (directions, 2, height, width),
                device=frames.device,
                dtype=torch.float32,
            )
            inputs = (frames.contiguous(), sources.contiguous())
            frames.record_stream(worker)
            sources.record_stream(worker)
            for name, contiguous in zip(("frames", "sources"), inputs, strict=True):
                if not context.set_tensor_address(name, contiguous.data_ptr()):
                    raise RuntimeError(
                        f"TensorRT could not bind the NeuFlow v2 input {name}."
                    )
            if not context.set_tensor_address("flow", output.data_ptr()):
                raise RuntimeError("TensorRT could not bind the NeuFlow v2 output.")
            if not context.execute_async_v3(worker.cuda_stream):
                raise RuntimeError("TensorRT NeuFlow v2 execution failed.")
        caller.wait_stream(worker)
        output.record_stream(caller)
        return output

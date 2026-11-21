"""VOSR 2.0 (github.com/cswry/VOSR, Apache-2.0) on 512x512 windows with the FP8 TensorRT stage engines.

Adapted from the vosr2-512-kit runner (chaiNNer-C v0.3.11's VOSR 2.0 FP8 stage engines rebuilt for one fixed 512 px
window). Changes: `Vosr2Window.__call__` takes an optional generator that refills each window's noise;
`pad_to_window` builds its indices on the image's device and `wavelet_colour` caches its kernel per device (no
host-to-device copy per call); the Pillow `prepare()` helper and the Windows DLL preload are left out.

One window: VAE encoder, latent normalization, noise, DINO on the window resized to 448 px, the DiT's keys/values and
one prediction, latent = noise - velocity, VAE decoder. No scene routing, face windows, colour correction or grain.

`lanes` windows run side by side on their own CUDA streams over one copy of the weights; each lane has fixed buffers,
one shared scratch block for its stages (they run one after another) and one CUDA graph for its whole window.
Measured in WSL on an RTX 5090 at a 600 W limit (TensorRT 11.4.0.106): 2 lanes = 24.3 ms per window, 2.99 GB of GPU
memory for the whole process.

Input: the low-resolution patch already upscaled to the output size with Pillow BICUBIC on 8-bit RGB, as a float
tensor in -1..1, (N, 3, 512, 512). Smaller outputs: reflect-pad to 512 (pad_to_window()) and crop the result.
Output: (N, 3, 512, 512) in -1..1.

Without a generator the noise is the lane's own, drawn once from `seed`, so a given window gives the same output in any
lane; with a generator every window gets fresh noise.
"""

import ctypes
from functools import cache
from pathlib import Path

import tensorrt as trt  # first: the plugin needs TensorRT's runtime library loaded
import torch
from torch.nn import functional

STAGES = (
    "encoder_pre",
    "encoder_qkv",
    "encoder_post",
    "decoder_pre",
    "decoder_qkv",
    "decoder_post",
    "attention_grouped",
    "dino",
    "dit_kv",
    "dit_cached",
)
# Qwen-Image VAE 2D latent statistics (its config.json)
LATENTS_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
LATENTS_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
]
DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
}


def pad_to_window(image: torch.Tensor) -> torch.Tensor:
    """Reflect an output-size image smaller than 512 px up to 512 (chaiNNer-C's pad_image for these sizes)."""
    height, width = image.shape[-2:]
    if height > 512 or width > 512:
        raise ValueError("one window is at most 512x512")

    def indices(length: int) -> torch.Tensor:
        target = torch.arange(512, device=image.device)
        if length == 1:
            return torch.zeros_like(target)
        phase = target % (2 * length - 2)
        return torch.minimum(phase, 2 * length - 2 - phase)

    return image.index_select(2, indices(height)).index_select(3, indices(width))


def rotary_tables(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DiT's RoPE for its trained 32x32 token grid (a 64x64 latent window)."""
    frequency = 1 / (10000 ** (torch.arange(0, 32, 2, device=device).float() / 32))
    y = torch.arange(32, device=device).float()
    x = torch.arange(32, device=device).float()
    fy = (y[:, None] * frequency).repeat_interleave(2, -1)
    fx = (x[:, None] * frequency).repeat_interleave(2, -1)
    angles = torch.cat(
        (fy[:, None, :].expand(32, 32, -1), fx[None, :, :].expand(32, 32, -1)), dim=-1
    ).reshape(1024, 64)
    return angles.cos().to(dtype).contiguous(), angles.sin().to(dtype).contiguous()


class _Stage:
    def __init__(
        self, engine: trt.ICudaEngine, scratch: torch.Tensor, device: torch.device
    ) -> None:
        self.context = engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED
        )
        if self.context is None or engine.device_memory_size_v2 > scratch.numel():
            raise RuntimeError("could not create an execution context")
        self.context.set_device_memory(scratch.data_ptr(), scratch.numel())
        self.tensors = {}
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            self.bind(
                name,
                torch.empty(
                    tuple(engine.get_tensor_shape(name)),
                    dtype=DTYPES[engine.get_tensor_dtype(name)],
                    device=device,
                ),
            )

    def bind(self, name: str, tensor: torch.Tensor) -> None:
        if not self.context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"cannot bind {name}")
        self.tensors[name] = tensor

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def run(self, stream: torch.cuda.Stream) -> None:
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")


class _Lane:
    def __init__(
        self, engines: dict[str, trt.ICudaEngine], device: torch.device, seed: int
    ) -> None:
        self.device = device
        self.scratch = torch.empty(
            max(e.device_memory_size_v2 for e in engines.values()),
            dtype=torch.uint8,
            device=device,
        )
        s = {
            name: _Stage(engine, self.scratch, device)
            for name, engine in engines.items()
        }
        self.encoder_attention, self.decoder_attention = (
            s["attention_grouped"],
            _Stage(engines["attention_grouped"], self.scratch, device),
        )
        for prefix in ("encoder", "decoder"):
            s[prefix + "_qkv"].bind("x", s[prefix + "_pre"]["result"])
            s[prefix + "_post"].bind("x", s[prefix + "_pre"]["result"])
        s["dit_kv"].bind("conditioning", s["dino"]["result"])
        s["dit_cached"].bind("kv", s["dit_kv"]["result"])
        cos, sin = rotary_tables(device, torch.bfloat16)
        s["dit_cached"].bind("cos", cos)
        s["dit_cached"].bind("sin", sin)
        self.s = s
        self.mean = torch.tensor(LATENTS_MEAN, device=device).view(1, 16, 1, 1)
        self.std = torch.tensor(LATENTS_STD, device=device).view(1, 16, 1, 1)
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device)[
            None, :, None, None
        ]
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=device)[
            None, :, None, None
        ]
        self.noise = torch.randn(
            (1, 16, 64, 64), generator=torch.Generator(device="cpu").manual_seed(seed)
        ).to(device)
        self.image = torch.zeros((1, 3, 512, 512), device=device)
        self.stream = torch.cuda.Stream(device=device)
        self.output: torch.Tensor | None = None
        for _ in range(2):
            self._window()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self._window()

    def _vae(self, prefix: str, attention: _Stage) -> torch.Tensor:
        s = self.s
        s[prefix + "_pre"].run(self.stream)
        s[prefix + "_qkv"].run(self.stream)
        q, k, v = (part.unsqueeze(1) for part in s[prefix + "_qkv"]["result"].unbind(1))
        attention["k"].copy_(k.transpose(-1, -2).unflatten(-1, (-1, 8)))
        attention["v"].copy_(v.unflatten(2, (-1, 8)))
        feature = s[prefix + "_pre"]["result"]
        result = torch.empty(q.shape, dtype=q.dtype, device=self.device)
        for start in range(0, q.shape[2], 1024):
            attention["q"].copy_(q[:, :, start : start + 1024])
            attention.run(self.stream)
            result[:, :, start : start + 1024].copy_(attention["result"])
        s[prefix + "_post"]["attention"].copy_(
            result.squeeze(1).transpose(1, 2).reshape_as(feature)
        )
        s[prefix + "_post"].run(self.stream)
        return s[prefix + "_post"]["result"]

    def _window(self) -> None:
        s = self.s
        with torch.cuda.stream(self.stream):
            s["encoder_pre"]["x"].copy_(self.image)
            latent = (
                self._vae("encoder", self.encoder_attention).float() - self.mean
            ) / self.std
            s["dit_cached"]["latent"].copy_(torch.cat((latent, self.noise), dim=1))
            conditioning = functional.interpolate(
                self.image * 0.5 + 0.5,
                size=(448, 448),
                mode="bicubic",
                align_corners=False,
            ).clamp_(0, 1)
            s["dino"]["x"].copy_((conditioning - self.dino_mean) / self.dino_std)
            s["dino"].run(self.stream)
            s["dit_kv"].run(self.stream)
            s["dit_cached"].run(self.stream)
            velocity = s["dit_cached"]["result"].float()
            s["decoder_pre"]["x"].copy_((self.noise - velocity) * self.std + self.mean)
            self.output = self._vae("decoder", self.decoder_attention)


class Vosr2Window:
    """VOSR 2.0 for (N, 3, 512, 512) windows in -1..1 on the GPU, `lanes` at a time."""

    def __init__(
        self,
        engine_dir: str | Path,
        plugin: str | Path,
        lanes: int = 2,
        seed: int = 42,
        device: str = "cuda:0",
    ) -> None:
        ctypes.CDLL(str(Path(plugin).resolve()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        self.device = torch.device(device)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engines = {}
        for name in STAGES:
            engine = self.runtime.deserialize_cuda_engine(
                (Path(engine_dir) / f"{name}.plan").read_bytes()
            )
            if engine is None:
                raise RuntimeError(
                    f"cannot load {name}.plan (other TensorRT version or GPU?)"
                )
            if engine.streamable_weights_size:
                engine.weight_streaming_budget_v2 = (
                    engine.streamable_weights_size
                )  # every weight resident
            self.engines[name] = engine
        with torch.cuda.device(self.device):
            self.lanes = [_Lane(self.engines, self.device, seed) for _ in range(lanes)]

    @torch.inference_mode()
    def __call__(
        self, windows: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if windows.shape[1:] != (3, 512, 512):
            raise ValueError("windows must be (N, 3, 512, 512)")
        current = torch.cuda.current_stream(self.device)
        outputs = []
        for start in range(0, windows.shape[0], len(self.lanes)):
            group = windows[start : start + len(self.lanes)].to(
                self.device, torch.float32
            )
            busy = self.lanes[: group.shape[0]]
            for lane, window in zip(busy, group, strict=True):
                lane.stream.wait_stream(current)
                with torch.cuda.stream(lane.stream):
                    lane.image.copy_(window[None])
                    if generator is not None:
                        # In place: the captured graph reads this buffer's address
                        lane.noise.normal_(generator=generator)
                    lane.graph.replay()
                    assert (
                        lane.output is not None
                    )  # set by the warm-up windows in _Lane.__init__
                    outputs.append(lane.output.float().clone())
            for lane in busy:
                current.wait_stream(lane.stream)
        return torch.cat(outputs)


@cache
def _gaussian_kernel(device: torch.device) -> torch.Tensor:
    """OpenCV's float32 Gaussian for sigma 5 (41 taps, the size GaussianBlur picks for that sigma), on `device`."""
    import cv2

    return torch.from_numpy(cv2.getGaussianKernel(41, 5, cv2.CV_32F)).to(device)


def wavelet_colour(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Upstream VOSR's wavelet_color_fix: output - blur(output) + blur(reference), Gaussian sigma 5 with reflect-101
    borders, both (N, 3, H, W) in -1..1 (reference = the bicubic input), clamped."""
    kernel = _gaussian_kernel(output.device)

    def blur(value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        horizontal = kernel.reshape(1, 1, 1, 41).expand(channels, 1, 1, 41)
        vertical = kernel.reshape(1, 1, 41, 1).expand(channels, 1, 41, 1)
        value = functional.conv2d(
            functional.pad(value, (20, 20, 0, 0), mode="reflect"),
            horizontal,
            groups=channels,
        )
        return functional.conv2d(
            functional.pad(value, (0, 0, 20, 20), mode="reflect"),
            vertical,
            groups=channels,
        )

    return (output - blur(output) + blur(reference)).clamp(-1, 1)

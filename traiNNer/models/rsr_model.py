"""Training of the recurrent video model RSR on clips with on-the-fly degradations.

RDUALModel's clip training (one set of degradation settings per clip, the image
loss stack per frame averaged over the clip, the discriminator on every frame)
with RSR as the generator. backward_clip gets the clip alone: the block read
takes no guidance, and the aligned read's frozen NeuFlow v2 (part of the model
and its checkpoint) computes it from the degraded clip. An aligned generator
whose start file has no flow weights gets the upstream ones, as RDUAL does.
"""

import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from traiNNer.archs.rsr_arch import RSR
from traiNNer.models.rdual_model import (
    ENCODE_CHUNK,
    GRAD_CHUNK,
    NEUFLOW_PRETRAIN_PATH,
    RDUALModel,
    clip_training_problems,
)
from traiNNer.models.realesrgan_model import DegradingBatch
from traiNNer.utils.misc import loss_type_to_label
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


def check_rsr_training_options(opt: ReduxOptions) -> None:
    """Reject settings the RSR clip training does not support."""
    problems = clip_training_problems(opt)
    train = opt.train
    if opt.is_train is not False and train is not None:
        for loss in train.losses or []:
            if float(loss.get("loss_weight", 0)) == 0:
                continue
            if loss_type_to_label(loss["type"]) == "l_g_vosr":
                problems.append(f"the {loss['type']} loss is not supported for RSR yet")
    if problems:
        raise ValueError(
            "RSR clip training (network_g type rsr_compact): " + "; ".join(problems)
        )


class RSRModel(RDUALModel):
    check_options = staticmethod(check_rsr_training_options)

    def prepare_generator(self, compile: Callable[[Callable], Callable] | None) -> Any:
        bare = self.get_bare_model(self.net_g)
        assert isinstance(bare, RSR)
        collect_before_capture = (
            compile is not None
            and bare.temporal_read == "aligned"
            and self.opt.compile_mode in ("max-autotune", "reduce-overhead")
            and "TORCH_CUDAGRAPH_GC" not in os.environ
        )
        self._capture_gc = (
            torch.compiler.config.patch(force_cudagraph_gc=True)
            if collect_before_capture else nullcontext()
        )
        if bare.temporal_read == "aligned":
            # The generator and its EMA copy exist and the pretrain is loaded:
            # every copy whose checkpoint lacked the frozen flow gets the
            # upstream weights.
            networks = [bare]
            if self.net_g_ema is not None:
                networks.append(self.net_g_ema.ema_model)
            missing = [net for net in networks if not bool(net.flow_loaded)]
            if missing:
                if not os.path.exists(NEUFLOW_PRETRAIN_PATH):
                    raise FileNotFoundError(
                        "The aligned RSR checkpoint has no NeuFlow v2 weights and "
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
        # RSR compiles its per-frame step functions, never the whole network.
        if isinstance(net, RSR):
            compile = False
        return super().model_to_device(net, compile, wrap_distributed)

    def _delivered(self, batch: DegradingBatch) -> DataFeed:
        """RDUALModel's batch; in channels-last training the LQ clips are stored
        frames first (T,B,H,W,C), so every frame lq[:, t] backward_clip takes
        is a dense channels-last image without a copy."""
        data = super()._delivered(batch)
        if self.memory_format == torch.channels_last:
            clips = data["lq"]
            data["lq"] = (
                clips.permute(1, 0, 3, 4, 2).contiguous().permute(1, 0, 4, 2, 3)
            )
        return data

    def backward_clip(self, loss: Callable[[Tensor, int], Tensor]) -> Tensor:
        assert self.lq is not None
        bare = self.get_bare_model(self.net_g)
        assert isinstance(bare, RSR)
        # An unreachable Inductor kernel can unload its CUDA module during capture.
        # PyTorch's capture-time collector runs before recording and skips replays.
        with self._capture_gc:
            return bare.backward_clip(
                self.lq,
                loss,
                shot_ids=self.shot_ids,
                keep=self.opt.clip_keep,
                functions=self.step_functions,
                autocast=self._autocast,
                encode_chunk=ENCODE_CHUNK,
                grad_chunk=GRAD_CHUNK,
            )

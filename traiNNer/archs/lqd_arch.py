"""LQ-only conditioned DUnet: candidate RGB plus shared degraded RGB."""

from typing import Any

import torch
from torch import Tensor

from traiNNer.archs.dunet_arch import DUnet
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class LQD(DUnet):
    """Select with network_d.type: lqd; the trainer supplies six channels.

    The backbone, DySample precision and spectral normalization are unchanged.
    Only native LQ-D checkpoints may be loaded, including in non-strict mode.
    Switching from MC-D or RGB DUnet requires a fresh D in a new experiment.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64) -> None:
        if num_in_ch != 3:
            raise ValueError("LQ-D expects RGB images; it adds three LQ channels")
        super().__init__(num_in_ch=6, num_feat=num_feat)
        self.register_buffer("_lqd_version", torch.tensor(1, dtype=torch.int64))
        # Initially retain an RGB-only forward; these columns receive gradients
        # during D updates and can learn to use the observed input.
        with torch.no_grad():
            self.in_to_dim.weight[:, 3:].zero_()

    def map_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Validate the conditioning contract without changing the source state."""
        weight = state_dict.get("in_to_dim.weight")
        version = state_dict.get("_lqd_version")
        if (
            "_mcd_version" in state_dict
            or weight is None
            or weight.shape != self.in_to_dim.weight.shape
            or version is None
            or version.ndim != 0
            or version.dtype != torch.int64
            or version.item() != 1
        ):
            raise ValueError(
                "LQ-D requires a compatible six-channel LQ-D v1 checkpoint. "
                "Do not load or resume MC-D/RGB DUnet states, even with "
                "strict_load_d: false. Start a new run with resume_state: null "
                "and pretrain_network_d: null; generator weights may be reused."
            )
        return state_dict

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        return super().load_state_dict(
            self.map_state_dict(state_dict), strict=strict, assign=assign
        )

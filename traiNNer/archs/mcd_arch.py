"""MC-D: target-medium conditioning, optionally with the degraded RGB input."""

from copy import copy
from typing import Any

import torch
from torch import Tensor

from traiNNer.archs.dunet_arch import DUnet
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MCD(DUnet):
    """Select with network_d.type: mcd; switch back with type: dunet.

    Channel expansion is for weight-only warm starts, NEVER optimizer resumes.
    SRModel disables conversion when path.resume_state is present.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        input_conditioning: bool = False,
    ) -> None:
        if num_in_ch != 3:
            raise ValueError("MC-D expects RGB images; it adds six condition channels")
        if not isinstance(input_conditioning, bool):
            raise ValueError("MC-D input_conditioning must be true or false")
        super().__init__(num_in_ch=12 if input_conditioning else 9, num_feat=num_feat)
        self.input_conditioning = input_conditioning
        self.allow_rgb_warm_start = True
        self.allow_input_warm_start = True
        self.register_buffer(
            "_mcd_version",
            torch.tensor(2 if input_conditioning else 1, dtype=torch.int64),
        )
        with torch.no_grad():
            self.in_to_dim.weight[:, 3:].zero_()

    def map_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Keep the source dictionary/checkpoint untouched during conversion."""
        weight = state_dict.get("in_to_dim.weight")
        if weight is None:
            raise ValueError("MC-D checkpoint is missing in_to_dim.weight")
        if weight.shape == (self.in_to_dim.out_channels, 3, 3, 3):
            if not self.allow_rgb_warm_start:
                raise ValueError(
                    "Cannot resume a DUnet training state as MC-D. Start a new run "
                    "with resume_state: null and optional pretrain_network_d instead."
                )
            if "_mcd_version" in state_dict:
                raise ValueError("MC-D version marker with RGB-only weights is invalid")
            converted = copy(state_dict)
            expanded = weight.new_zeros(self.in_to_dim.weight.shape)
            expanded[:, :3].copy_(weight)
            converted["in_to_dim.weight"] = expanded
            converted["_mcd_version"] = self._mcd_version.detach().clone()
            return converted
        version = state_dict.get("_mcd_version")
        source_version = (
            version.item() if version is not None and version.numel() == 1 else None
        )
        if (
            self.input_conditioning
            and weight.shape == (self.in_to_dim.out_channels, 9, 3, 3)
            and source_version == 1
        ):
            if not self.allow_input_warm_start:
                raise ValueError(
                    "Cannot resume nine-channel MC-D as input-conditioned MC-D. "
                    "Start a new run with resume_state: null and optional "
                    "pretrain_network_d for a weight-only warm start."
                )
            converted = copy(state_dict)
            expanded = weight.new_zeros(self.in_to_dim.weight.shape)
            expanded[:, :9].copy_(weight)
            converted["in_to_dim.weight"] = expanded
            converted["_mcd_version"] = self._mcd_version.detach().clone()
            return converted
        expected_version = 2 if self.input_conditioning else 1
        if (
            weight.shape != self.in_to_dim.weight.shape
            or source_version != expected_version
        ):
            expected = (
                "twelve-channel v2" if self.input_conditioning else "nine-channel v1"
            )
            raise ValueError(
                f"MC-D requires compatible {expected} weights. Changing "
                "input_conditioning requires a new run; do not resume across modes "
                "or load twelve-channel weights into nine-channel MC-D."
            )
        return state_dict

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        return super().load_state_dict(
            self.map_state_dict(state_dict), strict=strict, assign=assign
        )

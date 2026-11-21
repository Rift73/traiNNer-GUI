"""Mixed-precision network forwards for dense and scalar supervised logits."""

import torch
from torch import Tensor, nn


def check_bfloat16(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This CUDA device lacks native BF16. Disable AMP for FP32 supervised training.")


def forward_logits(network: nn.Module, image: Tensor, *, mixed: bool):
    if not mixed:
        return network(image)
    with torch.autocast(image.device.type, dtype=torch.bfloat16):
        return network(image)

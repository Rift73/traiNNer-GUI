from typing import Literal

import numpy as np
import torch
from torch import Tensor

from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.metrics.learned import learned_rgb
from traiNNer.metrics.session import pretrained_metric
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(
    img: np.ndarray | Tensor,
    img2: np.ndarray | Tensor,
    device: torch.device,
    net: Literal["alex", "vgg", "squeeze"] = "alex",
    **kwargs,
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    loss = pretrained_metric(("lpips", net), device, lambda: LPIPS(net=net))

    with torch.inference_mode():
        return loss(
            learned_rgb(img, device),
            learned_rgb(img2, device),
            normalize=True,
        ).view(())

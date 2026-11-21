import numpy as np
import torch
from torch import Tensor

from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.metrics.learned import learned_rgb
from traiNNer.metrics.session import pretrained_metric
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_dists(
    img: np.ndarray | Tensor, img2: np.ndarray | Tensor, device: torch.device, **kwargs
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    loss = pretrained_metric(("dists",), device, lambda: DISTSLoss(loss_weight=1.0, as_loss=False))
    with torch.inference_mode():
        return loss(
            learned_rgb(img, device),
            learned_rgb(img2, device),
        )

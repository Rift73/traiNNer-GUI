import numpy as np
import torch
from torch import Tensor

from traiNNer.archs.topiq_arch import CFANet
from traiNNer.metrics.learned import learned_rgb
from traiNNer.metrics.session import pretrained_metric
from traiNNer.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_topiq(
    img: np.ndarray | Tensor, img2: np.ndarray | Tensor, device: torch.device, **kwargs
) -> Tensor:
    assert img.shape == img2.shape, (
        f"Image shapes are different: {img.shape}, {img2.shape}."
    )

    topiq = pretrained_metric(("topiq",), device, lambda: CFANet(backbone_pretrain=False))
    with torch.inference_mode():
        return topiq(
            learned_rgb(img, device),
            learned_rgb(img2, device),
        )


@METRIC_REGISTRY.register()
def calculate_neosr_topiq(
    img: np.ndarray | Tensor, img2: np.ndarray | Tensor, device: torch.device, dataset: str = "kadid"
) -> Tensor:
    if dataset not in ("kadid", "pipal"):
        raise ValueError("NeoSR TOPIQ requires the kadid or pipal weight set.")
    if img.shape != img2.shape:
        raise ValueError("TOPIQ requires matching image geometry.")
    topiq = pretrained_metric(
        ("neosr_topiq", dataset), device,
        lambda: CFANet(model_name=f"cfanet_fr_{dataset}_res50", backbone_pretrain=False, pool_first_feature=True),
    )
    with torch.inference_mode():
        return topiq(
            learned_rgb(img, device),
            learned_rgb(img2, device),
        )


@METRIC_REGISTRY.register()
def calculate_topiq_nr(img: np.ndarray | Tensor, device: torch.device, **kwargs) -> Tensor:
    topiq = pretrained_metric(("topiq_nr",), device, lambda: CFANet(model_name="cfanet_nr_koniq_res50", use_ref=False, backbone_pretrain=False))
    with torch.inference_mode():
        return topiq(
            learned_rgb(img, device),
        )

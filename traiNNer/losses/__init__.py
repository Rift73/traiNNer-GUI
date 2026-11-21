from copy import deepcopy
from pathlib import Path
from typing import Any

from torch import nn

from trainner_config.registry_index import install

from traiNNer.losses.gan_loss import (
    g_path_regularize,
    gradient_penalty_loss,
    r1_penalty,
)
from traiNNer.utils import get_root_logger
from traiNNer.utils.registry import LOSS_REGISTRY

__all__ = ["build_loss", "g_path_regularize", "gradient_penalty_loss", "r1_penalty"]

install(__name__, Path(__file__).parent, "_loss.py", {"loss": LOSS_REGISTRY})


def build_loss(loss_opt: dict[str, Any]) -> nn.Module:
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(loss_opt)
    loss_type = opt.pop("type")
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(
        "Loss [bold]%s[/bold](%s) is created.",
        loss.__class__.__name__,
        opt,
        extra={"markup": True},
    )
    return loss

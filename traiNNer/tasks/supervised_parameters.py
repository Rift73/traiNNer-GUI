"""Parameter selection and gradient checks for supervised training."""

from torch import nn
from trainner_config.options import TrainOptions
from trainner_config.parameter_filters import parameter_filter


def select_trainable(network: nn.Module, train: TrainOptions) -> None:
    include, exclude = train.optim_include_params_g, train.optim_exclude_params_g
    if not (include or exclude):
        return
    selected = parameter_filter(include, exclude)
    kept = 0
    for name, parameter in network.named_parameters():
        if parameter.requires_grad and selected(name):
            kept += 1
        else:
            parameter.requires_grad_(False)
    if not kept:
        raise ValueError("Generator optimizer parameter filters excluded all trainable parameters.")


def check_grad_norm(network: nn.Module, max_norm: float | None) -> None:
    """Validate the full gradient norm, clipping only when a limit was requested."""
    if max_norm is None:
        grads = [p.grad for p in network.parameters() if p.grad is not None]
        nn.utils.get_total_norm(grads, error_if_nonfinite=True)
    else:
        nn.utils.clip_grad_norm_(network.parameters(), max_norm, error_if_nonfinite=True)

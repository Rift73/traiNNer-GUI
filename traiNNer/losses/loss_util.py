import functools
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


class _RestoringClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, image: Tensor, lo: float, hi: float) -> Tensor:
        ctx.save_for_backward(image)
        ctx.lo, ctx.hi = lo, hi
        return image.clamp(lo, hi)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None]:
        (image,) = ctx.saved_tensors
        outward = ((image < ctx.lo) & (grad > 0)) | ((image > ctx.hi) & (grad < 0))
        return grad.masked_fill(outward, 0), None, None


def restoring_clamp(image: Tensor, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    """Clamp the forward value and pass inward gradients outside the interval."""
    return _RestoringClamp.apply(image, lo, hi)


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)  # type: ignore # noqa: SLF001
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(
    loss: Tensor, weight: Tensor | None = None, reduction: str = "mean"
) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == "mean":
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func: Callable) -> Callable:
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: str = "mean",
        **kwargs,
    ) -> Tensor:
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper

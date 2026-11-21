"""Convolution input gradients without retaining a frozen layer's activations.

This is an ordinary convolution with its ordinary native input-gradient
operation. It neither recomputes activations nor changes arithmetic precision.
Trainable weights/bias and inference calls retain nn.Conv2d's original path.
"""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class _InputOnlyConv(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ) -> Tensor:
        output = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        # Autocast may use BF16 operands even though the parameters/input are
        # FP32. Save exactly that operand dtype, not a lower-precision surrogate.
        ctx.save_for_backward(weight.to(output.dtype))
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.channels_last = x.is_contiguous(memory_format=torch.channels_last)
        ctx.geometry = (stride, padding, dilation, groups)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:
        (weight,) = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.geometry
        memory_format = (
            torch.channels_last if ctx.channels_last else torch.contiguous_format
        )
        # With output_mask=(True, False, False), convolution_backward consumes
        # only input geometry, not input values. This temporary is not a saved
        # activation; it also preserves the original convolution's layout.
        geometry = torch.empty(
            ctx.input_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
            memory_format=memory_format,
        )
        grad_input, _, _ = torch.ops.aten.convolution_backward.default(
            grad_output,
            geometry,
            weight,
            None,
            stride,
            padding,
            dilation,
            False,
            (0, 0),
            groups,
            (True, False, False),
        )
        return grad_input.to(ctx.input_dtype), None, None, None, None, None, None


class InputOnlyConv2d(nn.Conv2d):
    """Use input-only autograd only while both convolution parameters are frozen.

    Parameter registration/order, state-dict keys, initialization and module hooks
    are inherited unchanged. In particular, spectral-normalization parametrizations
    must be evaluated ONCE per forward, including the ordinary trainable branch.
    """

    def forward(self, x: Tensor) -> Tensor:
        weight, bias = self.weight, self.bias
        if (
            not torch.is_grad_enabled()
            or not x.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
            or x.ndim != 4
            or isinstance(self.padding, str)
        ):
            return self._conv_forward(x, weight, bias)
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return _InputOnlyConv.apply(
            x, weight, bias, self.stride, padding, self.dilation, self.groups
        )

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "InputOnlyConv2d":
        """Wrap a fresh plain pretrained convolution without RNG or weight copies.

        Used while constructing VGG/DySample, before callers can attach hooks.
        Do not strip parametrizations or discard existing instrumentation.
        """
        if type(conv) is not nn.Conv2d:
            raise TypeError("from_conv requires a plain nn.Conv2d")
        if (
            conv._forward_hooks  # noqa: SLF001 - preserve module instrumentation
            or conv._forward_pre_hooks  # noqa: SLF001
            or conv._backward_hooks  # noqa: SLF001
            or conv._backward_pre_hooks  # noqa: SLF001
        ):
            raise ValueError("Wrap the convolution before attaching module hooks")
        result = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            device="meta",
            dtype=conv.weight.dtype,
        )
        result.weight = conv.weight
        result.bias = conv.bias
        result.train(conv.training)
        return result

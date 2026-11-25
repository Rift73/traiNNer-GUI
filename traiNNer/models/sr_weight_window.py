"""Reuse trainable convolution kernels while their branch factors stay fixed."""

# Inspect registered autograd hooks and manage the private per-layer training cache.
# ruff: noqa: SLF001
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from traiNNer.archs.dual_arch import EDBBReparamConv3x3
    from traiNNer.archs.wsr_arch import Conv3XC, FoldableRGB5

_Layer = TypeVar("_Layer", bound=nn.Module)


class ComposedWeightWindow(Generic[_Layer]):
    def __init__(self, layers: list[_Layer], factors: Iterable[nn.Parameter]) -> None:
        self.layers = layers
        self.factors = tuple(factors)
        self.leaves: list[tuple[Tensor, Tensor | None]] = []
        self.active_grads: list[Tensor] = []
        self.pending = False

    def _compose(
        self, layer: _Layer, dtype: torch.dtype, *, backward: bool
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError

    def can_fold(self) -> bool:
        # A hook on a branch must still see each microbatch's backward.
        return (
            not torch.is_anomaly_enabled()
            and all(layer.training for layer in self.layers)
            and all(
                p.requires_grad
                and not p._backward_hooks
                and not p._post_accumulate_grad_hooks
                for p in self.factors
            )
            and all(
                not module._backward_hooks and not module._backward_pre_hooks
                for layer in self.layers
                for module in layer.modules()
            )
        )

    @contextmanager
    def unfolded(self) -> Iterator[None]:
        cached = [layer._training_kernel for layer in self.layers]
        for layer in self.layers:
            layer._training_kernel = None
        try:
            yield
        finally:
            for layer, value in zip(self.layers, cached, strict=True):
                layer._training_kernel = value

    def begin(self) -> tuple[Tensor, ...]:
        if self.pending:
            return ()
        first_window = not self.leaves
        # Compiled G consumes only composed leaves. Eager factor VJPs allocate
        # ordinary parameter gradients at commit, after the last image backward.
        for factor in self.factors:
            factor.grad = None
        for index, layer in enumerate(self.layers):
            weight = next(layer.parameters())
            dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
            with (
                torch.no_grad(),
                torch.autocast(device_type=weight.device.type, enabled=False),
            ):
                kernel, bias = self._compose(layer, dtype, backward=False)
            if index == len(self.leaves):
                # Stable addresses survive graph replay even as optimizer steps
                # produce fresh source tensors for the next accumulation window.
                self.leaves.append(
                    (
                        torch.empty_like(kernel).requires_grad_(True),
                        None
                        if bias is None
                        else torch.empty_like(bias).requires_grad_(True),
                    )
                )
            leaves = self.leaves[index]
            with torch.no_grad():
                for source, leaf in zip((kernel, bias), leaves, strict=True):
                    if source is None or leaf is None:
                        continue
                    leaf.copy_(source)
                    if leaf.grad is None:
                        leaf.grad = torch.zeros_like(leaf)
                    self.active_grads.append(leaf.grad)
            layer._training_kernel = leaves
        self.pending = True
        if first_window:
            return tuple(
                t
                for group in self.leaves
                for t in group
                if t is not None and t.requires_grad
            )
        return ()

    def backward(self) -> None:
        # The image adjoint is linear in each effective kernel. Propagate its
        # summed cotangent through one layer at a time, after image activations
        # have been released. Its factors are constant throughout this window.
        for layer, (kernel_leaf, bias_leaf) in zip(
            self.layers, self.leaves, strict=True
        ):
            weight = next(layer.parameters())
            dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type=weight.device.type, enabled=False):
                with torch.enable_grad():
                    kernel, bias = self._compose(layer, dtype, backward=True)
                assert kernel_leaf.grad is not None
                if bias is None:
                    torch.autograd.backward(kernel, kernel_leaf.grad)
                else:
                    assert bias_leaf is not None and bias_leaf.grad is not None
                    torch.autograd.backward(
                        (kernel, bias), (kernel_leaf.grad, bias_leaf.grad)
                    )
            del kernel, bias
        self.clear()

    def clear(self) -> None:
        for layer in self.layers:
            layer._training_kernel = None
        if self.active_grads:
            torch._foreach_zero_(self.active_grads)
        self.active_grads.clear()
        self.pending = False


class EDBBWeightWindow(ComposedWeightWindow["EDBBReparamConv3x3"]):
    def __init__(self, network: nn.Module) -> None:
        from traiNNer.archs.dual_arch import _HAS_DIRECT_ATTENTION, EDBBReparamConv3x3

        layers = [
            layer
            for layer in network.modules()
            if isinstance(layer, EDBBReparamConv3x3)
        ]
        super().__init__(layers, (p for layer in layers for p in layer.parameters()))
        self._can_compile = _HAS_DIRECT_ATTENTION
        self._compiled_kernel = None

    def _compose(
        self, layer: EDBBReparamConv3x3, dtype: torch.dtype, *, backward: bool
    ) -> tuple[Tensor, Tensor | None]:
        weight = layer.direct.expand.weight
        if (
            backward
            and dtype == torch.float32
            and weight.is_cuda
            and self._can_compile
            and torch.cuda.get_device_capability(weight.device) == (12, 0)
            and not torch.are_deterministic_algorithms_enabled()
        ):
            if self._compiled_kernel is None:
                self._compiled_kernel = torch.compile(
                    type(layer).equivalent_kernel,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
            precision = (
                "tf32"
                if torch.backends.cuda.matmul.fp32_precision == "tf32"
                else "ieee"
            )
            return self._compiled_kernel(layer, dtype, precision)
        return layer.equivalent_kernel(dtype)


class WSRWeightWindow(ComposedWeightWindow["Conv3XC | FoldableRGB5"]):
    def __init__(self, network: nn.Module) -> None:
        from traiNNer.archs.wsr_arch import Conv3XC, FoldableRGB5, ValidConv3XC

        # A FoldableRGB5 owns both valid 3x3 branches. Their standalone forwards
        # are bypassed when the enclosing layer composes its 5x5 training kernel.
        layers = [
            layer
            for layer in network.modules()
            if type(layer) is Conv3XC
            or (
                type(layer) is FoldableRGB5
                and type(layer.first) is ValidConv3XC
                and type(layer.second) is ValidConv3XC
            )
        ]
        factors = (
            p
            for layer in layers
            for name, p in layer.named_parameters()
            if "eval_conv" not in name.split(".")
        )
        super().__init__(layers, factors)

    def _compose(
        self, layer: Conv3XC | FoldableRGB5, dtype: torch.dtype, *, backward: bool
    ) -> tuple[Tensor, Tensor]:
        return layer.composed(dtype)

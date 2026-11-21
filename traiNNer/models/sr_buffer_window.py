"""Accumulation-window rollback for the plain still-image G/D finite guard."""

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain

import torch
from torch import nn


class BufferWindow:
    def __init__(self, *nets: nn.Module | None) -> None:
        self.nets = [net for net in nets if net is not None]
        self.live: list[torch.Tensor] = []
        for net in self.nets:
            for name, buffer in net.named_buffers():
                if (
                    nn.parameter.is_lazy(buffer)
                    or buffer.layout != torch.strided
                    or buffer.is_inference()
                    or buffer.is_quantized
                    or buffer.is_meta
                    or buffer.requires_grad
                    or torch._debug_has_internal_overlap(buffer) != 0
                ):
                    raise ValueError(
                        f"train.rollback_buffers_on_reject requires writable, non-overlapping "
                        f"strided buffers without gradients. Unsupported {type(net).__name__}.{name}."
                    )
                self.live.append(buffer)
        self.saved = [torch.empty_like(buffer) for buffer in self.live]
        self.pending = False

    def check_layout(self) -> None:
        # Holding references alone could silently restore stale tensors after a
        # module rebinds a buffer. Check before either optimizer can commit.
        current = list(chain.from_iterable(net.buffers() for net in self.nets))
        if len(current) != len(self.live) or any(
            buf is not live
            or buf.shape != saved.shape
            or buf.dtype != saved.dtype
            or buf.device != saved.device
            for buf, live, saved in zip(current, self.live, self.saved, strict=True)
        ):
            raise RuntimeError(
                "train.rollback_buffers_on_reject requires a fixed buffer layout. "
                "A network added, removed, replaced or resized a registered buffer."
            )

    @torch.no_grad()
    def begin(self) -> None:
        if not self.pending:
            torch._foreach_copy_(self.saved, self.live)
            self.pending = True

    @torch.no_grad()
    def rollback(self) -> None:
        if self.pending:
            self.check_layout()
            torch._foreach_copy_(self.live, self.saved)
            self.pending = False

    @contextmanager
    @torch.no_grad()
    def checkpoint_view(self) -> Iterator[None]:
        if not self.pending:
            yield
            return
        self.check_layout()
        partial = [buffer.clone() for buffer in self.live]
        try:
            torch._foreach_copy_(self.live, self.saved)
            yield
        finally:
            torch._foreach_copy_(self.live, partial)

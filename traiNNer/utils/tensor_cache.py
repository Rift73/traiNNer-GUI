"""Per-device caches for small constant tensors used in hot paths.

Building a constant on the CPU and moving it to the GPU on every call (``torch.tensor(...).to(x)``)
is a blocking host-to-device copy, so the CPU waits for the GPU to drain its queue each time.
These helpers build each constant once per (args, device, dtype) and reuse it.
"""

from collections.abc import Callable, Hashable
from functools import wraps
from typing import ParamSpec

import torch
from torch import Tensor

P = ParamSpec("P")
_MAX_ENTRIES = 64


def cached_constant(
    fn: Callable[P, Tensor],
) -> Callable[..., Tensor]:
    """Decorate a function that builds a constant CPU tensor from hashable args.

    The wrapped function takes the same positional args plus a keyword-only ``like``
    tensor (or ``device``/``dtype``) and returns the constant on that device and
    dtype, building it only on the first call for each key. Callers must not modify
    the returned tensor in place.
    """
    cache: dict[Hashable, Tensor] = {}

    @wraps(fn)
    def wrapper(
        *args: Hashable,
        like: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if like is not None:
            device = like.device if device is None else device
            dtype = like.dtype if dtype is None else dtype
        if torch.compiler.is_compiling():
            # Preserve fullgraph callers. Inductor can fold these fixed tensors
            # into the compiled program without tracing the eager dictionary.
            return fn(*args).to(device=device, dtype=dtype)
        key = (args, device, dtype)
        t = cache.get(key)
        if t is None:
            if len(cache) >= _MAX_ENTRIES:
                cache.pop(next(iter(cache)))
            # A tensor made in inference mode can't be saved for backward later, so
            # build it as a normal tensor even when first requested there.
            with torch.inference_mode(False):
                t = fn(*args).to(device=device, dtype=dtype)  # pyright: ignore[reportCallIssue]
            cache[key] = t
        return t

    return wrapper

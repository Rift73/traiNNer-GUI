from collections.abc import Callable, Sequence
from copy import deepcopy
from importlib import import_module
from os import path as osp
from typing import Any

from torch import Tensor
from torch.optim import Adam, AdamW, NAdam, Optimizer
from torch.optim.optimizer import ParamsT

from traiNNer.utils import get_root_logger, scandir
from traiNNer.utils.registry import OPTIMIZER_REGISTRY

__all__ = ["build_optimizer"]

AUTO_FUSED_OPTIMIZERS = (Adam, AdamW)

# automatically scan and import arch modules for registry
# scan all the files under the 'optimizers' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(arch_folder)
    if v.endswith("_optim.py")
]
# import all the arch modules
_arch_modules = [
    import_module(f"traiNNer.optimizers.{file_name}") for file_name in arch_filenames
]


# register built in optimizers
for o in [Adam, AdamW, NAdam]:
    OPTIMIZER_REGISTRY.register(o)

_EXTERNAL = ("StableAdamW", "SOAP", "ADOPT")
for name in _EXTERNAL:
    OPTIMIZER_REGISTRY.defer(name.lower(), f"pytorch_optimizer:{name}")


def __getattr__(name: str) -> Callable[..., Optimizer]:
    if name in _EXTERNAL:
        return OPTIMIZER_REGISTRY.get(name)
    raise AttributeError(name)


def build_optimizer(params: ParamsT, opt: dict[str, Any]) -> Optimizer:
    opt = deepcopy(opt)
    optimizer_type = opt.pop("type")
    logger = get_root_logger()
    optimizer_class = OPTIMIZER_REGISTRY.get(optimizer_type)

    # Without fused or foreach torch picks its multi-tensor path, which keeps
    # step counters on the CPU and builds bias corrections per tensor on the
    # host. The fused kernel gives the same update up to rounding. Only a
    # sequence is inspected: checking a generator would consume it.
    if (
        optimizer_class in AUTO_FUSED_OPTIMIZERS
        and "fused" not in opt
        and "foreach" not in opt
        and isinstance(params, Sequence)
        and params
        and all(
            isinstance(param, Tensor) and param.is_cuda and param.is_floating_point()
            for param in params
        )
    ):
        opt["fused"] = True

    optimizer = optimizer_class(params, **opt)
    logger.info(
        "Optimizer [bold]%s[/bold](%s) is created.",
        optimizer.__class__.__name__,
        opt,
        extra={"markup": True},
    )

    return optimizer

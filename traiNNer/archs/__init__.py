from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any

from torch import nn

from trainner_config.registry_index import install

from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY, TESTARCH_REGISTRY

__all__ = ["build_network"]
spandrel_name = "spandrel"

install(__name__, Path(__file__).parent, "_arch.py", {"arch": ARCH_REGISTRY, "spandrel": SPANDREL_REGISTRY, "testarch": TESTARCH_REGISTRY})


def build_network(opt: dict[str, Any]) -> nn.Module:
    from traiNNer.utils.logger import get_root_logger

    opt = deepcopy(opt)
    network_type = opt.pop("type")
    logger = get_root_logger()

    # A legacy recipe or direct builder can use a factory registered by a task
    # without naming that task. Keep catalog discovery as the last resort.
    def registered() -> bool:
        return any(network_type in registry for registry in (SPANDREL_REGISTRY, ARCH_REGISTRY, TESTARCH_REGISTRY))

    if not registered():
        from traiNNer.tasks import discover_task_factory

        discover_task_factory(network_type, registered, dataset=False)

    # try loading from spandrel first
    if network_type in SPANDREL_REGISTRY:
        net = SPANDREL_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [bold]%s[/bold](%s) is created from [bold]%s[/bold] v%s.",
            net.__class__.__name__,
            opt,
            spandrel_name,
            version(spandrel_name),
            extra={"markup": True},
        )
    elif network_type in ARCH_REGISTRY:
        net = ARCH_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [bold]%s[/bold](%s) is created from [bold]traiNNer-redux[/bold].",
            net.__class__.__name__,
            opt,
            extra={"markup": True},
        )
    else:
        net = TESTARCH_REGISTRY.get(network_type)(**opt)
        logger.info(
            "Network [bold]%s[/bold](%s) is created from [bold]traiNNer-redux-test[/bold].",
            net.__class__.__name__,
            opt,
            extra={"markup": True},
        )

    return net

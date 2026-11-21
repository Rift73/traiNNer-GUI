from __future__ import annotations

import sys
from collections.abc import Callable
from copy import deepcopy
from importlib import import_module
from typing import TYPE_CHECKING, TypeGuard

from msgspec import Struct
from trainner_config.tasks import TrainingTaskSpec, resolve_task

from traiNNer.tasks import load_task_provider
from trainner_config.options import ReduxOptions

if TYPE_CHECKING:
    from .base_model import BaseModel
    from .realesrgan_model import RealESRGANModel

__all__ = ["build_model", "degrades_ahead"]


_MODEL_MODULES = {
    "AEModel": "ae_model", "BaseModel": "base_model", "SRModel": "sr_model",
    "RealESRGANModel": "realesrgan_model", "RealESRGANPairedModel": "realesrgan_paired_model",
    "RDUALModel": "rdual_model", "RSRModel": "rsr_model",
}
_RECURRENT_MODELS = {"rdual": "RDUALModel", "rsr_compact": "RSRModel"}


def _model_class(name: str) -> type[BaseModel]:
    return getattr(import_module("." + _MODEL_MODULES[name], __name__), name)


def __getattr__(name: str):
    # Keep existing direct class imports without loading every task family.
    if name not in _MODEL_MODULES:
        raise AttributeError(name)
    value = _model_class(name)
    globals()[name] = value
    return value


def _image_model_type(opt: ReduxOptions) -> type[BaseModel]:
    if opt.high_order_degradation:
        return _model_class("RealESRGANPairedModel" if opt.dataroot_lq_prob > 0 else "RealESRGANModel")
    return _model_class("SRModel")


def _build_image_model(opt: ReduxOptions, _settings: Struct | None) -> BaseModel:
    return _image_model_type(opt)(opt)


def _build_autoencoder(opt: ReduxOptions, _settings: Struct | None) -> BaseModel:
    return _model_class("AEModel")(opt)


def _build_recurrent_model(opt: ReduxOptions, _settings: Struct | None) -> BaseModel:
    assert opt.network_g is not None
    return _model_class(_RECURRENT_MODELS[opt.network_g["type"].lower()])(opt)


def _legacy_model_type(opt: ReduxOptions) -> type[BaseModel]:
    architecture = opt.network_g.get("type") if opt.network_g is not None else None
    if isinstance(architecture, str) and architecture in _RECURRENT_MODELS:
        return _model_class(_RECURRENT_MODELS[architecture])
    if not opt.high_order_degradation and opt.network_ae and not opt.network_g:
        return _model_class("AEModel")
    return _image_model_type(opt)


def _task_factory(
    spec: TrainingTaskSpec,
) -> Callable[[ReduxOptions, Struct | None], BaseModel]:
    factory = getattr(load_task_provider(spec), spec.model_factory.partition(":")[2])
    if not callable(factory):
        raise TypeError(f"Task model factory {spec.model_factory!r} must be callable.")
    return factory


def build_model(opt: ReduxOptions) -> BaseModel:
    """Build the trainer for the explicit task or the legacy recipe."""
    from traiNNer.utils.logger import get_root_logger

    opt = deepcopy(opt)
    logger = get_root_logger()
    selection = resolve_task(opt)
    if selection is None:
        model = _legacy_model_type(opt)(opt)
    else:
        spec, _, settings = selection
        model = _task_factory(spec)(opt, settings)

    logger.info(
        "Model [bold]%s[/bold] is created.",
        model.__class__.__name__,
        extra={"markup": True},
    )
    return model


def degrades_ahead(model: object) -> TypeGuard[RealESRGANModel]:
    """Queue training batches only for models that consume prepared degradations."""
    # An instance of either class already imported its defining module. Other
    # tasks need no codec or recurrent architecture to answer this question.
    real = sys.modules.get("traiNNer.models.realesrgan_model")
    if not isinstance(model, getattr(real, "RealESRGANModel", ())) or not model.is_train:
        return False
    if type(model).feed_data is real.RealESRGANModel.feed_data:
        return True
    recurrent = sys.modules.get("traiNNer.models.rdual_model")
    recurrent_type = getattr(recurrent, "RDUALModel", None)
    return recurrent_type is not None and type(model).feed_data is recurrent_type.feed_data

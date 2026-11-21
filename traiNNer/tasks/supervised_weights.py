"""Network selection and checkpoint association for supervised task providers."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from traiNNer.training.checkpoint_association import observe_inference
from traiNNer.training.schedule_free_weights import evaluation_view, preserve_module_modes

if TYPE_CHECKING:
    from traiNNer.models.base_model import BaseModel


def initialize_ema(model: "BaseModel", *, expected_metadata: dict[str, object] | None = None) -> None:
    train = model.opt.train
    assert train is not None and model.net_g is not None
    if train.ema_decay == 0:
        return
    if not 0 < train.ema_decay < 1:
        raise ValueError("train.ema_decay must be zero or a positive decay below one.")
    from ema_pytorch import EMA
    from traiNNer.models.checkpoint_payload import load_matching_network_checkpoint
    from traiNNer.models.neosr_ema import NeoSREMA

    online = model.get_bare_model(model.net_g)
    if train.ema_mode == "neosr":
        ema = NeoSREMA(online, train.ema_decay, train.ema_power)
    else:
        ema = EMA(online, beta=train.ema_decay, allow_different_devices=True, update_every=1,
                  update_after_step=train.ema_update_after_step, power=10 if train.ema_power is None else train.ema_power,
                  use_foreach=train.ema_foreach, update_model_with_ema_every=train.ema_switch_iter or None)
    source = model.opt.path.pretrain_network_g_ema
    if source:
        metadata = online.hyperparameters
        load_matching_network_checkpoint(ema.ema_model, Path(source), "params_ema",
                                         class_names=metadata.get("class_names"), category_ids=metadata.get("category_ids"),
                                         expected_task=metadata["task"], expected_metadata=expected_metadata)
    elif model.opt.path.resume_state:
        raise ValueError("Resuming supervised EMA requires the matching numbered EMA weights.")
    model.net_g_ema = ema.to(device=model.device)
    ema.step = ema.step.to(device=torch.device("cpu"))
    ema.initted = ema.initted.to(device="cpu")


def save_supervised(model: "BaseModel", epoch: int, step: int) -> None:
    paths = model.opt.path
    assert paths.models is not None and model.net_g is not None
    raw_dir = paths.models
    if model.net_g_ema is not None:
        assert paths.resume_models is not None
        raw_dir = paths.resume_models
        model.save_network(model.net_g_ema.ema_model, "net_g_ema", paths.models, step, "params_ema")
    model.save_network(model.net_g, "net_g", raw_dir, step, "params")
    model.save_training_state(epoch, step)


@contextmanager
def evaluation_weights(model: "BaseModel", step: int) -> Iterator[nn.Module]:
    network = model.net_g_ema.ema_model if model.net_g_ema is not None else model.net_g
    assert network is not None
    optimizer = model.optimizers[0] if model.optimizers and model.net_g_ema is None else None
    with preserve_module_modes(network), evaluation_view(optimizer):
        network.eval()
        with observe_inference(model, network, step):
            yield network

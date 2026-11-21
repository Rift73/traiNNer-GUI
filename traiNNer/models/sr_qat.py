"""AetherNet QAT phases for training, validation and portable weights."""
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from safetensors import safe_open
from torch.ao.quantization import FakeQuantize, disable_observer

from traiNNer.archs.aether_arch import DeploymentNorm, aether
from traiNNer.models.checkpoint_payload import (
    _checkpoint_tensors,
    read_network_checkpoint,
    strip_common_prefix,
    transfer_generator_state,
)
from traiNNer.utils.logger import get_root_logger

if TYPE_CHECKING:
    from traiNNer.models.sr_model import SRModel


def load_aether_for_inference(
    net: aether, checkpoint: Path, param_key: str | None, *,
    strict: bool = True, allow_qat: bool = True, quantize_residual: bool | None = None,
) -> None:
    """Reopen a validation winner in its measured phase. Regular QAT files deploy with INT8 fake quant."""
    retained = False
    if checkpoint.suffix.lower() == ".safetensors":
        # Winners can be replaced atomically. Read the receipt and tensors from one file.
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            retained = "trainner_best_validation" in (file.metadata() or {})
            weights = {name: file.get_tensor(name) for name in file.keys()}
    else:
        weights = read_network_checkpoint(checkpoint)
    state, _ = _checkpoint_tensors(weights, checkpoint, param_key)
    state = strip_common_prefix(state, ("module.", "_orig_mod.", "netG."))
    topology = net.qat_state(state)
    if topology is not None:
        if not allow_qat:
            raise ValueError("Aether fake-quant evaluation requires use_compile: false and use_amp: false.")
        if not strict:
            raise ValueError("Loading Aether QAT weights requires a strict, complete generator checkpoint.")
        if quantize_residual is not None and quantize_residual != topology["quantize_residual"]:
            raise ValueError(f"Aether residual quantization in {checkpoint} disagrees with the requested topology.")
        net.prepare_qat(**topology)
    elif not net.fused_init and any(key.endswith(".fused_conv.weight") for key in state):
        net.fuse_model()
    net.load_state_dict(state, strict=strict)
    net.eval()
    if topology is not None:
        if retained:
            # eval() freezes DeploymentNorm. Preserve each loaded fake_quant_enabled
            # bit, including a calibration winner, while keeping its ranges fixed.
            net.apply(disable_observer)
        else:
            net.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True)


def aether_export_precision(net: aether) -> Literal["float32", "int8_qdq"]:
    """A prepared calibration network can still execute entirely in float."""
    if net.is_quantized and any(
        bool(m.fake_quant_enabled.item()) for m in net.modules() if isinstance(m, FakeQuantize)
    ):
        return "int8_qdq"
    return "float32"


def initialize_aether(model: "SRModel", net: aether, *, ema: bool = False) -> None:
    """Load float weights before fusion, or full observer weights after preparation."""
    opt = model.opt
    policy = opt.train.qat if opt.train and opt.is_train else None
    paths = opt.path
    path = paths.pretrain_network_g_ema if ema else paths.pretrain_network_g
    param_key = "params_ema" if ema else paths.param_key_g
    if not opt.is_train:
        if path:
            load_aether_for_inference(
                net, Path(path), param_key, strict=paths.strict_load_g,
                allow_qat=not (opt.use_compile or opt.use_amp),
                quantize_residual=(opt.network_g or {}).get("quantize_residual"),
            )
        return

    state = saved = payload = None
    if path:
        payload = _checkpoint_tensors(read_network_checkpoint(path), Path(path), param_key)
        state = strip_common_prefix(payload[0], ("module.", "_orig_mod.", "netG."))
        saved = net.qat_state(state)
    if saved is not None and not policy:
        raise ValueError(f"{path} contains Aether QAT weights. Set train.qat to continue training.")
    if saved is not None and (not paths.strict_load_g or model._initial_transfer_g is not None):
        raise ValueError("Loading Aether QAT weights requires a strict, complete generator checkpoint.")
    if policy and paths.resume_state and saved is None:
        raise ValueError("Full QAT resume requires the corresponding prepared QAT network weights.")
    if policy and net.fused_init and not path:
        raise ValueError("QAT from scratch requires network_g.fused_init: false for live normalization.")

    prepared_ema = None
    if ema and net.is_quantized and saved is None:
        # A float EMA warm start has to load its own unfused branches before preparation.
        from traiNNer.archs import build_network

        prepared_ema = net
        assert opt.network_g is not None
        net = build_network({**opt.network_g, "scale": opt.scale})
    if state is not None and saved is None:
        if not ema and model._initial_transfer_g is not None:
            assert path is not None and payload is not None
            model.transfer_g_receipt = transfer_generator_state(net, Path(path), payload,
                                                                model._initial_transfer_g.exclude, get_root_logger())
        else:
            net.load_state_dict(state, strict=paths.strict_load_g)
    if policy is None and saved is None:
        return

    residual = (opt.network_g or {}).get("quantize_residual")
    if policy and policy.quantize_residual is not None:
        residual = policy.quantize_residual
    if ema:
        online = model.get_bare_model(model.net_g)
        residual = online.quantize_residual
    if saved is not None:
        if residual is not None and residual != saved["quantize_residual"]:
            raise ValueError(f"Aether residual quantization in {path} disagrees with the requested topology.")
        residual = saved["quantize_residual"]
    if not net.is_quantized:
        net.prepare_qat(qat_type=policy.type if policy else "int8", quantize_residual=residual)
    if prepared_ema is not None:
        prepared_ema.load_state_dict(net.state_dict(), strict=True)
    if saved is not None:
        net.load_state_dict(state, strict=True)


@contextmanager
def frozen_observers(net: torch.nn.Module) -> Iterator[None]:
    """Hold ranges fixed for validation and LDL without changing saved fake-quant flags."""
    states = [(m, m.observer_enabled) for m in net.modules() if isinstance(m, FakeQuantize)]
    try:
        for module, enabled in states:
            module.observer_enabled = torch.zeros_like(enabled)
        yield
    finally:
        for module, enabled in states:
            module.observer_enabled = enabled


class QATWindow:
    """Commit mutable ranges and norm statistics at the same boundary as generator weights."""

    def __init__(self, net: aether, schedule: tuple[int, int, int]):
        self.net = net
        self.schedule = schedule
        self.buffers = list(net.buffers())
        self.saved = [torch.empty_like(b) for b in self.buffers]
        self.norms = [m for m in net.modules() if isinstance(m, DeploymentNorm)]
        self.saved_modes: list[bool] = []
        self.phase: tuple[bool, bool, bool] | None = None
        self.pending = False

    def set_phase(self, step: int) -> None:
        calib, norm, obs = self.schedule
        phase = (step > calib, not 0 <= obs <= step, 0 <= norm <= step)
        if phase != self.phase:
            self.net.set_qat_phase(*phase)
            self.phase = phase

    def begin(self, step: int) -> None:
        if not self.pending:
            torch._foreach_copy_(self.saved, self.buffers)
            self.saved_modes = [norm.training for norm in self.norms]
            self.pending = True
            self.set_phase(step)

    def rollback(self) -> None:
        if self.pending:
            torch._foreach_copy_(self.buffers, self.saved)
            for norm, was_training in zip(self.norms, self.saved_modes, strict=True):
                norm.train(was_training)
            self.phase = None
            self.pending = False

    @contextmanager
    def checkpoint_view(self) -> Iterator[None]:
        """A save between microbatches contains only the last accepted update."""
        if not self.pending:
            yield
            return
        partial = [b.clone() for b in self.buffers]
        try:
            torch._foreach_copy_(self.buffers, self.saved)
            yield
        finally:
            torch._foreach_copy_(self.buffers, partial)

"""Inspect the real restoration objective in an isolated process without applying an update."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from .preflight import launch_text
from .yaml_edit import Loader, read_recipe


def _finite(value) -> float | None:
    x = float(value)
    return x if math.isfinite(x) else None


def check_loss_report_recipe(data: dict) -> str:
    network = data.get("network_g")
    arch = network.get("type") if isinstance(network, dict) else None
    if not isinstance(arch, str) or not arch or data.get("task") not in (None, "image_restoration") or arch.lower() in ("rdual", "rsr_compact"):
        raise ValueError("Loss attribution currently supports the still-image restoration generator objective.")
    paths = data.get("path")
    if paths is not None and not isinstance(paths, dict):
        raise ValueError("The recipe's path section must be a mapping.")
    if paths and paths.get("resume_state"):
        raise ValueError("Use explicit generator/discriminator weights from the checkpoint for an offline diagnostic. Clear path.resume_state in the diagnostic recipe.")
    return arch


def _measure(probe, total, terms):
    import torch
    # Gradients are unscaled, before accumulation averaging. The same multiplier
    # would cancel in cosines and squared-gradient shares.
    def gradient(scalar):
        if not isinstance(scalar, torch.Tensor) or not scalar.requires_grad:
            return None
        value = torch.autograd.grad(scalar, probe, retain_graph=True, allow_unused=True)[0]
        return None if value is None else value.detach().float().reshape(-1)

    def cosine(left, right, left_norm, right_norm):
        return _finite(torch.dot(left, right) / (left_norm * right_norm))

    total_grad = gradient(total)
    total_norm = total_grad.norm() if total_grad is not None else None
    # A custom backward can apply a gate to the sum arriving at a shared
    # activation. Independent term derivatives need not add to that result.
    term_sum = torch.zeros_like(total_grad) if total_grad is not None else None
    adversarial = None
    result = []
    for name, scalar in terms.items():
        if not name.startswith("l_g_") or not isinstance(scalar, torch.Tensor):
            continue
        grad = gradient(scalar)
        item = {"term":name, "weightedLoss":_finite(scalar.detach()), "connected":grad is not None}
        if grad is not None:
            norm = grad.norm()
            item["gradientNorm"] = _finite(norm)
            if term_sum is not None:
                term_sum.add_(grad)
                item["cosineWithTotal"] = cosine(grad, total_grad, norm, total_norm)
            if name == "l_g_gan":
                # Measure the other objectives grouped in one backward, so their
                # common prediction clamp sees their combined upstream gradient.
                others = gradient(total - scalar)
                if others is not None:
                    other_norm = others.norm()
                    adversarial = {"ganNorm": item["gradientNorm"], "otherNorm": _finite(other_norm),
                                   "ganToOtherNorm": _finite(norm / other_norm),
                                   "cosineWithOther": cosine(grad, others, norm, other_norm)}
        result.append(item)
    sum_sq = sum(item["gradientNorm"]**2 for item in result if item.get("gradientNorm") is not None)
    for item in result:
        norm = item.get("gradientNorm")
        item["squaredNormShare"] = norm**2 / sum_sq if sum_sq and norm is not None else None
    residual = (total_grad - term_sum).norm() if term_sum is not None else None
    return {"totalLoss": _finite(total.detach()), "totalGradientNorm": _finite(total_norm) if total_norm is not None else None,
            "independentTermResidualNorm": _finite(residual) if residual is not None else None,
            "adversarialBalance": adversarial, "terms": result}


def loss_report(recipe: Path, root: Path, *, device: str, batches: int,
                step: int, parameter: str | None, weights: Path | None,
                discriminator: Path | None) -> dict:
    if device not in ("cpu", "cuda") or not 1 <= batches <= 32 or step < 1:
        raise ValueError("Choose cpu or cuda, 1 to 32 batches, and an update number starting at 1.")
    text = recipe.read_text(encoding="utf-8-sig")
    data = read_recipe(text)
    arch = check_loss_report_recipe(data)
    import os
    import yaml
    if device == "cpu":
        # This module runs in its own CLI process. Keep even library probes off CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from traiNNer.training.data import create_train_val_dataloader
    from traiNNer.utils.logger import get_root_logger
    from traiNNer.models import build_model
    from traiNNer.models.sr_model import SRModel
    from traiNNer.utils.config import Config
    from traiNNer.utils.misc import set_random_seed

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected and is unavailable.")
    data = yaml.load(launch_text(text, data, root), Loader=Loader)
    paths = data.get("path") or {}
    data["path"] = paths
    if weights:
        paths["pretrain_network_g"] = str(weights)
    if discriminator:
        paths["pretrain_network_d"] = str(discriminator)
    data["num_gpu"] = int(device == "cuda")
    data["use_compile"] = False
    data["compile_discriminator"] = False
    if device == "cpu":
        data["use_amp"] = False
    # Use the configured training crop, augmentations and batch size. Read
    # synchronously so a short report never starts a pool of persistent workers.
    ds = data["datasets"]["train"]
    ds.update(num_worker_per_gpu=0, persistent_workers=False, prefetch_factor=None,
              pin_memory=False, prefetch_mode=None)
    with tempfile.TemporaryDirectory(prefix="trainner-objective-") as tmp:
        effective = Path(tmp) / "recipe.yml"
        effective.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        opt, args = Config.load_config_from_file(tmp, True, ["-opt",str(effective)])
        opt.root_path = tmp
        set_random_seed(opt.manual_seed)
        loader, _, _, _, _ = create_train_val_dataloader(opt, args, False, get_root_logger())
        if loader is None:
            raise ValueError("The recipe did not create a training loader.")
        model = build_model(opt)
        if not isinstance(model,SRModel) or type(model).optimize_parameters is not SRModel.optimize_parameters:
            raise ValueError("This task supplies a different generator objective; the still-image observer cannot inspect it.")
        selected = None
        if parameter:
            named = dict(model.get_bare_model(model.net_g).named_parameters())
            selected = named.get(parameter)
            if selected is None or not selected.requires_grad:
                raise ValueError(f"Generator parameter {parameter!r} is absent or frozen.")
        records = []
        def observe(output, total, terms):
            records.append(_measure(output if selected is None else selected,total,terms))
        iterator = iter(loader)
        for _ in range(batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            with model.data_feed_context():
                model.feed_data(batch)
            before = len(records)
            model.optimize_parameters(step,0,True,observe_objective=observe)
            if len(records) == before:
                raise ValueError("This update does not evaluate a generator objective. Select another iteration.")
        if not records:
            raise ValueError("No complete batch was available at the configured batch size.")
        return {"task":"image_restoration", "architecture":arch, "device":device, "compiled":False,
                "precision": str(model.amp_dtype) if model.use_amp else "float32",
                "update":step, "seed":opt.manual_seed, "target":parameter or "generator_output",
                "generatorWeights":paths.get("pretrain_network_g"),
                "discriminatorWeights":paths.get("pretrain_network_d"),
                "optimizerUpdates":0, "batches":records}

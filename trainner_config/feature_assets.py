"""Local inventory of known frozen feature extractors used by restoration recipes."""

import os
import stat
from pathlib import Path

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_REPOSITORY = f"facebookresearch/dinov2:{DINO_REVISION}"
_DINO_MODEL = "dinov2_vitb14_pretrain.pth"
_FDL_WEIGHTS = {
    "vgg": "vgg19-dcbb9e9d.pth",
    "resnet": "resnet101-cd907fc2.pth",
    "effnet": "efficientnet_b7_lukemelas-c5b4e57e.pth",
}
_AA_VGG = "vgg19_lpf4_finetune-7ab2cf45.pth"


def feature_assets(objectives, metrics, root: Path, *, hub: Path | None = None) -> list[dict]:
    """Report known file locations without importing torch, contacting a host or loading weights."""
    if hub is None:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        hub = Path(os.environ.get("TORCH_HOME", cache_home / "torch")).expanduser() / "hub"
    cached = hub / "checkpoints"
    assets = {}

    def add(path: Path, used_by: str, provider: str) -> None:
        if path in assets:
            if used_by not in assets[path]["usedBy"]:
                assets[path]["usedBy"].append(used_by)
            return
        item = {"file": str(path), "provider": provider, "usedBy": [used_by]}
        try:
            info = path.stat()
        except FileNotFoundError:
            item["status"] = "missing"
        except OSError as exc:
            item.update(status="unreadable", error=str(exc))
        else:
            if stat.S_ISREG(info.st_mode) and info.st_size > 0:
                item.update(status="present", bytes=info.st_size, mtimeNs=str(info.st_mtime_ns))
            else:
                item["status"] = "invalid"
        assets[path] = item

    def dists(used_by, trained_coefficients):
        add(cached / "vgg16-397923af.pth", used_by, "torchvision")
        if trained_coefficients:
            add(root / "traiNNer/losses/dists_weights.pt", used_by, "checkout")

    for objective in objectives.values():
        kind = objective.options["type"].lower()
        if kind == "neosrfdlloss":
            model = objective.options.get("model", "dinov2")
            if not isinstance(model, str):
                continue
            if model.lower() == "dinov2":
                add(hub / f"facebookresearch_dinov2_{DINO_REVISION}" / "hubconf.py", objective.path, "torch.hub")
                add(cached / _DINO_MODEL, objective.path, "dinov2")
            elif model.lower() in _FDL_WEIGHTS:
                add(cached / _FDL_WEIGHTS[model.lower()], objective.path, "torchvision")
        elif kind in {"perceptualloss", "perceptualfp16loss"}:
            antialiased = objective.options.get("antialiased", False)
            add(cached / (_AA_VGG if antialiased else _FDL_WEIGHTS["vgg"]), objective.path,
                "antialiased_cnns" if antialiased else "torchvision")
        elif kind == "distsloss":
            dists(objective.path, objective.options.get("load_weights", True))
    for name, metric in (metrics or {}).items():
        if not isinstance(metric, dict):
            continue
        kind = str(metric.get("type", "")).lower()
        if kind == "calculate_dists":
            dists(f"val.metrics.{name}", True)
        elif kind == "calculate_lpips":
            net = metric.get("net", "alex")
            backbone = {"alex": "alexnet-owt-7be5be79.pth", "vgg": "vgg16-397923af.pth",
                        "squeeze": "squeezenet1_1-b8a52dc0.pth"}.get(net) if isinstance(net, str) else None
            if backbone:
                add(cached / backbone, f"val.metrics.{name}", "torchvision")
                add(root / "traiNNer/archs/lpips/weights/v0.1" / f"{net}.pth", f"val.metrics.{name}", "checkout")
        elif kind in ("calculate_topiq", "calculate_neosr_topiq", "calculate_topiq_nr"):
            if kind == "calculate_topiq_nr":
                filename = "cfanet_nr_koniq_res50-9a73138b.pth"
            elif kind == "calculate_neosr_topiq" and metric.get("dataset") == "pipal":
                filename = "cfanet_fr_pipal_res50-69bbe5ba.pth"
            else:
                filename = "cfanet_fr_kadid_res50-2c4cc61d.pth"
            add(cached / filename, f"val.metrics.{name}", "TOPIQ")
    return list(assets.values())

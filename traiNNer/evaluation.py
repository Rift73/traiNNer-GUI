"""Dataset evaluation shared by the headless trainer and local analysis jobs."""

import math
from collections.abc import Callable
from os import path as osp
from pathlib import Path

import torch
from rich.pretty import pretty_repr
from torch.utils.data import DataLoader
from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.prefetch_dataloader import PrefetchDataLoader
from traiNNer.models import build_model
from traiNNer.tasks import load_task_provider
from traiNNer.utils import get_env_info, get_root_logger, get_time_str
from trainner_config.options import ReduxOptions
from trainner_config.tasks import resolve_task


def evaluate(opt: ReduxOptions, *, on_dataset: Callable[[dict], None] | None = None) -> list[dict]:
    """Evaluate a configured inference network in dataset order and return observed metrics."""
    assert opt.val is not None
    assert opt.dist is not None
    assert opt.path.log is not None
    selection = resolve_task(opt)
    if selection is None:
        uses_autoencoder = bool(
            opt.network_ae and not opt.network_g and not opt.high_order_degradation
        )
        task_options = None
    else:
        spec, _, settings = selection
        load_task_provider(spec)
        uses_autoencoder = spec.network_field == "network_ae"
        task_options = settings if spec.dataset_geometry != "aligned_patches" else None

    if uses_autoencoder:
        if not opt.path.pretrain_network_ae:
            raise ValueError(
                "Autoencoder evaluation requires path.pretrain_network_ae."
            )
        opt.path.pretrain_network_ae = osp.expanduser(opt.path.pretrain_network_ae)
    elif not opt.path.pretrain_network_g:
        raise ValueError("Evaluation requires path.pretrain_network_g.")

    test_datasets = [
        dataset
        for name, dataset in sorted(opt.datasets.items())
        if name.split("_", 1)[0] in ("val", "test")
    ]
    if not test_datasets:
        raise ValueError("Evaluation requires a val or test dataset.")
    if len({data.name for data in test_datasets}) != len(test_datasets):
        raise ValueError("Each evaluation dataset needs a distinct name.")
    assert isinstance(opt.num_gpu, int)
    if not opt.val.metrics_enabled and not opt.val.save_img:
        from traiNNer.tasks.temporal_results import recording
        if not recording():
            raise ValueError("Select validation metrics, saved predictions or a temporal error report.")

    if opt.num_gpu > 0:
        torch.cuda.set_per_process_memory_fraction(fraction=1.0)
        torch.backends.cudnn.benchmark = not opt.deterministic

    # Reuse the results directory without moving existing files out of it.
    Path(opt.path.log).mkdir(parents=True, exist_ok=True)
    log_file = osp.join(opt.path.log, f"test_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(logger_name="traiNNer", log_file=log_file)
    logger.info(get_env_info(use_cuda=opt.num_gpu != 0))
    logger.debug(pretty_repr(opt))

    # create test dataset and dataloader
    test_loaders: list[PrefetchDataLoader | DataLoader] = []
    for dataset_opt in test_datasets:
        test_set = build_dataset(dataset_opt, task_options=task_options)
        if not len(test_set):
            raise ValueError(f"Evaluation dataset {dataset_opt.name!r} is empty.")
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt.num_gpu,
            dist=opt.dist,
            sampler=None,
            seed=opt.manual_seed,
        )
        logger.info("Number of test %s in %s: %d", test_set.label, dataset_opt.name, len(test_set))
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    model.validate_validation_datasets(test_loaders)

    reports = []
    for i, test_loader in enumerate(test_loaders, 1):
        assert isinstance(test_loader.dataset, BaseDataset)
        dataset = test_loader.dataset.opt
        logger.info("Testing %s...", dataset.name)
        model.validation(test_loader, current_iter=-1, tb_logger=None,
                         save_img=opt.val.should_save_img(-1, dataset),
                         multi_val_datasets=len(test_loaders) > 1)
        scores = {k: float(v) for k, v in model.metric_results.items()} if model.with_metrics else {}
        channels = getattr(model, "validation_output_channels", None)
        pixel_format = "y" if channels == {1} else "rgb" if channels == {3} else None
        reports.append({"dataset": dataset.name, "samples": len(test_loader.dataset),
                        **({"outputPixelFormat": pixel_format} if pixel_format else {}),
                        "metrics": {k: v if math.isfinite(v) else None for k, v in scores.items()},
                        "nonfinite": {k: str(v) for k, v in scores.items() if not math.isfinite(v)}})
        if on_dataset is not None:
            on_dataset({"stage": "evaluating", "completed": i, "total": len(test_loaders),
                        "dataset": dataset.name})
    return reports

"""Dataset construction and epoch sizing for the training entry point.

This module resolves aligned patch sizes or whole-image task settings, then
builds the train sampler and enabled validation loaders. The epoch calculations
are intentionally kept alongside the loader configuration that determines them.
"""

import argparse
import logging
import math

from torch.utils.data import DataLoader
from trainner_config.devices import replay_worker_mode
from trainner_config.tasks import TRAINING_TASKS, resolve_task

from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.tasks import load_task_provider
from traiNNer.utils.redux_options import DatasetOptions, ReduxOptions


def _resolve_aligned_patch_sizes(scale: int, dataset_opt: DatasetOptions) -> None:
    if scale <= 0:
        raise ValueError("Training scale must be a positive integer.")
    if dataset_opt.gt_size is not None and (dataset_opt.gt_size < 1 or dataset_opt.gt_size % scale):
        raise ValueError("gt_size must be positive and exactly divisible by scale for aligned training crops.")
    if dataset_opt.lq_size is not None and dataset_opt.lq_size < 1:
        raise ValueError("lq_size must be a positive integer.")
    if dataset_opt.gt_size is None and dataset_opt.lq_size is not None:
        dataset_opt.gt_size = dataset_opt.lq_size * scale
    elif dataset_opt.lq_size is None and dataset_opt.gt_size is not None:
        dataset_opt.lq_size = dataset_opt.gt_size // scale
    else:
        raise ValueError(
            "Exactly one of gt_size or lq_size must be defined in the train dataset"
        )


def create_train_val_dataloader(
    opt: ReduxOptions,
    args: argparse.Namespace,
    val_enabled: bool,
    logger: logging.Logger,
) -> tuple[DataLoader | None, EnlargedSampler | None, list[DataLoader], int, int]:
    """Build loaders and return the sampler plus epoch/iteration limits.

    The args parameter is retained for callers of the original train.py API.
    Dataset options are resolved in place, as before; this is a one-time setup
    step, not a repeatable query against an already-resolved options object.
    """
    assert isinstance(opt.num_gpu, int)
    assert opt.world_size is not None
    assert opt.dist is not None

    spec = TRAINING_TASKS.get(opt.task) if opt.task is not None else None
    geometry_kind = spec.dataset_geometry if spec else "aligned_patches"
    selection = resolve_task(opt) if geometry_kind != "aligned_patches" else None
    task_options = selection[2] if selection is not None else None
    if spec is not None:
        load_task_provider(spec)  # Datasets are built before the model.

    # create train and val dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = (
        None,
        None,
        [],
        0,
        0,
    )
    val_names = set()
    for phase, dataset_opt in opt.datasets.items():
        if phase == "train":
            assert opt.train is not None
            assert dataset_opt.batch_size_per_gpu is not None

            if geometry_kind == "aligned_patches":
                _resolve_aligned_patch_sizes(opt.scale, dataset_opt)

            train_set = build_dataset(dataset_opt, task_options=task_options)
            if not len(train_set):
                raise ValueError(f"Training dataset {dataset_opt.name!r} is empty.")
            if dataset_opt.accum_iter < 1:
                raise ValueError("Gradient accumulation requires at least one microbatch.")
            dataset_enlarge_ratio = dataset_opt.dataset_enlarge_ratio
            if dataset_enlarge_ratio == "auto":
                dataset_enlarge_ratio = max(
                    2000 * dataset_opt.batch_size_per_gpu // len(train_set), 1
                )
            seed = None
            if replay_worker_mode(opt):
                from trainner_config.devices import WORKER_REPLAY_DATASETS
                expected = WORKER_REPLAY_DATASETS[dataset_opt.type.casefold()][1]
                cls = type(train_set)
                if f"{cls.__module__}.{cls.__qualname__}" != expected or opt.manual_seed is None:
                    raise ValueError("Position-seeded worker replay requires the registered dataset implementation and a seed.")
                seed = opt.manual_seed
            train_sampler = EnlargedSampler(
                train_set, opt.world_size, opt.rank, dataset_enlarge_ratio, paired_replay_seed=seed
            )
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt.num_gpu,
                dist=opt.dist,
                sampler=train_sampler,
                seed=opt.manual_seed,
            )

            microbatches = len(train_loader)
            if microbatches == 0:
                raise ValueError("No complete training batch fits. Reduce the batch size or increase the dataset size.")
            # Accumulation spans sampler epochs, so fractional updates per epoch are valid.
            iter_per_epoch = microbatches / dataset_opt.accum_iter

            opt.switch_iter_per_epoch = len(train_set) // (
                dataset_opt.batch_size_per_gpu * dataset_opt.accum_iter * opt.world_size
            )

            total_iters = int(opt.train.total_iter)
            total_epochs = math.ceil(total_iters * dataset_opt.accum_iter / microbatches)
            if geometry_kind == "records":
                geometry = ""
            elif geometry_kind == "whole_images":
                geometry = f"\t{'Image geometry:':<30} {'whole image':>9}\n"
            else:
                assert dataset_opt.gt_size is not None and dataset_opt.lq_size is not None
                geometry = (
                    f"\t{'HR crop size:':<30} {dataset_opt.gt_size:>9,}"
                    f"\t{'LR crop size:':<30} {dataset_opt.lq_size:>9,}\n"
                )
            logger.info(
                "Training statistics for [b]%s[/b]:\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s\n"
                "%s"
                "\t%-30s %9s\t%-30s %9s\n"
                "\t%-30s %9s\t%-30s %9s",
                opt.name,
                f"Number of train {train_set.label}:",
                f"{len(train_set):,}",
                "Dataset enlarge ratio:",
                f"{dataset_enlarge_ratio:,}",
                "Batch size per gpu:",
                f"{dataset_opt.batch_size_per_gpu:,}",
                "Accumulate iterations:",
                f"{dataset_opt.accum_iter:,}",
                geometry,
                "World size (gpu number):",
                f"{opt.world_size:,}",
                "Require iter per epoch:",
                f"{iter_per_epoch:,g}",
                "Total epochs:",
                f"{total_epochs:,}",
                "Total iters:",
                f"{total_iters:,}",
            )
            if len(train_set) < 100:
                logger.warning(
                    "Number of training %s is low: %d, training quality may be impacted. Please use more training %s for best training results.",
                    train_set.label,
                    len(train_set),
                    train_set.label,
                )
        elif phase.split("_")[0] == "val":
            if val_enabled:
                if dataset_opt.name in val_names:
                    raise ValueError(f"Repeated validation dataset name {dataset_opt.name!r}. Each dataset needs a unique name.")
                val_names.add(dataset_opt.name)
                val_set = build_dataset(dataset_opt, task_options=task_options)
                if len(val_set) == 0:
                    raise ValueError(f"Validation dataset {dataset_opt.name!r} is empty.")
                val_loader = build_dataloader(
                    val_set,
                    dataset_opt,
                    num_gpu=opt.num_gpu,
                    dist=opt.dist,
                    sampler=None,
                    seed=opt.manual_seed,
                )
                logger.info(
                    "Number of val images/folders in %s: %d",
                    dataset_opt.name,
                    len(val_set),
                )
                val_loaders.append(val_loader)
            else:
                logger.info(
                    "Validation is disabled, skip building val dataset %s.",
                    dataset_opt.name,
                )
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

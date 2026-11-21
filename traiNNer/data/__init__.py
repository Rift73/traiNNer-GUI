import os
import random
import sys
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from msgspec import Struct
from torch.utils.data import default_collate
from torch.utils.data.dataset import Dataset
from trainner_config.registry_index import install

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_sampler import EnlargedSampler
from traiNNer.data.prefetch_dataloader import PrefetchDataLoader
from traiNNer.data.shared_release import collate_shared, releases_shared_memory
from traiNNer.utils import get_root_logger
from traiNNer.utils.dist_util import get_dist_info
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.rng import RNG
from traiNNer.utils.types import DataLoaderArgs

__all__ = ["build_dataloader", "build_dataset"]

install(__name__, Path(__file__).parent, "_dataset.py", {"dataset": DATASET_REGISTRY})


def __getattr__(name: str):
    # Preserve explicit imports from the package without loading OTF on every import.
    modules = {"RealESRGANDataset": "realesrgan_dataset", "collate_dvd": "dvd_response",
               "collate_still_source": "still_source", "uses_still_sources": "still_source"}
    if name in modules:
        return getattr(import_module(f".{modules[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _image_collate(dataset: Dataset, opt: DatasetOptions):
    # An instance of RealESRGANDataset or any subclass already loaded its base
    # class's module. Keep the type check independent of user-supplied YAML names.
    module = sys.modules.get("traiNNer.data.realesrgan_dataset")
    if module is not None and isinstance(dataset, getattr(module, "RealESRGANDataset", ())):
        from .still_source import collate_still_source, uses_still_sources

        if uses_still_sources(opt):
            dataset.still_sources = True
            return collate_still_source
    if opt.dvd_prob > 0:
        from .dvd_response import collate_dvd

        return collate_dvd
    return None


def build_dataset(
    dataset_opt: DatasetOptions, *, task_options: Struct | None = None
) -> BaseDataset:
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
        task_options: Decoded settings for a whole-image or record-based provider.
    """
    logger = get_root_logger()
    logger.info("Building Dataset %s...", dataset_opt.name)

    # Legacy and direct callers may have no selected task at this boundary.
    name = dataset_opt.type.lower()

    def registered() -> bool:
        return name in DATASET_REGISTRY or name + "_traiNNer" in DATASET_REGISTRY.keys()

    if not registered():
        from traiNNer.tasks import discover_task_factory

        discover_task_factory(name, registered, dataset=True)
    factory = DATASET_REGISTRY.get(dataset_opt.type)
    dataset = (
        factory(dataset_opt)
        if task_options is None
        else factory(dataset_opt, task_options)
    )

    logger.info(
        "Dataset [bold]%s[/bold] - %s is built.",
        dataset.__class__.__name__,
        dataset_opt.name,
        extra={"markup": True},
    )
    return dataset


def build_dataloader(
    dataset: Dataset,
    dataset_opt: DatasetOptions,
    num_gpu: int = 1,
    dist: bool = False,
    sampler: EnlargedSampler | None = None,
    seed: int | None = None,
) -> PrefetchDataLoader | torch.utils.data.DataLoader:
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train', 'val' or 'test'.
            num_worker_per_gpu (int | "auto"): Integer training counts are per GPU.
                Evaluation counts are per loader and default to zero.
            batch_size_per_gpu (int): Training batch size per GPU. Datasets that
                support batched evaluation can also use it on validation/test splits.
        num_gpu (int): Number of GPUs for training batch sizing and memory pinning.
        dist (bool): One process per distributed rank. Auto shares CPUs across ranks.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    from trainner_config.devices import auto_workers

    rank, world_size = get_dist_info()
    train = dataset_opt.phase == "train"
    multiplier = max(1, num_gpu) if train and not dist else 1
    workers = dataset_opt.num_worker_per_gpu
    if workers == "auto":
        num_workers = auto_workers(world_size if dist else 1)
        get_root_logger().info(
            "Dataset %s: auto selected %d loader workers per process.",
            dataset_opt.name, num_workers,
        )
    elif workers is None and not train:
        num_workers = 0
    elif type(workers) is int and workers >= 0:
        num_workers = workers * multiplier
    else:
        raise ValueError("num_worker_per_gpu must be a nonnegative integer or auto.")
    dataset_collate = getattr(dataset, "collate_fn", None)
    if train:
        assert dataset_opt.batch_size_per_gpu is not None
        dataloader_args: DataLoaderArgs = {
            "dataset": dataset,
            "batch_size": dataset_opt.batch_size_per_gpu * multiplier,
            "shuffle": sampler is None,
            "sampler": sampler,
            "drop_last": True,
        }
        if dataset_collate is not None:
            dataloader_args["collate_fn"] = dataset_collate
        elif (image_collate := _image_collate(dataset, dataset_opt)) is not None:
            dataloader_args["collate_fn"] = image_collate
        release = dataset_collate is None and releases_shared_memory(
            num_workers, bool(num_gpu) and dataset_opt.pin_memory)
        if dataset_opt.native_paired_crop and dataset_collate is None:
            from .paired_image_dataset import PairedImageDataset
            if type(dataset) is PairedImageDataset:
                from .paired_crop import collate_crop_samples
                dataset_collate = partial(collate_crop_samples, release_shared=release)
                dataloader_args["collate_fn"] = dataset_collate
        if dataset_collate is None and release:
            # The batch goes to the pinned-memory thread in a carrier whose
            # pin_memory() hands its shared storages to the release thread.
            dataloader_args["collate_fn"] = partial(
                collate_shared,
                collate=dataloader_args.get("collate_fn") or default_collate,
            )
    elif dataset_opt.phase in ["val", "test"]:  # validation
        dataloader_args = {
            "dataset": dataset,
            "batch_size": dataset_opt.batch_size_per_gpu
            if getattr(dataset, "supports_batched_evaluation", False) and dataset_opt.batch_size_per_gpu is not None
            else 1,
            "shuffle": False,
        }
        if dataset_collate is not None:
            dataloader_args["collate_fn"] = dataset_collate
    else:
        raise ValueError(
            f"Wrong dataset phase: {dataset_opt.phase}. Supported ones are 'train', 'val' and 'test'."
        )

    worker_context = None
    dataloader_args["num_workers"] = num_workers
    if num_workers:
        # Automatic counts can differ between ranks with different affinity masks.
        auto_world = world_size if workers == "auto" and dist else None
        if seed is not None:
            last = _worker_seed(num_workers - 1, num_workers, rank, seed, auto_world)
            if seed < 0 or last >= 2**32:
                raise ValueError(
                    "The manual seed plus loader worker offsets exceeds the NumPy seed range."
                )
        dataloader_args["worker_init_fn"] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed, auto_world=auto_world,
        )
        dataloader_args["persistent_workers"] = dataset_opt.persistent_workers
        dataloader_args["prefetch_factor"] = dataset_opt.prefetch_factor
        # Validation can first start after the parent has already decoded images.
        # Forking that process can strand a worker on an inherited native lock.
        if not train and torch.multiprocessing.get_start_method() == "fork":
            worker_context = "spawn"
    dataloader_args["pin_memory"] = bool(num_gpu) and dataset_opt.pin_memory
    prefetch_mode = dataset_opt.prefetch_mode
    if prefetch_mode == "cpu":  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.num_prefetch_queue
        logger = get_root_logger()
        logger.info(
            "Use %s prefetch dataloader: num_prefetch_queue = %d",
            prefetch_mode,
            num_prefetch_queue,
        )
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue,
            multiprocessing_context=worker_context,
            **dataloader_args,
        )
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(multiprocessing_context=worker_context, **dataloader_args)


def _worker_seed(
    worker_id: int, num_workers: int, rank: int, seed: int, auto_world: int | None,
) -> int:
    if auto_world is not None:
        return seed + rank + worker_id * auto_world
    return seed + num_workers * rank + worker_id


def worker_init_fn(
    worker_id: int,
    num_workers: int,
    rank: int,
    seed: int | None,
    auto_world: int | None = None,
) -> None:
    # The loader's parallelism is its workers: chainner_ext's resize would start
    # a thread per CPU in each of them. Its pool reads this when the worker
    # first resizes.
    os.environ["RAYON_NUM_THREADS"] = "1"
    if seed is None:
        # PyTorch already seeded Python and NumPy in this worker. Give the
        # project generator its own stream even when the parent initialized it.
        RNG.init_rng(torch.initial_seed())
        return
    worker_seed = _worker_seed(worker_id, num_workers, rank, seed, auto_world)
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)
    RNG.init_rng(worker_seed)

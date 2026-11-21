"""Deterministic rank-local sampling and checkpointable loader positions."""

import math
from collections.abc import Iterator, Sized
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset
from torch.utils.data.sampler import Sampler

_INDEX_CHUNK = 65_536
_CURSOR_VERSION = 1


def _python_indices(indices: Tensor, dataset_size: int) -> Iterator[int]:
    # DataLoader still receives Python ints. Materialize only a bounded chunk
    # after rank-striding and skipping the consumed prefix in the CPU tensor.
    for start in range(0, indices.numel(), _INDEX_CHUNK):
        yield from (
            indices[start : start + _INDEX_CHUNK].remainder(dataset_size).tolist()
        )


class EnlargedSampler(Sampler[int | tuple[int, str]]):
    """Shuffle an enlarged range by epoch, then stride it by rank.

    Repeated dataset entries are produced by taking the permutation elements
    modulo dataset size. Resuming slices the rank-local stream before yielding
    any indices to the data loader.
    """

    def __init__(
        self, dataset: Dataset, num_replicas: int, rank: int | None, ratio: int = 1,
        *, paired_replay_seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.ratio = ratio
        self.epoch = 0
        self.start_sample = 0
        self.paired_replay_seed = paired_replay_seed
        assert isinstance(dataset, Sized), "Unable to determine length of dataset"
        self.dataset_size = len(dataset)
        self.num_samples = math.ceil(self.dataset_size * ratio / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[int | tuple[int, str]]:
        generator = torch.Generator(device="cpu").manual_seed(self.epoch)
        permutation = torch.randperm(self.total_size, generator=generator, device="cpu")
        rank = 0 if self.rank is None else self.rank
        start = rank + self.start_sample * self.num_replicas
        indices = permutation[start : self.total_size : self.num_replicas]
        assert len(indices) == len(self)
        assert isinstance(self.dataset, Sized), "Unable to determine length of dataset"
        selected = _python_indices(indices, len(self.dataset))
        if self.paired_replay_seed is None:
            return selected
        # The index encoding is shared by prepared datasets. Keep its existing
        # domain string so previously saved paired-image worker cursors replay.
        prefix = f"paired-v1/{self.paired_replay_seed}/{self.epoch}/{rank}/"
        return ((index, prefix + str(position))
                for position, index in enumerate(selected, self.start_sample))

    def __len__(self) -> int:
        return self.num_samples - self.start_sample

    def set_epoch(self, epoch: int, *, start_sample: int = 0) -> None:
        if not 0 <= start_sample <= self.num_samples:
            raise ValueError("The rank-local sampler offset is outside this epoch.")
        self.epoch = epoch
        self.start_sample = start_sample


class SamplerGeometry(TypedDict):
    dataset_type: str
    dataset_size: int
    enlarge_ratio: int
    world_size: int
    samples_per_rank: int
    batch_size: int
    drop_last: bool
    accum_iters: int


class SamplerCursor(TypedDict):
    """Next rank-local batch of the epoch-seeded enlarged permutation.

    All ranks share the same cursor because their loader geometry gives them
    equal numbers of batches. The rank itself determines the stride at load
    time. Numeric indices describe order only. Dataset contents, worker RNG,
    prefetched work, and augmentation state are outside this checkpoint.
    """

    version: int
    epoch: int
    next_batch: int
    geometry: SamplerGeometry


def sampler_geometry(
    loader: DataLoader, sampler: EnlargedSampler, accum_iters: int
) -> SamplerGeometry | None:
    """Describe the ordered finite map-loader path supported by this cursor."""
    dataset = loader.dataset
    batches = loader.batch_sampler
    if (
        dataset is not sampler.dataset
        or isinstance(dataset, IterableDataset)
        or not isinstance(dataset, Sized)
        or len(dataset) != sampler.dataset_size
        or type(batches) is not BatchSampler
        or batches.sampler is not sampler
        or not getattr(loader, "in_order", True)
    ):
        return None
    return {
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "dataset_size": sampler.dataset_size,
        "enlarge_ratio": sampler.ratio,
        "world_size": sampler.num_replicas,
        "samples_per_rank": sampler.num_samples,
        "batch_size": batches.batch_size,
        "drop_last": batches.drop_last,
        "accum_iters": accum_iters,
    }


def _batches_per_epoch(geometry: SamplerGeometry) -> int:
    full, remainder = divmod(geometry["samples_per_rank"], geometry["batch_size"])
    return full + int(not geometry["drop_last"] and remainder > 0)


def make_sampler_cursor(
    geometry: SamplerGeometry | None, epoch: int, next_batch: int
) -> SamplerCursor | None:
    if geometry is None:
        return None
    num_batches = _batches_per_epoch(geometry)
    if next_batch == num_batches:
        epoch, next_batch = epoch + 1, 0
    elif not 0 <= next_batch < num_batches:
        raise ValueError("The saved sampler position is outside the loader epoch.")
    return {
        "version": _CURSOR_VERSION,
        "epoch": epoch,
        "next_batch": next_batch,
        "geometry": geometry,
    }


def sampler_cursor_epoch(cursor: object) -> int | None:
    """Read the recorded next-batch epoch even when the old geometry is reset."""
    if (
        isinstance(cursor, dict)
        and type(cursor.get("version")) is int
        and cursor["version"] == _CURSOR_VERSION
        and type(epoch := cursor.get("epoch")) is int
        and epoch >= 0
    ):
        return epoch
    return None


def restore_sampler_cursor(
    cursor: object, geometry: SamplerGeometry | None
) -> tuple[int, int]:
    """Validate the stored position before creating a loader iterator."""
    restart = "Use --restart-epoch to start the recorded epoch at its first batch."
    if geometry is None:
        raise ValueError(
            f"This loader does not support sampler cursor resume. {restart}"
        )
    epoch = sampler_cursor_epoch(cursor)
    if (
        epoch is None
        or not isinstance(cursor, dict)
        or not isinstance(saved := cursor.get("geometry"), dict)
    ):
        raise ValueError(f"Missing or unsupported sampler cursor. {restart}")

    changed = [
        f"{key} (saved {saved.get(key)!r}, current {value!r})"
        for key, value in geometry.items()
        if type(saved.get(key)) is not type(value) or saved[key] != value
    ]
    if changed:
        raise ValueError(
            f"Sampler cursor geometry changed: {', '.join(changed)}. {restart}"
        )

    next_batch = cursor.get("next_batch")
    if type(next_batch) is not int or not 0 <= next_batch < _batches_per_epoch(
        geometry
    ):
        raise ValueError(f"Invalid sampler cursor position. {restart}")
    return epoch, next_batch

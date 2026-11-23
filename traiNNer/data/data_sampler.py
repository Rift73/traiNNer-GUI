"""Deterministic rank-local sampling and checkpointable loader positions."""

import math
from collections.abc import Iterator, Sized
from typing import NamedTuple, NotRequired, TypedDict, cast

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset
from torch.utils.data.sampler import Sampler

_INDEX_CHUNK = 65_536
_BATCH_CURSOR_VERSION = 1
_SAMPLE_CURSOR_VERSION = 2


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
    """Next rank-local position of the epoch-seeded enlarged permutation.

    All ranks share the same cursor because their loader geometry gives them
    equal numbers of batches. The rank itself determines the stride at load
    time. Numeric indices describe order only. Dataset contents, worker RNG,
    prefetched work, and augmentation state are outside this checkpoint.
    """

    version: int
    epoch: int
    next_batch: int
    # Version 2 adds the offset when it is not a multiple of batch size.
    next_sample: NotRequired[int]
    geometry: SamplerGeometry


class SamplerResume(NamedTuple):
    epoch: int
    next_sample: int
    saved_batch_size: int
    restarted_epoch: bool


def auto_enlarge_ratio(dataset_size: int, batch_size_per_gpu: int) -> int:
    return max(2000 * batch_size_per_gpu // dataset_size, 1)


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


def _has_batch_at(geometry: SamplerGeometry, sample: int) -> bool:
    required = geometry["batch_size"] if geometry["drop_last"] else 1
    return geometry["samples_per_rank"] - sample >= required


def make_sampler_cursor(
    geometry: SamplerGeometry | None, epoch: int, next_sample: int
) -> SamplerCursor | None:
    if geometry is None:
        return None
    if not 0 <= next_sample <= geometry["samples_per_rank"]:
        raise ValueError("The saved sampler position is outside the loader epoch.")
    if not _has_batch_at(geometry, next_sample):
        epoch, next_sample = epoch + 1, 0

    next_batch, offset = divmod(next_sample, geometry["batch_size"])
    cursor: SamplerCursor = {
        "version": _BATCH_CURSOR_VERSION,
        "epoch": epoch,
        "next_batch": next_batch,
        "geometry": geometry,
    }
    if offset:
        cursor["version"] = _SAMPLE_CURSOR_VERSION
        cursor["next_sample"] = next_sample
    return cursor


def sampler_cursor_epoch(cursor: object) -> int | None:
    """Read the recorded epoch even when the old geometry is reset."""
    if (
        isinstance(cursor, dict)
        and type(cursor.get("version")) is int
        and cursor["version"] in (_BATCH_CURSOR_VERSION, _SAMPLE_CURSOR_VERSION)
        and type(epoch := cursor.get("epoch")) is int
        and epoch >= 0
    ):
        return epoch
    return None


def _matches_auto_enlargement(geometry: SamplerGeometry, multiplier: int) -> bool:
    batch_per_gpu, remainder = divmod(geometry["batch_size"], multiplier)
    dataset_size = geometry["dataset_size"]
    ratio = geometry["enlarge_ratio"]
    samples = math.ceil(dataset_size * ratio / geometry["world_size"])
    return (
        remainder == 0
        and ratio == auto_enlarge_ratio(dataset_size, batch_per_gpu)
        and geometry["samples_per_rank"] == samples
    )


def restore_sampler_cursor(
    cursor: object,
    geometry: SamplerGeometry | None,
    *,
    auto_batch_size_per_gpu: int | None = None,
) -> SamplerResume:
    """Validate the saved position and rebase it after a batch-size edit."""
    restart = "Use --restart-epoch to start the recorded epoch at its first batch."
    if geometry is None:
        raise ValueError(
            f"This loader does not support sampler cursor resume. {restart}"
        )
    epoch = sampler_cursor_epoch(cursor)
    if (
        epoch is None
        or not isinstance(cursor, dict)
        or not isinstance(saved_geometry := cursor.get("geometry"), dict)
    ):
        raise ValueError(f"Missing or unsupported sampler cursor. {restart}")

    if any(
        type(saved_geometry.get(key)) is not type(value)
        for key, value in geometry.items()
    ):
        raise ValueError(f"Invalid sampler cursor geometry. {restart}")
    saved = cast(SamplerGeometry, saved_geometry)
    saved_batch_size = saved["batch_size"]
    next_batch = cursor.get("next_batch")
    if saved_batch_size < 1 or type(next_batch) is not int:
        raise ValueError(f"Invalid sampler cursor position. {restart}")
    next_sample = (
        cursor.get("next_sample")
        if cursor["version"] == _SAMPLE_CURSOR_VERSION
        else next_batch * saved_batch_size
    )
    if (
        type(next_sample) is not int
        or next_sample < 0
        or not _has_batch_at(saved, next_sample)
        or next_batch != next_sample // saved_batch_size
    ):
        raise ValueError(f"Invalid sampler cursor position. {restart}")

    changed = {key for key in geometry if saved[key] != geometry[key]}
    if changed - {"batch_size"}:
        if (
            auto_batch_size_per_gpu is not None
            and {"batch_size", "enlarge_ratio"} <= changed
            and changed <= {"batch_size", "enlarge_ratio", "samples_per_rank"}
            and _matches_auto_enlargement(
                saved, geometry["batch_size"] // auto_batch_size_per_gpu
            )
        ):
            # An offset into the saved permutation cannot locate a suffix
            # when automatic enlargement changes the permutation length.
            return SamplerResume(epoch, 0, saved_batch_size, True)
        detail = ", ".join(
            f"{key} (saved {saved[key]!r}, current {geometry[key]!r})"
            for key in geometry
            if key in changed
        )
        raise ValueError(f"Sampler cursor geometry changed: {detail}. {restart}")

    if not _has_batch_at(geometry, next_sample):
        # With drop_last, a larger batch can exhaust the remaining suffix.
        epoch, next_sample = epoch + 1, 0
    return SamplerResume(epoch, next_sample, saved_batch_size, False)

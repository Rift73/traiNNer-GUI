"""Process-local random streams at an accepted sampler checkpoint."""
from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch

from traiNNer.utils.rng import RNG


def seed_legacy_numpy(seed: int) -> None:
    # The legacy function accepts uint32, while the project generator accepts
    # larger seeds. Only the explicitly replayable lane changes this stream.
    np.random.seed(seed % (1 << 32))


def snapshot(seed: int, cuda_device: torch.device | None) -> dict:
    algorithm, keys, index, has_gaussian, cached = np.random.get_state()
    return {
        "version": 1,
        "seed": seed,
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), index, has_gaussian, cached),
        "project": RNG.capture_streams(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None if cuda_device is None else torch.cuda.get_rng_state(cuda_device),
    }


def restore(state: dict, seed: int, cuda_device: torch.device | None) -> None:
    if state.get("version") != 1 or state.get("seed") != seed:
        raise ValueError("Checkpoint random streams require the original manual_seed and snapshot version. "
                         "Set checkpoint_rng to false to continue with fresh streams.")
    saved_cuda = state["torch_cuda"]
    if (cuda_device is None) != (saved_cuda is None):
        raise ValueError("Checkpoint random streams require the original CPU or CUDA device type.")
    algorithm, keys, index, has_gaussian, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), index, has_gaussian, cached))
    RNG.restore_streams(state["project"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda_device is not None:
        torch.cuda.set_rng_state(saved_cuda, cuda_device)


@contextmanager
def preserve(seed: int, cuda_device: torch.device | None):
    state = snapshot(seed, cuda_device)
    try:
        yield
    finally:
        restore(state, seed, cuda_device)

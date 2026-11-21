"""Release of a training batch's shared memory after its pinned copy.

With the file_descriptor sharing strategy, a batch a loader worker sends
arrives as tensors whose storages map shared-memory files. Dropping the last
reference to such a storage unmaps it and frees the file's pages while the
interpreter lock is held; for a batch of tens of MB that takes milliseconds in
which no other Python thread runs. torch's pinned-memory thread calls
pin_memory() on a batch object that has that method, so a batch carried in
SharedBatch is pinned exactly as torch pins it. The storages the collation
created for the batch then go to a thread that truncates each file to length 0
(os.ftruncate frees the pages without the interpreter lock) before it drops
the reference, whose unmap then has no pages left to free.
"""

import os
import queue
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch
from torch.utils.data._utils.pin_memory import pin_memory

from traiNNer.utils import get_root_logger

# Smaller storages keep the default release: freeing a storage's pages costs
# about 0.2 ms per MiB, too little below 1 MiB to be worth a system call.
RELEASE_MIN_BYTES = 1 << 20
# Storages waiting for the release thread; one large storage per batch, so this
# covers a few batches of delay while bounding the shared memory held back.
RELEASE_QUEUE_SIZE = 8


def shared_fd(storage: torch.UntypedStorage) -> int:
    """The descriptor of a storage shared through the file_descriptor strategy;
    RuntimeError for a storage without one."""
    # torch has no public call that returns a shared storage's descriptor.
    return storage._get_shared_fd()  # pyright: ignore[reportAttributeAccessIssue] # noqa: SLF001


class Releaser:
    """The process's release thread, started on first use."""

    def __init__(self) -> None:
        self.enabled = True
        self.released = 0
        self.storages: queue.Queue[torch.UntypedStorage] = queue.Queue(
            RELEASE_QUEUE_SIZE
        )
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def switch_off(self, reason: object) -> None:
        with self.lock:
            if not self.enabled:
                return
            self.enabled = False
        get_root_logger().warning(
            "Releasing a batch's shared memory failed (%s); batches use the "
            "default release from now on.",
            reason,
        )

    def submit(self, storages: list[torch.UntypedStorage]) -> None:
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(
                    target=self._run, name="shared_release", daemon=True
                )
                self.thread.start()
            thread = self.thread
        for storage in storages:
            while self.enabled:
                try:
                    self.storages.put(storage, timeout=1.0)
                    break
                except queue.Full:
                    if not thread.is_alive():
                        self.switch_off("the release thread is not running")

    def _run(self) -> None:
        while True:
            storage = self.storages.get()
            if self.enabled:
                try:
                    os.ftruncate(shared_fd(storage), 0)
                    self.released += 1
                except (OSError, RuntimeError) as error:
                    self.switch_off(error)
            del storage


RELEASER = Releaser()


def _tensors(data: object) -> Iterator[torch.Tensor]:
    if isinstance(data, torch.Tensor):
        yield data
    elif isinstance(data, Mapping):
        for value in data.values():
            yield from _tensors(value)
    elif isinstance(data, list | tuple):
        for value in data:
            yield from _tensors(value)


class SharedBatch:
    """A collated training batch on its way from a loader worker to the
    pinned-memory thread, with the tensors whose storages the collation made
    for this batch; pin_memory() returns the plain pinned batch."""

    def __init__(self, batch: Any, owned: list[torch.Tensor]) -> None:
        self.batch = batch
        self.owned = owned

    def pin_memory(self) -> Any:
        pinned = pin_memory(self.batch)
        if RELEASER.enabled:
            RELEASER.submit(
                [
                    tensor.untyped_storage()
                    for tensor in self.owned
                    if tensor.untyped_storage().is_shared()
                ]
            )
        return pinned


def collate_shared(samples: Any, collate: Callable[[Any], Any]) -> SharedBatch:
    """The collation's batch, carried to the pinned-memory thread with the
    large CPU tensors whose storages the collation created: a storage that
    also backs a tensor of the samples stays the worker's."""
    inputs = {tensor.untyped_storage().data_ptr() for tensor in _tensors(samples)}
    batch = collate(samples)
    owned: dict[int, torch.Tensor] = {}
    for tensor in _tensors(batch):
        storage = tensor.untyped_storage()
        if (
            tensor.device.type == "cpu"
            and storage.nbytes() >= RELEASE_MIN_BYTES
            and storage.data_ptr() not in inputs
        ):
            owned.setdefault(storage.data_ptr(), tensor)
    return SharedBatch(batch, list(owned.values()))


def releases_shared_memory(num_workers: int, pin: bool) -> bool:
    """Whether a training loader's batches reach the main process as
    shared-memory files that its pinned-memory thread copies: workers, pinned
    memory with an accelerator (as DataLoader decides), Linux, the
    file_descriptor strategy, and a torch whose storages expose their
    descriptor."""
    return (
        num_workers > 0
        and pin
        and torch.accelerator.is_available()
        and sys.platform.startswith("linux")
        and torch.multiprocessing.get_sharing_strategy() == "file_descriptor"
        and hasattr(torch.UntypedStorage, "_get_shared_fd")
    )

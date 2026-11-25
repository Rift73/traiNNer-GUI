import queue
import threading
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader

from traiNNer.utils.redux_options import ReduxOptions


def _memory_format_for_tensor(
    tensor: torch.Tensor, memory_format: torch.memory_format
) -> torch.memory_format:
    """Apply channels_last only to rank-4 image tensors."""
    if memory_format is torch.channels_last and tensor.dim() != 4:
        return torch.preserve_format
    return memory_format


_END = object()


class PrefetchGenerator(threading.Thread):
    """Deliver loader results and its original exception in order."""

    def __init__(self, generator: Iterator, num_prefetch_queue: int) -> None:
        super().__init__(daemon=True, name="trainner-data-prefetch")
        if num_prefetch_queue < 1:
            raise ValueError("num_prefetch_queue must be positive.")
        self.queue = queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.cancelled = threading.Event()
        self.error: BaseException | None = None
        self.start()

    def _put(self, value: Any) -> bool:
        while not self.cancelled.is_set():
            try:
                self.queue.put(value, timeout=0.2)
            except queue.Full:
                continue
            return True
        return False

    def run(self) -> None:
        try:
            for item in self.generator:
                if not self._put(item):
                    return
        except BaseException as error:
            self.error = error
        finally:
            self._put(_END)

    def __next__(self) -> Any:
        if self.cancelled.is_set():
            raise StopIteration
        item = self.queue.get()
        if item is _END:
            self.cancelled.set()
            if self.error is not None:
                error, self.error = self.error, None
                raise error
            raise StopIteration
        return item

    def __iter__(self) -> Iterator:
        return self

    def close(self) -> None:
        self.cancelled.set()
        # Free a slot if the producer is blocked. Joining before reusing a
        # persistent DataLoader prevents two threads consuming the same iterator.
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        self.join()


class PrefetchDataLoader(DataLoader):
    """Prefetch on a host thread and release it when iteration ends early."""

    def __init__(self, num_prefetch_queue: int, **kwargs) -> None:
        self.num_prefetch_queue = num_prefetch_queue
        super().__init__(**kwargs)

    def __iter__(self) -> Iterator:
        producer = PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)
        try:
            yield from producer
        finally:
            producer.close()


class CPUPrefetcher:
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader: DataLoader) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self) -> Any:
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def close(self) -> None:
        if hasattr(self.loader, "close"):
            self.loader.close()

    def reset(self) -> None:
        self.close()
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader: DataLoader, opt: ReduxOptions) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device("cuda" if opt.num_gpu != 0 else "cpu")
        self.batch = None
        self.memory_format = (
            torch.channels_last
            if opt.use_amp and opt.use_channels_last
            else torch.preserve_format
        )
        self.preload()

    def preload(self) -> None:
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):  # type: ignore
            for k, v in self.batch.items():
                # Crop and replay-pool decisions read these flags on the host.
                if torch.is_tensor(v) and k not in (
                    "normalized_byte_gt",
                    "otf_normalized_byte_gt",
                    "otf_padded_height",
                    "otf_padded_width",
                    "gt_geometry",
                ):
                    self.batch[k] = self.batch[k].to(
                        device=self.device,
                        memory_format=_memory_format_for_tensor(v, self.memory_format),
                        non_blocking=True,
                    )

    def next(self) -> Any:
        consumer_stream = torch.cuda.current_stream()
        consumer_stream.wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            # The copy stream created these allocations. Waiting makes their
            # contents ready, but the allocator must also know that training
            # uses them before it can recycle their storage on the copy stream.
            for value in batch.values():
                if torch.is_tensor(value) and value.is_cuda:
                    value.record_stream(consumer_stream)
        self.preload()
        return batch

    def close(self) -> None:
        if hasattr(self.loader, "close"):
            self.loader.close()

    def reset(self) -> None:
        self.close()
        self.loader = iter(self.ori_loader)
        self.preload()

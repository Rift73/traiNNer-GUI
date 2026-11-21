"""Dispatch persisted, explicitly released queue entries while the local service runs."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .update_gate import UpdateDeferred

LOG = logging.getLogger("trainner-gui.queue")


class QueueDispatcher:
    def __init__(self, state: Path, manager):
        self.index = state / "queue/queue.json"
        self.manager = manager
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="trainer-queue", daemon=True)

    def _run(self):
        delay = 3
        previous_error = None
        while not self.stopping.is_set():
            if self.index.is_file():
                try:
                    self.manager().dispatch_once()
                    delay = 3
                    previous_error = None
                except UpdateDeferred:
                    delay = 3
                except Exception as exc:
                    # A corrupt queue or inaccessible volume must stay visible,
                    # while requests for unrelated recipes keep working.
                    signature = (type(exc), str(exc))
                    if signature != previous_error:
                        LOG.exception("Queued training dispatch failed")
                    previous_error = signature
                    delay = min(delay*2, 60)
            self.wake.wait(delay)
            self.wake.clear()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopping.set()
        self.wake.set()
        self.thread.join(timeout=5)

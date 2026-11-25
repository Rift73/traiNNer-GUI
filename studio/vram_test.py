"""Transient GPU measurements for the recipe editor. Nothing enters the run store."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .recipes import safe_name

_PREFIX = "TRAINNER_VRAM_RESULT "
_UPDATES = 20


class VramTest:
    def __init__(self, services):
        self.services = services
        self.lock = threading.Lock()
        self.process = None
        self.reader = None
        self.closed = False
        self.result = {"status": "idle"}

    def status(self):
        with self.lock:
            return self.result.copy()

    def start(self, text, name):
        name = safe_name(name)
        with self.lock:
            if self.closed or self.process is not None:
                raise FileExistsError("A VRAM test is already running or the service is closing.")
            with self.services.runs.execution_slot(exclusive=True) as lease:
                temp = TemporaryDirectory(prefix="trainner-vram-")
                directory = Path(temp.name)
                try:
                    options = directory / "options.yml"
                    options.write_text(text, encoding="utf-8")
                    env = os.environ.copy()
                    for key in ("TRAINNER_STUDIO_ATTEMPT", "TRAINNER_STUDIO_CONTROL", "TRAINNER_STUDIO_PROGRESS",
                                "TRAINNER_NONFINITE_REPORT", "TRAINNER_RECOVERY_RESUME_STATE", "TRAINNER_RECOVERY_MANUAL_SEED"):
                        env.pop(key, None)
                    env.update(TRAINNER_MANAGED_RUN_ROOT=temp.name, TMPDIR=temp.name,
                               TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor"),
                               TRITON_CACHE_DIR=str(directory / "triton"), CUDA_CACHE_PATH=str(directory / "cuda"),
                               TORCH_COMPILE_DEBUG_DIR=str(directory / "compile-debug"), PYTHONDONTWRITEBYTECODE="1")
                    process = subprocess.Popen(
                        [sys.executable, "-B", "-u", "-m", "studio.vram_test", "-opt", str(options), "--no-auto-recover"],
                        cwd=self.services.root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True, pass_fds=(lease,),
                    )
                except BaseException:
                    temp.cleanup()
                    raise
            self.process = process
            self.result = {"id": uuid4().hex, "status": "running", "name": name, "iterations": 0, "target": _UPDATES}
            self.reader = threading.Thread(target=self._collect, args=(process, temp), daemon=True)
            self.reader.start()
            return self.result.copy()

    def _collect(self, process, temp):
        status = "failed"
        tail = deque(maxlen=8)
        try:
            with process.stdout as output:
                for line in output:
                    if not line.startswith(_PREFIX):
                        if line.strip():
                            tail.append(line.strip()[-480:])
                        continue
                    update = json.loads(line[len(_PREFIX):])
                    with self.lock:
                        self.result.update(update)
            code = process.wait()
            with self.lock:
                if self.result["status"] == "cancelling":
                    status = "cancelled"
                elif code == 0 and self.result["iterations"] == _UPDATES and self.result.get("devices"):
                    status = "completed"
                else:
                    status = "failed"
                    self.result.setdefault("error", "\n".join(tail) or f"Trainer exited with code {code}.")
        except (OSError, ValueError) as exc:
            status = "failed"
            self._kill(process)
            process.wait()
            with self.lock:
                self.result["error"] = str(exc)
        finally:
            try:
                temp.cleanup()
            except OSError as exc:
                status = "failed"
                with self.lock:
                    self.result["error"] = f"Temporary test cleanup failed: {exc}"
            finally:
                with self.lock:
                    self.result["status"] = status
                    self.process = None

    @staticmethod
    def _kill(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cancel(self, test_id):
        with self.lock:
            if self.result.get("id") != test_id:
                raise ValueError("This VRAM test is no longer current.")
            if self.process is not None:
                self.result["status"] = "cancelling"
                self._kill(self.process)
            return self.result.copy()

    def close(self):
        with self.lock:
            self.closed = True
            if self.process is not None:
                self.result["status"] = "cancelling"
                self._kill(self.process)
            reader = self.reader
        if reader is not None:
            reader.join()


class VramMeasurement:
    iterations = _UPDATES

    def __init__(self):
        self.devices = []
        self.used = {}

    def configure(self, opt):
        import torch

        if opt.num_gpu == 0 or not torch.cuda.is_available():
            raise ValueError("VRAM test requires a CUDA training recipe and an available GPU.")
        # DataParallel uses all visible devices when the trainer enables it.
        self.devices = list(range(torch.cuda.device_count())) if opt.num_gpu > 1 else [torch.cuda.current_device()]
        opt.high_order_degradations_debug = False
        opt.train.moa_debug = False
        opt.logger.training_sample_steps = []
        self.update(0)

    def stats(self):
        import torch

        rows = []
        for index in self.devices:
            free, total = torch.cuda.mem_get_info(index)
            self.used[index] = max(self.used.get(index, 0), total - free)
            rows.append({"device": index, "name": torch.cuda.get_device_name(index), "total": total,
                         "allocated": torch.cuda.max_memory_allocated(index),
                         "reserved": torch.cuda.max_memory_reserved(index), "sampledUsed": self.used[index]})
        return rows

    def update(self, iterations):
        print(_PREFIX + json.dumps({"iterations": iterations, "devices": self.stats()}), flush=True)


def _worker():
    import torch
    from train import _train_pipeline

    measure = VramMeasurement()
    try:
        _train_pipeline(os.environ["TRAINNER_MANAGED_RUN_ROOT"], probe=measure)
        for index in measure.devices:
            torch.cuda.synchronize(index)
        measure.update(_UPDATES)
    except Exception as exc:
        error = {"error": str(exc) or type(exc).__name__}
        if isinstance(exc, torch.cuda.OutOfMemoryError) or isinstance(exc.__cause__, torch.cuda.OutOfMemoryError):
            error["outOfMemory"] = True
        try:
            error["devices"] = measure.stats()
        except RuntimeError:
            pass  # A failed CUDA context can also reject diagnostic queries.
        print(_PREFIX + json.dumps(error), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_worker())

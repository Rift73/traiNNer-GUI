"""Five-second host GPU readings shared by open run monitors."""
from __future__ import annotations

import csv
import ctypes
import math
import re
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

from .paths import clipboard_paths


_INTERVAL = 5.0
_lock = threading.Lock()
_next = 0.0
_sample = {}
_WSL = Path("/usr/lib/wsl/lib")
_COUNTER = re.compile(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)_phys_(\d+) (\d+)", re.I)


def _read(args):
    try:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.decode("utf-8", errors="replace") if result.returncode == 0 else ""


@lru_cache(maxsize=32)
def _windows_adapter(bus):
    # Match CUDA's PCI device to the Windows counter instance, including its physical node.
    try:
        cuda = ctypes.CDLL(str(_WSL / "libcuda.so.1"))
        device = ctypes.c_int()
        luid = ctypes.create_string_buffer(8)
        mask = ctypes.c_uint()
        if (cuda.cuInit(0) or cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), bus.encode("ascii"))
                or cuda.cuDeviceGetLuid(luid, ctypes.byref(mask), device)):
            return None
    except (OSError, AttributeError):
        return None
    if not mask.value or mask.value & (mask.value - 1):
        return None
    return int.from_bytes(luid.raw, "little"), mask.value.bit_length() - 1


def _dedicated_bytes():
    host = Path("/mnt/c/Windows/System32/cscript.exe")
    script = clipboard_paths(Path(__file__).with_name("gpu_memory.js")).get("windows")
    if not script or not host.is_file():
        return {}
    values = {}
    for line in _read([str(host), "//Nologo", script]).splitlines():
        match = _COUNTER.fullmatch(line.strip())
        if match:
            high, low, node, used = match.groups()
            values[(int(high, 16) << 32 | int(low, 16), int(node))] = int(used)
    return values


def _number(text):
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _collect():
    wsl = Path("/dev/dxg").exists() and _WSL.is_dir()
    smi = str(_WSL / "nvidia-smi") if wsl else shutil.which("nvidia-smi")
    adapters = []
    if smi:
        output = _read([smi, "--query-gpu=index,name,pci.bus_id,power.draw,temperature.gpu,memory.used",
                        "--format=csv,noheader,nounits"])
        dedicated = _dedicated_bytes() if output and wsl else {}
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(row) != 6:
                continue
            index, name, bus, power, temperature, memory = row
            mib = _number(memory)
            used = dedicated.get(_windows_adapter(bus)) if wsl else (int(mib * 2**20) if mib is not None else None)
            adapters.append({"index": index, "name": name, "watts": _number(power),
                             "celsius": _number(temperature), "usedBytes": used})
    return {"adapters": adapters, "memorySource": "windowsDedicated" if wsl else "nvidia",
            "sampledAt": time.time()}


def readings():
    global _next, _sample
    with _lock:
        now = time.monotonic()
        if now >= _next:
            _sample = _collect()
            _next = now + _INTERVAL
        return {**_sample, "nextPollMs": max(1, math.ceil((_next - time.monotonic()) * 1000))}

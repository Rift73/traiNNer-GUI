"""CUDA kernels shipped as prebuilt PTX in traiNNer/csrc, launched on Torch's stream.

The PTX is loaded through the CUDA driver API (ctypes, no build step) into the
CUDA context that is current on the calling thread, which is Torch's primary
context of the tensor's device, once per context; kernels run on Torch's
current stream of that device. traiNNer/csrc/README.md has the build command.

Usage:
    from traiNNer.data.ptx_cuda import PtxModule
    module = PtxModule("effect_cuda.ptx")
    module.launch("effect_shift_yuv_planes", grid, block, image, ctypes.c_int(w), ...)
"""

from __future__ import annotations

import ctypes
import os
import threading
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

_csrc_dir = Path(__file__).resolve().parent.parent / "csrc"

_POINTER = ctypes.c_void_p
_UINT = ctypes.c_uint
# Kernel arguments other than tensors.
KernelScalar = ctypes.c_int | ctypes.c_uint | ctypes.c_int64 | ctypes.c_float
_SIGNATURES = {
    "cuCtxGetCurrent": [ctypes.POINTER(_POINTER)],
    "cuModuleLoadData": [ctypes.POINTER(_POINTER), _POINTER],
    "cuModuleGetFunction": [ctypes.POINTER(_POINTER), _POINTER, ctypes.c_char_p],
    "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuLaunchKernel": [
        _POINTER,
        _UINT,
        _UINT,
        _UINT,
        _UINT,
        _UINT,
        _UINT,
        _UINT,
        _POINTER,
        ctypes.POINTER(_POINTER),
        ctypes.POINTER(_POINTER),
    ],
}


@cache
def _driver() -> ctypes.CDLL:
    driver = (
        ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    )
    for name, argtypes in _SIGNATURES.items():
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    return driver


def _check(status: int) -> None:
    if status:
        message = ctypes.c_char_p()
        _driver().cuGetErrorString(status, ctypes.byref(message))
        text = message.value.decode() if message.value else "unknown error"
        raise RuntimeError(f"CUDA driver error {status}: {text}")


class PtxModule:
    """One PTX file of traiNNer/csrc, loaded on first use in each CUDA context.

    A loaded module stays loaded for the lifetime of its context.
    """

    def __init__(self, name: str) -> None:
        self.path = _csrc_dir / name
        self._lock = threading.Lock()
        # Per context: the loaded module and the kernels looked up in it.
        self._loaded: dict[int, tuple[_POINTER, dict[str, _POINTER]]] = {}

    def _function(self, symbol: str) -> _POINTER:
        driver = _driver()
        context = _POINTER()
        _check(driver.cuCtxGetCurrent(ctypes.byref(context)))
        if not context.value:
            raise RuntimeError("No CUDA context is current on this thread")
        with self._lock:
            entry = self._loaded.get(context.value)
            if entry is None:
                module = _POINTER()
                ptx = ctypes.create_string_buffer(self.path.read_bytes())
                _check(
                    driver.cuModuleLoadData(
                        ctypes.byref(module), ctypes.cast(ptx, _POINTER)
                    )
                )
                entry = self._loaded[context.value] = (module, {})
            module, functions = entry
            function = functions.get(symbol)
            if function is None:
                function = _POINTER()
                _check(
                    driver.cuModuleGetFunction(
                        ctypes.byref(function), module, symbol.encode()
                    )
                )
                functions[symbol] = function
            return function

    def launch(
        self,
        symbol: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        *args: Tensor | KernelScalar,
    ) -> None:
        """Launch kernel symbol on the current stream of the first tensor's device.

        A tensor argument passes its data pointer and is recorded on that
        stream; every other argument is passed as the ctypes value it is.
        """
        device = next(arg.device for arg in args if isinstance(arg, Tensor))
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            values: list[KernelScalar | _POINTER] = []
            for arg in args:
                if isinstance(arg, Tensor):
                    arg.record_stream(stream)
                    values.append(_POINTER(arg.data_ptr()))
                else:
                    values.append(arg)
            pointers = (_POINTER * len(values))(
                *(ctypes.addressof(value) for value in values)
            )
            _check(
                _driver().cuLaunchKernel(
                    self._function(symbol),
                    *grid,
                    *block,
                    0,
                    _POINTER(stream.cuda_stream),
                    pointers,
                    None,
                )
            )

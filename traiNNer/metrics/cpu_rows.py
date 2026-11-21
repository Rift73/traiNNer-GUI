"""CPU holdout metric implementation for the trainer's prepared image tensors."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import cv2
import torch
from torch import Tensor

_LIBRARY = Path(__file__).resolve().parents[2] / "native/studio/metrics/build/clang/metrics.so"


@lru_cache(maxsize=2)
def _operator(on_grid: bool):
    if not _LIBRARY.is_file():
        return None
    lib = ctypes.CDLL(str(_LIBRARY))
    fn = getattr(lib, "trainner_ssim_grid_rows" if on_grid else "trainner_ssim_rows", None)
    if fn is None:
        return None
    ptr = ctypes.c_void_p
    fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ptr, ptr, ptr]
    fn.restype = ctypes.c_int
    # Use the same coefficients as the Torch convolution.
    kernel = cv2.getGaussianKernel(11, 1.5)
    return fn, kernel


def _row_inputs(left: Tensor, right: Tensor, window: int) -> bool:
    return (left.ndim == 4 and right.shape == left.shape and min(left.shape[:2]) > 0
            and min(left.shape[-2:]) >= window and left.device.type == "cpu" and right.device.type == "cpu"
            and right.dtype == left.dtype and left.dtype in (torch.float32, torch.float64)
            and left.layout == torch.strided and right.layout == torch.strided
            and not (left.is_neg() or right.is_neg() or left.requires_grad or right.requires_grad))


@lru_cache(maxsize=1)
def _mse_operator():
    if not _LIBRARY.is_file():
        return None
    fn = getattr(ctypes.CDLL(str(_LIBRARY)), "trainner_mse_rows", None)
    if fn:
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ptr, ptr]
        fn.restype = ctypes.c_int
    return fn


def streamed_mse(left: Tensor, right: Tensor) -> Tensor | None:
    if not _row_inputs(left, right, 1):
        return None
    result = torch.zeros(left.shape[0], dtype=torch.float64, device="cpu")
    fn = _mse_operator()
    if fn is None:
        for row in range(0, left.shape[-2], 64):
            region = (..., slice(row, row + 64), slice(None))
            difference = left[region].to(dtype=torch.float64, copy=True).sub_(right[region])
            result.add_(difference.square_().sum((1, 2, 3)))
        return result / left[0].numel()
    shape = ctypes.c_int64 * 4
    if fn(left.data_ptr(), shape(*left.stride()), right.data_ptr(), shape(*right.stride()),
          left.dtype == torch.float64, shape(*left.shape), result.data_ptr()):
        raise ValueError("Invalid input geometry for the CPU MSE evaluator.")
    return result


def streamed_ssim(left: Tensor, right: Tensor, *, on_grid: bool = False) -> Tensor | None:
    if not _row_inputs(left, right, 11):
        return None
    backend = _operator(on_grid)
    if backend is None:
        # An unprepared checkout can still score a full-resolution image. Each
        # tile includes the 10-pixel input halo for its valid 11x11 windows.
        from .psnr_ssim import _ssim_pth

        rows, cols = left.shape[-2] - 10, left.shape[-1] - 10
        total = torch.zeros(left.shape[0], dtype=torch.float64, device="cpu")
        for y in range(0, rows, 128):
            height = min(rows - y, 128)
            for x in range(0, cols, 192):
                width = min(cols - x, 192)
                region = (..., slice(y, y + height + 10), slice(x, x + width + 10))
                source, target = left[region].double(), right[region].double()
                if not on_grid:
                    source, target = source * 255, target * 255
                score = _ssim_pth(source, target)
                total.add_(score, alpha=height * width)
        return total / (rows * cols)
    fn, kernel = backend
    shape = ctypes.c_int64 * 4
    result = torch.empty(left.shape[0], dtype=torch.float64, device="cpu")
    status = fn(left.data_ptr(), shape(*left.stride()), right.data_ptr(), shape(*right.stride()),
                left.dtype == torch.float64, shape(*left.shape), kernel.ctypes.data, result.data_ptr())
    if status == 2:
        raise MemoryError("Could not allocate the CPU SSIM row buffers.")
    if status:
        raise ValueError("Invalid input geometry for the CPU SSIM evaluator.")
    return result

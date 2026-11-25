"""SSIMULACRA2 validation with the libjxl recursive Gaussian (BSD-3-Clause)."""

import numpy as np
import torch

from traiNNer.losses._ssimulacra2 import Ssimulacra2, _xyb


def _gaussian_coefficients() -> tuple[tuple[float, ...], ...]:
    """Four-lane recurrence coefficients for libjxl's sigma-1.5 Gaussian."""
    f = np.arange(1, 6, 2) * np.pi / 10
    p = np.array([1, -1, 1]) / np.tan(0.5 * f)
    r = p * p / np.sin(f) * np.array([1, -1, 1])
    rho = np.exp(-1.125 * f * f) * 0.2
    z = np.cross(p, r)
    z /= z[2]
    beta = np.linalg.solve(np.stack((p, r, z)), np.array([1.0, 22.75, z @ rho]))
    d = -2 * np.cos(f)
    n = -beta * np.cos(f * 6.0)
    return tuple(
        tuple(map(float, x))
        for x in zip(
            n,
            -d,
            d * d - 1,
            -(d**3) + 2 * d,
            d**4 - 3 * d * d + 1,
            d,
            -d * d + 1,
            d**3 - 2 * d,
            -d * n,
            (d * d - 1) * n,
            (-(d**3) + 2 * d) * n,
            strict=True,
        )
    )


GAUSSIAN_COEFFICIENTS = _gaussian_coefficients()


def _scan_cpu(image: np.ndarray, axis: int, horizontal: bool) -> np.ndarray:
    # Adjacent lines (and their moments) are independent. Preserve the FP32
    # recurrence along each line including its decaying rounding error.
    src = np.moveaxis(image, axis, 0)
    length = src.shape[0]
    padded = np.pad(src, [(10, 10)] + [(0, 0)] * (src.ndim - 1))
    inputs = padded[: length + 4] + padded[10 : length + 14]
    previous = np.zeros((3, *src.shape[1:]), dtype=np.float32)
    older = np.zeros_like(previous)
    output = np.empty_like(src)
    coeff = np.asarray(GAUSSIAN_COEFFICIENTS, dtype=np.float32).T.reshape(
        11, 3, *([1] * (src.ndim - 1))
    )

    def fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (a.astype(np.float64) * b + c).astype(np.float32)

    for j, v in enumerate(inputs):
        if horizontal:
            current = fma(coeff[1], previous, coeff[0] * v - older)
        else:
            current = fma(coeff[0], v, fma(coeff[1], previous, -older))
        if j >= 4:
            output[j - 4] = current[0] + (current[1] + current[2])
        older, previous = previous, current
    return np.moveaxis(output, 0, axis)


class CpuSsimulacra2(Ssimulacra2):
    def features(
        self, reference: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        r, d = _xyb(reference), _xyb(prediction)
        moments = torch.cat((r, d, r * r, d * d, r * d)).numpy()
        blurred = _scan_cpu(_scan_cpu(moments, -1, True), -2, False)
        m, n, rr, dd, rd = torch.from_numpy(blurred).chunk(5)
        denominator = (rr - m * m) + (dd - n * n) + 0.0009
        s = (
            1 - (1 - (m - n).square()) * (2 * (rd - m * n) + 0.0009) / denominator
        ).relu()
        edge = ((d - n).abs() - (r - m).abs()) / (1 + (r - m).abs())
        maps = torch.stack((s, edge.relu(), (-edge).relu()), dim=2)
        fourth = maps.square().square().mean((-1, -2)).sqrt().sqrt()
        return torch.stack((maps.mean((-1, -2)), fourth), dim=2)

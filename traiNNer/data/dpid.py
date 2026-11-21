"""CPU half-size Rapid, Detail-Preserving Image Downscaling (lambda=1).

Adapted from mergian/dpid, matab/dpid.m, commit
dba82881a1797a12df123a7de4079e54da24bcc1:
https://github.com/mergian/dpid
Copyright (c) 2016, Nicolas Weber, Sandra C. Amend / GCC / TU-Darmstadt.
All rights reserved. BSD-3-Clause; see LICENSE/LICENSE-DPID.

Like the MATLAB reference, retain floating-point area averages and guidance;
unlike its file output, do not quantize to uint8. The upstream CUDA example
quantizes intermediate guidance and is therefore not bitwise equivalent.
"""

from collections.abc import Iterator

import cv2
import numpy as np


def _area_axis(length: int, output: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact fractional pixel coverage, including odd and one-pixel dimensions."""
    # Work in integer coordinates scaled by output to avoid rounding the last
    # footprint past the source boundary. Half-size reduction needs <=4 taps.
    starts = np.arange(output, dtype=np.int64) * length
    ends = starts + length
    taps = (length + output - 1) // output + 1
    indices = starts[:, None] // output + np.arange(taps)
    coverage = np.minimum((indices + 1) * output, ends[:, None]) - np.maximum(
        indices * output, starts[:, None]
    )
    coverage = np.maximum(coverage, 0).astype(np.float32) / output
    return indices.clip(0, length - 1), coverage


def _footprints(
    image: np.ndarray, height: int, width: int
) -> Iterator[tuple[np.ndarray, np.ndarray | float]]:
    if image.shape[:2] == (2 * height, 2 * width):
        # Common path: four strided views, no per-pixel Python loops, index
        # gathers, or full-resolution weight/patch arrays.
        for y in range(2):
            for x in range(2):
                yield image[y::2, x::2], 1.0
        return

    yi, yw = _area_axis(image.shape[0], height)
    xi, xw = _area_axis(image.shape[1], width)
    for y in range(yi.shape[1]):
        if not np.any(yw[:, y]):
            continue
        for x in range(xi.shape[1]):
            if np.any(xw[:, x]):
                yield (
                    image[yi[:, y, None], xi[None, :, x]],
                    yw[:, y, None] * xw[None, :, x],
                )


def dpid_half(image: np.ndarray) -> np.ndarray:
    """Reduce normalized float32 HWC RGB/grayscale to floor(H/2), floor(W/2).

    The reference forms an area-averaged image, smooths it with a border-normalized
    3x3 [1,2,1] outer-product kernel, then weights each footprint's source colors by
    their Euclidean distance from that guide. The common distance normalization
    factor cancels in the weighted mean at lambda=1 and is omitted here.
    """
    if (
        image.ndim != 3
        or image.shape[2] not in (1, 3)
        or image.dtype != np.float32
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("DPID expects a nonempty float32 HWC RGB or grayscale image")

    height, width = max(1, image.shape[0] // 2), max(1, image.shape[1] // 2)
    shape = (height, width, image.shape[2])
    average = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    kernel = np.array([1, 2, 1], dtype=np.float32)
    guide = cv2.sepFilter2D(
        average, -1, kernel, kernel, borderType=cv2.BORDER_CONSTANT
    ).reshape(shape)
    # Discard out-of-bounds neighbors and renormalize, as in the reference.
    # Subtract twice for a singleton axis: only the center weight (2) remains.
    ynorm = np.full(height, 4, dtype=np.float32)
    xnorm = np.full(width, 4, dtype=np.float32)
    ynorm[0] -= 1
    ynorm[-1] -= 1
    xnorm[0] -= 1
    xnorm[-1] -= 1
    guide /= (ynorm[:, None] * xnorm[None, :])[:, :, None]

    result = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for pixels, coverage in _footprints(image, height, width):
        difference = pixels - guide
        np.square(difference, out=difference)
        weight = difference.sum(axis=2)
        np.sqrt(weight, out=weight)
        weight *= coverage
        result += pixels * weight[:, :, None]
        weight_sum += weight

    # Constant footprints can have zero distance. Use the guidance color,
    # rather than introducing epsilon bias or NaNs in flat image regions.
    return np.divide(
        result,
        weight_sum[:, :, None],
        out=guide,
        where=weight_sum[:, :, None] > 0,
    )

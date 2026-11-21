"""CPU descale: undo a known upscale by least squares, one axis at a time.

Follows vapoursynth-descale (Release 12, MIT), as chaiNNer's Descale node runs
it: the same tap positions, kernels, borders and normalization, the normal
equations A'A x = A'b of each axis solved with A'A factored in double and
applied in float32, the horizontal pass first, and an axis whose size is
unchanged skipped. The kernels take the node's default parameters (Bicubic
b=0, c=0.5; Lanczos 3 taps) and Blur 1, with no source shift or window.

Linear light uses chaiNNer's transfer curves on |value| with the value's sign,
so negative ringing and values above 1 survive a round trip.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse import coo_matrix, csr_matrix

_SUPPORT = {
    "bilinear": 1,
    "bicubic": 2,
    "lanczos": 3,
    "spline16": 2,
    "spline36": 3,
    "spline64": 4,
    "point": 0,
}
_BORDERS = ("mirror", "zero", "extend")
TRANSFERS = ("gamma24", "srgb", "gamma22", "bt709")


def _spline(
    d: np.ndarray, pieces: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """Piecewise cubic in d - k on [k, k + 1), zero past the last piece."""
    out = np.zeros_like(d)
    for k, (c0, c1, c2, c3) in enumerate(pieces):
        t = d - k
        out = np.where((d >= k) & (d < k + 1), c0 + c1 * t + c2 * t**2 + c3 * t**3, out)
    return out


def _weight(kernel: str, d: np.ndarray) -> np.ndarray:
    """The kernel's weight at absolute distance d (descale's calculate_weight)."""
    if kernel == "bilinear":
        return np.maximum(1.0 - d, 0.0)
    if kernel == "bicubic":
        b, c = 0.0, 0.5
        near = (
            (12 - 9 * b - 6 * c) * d**3 + (-18 + 12 * b + 6 * c) * d**2 + (6 - 2 * b)
        ) / 6.0
        far = (
            (-b - 6 * c) * d**3
            + (6 * b + 30 * c) * d**2
            + (-12 * b - 48 * c) * d
            + (8 * b + 24 * c)
        ) / 6.0
        return np.where(d < 1, near, np.where(d < 2, far, 0.0))
    if kernel == "lanczos":
        # np.sinc(0) is 1 and sin(pi x) / (pi x) elsewhere, as descale's sinc.
        return np.where(d < 3, np.sinc(d) * np.sinc(d / 3), 0.0)
    if kernel == "spline16":
        return _spline(d, [(1.0, -1 / 5, -9 / 5, 1.0), (0.0, -7 / 15, 4 / 5, -1 / 3)])
    if kernel == "spline36":
        return _spline(
            d,
            [
                (1.0, -3 / 209, -453 / 209, 13 / 11),
                (0.0, -156 / 209, 270 / 209, -6 / 11),
                (0.0, 26 / 209, -45 / 209, 1 / 11),
            ],
        )
    if kernel == "spline64":
        return _spline(
            d,
            [
                (1.0, -3 / 2911, -6387 / 2911, 49 / 41),
                (0.0, -2328 / 2911, 4032 / 2911, -24 / 41),
                (0.0, 582 / 2911, -1008 / 2911, 6 / 41),
                (0.0, -97 / 2911, 168 / 2911, -1 / 41),
            ],
        )
    return np.ones_like(d)


def upscale_matrix(native: int, size: int, kernel: str, border: str) -> csr_matrix:
    """The size x native float64 matrix of the upscale that descale inverts."""
    if kernel not in _SUPPORT:
        raise ValueError(
            f"Unsupported descale kernel {kernel!r}; expected one of {list(_SUPPORT)}"
        )
    if border not in _BORDERS:
        raise ValueError(
            f"Unsupported descale border {border!r}; expected one of {list(_BORDERS)}"
        )
    support = _SUPPORT[kernel]
    taps = 2 * support if support > 0 else 1
    pos = (np.arange(size) + 0.5) * native / size
    start = pos - taps / 2.0
    # zimg's round_halfup keeps round(x - 1) == round(x) - 1 on the pixel grid.
    start = np.where(
        start < 0, np.floor(start + 0.5), np.floor(start + 0.49999999999999994)
    )
    xpos = start[:, None] + 0.5 + np.arange(taps)
    weights = _weight(kernel, np.abs(xpos - pos[:, None]))
    total = weights.sum(axis=1, keepdims=True)
    weights = weights / np.where(total == 0, np.finfo(np.float64).eps, total)
    pixel = np.floor(xpos)
    rows = np.broadcast_to(np.arange(size)[:, None], xpos.shape)
    if border == "mirror":
        # Half-sample symmetric, repeated past one reflection.
        fold = np.mod(pixel, 2 * native)
        pixel = np.where(fold < native, fold, 2 * native - 1 - fold)
    elif border == "extend":
        pixel = np.clip(pixel, 0, native - 1)
    else:
        keep = (xpos >= 0) & (xpos <= native)
        rows, pixel, weights = rows[keep], pixel[keep], weights[keep]
    matrix = coo_matrix(
        (weights.ravel(), (rows.ravel(), pixel.astype(np.int64).ravel())),
        shape=(size, native),
    )
    return matrix.tocsr()


@lru_cache(maxsize=16)
def _system(
    native: int, size: int, kernel: str, border: str
) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    """The float32 upscale matrix, its transpose and the banded Cholesky factor
    of its normal matrix."""
    matrix = upscale_matrix(native, size, kernel, border)
    if np.any(abs(matrix).max(axis=0).toarray().ravel() < 1e-12):
        raise ValueError(
            f"Descale {size} -> {native} with the {kernel} kernel leaves target "
            "pixels with no weight"
        )
    normal = (matrix.T @ matrix).todia()
    upper = int(normal.offsets.max())
    banded = np.zeros((upper + 1, native))
    for offset in range(upper + 1):
        banded[upper - offset, offset:] = normal.diagonal(offset)
    # descale adds DBL_EPSILON to each pivot of its LDL' factorization.
    banded[upper] += np.finfo(np.float64).eps
    factor = cholesky_banded(banded, lower=False).astype(np.float32)
    single = matrix.astype(np.float32)
    return single, single.T.tocsr(), factor


@lru_cache(maxsize=16)
def descale_matrix(native: int, size: int, kernel: str, border: str) -> np.ndarray:
    """The native x size float64 matrix of the descale along one axis: the
    least-squares inverse (A'A + eps I)^-1 A' of the upscale A, with the
    DBL_EPSILON that the solve adds to the diagonal; the identity when the size
    is unchanged, since descale then skips the axis. Read-only, as it is cached.
    """
    if native == size:
        out = np.eye(size)
    else:
        matrix = upscale_matrix(native, size, kernel, border)
        normal = (matrix.T @ matrix).toarray()
        normal[np.diag_indices_from(normal)] += np.finfo(np.float64).eps
        out = np.linalg.solve(normal, matrix.T.toarray())
    out.flags.writeable = False
    return out


def _resize_axis(
    image: np.ndarray, size: int, axis: int, kernel: str, border: str, solve: bool
) -> np.ndarray:
    length = image.shape[axis]
    if length == size:
        return image
    moved = np.moveaxis(image, axis, 0)
    rest = moved.shape[1:]
    flat = np.ascontiguousarray(moved).reshape(length, -1)
    if solve:
        _, transposed, factor = _system(size, length, kernel, border)
        out = cho_solve_banded(
            (factor, False), transposed @ flat, overwrite_b=True, check_finite=False
        )
    else:
        out = _system(length, size, kernel, border)[0] @ flat
    return np.moveaxis(out.reshape(size, *rest), 0, axis)


def _check_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.dtype != np.float32:
        raise ValueError(
            f"Expected an H,W,C float32 image, got {image.dtype} {image.shape}"
        )


def linearize(image: np.ndarray, transfer: str) -> np.ndarray:
    """Encoded values to linear light with the named transfer curve."""
    a = np.abs(image)
    if transfer == "gamma24":
        v = a**2.4
    elif transfer == "srgb":
        v = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    elif transfer == "gamma22":
        v = a**2.2
    elif transfer == "bt709":
        v = np.where(a < 0.081, a / 4.5, ((a + 0.099) / 1.099) ** (1 / 0.45))
    else:
        raise ValueError(f"Unknown transfer {transfer!r}; expected one of {TRANSFERS}")
    return np.copysign(v, image).astype(np.float32)


def encode_light(image: np.ndarray, transfer: str) -> np.ndarray:
    """Linear light back to encoded values with the named transfer curve."""
    a = np.abs(image)
    if transfer == "gamma24":
        v = a ** (1 / 2.4)
    elif transfer == "srgb":
        v = np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)
    elif transfer == "gamma22":
        v = a ** (1 / 2.2)
    elif transfer == "bt709":
        v = np.where(a < 0.018, 4.5 * a, 1.099 * a**0.45 - 0.099)
    else:
        raise ValueError(f"Unknown transfer {transfer!r}; expected one of {TRANSFERS}")
    return np.copysign(v, image).astype(np.float32)


def descale(
    image: np.ndarray,
    width: int,
    height: int,
    kernel: str = "bilinear",
    *,
    border: str = "mirror",
    linear_light: str | None = None,
) -> np.ndarray:
    """The height,width,C image whose upscale to the image's size with this
    kernel and border is closest to the image in least squares.

    Channels are independent planes, so the frames of a clip can be stacked
    along C and solved together. linear_light names a transfer curve (chaiNNer's
    Descale default is "gamma24"); the solve then runs on the linear copy and
    the result is encoded back. None solves on the stored values.
    """
    _check_image(image)
    h, w = image.shape[:2]
    if not (1 <= width <= w and 1 <= height <= h):
        raise ValueError(
            f"Descale needs a target no larger than the image ({w}x{h}), "
            f"got {width}x{height}"
        )
    source = image if linear_light is None else linearize(image, linear_light)
    out = _resize_axis(source, width, 1, kernel, border, solve=True)
    out = np.ascontiguousarray(_resize_axis(out, height, 0, kernel, border, solve=True))
    return out if linear_light is None else encode_light(out, linear_light)


def upscale(
    image: np.ndarray,
    width: int,
    height: int,
    kernel: str = "bilinear",
    *,
    border: str = "mirror",
    linear_light: str | None = None,
) -> np.ndarray:
    """The upscale that descale inverts, with the same conventions."""
    _check_image(image)
    h, w = image.shape[:2]
    if not (width >= w and height >= h):
        raise ValueError(
            f"Upscale needs a target no smaller than the image ({w}x{h}), "
            f"got {width}x{height}"
        )
    source = image if linear_light is None else linearize(image, linear_light)
    out = _resize_axis(source, width, 1, kernel, border, solve=False)
    out = np.ascontiguousarray(
        _resize_axis(out, height, 0, kernel, border, solve=False)
    )
    return out if linear_light is None else encode_light(out, linear_light)

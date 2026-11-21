"""Triton kernel of still_source.device_canvas, in its own module so that
still_source imports without Triton."""

import triton
import triton.language as tl
from torch import Tensor

_BLOCK = 256


@triton.jit
def _canvas_kernel(
    source: tl.tensor,
    flags: tl.tensor,
    taps: tl.tensor,
    canvas: tl.tensor,
    source_batch_stride: int,
    source_channel_stride: int,
    source_row_stride: int,
    source_column_stride: int,
    canvas_batch_stride: int,
    canvas_channel_stride: int,
    canvas_row_stride: int,
    canvas_column_stride: int,
    size: tl.constexpr,
    canvas_size: tl.constexpr,
    block: tl.constexpr,
) -> None:
    flat = tl.program_id(0) * block + tl.arange(0, block)
    b = tl.program_id(1)
    c = tl.program_id(2)
    inside = flat < canvas_size * canvas_size
    row = flat // canvas_size
    column = flat % canvas_size
    # Position in the oriented halved image; the padding mirrors it with the
    # edge pixel repeated, as libvips' mirror extend does.
    x = column - (canvas_size - size) // 2
    y = row - (canvas_size - size) // 2
    x = tl.where(x < 0, -x - 1, tl.where(x >= size, 2 * size - x - 1, x))
    y = tl.where(y < 0, -y - 1, tl.where(y >= size, 2 * size - y - 1, y))
    # Undo the clockwise rotation, then the flips, to reach the halved source.
    orientation = tl.load(flags + b)
    rotated = (orientation & 4) != 0
    x, y = tl.where(rotated, y, x), tl.where(rotated, size - 1 - x, y)
    x = tl.where((orientation & 1) != 0, size - 1 - x, x)
    y = tl.where((orientation & 2) != 0, size - 1 - y, y)
    plane = source + b * source_batch_stride + c * source_channel_stride
    value = tl.zeros((block,), tl.float32)
    for dy in tl.static_range(4):
        source_y = tl.minimum(tl.maximum(2 * y - 1 + dy, 0), 2 * size - 1)
        line = tl.zeros((block,), tl.float32)
        for dx in tl.static_range(4):
            source_x = tl.minimum(tl.maximum(2 * x - 1 + dx, 0), 2 * size - 1)
            pixel = tl.load(
                plane + source_y * source_row_stride + source_x * source_column_stride,
                inside,
                0,
            ).to(tl.float32) * (1.0 / 255.0)
            line += pixel * tl.load(taps + x * 4 + dx, inside, 0.0)
        value += line * tl.load(taps + y * 4 + dy, inside, 0.0)
    value = tl.minimum(tl.maximum(value, 0.0), 1.0)
    tl.store(
        canvas
        + b * canvas_batch_stride
        + c * canvas_channel_stride
        + row * canvas_row_stride
        + column * canvas_column_stride,
        value,
        inside,
    )


def fill_canvas(source: Tensor, flags: Tensor, taps: Tensor, canvas: Tensor) -> None:
    """Write the canvases of uint8 (b, 3, 2 * size, 2 * size) sources into the
    float32 (b, 3, size + 32, size + 32) `canvas`, on the current stream."""
    batch, channels, canvas_size, _ = canvas.shape
    size = source.shape[-1] // 2
    # Fused multiply-adds would round differently from the CPU path's sums.
    _canvas_kernel[(triton.cdiv(canvas_size * canvas_size, _BLOCK), batch, channels)](
        source,
        flags,
        taps,
        canvas,
        *source.stride(),
        *canvas.stride(),
        size=size,
        canvas_size=canvas_size,
        block=_BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )

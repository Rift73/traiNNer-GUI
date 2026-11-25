"""Pack clamped DUAL neighborhoods for inference with the original SDPA math."""


# Triton infers device argument types; uppercase kernel parameters identify buffers and dimensions.
# ruff: noqa: ANN001, ANN202, N803

import math
from functools import lru_cache

import torch
import torch.nn.functional as F  # noqa: N812
import triton
import triton.language as tl


@triton.jit
def _axis_origins(
    tile,
    query_width,
    key_height,
    key_width,
    kh,
    kw,
    uh,
    uw,
    TILE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    tiles_wide = tl.cdiv(query_width, TILE)
    qy = tile // tiles_wide * TILE
    qx = tile % tiles_wide * TILE
    ky = tl.minimum(tl.maximum(qy // STRIDE - kh // 2, 0), key_height - uh)
    kx = tl.minimum(tl.maximum(qx // STRIDE - kw // 2, 0), key_width - uw)
    return qy, qx, ky, kx


@triton.jit
def _pack_tokens(
    Source,
    Value,
    Factors,
    Packed,
    PackedValue,
    height,
    width,
    key_height,
    key_width,
    kh,
    kw,
    uh,
    uw,
    first,
    count,
    heads,
    s_batch,
    s_head,
    s_token,
    s_channel,
    v_batch,
    v_head,
    v_token,
    v_channel,
    f_head,
    f_token,
    TILE: tl.constexpr,
    STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    QUERY: tl.constexpr,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
):
    group = tl.program_id(1)
    batch = group // (count * heads)
    head = group % heads
    tile = first + group // heads % count
    qy, qx, ky, kx = _axis_origins(
        tile, width, key_height, key_width, kh, kw, uh, uw, TILE, STRIDE
    )
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, LANES)
    if QUERY:
        tokens = TILE * TILE
        y = tl.minimum(qy + slot // TILE, height - 1)
        x = tl.minimum(qx + slot % TILE, width - 1)
        index = y * width + x
        local = (y - ky) * uw + x - kx
    else:
        tokens = uh * uw
        index = (ky + slot // uw) * key_width + kx + slot % uw
        local = slot
    valid = slot < tokens
    address = (
        batch.to(tl.int64) * s_batch
        + head * s_head
        + index[:, None].to(tl.int64) * s_token
    )
    z = tl.load(
        Source + address + lane[None, :] * s_channel,
        valid[:, None] & (lane[None, :] < DIM),
        0,
    )
    if RANK:
        positional = tl.load(
            Factors + head * f_head + local[:, None] * f_token + lane[None, :] - DIM,
            valid[:, None] & (lane[None, :] >= DIM) & (lane[None, :] < DIM + RANK),
            0,
        )
        z = tl.where(lane[None, :] < DIM, z, positional)
    dst = (group.to(tl.int64) * tokens + slot[:, None]) * (DIM + RANK) + lane[None, :]
    write = valid[:, None] & (lane[None, :] < DIM + RANK)
    tl.store(Packed + dst, z, write)
    if not QUERY:
        address = (
            batch.to(tl.int64) * v_batch
            + head * v_head
            + index[:, None].to(tl.int64) * v_token
        )
        v = tl.load(
            Value + address + lane[None, :] * v_channel,
            valid[:, None] & (lane[None, :] < DIM),
            0,
        )
        tl.store(PackedValue + dst, v, write)


@triton.jit
def _pack_mask(
    Bias,
    Out,
    height,
    width,
    key_height,
    key_width,
    kh,
    kw,
    uh,
    uw,
    first,
    pitch,
    heads,
    TILE: tl.constexpr,
    STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    RPB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(2)
    row = tl.program_id(1)
    tile = first + group // heads
    qy, qx, ky, kx = _axis_origins(
        tile, width, key_height, key_width, kh, kw, uh, uw, TILE, STRIDE
    )
    y = tl.minimum(qy + row // TILE, height - 1)
    x = tl.minimum(qx + row % TILE, width - 1)
    start_y = tl.minimum(tl.maximum(y // STRIDE - kh // 2, 0), key_height - kh)
    start_x = tl.minimum(tl.maximum(x // STRIDE - kw // 2, 0), key_width - kw)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    donor_y = ky + col // uw
    donor_x = kx + col % uw
    valid = col < uh * uw
    inside = (
        (donor_y >= start_y)
        & (donor_y < start_y + kh)
        & (donor_x >= start_x)
        & (donor_x < start_x + kw)
    )
    if RPB:
        index = (y - donor_y + WINDOW - 1) * (2 * WINDOW - 1) + x - donor_x + WINDOW - 1
        value = tl.load(Bias + index * heads + group % heads, valid & inside, 0)
    else:
        value = 0.0
    dst = (group.to(tl.int64) * TILE * TILE + row) * pitch + col
    tl.store(Out + dst, tl.where(inside, value, -float("inf")), valid)


@lru_cache(maxsize=256)
def _union(
    query_length: int, key_length: int, window: int, tile: int, stride: int
) -> int:
    size = min(key_length, window)

    def start(pos):
        return min(
            max(min(pos, query_length - 1) // stride - size // 2, 0), key_length - size
        )

    return size + max(
        start(origin + tile - 1) - start(origin)
        for origin in range(0, query_length, tile)
    )


@lru_cache(maxsize=1)
def _sm120_attention():
    try:
        from triton.experimental import gluon
        from triton.experimental.gluon import language
    except ImportError:
        return None
    if not hasattr(gluon, "constexpr_function") or not hasattr(language, "inline_asm"):
        return None
    from .dual_validation_mma import attend

    return attend


def attend(
    query,
    key,
    value,
    shape,
    key_shape,
    window,
    scale,
    *,
    fourier=None,
    relative_bias=None,
) -> torch.Tensor:
    """All positional terms and the SDPA scale are evaluated exactly as in the eager model."""
    batch, heads, _, dim = query.shape
    height, width = shape
    key_height, key_width = key_shape
    kh, kw = min(window, key_height), min(window, key_width)
    if fourier is not None:
        from .dual_arch import _inference_na_coordinates

        tile, stride = 16, 1
        uh, uw = min(height, kh + 15), min(width, kw + 15)
        coords = _inference_na_coordinates(uh, uw, query.device)
        qfactor, kfactor = fourier.factors(coords, query.dtype)
        rank = qfactor.shape[-1]
    else:
        tile, stride = 8, (1 if relative_bias is not None else 2)
        uh = _union(height, key_height, window, tile, stride)
        uw = _union(width, key_width, window, tile, stride)
        qfactor = kfactor = None
        rank = 0
    if (
        query.dtype == torch.bfloat16
        and dim == 32
        and rank in (0, 16)
        and (fourier is None or rank == 16)
        and torch.cuda.get_device_capability(query.device) == (12, 0)
        and max(query.numel(), key.numel(), value.numel()) < 2**31
    ):
        fused = _sm120_attention()
        if fused is not None:
            return fused(
                query,
                key,
                value,
                shape,
                key_shape,
                window,
                scale,
                qfactor,
                kfactor,
                relative_bias,
                tile,
                stride,
                (uh, uw),
            )
    ty, tx = triton.cdiv(height, tile), triton.cdiv(width, tile)
    tiles = ty * tx
    nq, nk = tile * tile, uh * uw
    mask_heads = heads if relative_bias is not None else 1
    pitch = triton.cdiv(nk, 8) * 8
    # Include Q, K, V and the additive mask in the per-chunk footprint.
    tile_bytes = query.element_size() * (
        batch * heads * (nq + 2 * nk) * (dim + rank)
        + mask_heads * (batch if rank else 1) * nq * pitch
    )
    group_size = max(1, (128 << 20) // tile_bytes)
    if rank:
        group_size = min(16, group_size)
    outputs = []
    with torch.cuda.device(query.device):
        for first in range(0, tiles, group_size):
            count = min(group_size, tiles - first)
            q = query.new_empty((batch * count, heads, nq, dim + rank))
            k = query.new_empty((batch * count, heads, nk, dim + rank))
            v = torch.empty_like(k)
            for src, factor, out, tokens, is_query in (
                (query, qfactor, q, nq, True),
                (key, kfactor, k, nk, False),
            ):
                fstrides = (0, 0) if factor is None else factor.stride()[:2]
                _pack_tokens[(triton.cdiv(tokens, 32), batch * count * heads)](
                    src,
                    value,
                    src if factor is None else factor,
                    out,
                    v,
                    height,
                    width,
                    key_height,
                    key_width,
                    kh,
                    kw,
                    uh,
                    uw,
                    first,
                    count,
                    heads,
                    *src.stride(),
                    *value.stride(),
                    *fstrides,
                    tile,
                    stride,
                    dim,
                    rank,
                    is_query,
                    32,
                    triton.next_power_of_2(dim + rank),
                )
            mask = query.new_empty((count, mask_heads, nq, pitch))[..., :nk]
            _pack_mask[(triton.cdiv(nk, 256), nq, count * mask_heads)](
                query if relative_bias is None else relative_bias,
                mask,
                height,
                width,
                key_height,
                key_width,
                kh,
                kw,
                uh,
                uw,
                first,
                pitch,
                mask_heads,
                tile,
                stride,
                window,
                relative_bias is not None,
                256,
            )
            if rank:
                if batch > 1:
                    mask = mask.repeat(batch, 1, 1, 1)
            else:
                q, k, v = (
                    t.reshape(batch, count * heads, t.shape[-2], dim) for t in (q, k, v)
                )
                mask = mask.reshape(1, count * mask_heads, nq, nk)
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale
            )
            outputs.append(a[..., :dim].reshape(batch, count, heads, tile, tile, dim))
    return (
        torch.cat(outputs, 1)
        .reshape(batch, ty, tx, heads, tile, tile, dim)
        .permute(0, 3, 6, 1, 4, 2, 5)
        .reshape(batch, heads * dim, ty * tile, tx * tile)[:, :, :height, :width]
    )


def spatial(
    module, shape, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, context: bool
) -> torch.Tensor:
    dim = q.shape[-1]
    a = attend(
        q * dim**-0.5,
        k,
        v,
        shape,
        shape,
        module.window,
        1.0,
        fourier=module.factored_bias if context else None,
        relative_bias=None if context else module.relative_bias,
    )
    return a.contiguous() if context else a


def retrieval(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_tau: torch.Tensor,
    shape,
    key_shape,
) -> torch.Tensor:
    images, channels, height, width = shape
    query = q * log_tau.exp().clamp(0.1, 10).to(q.dtype).repeat(images)[:, None, None]
    v = v.reshape(q.shape[0], 32, -1).transpose(1, 2)
    size = min(48, key_shape[0]) * min(48, key_shape[1])
    scale = math.log(size) / (math.log(1024) * math.sqrt(32))
    a = attend(
        query[:, None], k[:, None], v[:, None], (height, width), key_shape, 48, scale
    )
    return a.reshape(images, channels, height, width)

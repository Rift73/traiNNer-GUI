# ruff: noqa: ANN001, ANN202, I001, N803, N806
"""Fused dense-RPB attention for DRFT OCAB training.

This is a FlashAttention-style implementation of the exact OCAB equation

    softmax((Q @ K.T) * scale + dense_signed_relative_bias) @ V

The learned bias remains in its compact relative-position table and is loaded
inside the tiled attention kernels.  No dense bias or score tensor is
materialized.  Backward gives each Q, K, V, and relative-table output element
one owner, so it requires no atomics and remains deterministic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


_BF16 = 1
_FP16 = 2


_FWD_BLOCK_M = 64
_FWD_BLOCK_N = 64
_FWD_WARPS = 4
_FWD_STAGES = 3
_DQ_BLOCK_M = 16
_DQ_BLOCK_N = 64
_DQ_WARPS = 4
_DQ_STAGES = 3
_DKV_BLOCK_M = 64
_DKV_BLOCK_N = 64
_DKV_WARPS = 4
_DKV_STAGES = 3
_DBIAS_BLOCK_M = 32
_DBIAS_BLOCK_N = 64
_DBIAS_WARPS = 4
_DBIAS_STAGES = 3
_DELTA_WARPS = 8
_REDUCE_WARPS = 8

def ocab_attention_dtype_code(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return _BF16
    if dtype is torch.float16:
        return _FP16
    raise ValueError(f"unsupported fused OCAB attention dtype: {dtype}")


@triton.jit
def _rounded_bias(value, DTYPE_CODE: tl.constexpr):
    if DTYPE_CODE == 1:
        return value.to(tl.bfloat16).to(tl.float32)
    return value.to(tl.float16).to(tl.float32)


@triton.jit
def _attention_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    table_ptr,
    output_ptr,
    lse_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_channel,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_channel,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_channel,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_channel,
    table_stride_relative,
    table_stride_head,
    scale,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_height: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads

    query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    channel = tl.arange(0, BLOCK_D)
    query_valid = query_index < query_tokens
    channel_valid = channel < head_dim

    query_offsets = (
        batch * query_stride_batch
        + head * query_stride_head
        + query_index[:, None] * query_stride_token
        + channel[None, :] * query_stride_channel
    )
    query = tl.load(
        query_ptr + query_offsets,
        mask=query_valid[:, None] & channel_valid[None, :],
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    query_y = query_index // query_width
    query_x = query_index - query_y * query_width
    relative_width = query_width + key_width - 1

    for key_start in range(0, key_tokens, BLOCK_N):
        key_index = key_start + tl.arange(0, BLOCK_N)
        key_valid = key_index < key_tokens
        key_offsets = (
            batch * key_stride_batch
            + head * key_stride_head
            + key_index[:, None] * key_stride_token
            + channel[None, :] * key_stride_channel
        )
        key = tl.load(
            key_ptr + key_offsets,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)

        key_y = key_index // key_width
        key_x = key_index - key_y * key_width
        relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
        relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
        relative_index = relative_y * relative_width + relative_x
        bias = tl.load(
            table_ptr
            + relative_index * table_stride_relative
            + head * table_stride_head,
            mask=query_valid[:, None] & key_valid[None, :],
            other=0.0,
        )
        score += _rounded_bias(bias, DTYPE_CODE)
        score = tl.where(
            query_valid[:, None] & key_valid[None, :],
            score,
            -float("inf"),
        )

        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(row_max, block_max)
        probability = tl.exp(score - new_max[:, None])
        correction = tl.exp(row_max - new_max)
        block_sum = tl.sum(probability, axis=1)
        row_sum = row_sum * correction + block_sum
        accumulator *= correction[:, None]

        value_offsets = (
            batch * value_stride_batch
            + head * value_stride_head
            + key_index[:, None] * value_stride_token
            + channel[None, :] * value_stride_channel
        )
        value = tl.load(
            value_ptr + value_offsets,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        if DTYPE_CODE == 1:
            accumulator += tl.dot(probability.to(tl.bfloat16), value)
        else:
            accumulator += tl.dot(probability.to(tl.float16), value)
        row_max = new_max

    accumulator /= row_sum[:, None]
    output_offsets = (
        batch * output_stride_batch
        + head * output_stride_head
        + query_index[:, None] * output_stride_token
        + channel[None, :] * output_stride_channel
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=query_valid[:, None] & channel_valid[None, :],
    )
    lse_offset = (batch * num_heads + head) * query_tokens + query_index
    tl.store(
        lse_ptr + lse_offset,
        row_max + tl.log(row_sum),
        mask=query_valid,
    )


@triton.jit
def _attention_delta_kernel(
    output_ptr,
    grad_output_ptr,
    delta_ptr,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_channel,
    grad_stride_batch,
    grad_stride_head,
    grad_stride_token,
    grad_stride_channel,
    query_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_index = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads
    channel = tl.arange(0, BLOCK_D)
    valid = channel < head_dim
    output_offset = (
        batch * output_stride_batch
        + head * output_stride_head
        + query_index * output_stride_token
        + channel * output_stride_channel
    )
    grad_offset = (
        batch * grad_stride_batch
        + head * grad_stride_head
        + query_index * grad_stride_token
        + channel * grad_stride_channel
    )
    output = tl.load(output_ptr + output_offset, mask=valid, other=0.0).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr + grad_offset, mask=valid, other=0.0,
    ).to(tl.float32)
    delta = tl.sum(output * grad_output, axis=0)
    tl.store(delta_ptr + batch_head * query_tokens + query_index, delta)


@triton.jit
def _attention_backward_query_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    table_ptr,
    grad_output_ptr,
    lse_ptr,
    delta_ptr,
    grad_query_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_channel,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_channel,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_channel,
    grad_output_stride_batch,
    grad_output_stride_head,
    grad_output_stride_token,
    grad_output_stride_channel,
    grad_query_stride_batch,
    grad_query_stride_head,
    grad_query_stride_token,
    grad_query_stride_channel,
    table_stride_relative,
    table_stride_head,
    scale,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_height: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads
    query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    channel = tl.arange(0, BLOCK_D)
    query_valid = query_index < query_tokens
    channel_valid = channel < head_dim

    query_offset = (
        batch * query_stride_batch
        + head * query_stride_head
        + query_index[:, None] * query_stride_token
        + channel[None, :] * query_stride_channel
    )
    query = tl.load(
        query_ptr + query_offset,
        mask=query_valid[:, None] & channel_valid[None, :],
        other=0.0,
    )
    grad_output_offset = (
        batch * grad_output_stride_batch
        + head * grad_output_stride_head
        + query_index[:, None] * grad_output_stride_token
        + channel[None, :] * grad_output_stride_channel
    )
    grad_output = tl.load(
        grad_output_ptr + grad_output_offset,
        mask=query_valid[:, None] & channel_valid[None, :],
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + batch_head * query_tokens + query_index,
        mask=query_valid,
        other=0.0,
    )
    delta = tl.load(
        delta_ptr + batch_head * query_tokens + query_index,
        mask=query_valid,
        other=0.0,
    )
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    query_y = query_index // query_width
    query_x = query_index - query_y * query_width
    relative_width = query_width + key_width - 1

    for key_start in range(0, key_tokens, BLOCK_N):
        key_index = key_start + tl.arange(0, BLOCK_N)
        key_valid = key_index < key_tokens
        key_offset = (
            batch * key_stride_batch
            + head * key_stride_head
            + key_index[:, None] * key_stride_token
            + channel[None, :] * key_stride_channel
        )
        key = tl.load(
            key_ptr + key_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        value_offset = (
            batch * value_stride_batch
            + head * value_stride_head
            + key_index[:, None] * value_stride_token
            + channel[None, :] * value_stride_channel
        )
        value = tl.load(
            value_ptr + value_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )

        score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)
        key_y = key_index // key_width
        key_x = key_index - key_y * key_width
        relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
        relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
        relative_index = relative_y * relative_width + relative_x
        bias = tl.load(
            table_ptr
            + relative_index * table_stride_relative
            + head * table_stride_head,
            mask=query_valid[:, None] & key_valid[None, :],
            other=0.0,
        )
        score += _rounded_bias(bias, DTYPE_CODE)
        probability = tl.exp(score - lse[:, None])
        probability = tl.where(
            query_valid[:, None] & key_valid[None, :], probability, 0.0,
        )
        grad_probability = tl.dot(grad_output, tl.trans(value))
        grad_score = probability * (grad_probability - delta[:, None])
        if DTYPE_CODE == 1:
            grad_query += (
                tl.dot(grad_score.to(tl.bfloat16), key) * scale.to(tl.float32)
            )
        else:
            grad_query += (
                tl.dot(grad_score.to(tl.float16), key) * scale.to(tl.float32)
            )

    grad_query_offset = (
        batch * grad_query_stride_batch
        + head * grad_query_stride_head
        + query_index[:, None] * grad_query_stride_token
        + channel[None, :] * grad_query_stride_channel
    )
    tl.store(
        grad_query_ptr + grad_query_offset,
        grad_query,
        mask=query_valid[:, None] & channel_valid[None, :],
    )


@triton.jit
def _attention_backward_key_value_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    table_ptr,
    grad_output_ptr,
    lse_ptr,
    delta_ptr,
    grad_key_ptr,
    grad_value_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_channel,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_channel,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_channel,
    grad_output_stride_batch,
    grad_output_stride_head,
    grad_output_stride_token,
    grad_output_stride_channel,
    grad_key_stride_batch,
    grad_key_stride_head,
    grad_key_stride_token,
    grad_key_stride_channel,
    grad_value_stride_batch,
    grad_value_stride_head,
    grad_value_stride_token,
    grad_value_stride_channel,
    table_stride_relative,
    table_stride_head,
    scale,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_height: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads
    key_index = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    channel = tl.arange(0, BLOCK_D)
    key_valid = key_index < key_tokens
    channel_valid = channel < head_dim

    key_offset = (
        batch * key_stride_batch
        + head * key_stride_head
        + key_index[:, None] * key_stride_token
        + channel[None, :] * key_stride_channel
    )
    key = tl.load(
        key_ptr + key_offset,
        mask=key_valid[:, None] & channel_valid[None, :],
        other=0.0,
    )
    value_offset = (
        batch * value_stride_batch
        + head * value_stride_head
        + key_index[:, None] * value_stride_token
        + channel[None, :] * value_stride_channel
    )
    value = tl.load(
        value_ptr + value_offset,
        mask=key_valid[:, None] & channel_valid[None, :],
        other=0.0,
    )
    grad_key = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    grad_value = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    key_y = key_index // key_width
    key_x = key_index - key_y * key_width
    relative_width = query_width + key_width - 1

    for query_start in range(0, query_tokens, BLOCK_M):
        query_index = query_start + tl.arange(0, BLOCK_M)
        query_valid = query_index < query_tokens
        query_offset = (
            batch * query_stride_batch
            + head * query_stride_head
            + query_index[:, None] * query_stride_token
            + channel[None, :] * query_stride_channel
        )
        query = tl.load(
            query_ptr + query_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        grad_output_offset = (
            batch * grad_output_stride_batch
            + head * grad_output_stride_head
            + query_index[:, None] * grad_output_stride_token
            + channel[None, :] * grad_output_stride_channel
        )
        grad_output = tl.load(
            grad_output_ptr + grad_output_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch_head * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        delta = tl.load(
            delta_ptr + batch_head * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )

        score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
        relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
        relative_index = relative_y * relative_width + relative_x
        bias = tl.load(
            table_ptr
            + relative_index * table_stride_relative
            + head * table_stride_head,
            mask=query_valid[:, None] & key_valid[None, :],
            other=0.0,
        )
        score += _rounded_bias(bias, DTYPE_CODE)
        probability = tl.exp(score - lse[:, None])
        probability = tl.where(
            query_valid[:, None] & key_valid[None, :], probability, 0.0,
        )
        grad_probability = tl.dot(grad_output, tl.trans(value))
        grad_score = probability * (grad_probability - delta[:, None])
        if DTYPE_CODE == 1:
            grad_key += (
                tl.dot(tl.trans(grad_score.to(tl.bfloat16)), query)
                * scale.to(tl.float32)
            )
            grad_value += tl.dot(tl.trans(probability.to(tl.bfloat16)), grad_output)
        else:
            grad_key += (
                tl.dot(tl.trans(grad_score.to(tl.float16)), query)
                * scale.to(tl.float32)
            )
            grad_value += tl.dot(tl.trans(probability.to(tl.float16)), grad_output)

    grad_key_offset = (
        batch * grad_key_stride_batch
        + head * grad_key_stride_head
        + key_index[:, None] * grad_key_stride_token
        + channel[None, :] * grad_key_stride_channel
    )
    grad_value_offset = (
        batch * grad_value_stride_batch
        + head * grad_value_stride_head
        + key_index[:, None] * grad_value_stride_token
        + channel[None, :] * grad_value_stride_channel
    )
    mask = key_valid[:, None] & channel_valid[None, :]
    tl.store(grad_key_ptr + grad_key_offset, grad_key, mask=mask)
    tl.store(grad_value_ptr + grad_value_offset, grad_value, mask=mask)


@triton.jit
def _attention_backward_dense_bias_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    table_ptr,
    grad_output_ptr,
    lse_ptr,
    delta_ptr,
    grad_bias_ptr,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_channel,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_channel,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_channel,
    grad_output_stride_batch,
    grad_output_stride_head,
    grad_output_stride_token,
    grad_output_stride_channel,
    table_stride_relative,
    table_stride_head,
    scale,
    batch_windows: tl.constexpr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_height: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    key_block = tl.program_id(1)
    head = tl.program_id(2)
    query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_index = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    channel = tl.arange(0, BLOCK_D)
    query_valid = query_index < query_tokens
    key_valid = key_index < key_tokens
    channel_valid = channel < head_dim
    query_y = query_index // query_width
    query_x = query_index - query_y * query_width
    key_y = key_index // key_width
    key_x = key_index - key_y * key_width
    relative_width = query_width + key_width - 1
    relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
    relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
    relative_index = relative_y * relative_width + relative_x
    bias = tl.load(
        table_ptr
        + relative_index * table_stride_relative
        + head * table_stride_head,
        mask=query_valid[:, None] & key_valid[None, :],
        other=0.0,
    )
    bias = _rounded_bias(bias, DTYPE_CODE)
    grad_bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for batch in range(batch_windows):
        query_offset = (
            batch * query_stride_batch
            + head * query_stride_head
            + query_index[:, None] * query_stride_token
            + channel[None, :] * query_stride_channel
        )
        key_offset = (
            batch * key_stride_batch
            + head * key_stride_head
            + key_index[:, None] * key_stride_token
            + channel[None, :] * key_stride_channel
        )
        value_offset = (
            batch * value_stride_batch
            + head * value_stride_head
            + key_index[:, None] * value_stride_token
            + channel[None, :] * value_stride_channel
        )
        grad_output_offset = (
            batch * grad_output_stride_batch
            + head * grad_output_stride_head
            + query_index[:, None] * grad_output_stride_token
            + channel[None, :] * grad_output_stride_channel
        )
        query = tl.load(
            query_ptr + query_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        key = tl.load(
            key_ptr + key_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        value = tl.load(
            value_ptr + value_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        grad_output = tl.load(
            grad_output_ptr + grad_output_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + (batch * num_heads + head) * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        row_delta = tl.load(
            delta_ptr + (batch * num_heads + head) * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        score = (
            tl.dot(query, tl.trans(key)) * scale.to(tl.float32) + bias
        )
        probability = tl.exp(score - lse[:, None])
        probability = tl.where(
            query_valid[:, None] & key_valid[None, :], probability, 0.0,
        )
        grad_probability = tl.dot(grad_output, tl.trans(value))
        grad_bias += probability * (grad_probability - row_delta[:, None])

    output_offset = (
        (head * query_tokens + query_index[:, None]) * key_tokens
        + key_index[None, :]
    )
    tl.store(
        grad_bias_ptr + output_offset,
        grad_bias,
        mask=query_valid[:, None] & key_valid[None, :],
    )


@triton.jit
def _reduce_dense_bias_gradient_kernel(
    grad_bias_ptr,
    grad_table_ptr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_height: tl.constexpr,
    query_width: tl.constexpr,
    key_height: tl.constexpr,
    key_width: tl.constexpr,
    relative_width: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_QUERY: tl.constexpr,
):
    output_index = tl.program_id(0)
    head = output_index % num_heads
    relative_index = output_index // num_heads
    relative_y = relative_index // relative_width
    relative_x = relative_index - relative_y * relative_width
    delta_y = relative_y - (query_height - 1)
    delta_x = relative_x - (query_width - 1)

    query_index = tl.arange(0, BLOCK_QUERY)
    query_y = query_index // query_width
    query_x = query_index - query_y * query_width
    key_y = query_y + delta_y
    key_x = query_x + delta_x
    valid = (
        (query_index < query_tokens)
        & (key_y >= 0)
        & (key_y < key_height)
        & (key_x >= 0)
        & (key_x < key_width)
    )
    key_index = key_y * key_width + key_x
    dense_offset = (head * query_tokens + query_index) * key_tokens + key_index
    gradient = tl.load(
        grad_bias_ptr + dense_offset,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_table_ptr + relative_index * num_heads + head,
        tl.sum(gradient, axis=0),
    )


@triton_op("trainner_drft::ocab_dense_rpb_attention", mutates_args={})
def ocab_dense_rpb_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    query_height: int,
    query_width: int,
    key_height: int,
    key_width: int,
    scale: float,
    dtype_code: int,
    preserve_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_windows, num_heads, query_tokens, head_dim = query.shape
    if key.shape != (batch_windows, num_heads, key_height * key_width, head_dim):
        raise ValueError("fused OCAB key geometry does not match query geometry")
    if value.shape != key.shape:
        raise ValueError("fused OCAB value geometry must match key geometry")
    if query_tokens != query_height * query_width:
        raise ValueError("fused OCAB query geometry does not match token count")
    expected_relative = (query_height + key_height - 1) * (
        query_width + key_width - 1
    )
    if table.shape != (expected_relative, num_heads):
        raise ValueError("fused OCAB relative-bias table has the wrong shape")

    # Q arrives as a dense head view of token-major QKV storage.  Preserving
    # those strides makes the immediately following head->channel reshape a
    # view instead of a materialized transpose.  Keep the historical layout
    # available for controlled A/B validation.
    output = (
        torch.empty_like(query, memory_format=torch.preserve_format)
        if preserve_layout
        else torch.empty(query.shape, device=query.device, dtype=query.dtype)
    )
    lse = torch.empty(
        (batch_windows, num_heads, query_tokens),
        device=query.device,
        dtype=torch.float32,
    )
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(query_tokens, _FWD_BLOCK_M),
        batch_windows * num_heads,
    )
    wrap_triton(_attention_forward_kernel)[grid](
        query,
        key,
        value,
        table,
        output,
        lse,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *table.stride(),
        scale,
        query_tokens,
        key_height * key_width,
        query_height,
        query_width,
        key_width,
        num_heads,
        head_dim,
        dtype_code,
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_N=_FWD_BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=_FWD_WARPS,
        num_stages=_FWD_STAGES,
    )
    return output, lse


@triton_op("trainner_drft::ocab_dense_rpb_attention_backward", mutates_args={})
def ocab_dense_rpb_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    query_height: int,
    query_width: int,
    key_height: int,
    key_width: int,
    scale: float,
    dtype_code: int,
    preserve_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not preserve_layout:
        grad_output = grad_output.contiguous()
    batch_windows, num_heads, query_tokens, head_dim = query.shape
    key_tokens = key_height * key_width
    BLOCK_D = triton.next_power_of_2(head_dim)
    delta = torch.empty_like(lse)
    delta_grid = (query_tokens, batch_windows * num_heads)
    wrap_triton(_attention_delta_kernel)[delta_grid](
        output,
        grad_output,
        delta,
        *output.stride(),
        *grad_output.stride(),
        query_tokens,
        num_heads,
        head_dim,
        BLOCK_D=BLOCK_D,
        num_warps=_DELTA_WARPS,
    )

    # In the projection-ready mode, return gradients in the source Q/K/V
    # layouts so their inverse reshape/permute into the fused QKV-window
    # operator remains a view.  Every kernel already honors explicit strides.
    grad_query = (
        torch.empty_like(query, memory_format=torch.preserve_format)
        if preserve_layout
        else torch.empty(query.shape, device=query.device, dtype=query.dtype)
    )
    query_grid = (
        triton.cdiv(query_tokens, _DQ_BLOCK_M),
        batch_windows * num_heads,
    )
    wrap_triton(_attention_backward_query_kernel)[query_grid](
        query,
        key,
        value,
        table,
        grad_output,
        lse,
        delta,
        grad_query,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *grad_output.stride(),
        *grad_query.stride(),
        *table.stride(),
        scale,
        query_tokens,
        key_tokens,
        query_height,
        query_width,
        key_width,
        num_heads,
        head_dim,
        dtype_code,
        BLOCK_M=_DQ_BLOCK_M,
        BLOCK_N=_DQ_BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=_DQ_WARPS,
        num_stages=_DQ_STAGES,
    )

    grad_key = (
        torch.empty_like(key, memory_format=torch.preserve_format)
        if preserve_layout
        else torch.empty(key.shape, device=key.device, dtype=key.dtype)
    )
    grad_value = (
        torch.empty_like(value, memory_format=torch.preserve_format)
        if preserve_layout
        else torch.empty(value.shape, device=value.device, dtype=value.dtype)
    )
    key_value_grid = (
        triton.cdiv(key_tokens, _DKV_BLOCK_N),
        batch_windows * num_heads,
    )
    wrap_triton(_attention_backward_key_value_kernel)[key_value_grid](
        query,
        key,
        value,
        table,
        grad_output,
        lse,
        delta,
        grad_key,
        grad_value,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *grad_output.stride(),
        *grad_key.stride(),
        *grad_value.stride(),
        *table.stride(),
        scale,
        query_tokens,
        key_tokens,
        query_height,
        query_width,
        key_width,
        num_heads,
        head_dim,
        dtype_code,
        BLOCK_M=_DKV_BLOCK_M,
        BLOCK_N=_DKV_BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=_DKV_WARPS,
        num_stages=_DKV_STAGES,
    )

    grad_bias = torch.empty(
        (num_heads, query_tokens, key_tokens),
        device=query.device,
        dtype=query.dtype,
    )
    dense_bias_grid = (
        triton.cdiv(query_tokens, _DBIAS_BLOCK_M),
        triton.cdiv(key_tokens, _DBIAS_BLOCK_N),
        num_heads,
    )
    wrap_triton(_attention_backward_dense_bias_kernel)[dense_bias_grid](
        query,
        key,
        value,
        table,
        grad_output,
        lse,
        delta,
        grad_bias,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *grad_output.stride(),
        *table.stride(),
        scale,
        batch_windows,
        query_tokens,
        key_tokens,
        query_height,
        query_width,
        key_width,
        num_heads,
        head_dim,
        dtype_code,
        BLOCK_M=_DBIAS_BLOCK_M,
        BLOCK_N=_DBIAS_BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=_DBIAS_WARPS,
        num_stages=_DBIAS_STAGES,
    )

    grad_table = torch.empty_like(table, memory_format=torch.contiguous_format)
    relative_entries = table.shape[0]
    relative_width = query_width + key_width - 1
    reduction_grid = (relative_entries * num_heads,)
    wrap_triton(_reduce_dense_bias_gradient_kernel)[reduction_grid](
        grad_bias,
        grad_table,
        query_tokens,
        key_tokens,
        query_height,
        query_width,
        key_height,
        key_width,
        relative_width,
        num_heads,
        BLOCK_QUERY=triton.next_power_of_2(query_tokens),
        num_warps=_REDUCE_WARPS,
    )
    return grad_query, grad_key, grad_value, grad_table


def _setup_context(ctx, inputs, output) -> None:
    (
        query,
        key,
        value,
        table,
        query_height,
        query_width,
        key_height,
        key_width,
        scale,
        dtype_code,
        preserve_layout,
    ) = inputs
    attention_output, lse = output
    ctx.save_for_backward(query, key, value, table, attention_output, lse)
    ctx.query_height = query_height
    ctx.query_width = query_width
    ctx.key_height = key_height
    ctx.key_width = key_width
    ctx.scale = scale
    ctx.dtype_code = dtype_code
    ctx.preserve_layout = preserve_layout


def _backward(ctx, grad_output: torch.Tensor, grad_lse: torch.Tensor | None):
    del grad_lse
    query, key, value, table, attention_output, lse = ctx.saved_tensors
    grad_query, grad_key, grad_value, grad_table = (
        ocab_dense_rpb_attention_backward(
            grad_output,
            query,
            key,
            value,
            table,
            attention_output,
            lse,
            ctx.query_height,
            ctx.query_width,
            ctx.key_height,
            ctx.key_width,
            ctx.scale,
            ctx.dtype_code,
            ctx.preserve_layout,
        )
    )
    return (
        grad_query,
        grad_key,
        grad_value,
        grad_table,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    ocab_dense_rpb_attention,
    _backward,
    setup_context=_setup_context,
)


def ocab_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    query_height: int,
    query_width: int,
    key_height: int,
    key_width: int,
    scale: float,
    preserve_layout: bool = False,
) -> torch.Tensor:
    output, _lse = ocab_dense_rpb_attention(
        query,
        key,
        value,
        table,
        query_height,
        query_width,
        key_height,
        key_width,
        scale,
        ocab_attention_dtype_code(query.dtype),
        preserve_layout,
    )
    return output


__all__ = ["ocab_attention", "ocab_attention_dtype_code"]

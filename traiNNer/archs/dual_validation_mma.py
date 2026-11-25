"""Fused BF16 inference for the SM120 DUAL neighborhood-attention layouts.

Four warps each handle sixteen queries. Matching the reference SDPA kernel's
K=8 matrix instructions, 64-key softmax blocks and correctly rounded reciprocal
preserves its rounding even when neighboring query tiles overlap.
"""


# Triton infers device argument types; uppercase kernel parameters identify buffers and dimensions.
# ruff: noqa: ANN001, ANN202, N803

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import BlockedLayout, SliceLayout

LAY = gl.constexpr(BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0]))


@gluon.constexpr_function
def _qk_mma(outs, ins):
    """Accumulate QK in 8-channel steps, including 16 Fourier channels when present."""
    a, b, k, z = ins
    code = []
    for j in range(4):
        dst = "{" + ",".join(out[j] for out in outs) + "}"
        for i in range(6 if len(k) == 32 else 4):
            acc = "{" + ",".join([z[0]] * 4) + "}" if i == 0 else dst
            code.append(
                f"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {dst}, {{{a[i]},{b[i]}}}, {{{k[j * (len(k) // 4) + i]}}}, {acc};"
            )
    return "\n".join(code)


@gluon.constexpr_function
def _pv_mma(outs, ins):
    p0, p1, v, *state = ins
    code = []
    for group in range(4):
        dst = "{" + ",".join(out[group] for out in outs) + "}"
        for j in range(8):
            acc = "{" + ",".join(x[group] for x in state) + "}" if j == 0 else dst
            code.append(
                f"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {dst}, {{{p0[j]},{p1[j]}}}, {{{v[j * 4 + group]}}}, {acc};"
            )
    return "\n".join(code)


@gluon.constexpr_function
def _sum_softmax_row(outs, ins):
    """Sum two half-blocks in the same order as CUTLASS memory-efficient attention."""
    a, b, total = ins
    ds = outs[0][0]
    code = ["{ .reg .f32 lo,hi,tmp;"]
    for half, var in ((0, "lo"), (4, "hi")):
        code.append(f"add.rn.f32 {var},{a[half]},{b[half]};")
        for j in range(half + 1, half + 4):
            code.append(f"add.rn.f32 {var},{var},{a[j]};")
            code.append(f"add.rn.f32 {var},{var},{b[j]};")
        for t in (1, 2):
            code.append(f"shfl.sync.bfly.b32 tmp,{var},{t},31,0xffffffff;")
            code.append(f"add.rn.f32 {var},{var},tmp;")
    code.extend([f"add.rn.f32 {ds},{total[0]},lo;", f"add.rn.f32 {ds},{ds},hi;", "}"])
    return "\n".join(code)


@gluon.jit
def _max_quad(x):
    a = gl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0,$1,1,31,0xffffffff;",
        "=f,f",
        [x],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    x = gl.maximum(x, a)
    a = gl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0,$1,2,31,0xffffffff;",
        "=f,f",
        [x],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return gl.maximum(x, a)


@gluon.jit
def _load_q(Q, F, token, local, bits, head, s0, s1, s2, f0, f1, RFB: gl.constexpr):
    qptr = Q + s0 + head * s1 + token[:, None].to(gl.int64) * s2 + bits[None, :] * 8
    fptr = F + head * f0 + local[:, None] * f1 + (bits[None, :] - 4) * 8
    lane = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(1, LAY)))
    offset = 2 * (lane % 4)
    a = gl.load(
        (qptr + offset[:, None]).to(gl.pointer_type(gl.uint32)), bits[None, :] < 4, 0
    )
    if not RFB:
        return a
    b = gl.load(
        (fptr + offset[:, None]).to(gl.pointer_type(gl.uint32)),
        (bits[None, :] >= 4) & (bits[None, :] < 6),
        0,
    )
    return a | b


@gluon.jit
def _logit(
    score,
    col,
    qy,
    qx,
    origin_y,
    origin_x,
    keyh,
    keyw,
    kh,
    kw,
    uw,
    uh,
    bias,
    head,
    heads,
    scale,
    STRIDE: gl.constexpr,
    WINDOW: gl.constexpr,
    RPB: gl.constexpr,
):
    y0 = gl.minimum(gl.maximum(qy // STRIDE - kh // 2, 0), keyh - kh)
    x0 = gl.minimum(gl.maximum(qx // STRIDE - kw // 2, 0), keyw - kw)
    y = col // uw + origin_y
    x = col % uw + origin_x
    inside = (
        (col < uh * uw)
        & (y >= y0[:, None])
        & (y < y0[:, None] + kh)
        & (x >= x0[:, None])
        & (x < x0[:, None] + kw)
    )
    if RPB:
        index = (qy[:, None] - y + WINDOW - 1) * (2 * WINDOW - 1) + (
            qx[:, None] - x + WINDOW - 1
        )
        add = (
            gl.load(bias + index * heads + head, inside, 0)
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        score = score + add
    else:
        score = score * scale
    return gl.where(inside, score * 1.4426950408889634, -float("inf"))


@gluon.jit
def _pack_bf(a, b):
    return a.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) | (
        b.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )


@gluon.jit
def _query_key_half(
    K,
    G,
    qa,
    qb,
    block,
    quad,
    within,
    ky,
    kx,
    KeyW,
    OH,
    OW,
    batch,
    head,
    ks0,
    ks1,
    ks2,
    kg0,
    kg1,
    HALF: gl.constexpr,
    RFB: gl.constexpr,
):
    """Consume 32 keys at a time before loading the next key fragment."""
    lanes = gl.arange(0, 32 if RFB else 16, layout=gl.constexpr(SliceLayout(0, LAY)))
    words: gl.constexpr = 8 if RFB else 4
    idx = block * 64 + HALF * 32 + (lanes[None, :] // words) * 8 + quad[:, None]
    token = (idx // OW + ky) * KeyW + kx + idx % OW
    channel = (lanes[None, :] % words) * 8 + within[:, None] * 2
    kk = gl.load(
        (
            K
            + batch.to(gl.int64) * ks0
            + head * ks1
            + token.to(gl.int64) * ks2
            + channel
        ).to(gl.pointer_type(gl.uint32)),
        (idx < OH * OW) & (channel < 32),
        0,
    )
    if RFB:
        ff = gl.load(
            (G + head * kg0 + idx * kg1 + channel - 32).to(gl.pointer_type(gl.uint32)),
            (idx < OH * OW) & (channel >= 32) & (channel < 48),
            0,
        )
        kk = kk | ff
    tp: gl.constexpr = gl.distributed_type(gl.float32, [128, 4], LAY)
    a, b, c, e = gl.inline_asm(
        _qk_mma,
        ("=&f", "=&f", "=&f", "=&f", "r", "r", "r", "f"),
        [qa, qb, kk, 0.0],
        (tp, tp, tp, tp),
        is_pure=True,
    )
    return a, b, c, e


@gluon.jit
def _join_key_halves(first, second):
    return gl.convert_layout(
        gl.reshape(gl.permute(gl.join(first, second), (0, 2, 1)), (128, 8)), LAY
    )


# Image dimensions and token strides vary across validation images. Their alignment
# does not affect the packed BF16 channel pairs loaded by the MMA instructions.
@gluon.jit(
    do_not_specialize_on_alignment=[
        "H",
        "W",
        "KeyH",
        "KeyW",
        "OH",
        "OW",
        "OutH",
        "OutW",
        "qs0",
        "qs1",
        "qs2",
        "ks0",
        "ks1",
        "ks2",
        "vs0",
        "vs1",
        "vs2",
        "fg0",
        "fg1",
        "kg0",
        "kg1",
    ]
)
def _neighborhood_attention(
    Q,
    K,
    V,
    F,
    G,
    Bias,
    O,
    H,
    W,
    KeyH,
    KeyW,
    OH,
    OW,
    OutH,
    OutW,
    KH: gl.constexpr,
    KW: gl.constexpr,
    HEADS: gl.constexpr,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    fg0,
    fg1,
    kg0,
    kg1,
    scale,
    TILE: gl.constexpr,
    STRIDE: gl.constexpr,
    WINDOW: gl.constexpr,
    RFB: gl.constexpr,
    RPB: gl.constexpr,
):
    lane = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(1, LAY)))
    bits = gl.arange(0, 8, layout=gl.constexpr(SliceLayout(0, LAY)))
    i32 = gl.arange(0, 32, layout=gl.constexpr(SliceLayout(0, LAY)))
    g4 = gl.arange(0, 4, layout=gl.constexpr(SliceLayout(0, LAY)))
    t = gl.program_id(0)
    bh = gl.program_id(1)
    head = bh % HEADS
    batch = bh // HEADS
    ref = t // (TILE * TILE // 64)
    sub = t % (TILE * TILE // 64)
    nx = gl.cdiv(W, TILE)
    ty = ref // nx * TILE
    tx = ref % nx * TILE
    ky = gl.minimum(gl.maximum(ty // STRIDE - KH // 2, 0), KeyH - OH)
    kx = gl.minimum(gl.maximum(tx // STRIDE - KW // 2, 0), KeyW - OW)
    # Each warp consumes one 16-query row inside the original 16x16 tile.
    warp = lane // 32
    quad = (lane % 32) // 4
    within = lane % 4
    row0 = sub * 64 + warp * 16 + quad
    row1 = row0 + 8
    y0 = gl.minimum(ty + row0 // TILE, H - 1)
    x0 = gl.minimum(tx + row0 % TILE, W - 1)
    y1 = gl.minimum(ty + row1 // TILE, H - 1)
    x1 = gl.minimum(tx + row1 % TILE, W - 1)
    qa = _load_q(
        Q,
        F,
        y0 * W + x0,
        (y0 - ky) * OW + x0 - kx,
        bits,
        head,
        batch.to(gl.int64) * qs0,
        qs1,
        qs2,
        fg0,
        fg1,
        RFB,
    )
    qb = _load_q(
        Q,
        F,
        y1 * W + x1,
        (y1 - ky) * OW + x1 - kx,
        bits,
        head,
        batch.to(gl.int64) * qs0,
        qs1,
        qs2,
        fg0,
        fg1,
        RFB,
    )
    mx0 = gl.full(
        (128,), -float("inf"), gl.float32, layout=gl.constexpr(SliceLayout(1, LAY))
    )
    mx1 = mx0
    sum0 = gl.full((128,), 0.0, gl.float32, layout=gl.constexpr(SliceLayout(1, LAY)))
    sum1 = sum0
    o0 = gl.full((128, 4), 0.0, gl.float32, layout=LAY)
    o1 = o0
    o2 = o0
    o3 = o0
    first_y = gl.minimum(ty + sub * 64 // TILE, H - 1) // STRIDE
    last_y = gl.minimum(ty + (sub * 64 + 63) // TILE, H - 1) // STRIDE
    first_x = tx // STRIDE
    last_x = gl.minimum(tx + TILE - 1, W - 1) // STRIDE
    # Skip complete 64-key blocks outside every query neighborhood in this tile.
    begin = (
        (gl.minimum(gl.maximum(first_y - KH // 2, 0), KeyH - KH) - ky) * OW
        + gl.minimum(gl.maximum(first_x - KW // 2, 0), KeyW - KW)
        - kx
    ) // 64
    end = gl.cdiv(
        (gl.minimum(gl.maximum(last_y - KH // 2, 0), KeyH - KH) - ky + KH - 1) * OW
        + gl.minimum(gl.maximum(last_x - KW // 2, 0), KeyW - KW)
        - kx
        + KW,
        64,
    )
    for block in range(begin, end):
        top = _query_key_half(
            K,
            G,
            qa,
            qb,
            block,
            quad,
            within,
            ky,
            kx,
            KeyW,
            OH,
            OW,
            batch,
            head,
            ks0,
            ks1,
            ks2,
            kg0,
            kg1,
            0,
            RFB,
        )
        bottom = _query_key_half(
            K,
            G,
            qa,
            qb,
            block,
            quad,
            within,
            ky,
            kx,
            KeyW,
            OH,
            OW,
            batch,
            head,
            ks0,
            ks1,
            ks2,
            kg0,
            kg1,
            1,
            RFB,
        )
        a = _join_key_halves(top[0], bottom[0])
        b = _join_key_halves(top[1], bottom[1])
        c = _join_key_halves(top[2], bottom[2])
        e = _join_key_halves(top[3], bottom[3])
        # Each lane owns two adjacent keys for each of its two query rows.
        col = block * 64 + bits[None, :] * 8 + within[:, None] * 2
        a = _logit(
            a,
            col,
            y0,
            x0,
            ky,
            kx,
            KeyH,
            KeyW,
            KH,
            KW,
            OW,
            OH,
            Bias,
            head,
            HEADS,
            scale,
            STRIDE,
            WINDOW,
            RPB,
        )
        b = _logit(
            b,
            col + 1,
            y0,
            x0,
            ky,
            kx,
            KeyH,
            KeyW,
            KH,
            KW,
            OW,
            OH,
            Bias,
            head,
            HEADS,
            scale,
            STRIDE,
            WINDOW,
            RPB,
        )
        c = _logit(
            c,
            col,
            y1,
            x1,
            ky,
            kx,
            KeyH,
            KeyW,
            KH,
            KW,
            OW,
            OH,
            Bias,
            head,
            HEADS,
            scale,
            STRIDE,
            WINDOW,
            RPB,
        )
        e = _logit(
            e,
            col + 1,
            y1,
            x1,
            ky,
            kx,
            KeyH,
            KeyW,
            KH,
            KW,
            OW,
            OH,
            Bias,
            head,
            HEADS,
            scale,
            STRIDE,
            WINDOW,
            RPB,
        )
        # Online softmax rescales both existing sums before adding this block.
        m0 = gl.maximum(mx0, _max_quad(gl.max(gl.maximum(a, b), 1)))
        m1 = gl.maximum(mx1, _max_quad(gl.max(gl.maximum(c, e), 1)))
        safe0 = gl.where(m0 == -float("inf"), 0.0, m0)
        safe1 = gl.where(m1 == -float("inf"), 0.0, m1)
        alpha0 = gl.where(m0 > mx0, gl.exp2(mx0 - safe0), 1.0)
        alpha1 = gl.where(m1 > mx1, gl.exp2(mx1 - safe1), 1.0)
        a = gl.exp2(a - safe0[:, None])
        b = gl.exp2(b - safe0[:, None])
        c = gl.exp2(c - safe1[:, None])
        e = gl.exp2(e - safe1[:, None])
        sum0 = gl.inline_asm(
            _sum_softmax_row,
            ("=&f", "f", "f", "f"),
            [a, b, sum0 * alpha0],
            sum0.type,
            is_pure=True,
        )
        sum1 = gl.inline_asm(
            _sum_softmax_row,
            ("=&f", "f", "f", "f"),
            [c, e, sum1 * alpha1],
            sum1.type,
            is_pure=True,
        )
        # V fragments use the transposed ordering of the K fragments.
        vi = block * 64 + i32[None, :] // 4 * 8 + within[:, None] * 2
        ch = i32[None, :] % 4 * 8 + quad[:, None]
        tok0 = (vi // OW + ky) * KeyW + kx + vi % OW
        tok1 = ((vi + 1) // OW + ky) * KeyW + kx + (vi + 1) % OW
        vptr = V + batch.to(gl.int64) * vs0 + head * vs1 + ch
        vl = gl.load(vptr + tok0.to(gl.int64) * vs2, vi < OH * OW, 0)
        vh = gl.load(vptr + tok1.to(gl.int64) * vs2, vi + 1 < OH * OW, 0)
        o0, o1, o2, o3 = gl.inline_asm(
            _pv_mma,
            ("=&f", "=&f", "=&f", "=&f", "r", "r", "r", "f", "f", "f", "f"),
            [
                _pack_bf(a, b),
                _pack_bf(c, e),
                _pack_bf(vl, vh),
                o0 * alpha0[:, None],
                o1 * alpha0[:, None],
                o2 * alpha1[:, None],
                o3 * alpha1[:, None],
            ],
            (o0.type, o0.type, o0.type, o0.type),
            is_pure=True,
        )
        mx0 = m0
        mx1 = m1
    valid0 = (ty + row0 // TILE < H) & (tx + row0 % TILE < W)
    valid1 = (ty + row1 // TILE < H) & (tx + row1 % TILE < W)
    ch = g4[None, :] * 8 + within[:, None] * 2
    dest = (bh.to(gl.int64) * 32 + ch) * OutH * OutW
    norm0 = gl.inline_asm_elementwise(
        "rcp.rn.f32 $0,$1;", "=f,f", [sum0], dtype=gl.float32, is_pure=True, pack=1
    )[:, None]
    norm1 = gl.inline_asm_elementwise(
        "rcp.rn.f32 $0,$1;", "=f,f", [sum1], dtype=gl.float32, is_pure=True, pack=1
    )[:, None]
    gl.store(O + dest + (y0 * OutW + x0)[:, None], o0 * norm0, valid0[:, None])
    gl.store(
        O + dest + OutH * OutW + (y0 * OutW + x0)[:, None], o1 * norm0, valid0[:, None]
    )
    gl.store(O + dest + (y1 * OutW + x1)[:, None], o2 * norm1, valid1[:, None])
    gl.store(
        O + dest + OutH * OutW + (y1 * OutW + x1)[:, None], o3 * norm1, valid1[:, None]
    )


def attend(
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
    union_shape,
) -> torch.Tensor:
    batch, heads, _, dim = query.shape
    height, width = shape
    key_height, key_width = key_shape
    union_height, union_width = union_shape
    context = qfactor is not None
    ty, tx = triton.cdiv(height, tile), triton.cdiv(width, tile)
    out_height, out_width = (height, width) if context else (ty * tile, tx * tile)

    # The MMA operands load aligned pairs of BF16 channels. V is scalar-loaded.
    def paired(tensor: torch.Tensor) -> torch.Tensor:
        if (
            tensor.stride(-1) == 1
            and tensor.storage_offset() % 2 == 0
            and all(
                size == 1 or step % 2 == 0
                for size, step in zip(
                    tensor.shape[:-1], tensor.stride()[:-1], strict=True
                )
            )
        ):
            return tensor
        return tensor.clone(memory_format=torch.contiguous_format)

    query, key = paired(query), paired(key)
    if value.stride(-1) != 1:
        value = value.contiguous()
    qfactor = paired(qfactor) if context else query
    kfactor = paired(kfactor) if context else key
    bias = query if relative_bias is None else relative_bias.contiguous()
    output = query.new_empty((batch, heads * dim, out_height, out_width))
    with torch.cuda.device(query.device):
        _neighborhood_attention[(ty * tx * (tile * tile // 64), batch * heads)](
            query,
            key,
            value,
            qfactor,
            kfactor,
            bias,
            output,
            height,
            width,
            key_height,
            key_width,
            union_height,
            union_width,
            out_height,
            out_width,
            min(window, key_height),
            min(window, key_width),
            heads,
            *query.stride()[:3],
            *key.stride()[:3],
            *value.stride()[:3],
            *qfactor.stride()[:2],
            *kfactor.stride()[:2],
            scale,
            tile,
            stride,
            window,
            context,
            relative_bias is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output[:, :, :height, :width]

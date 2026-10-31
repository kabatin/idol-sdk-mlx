"""A fused Metal kernel for the denoiser's attention prologue (the costliest elementwise work while sampling).

RopeMultiHeadAttention rotates q and k by the learned RoPE (fp32), rounds them to the activation dtype, RMS-normalises
each head (fp32) and transposes q, k, v to [B, H, L, D] for the attention kernel: ~1.2 s of every 5.3 s sampling
step on an M5 when done op by op. This kernel does it in one pass: one SIMD group per (token, head), one thread per
rotation pair."""
from typing import Tuple

import mlx.core as mx

HEAD_DIM = 64   # 32 rotation pairs: one per SIMD lane

_SOURCE = """
    uint lane = thread_position_in_grid.x % 32;
    uint row = thread_position_in_grid.x / 32;            // (b, l, h) flattened
    uint h = row % H;
    uint bl = row / H;                                    // b * L + l
    uint b = bl / L;
    uint l = bl % L;
    if (bl >= B * L) return;

    uint pair = lane;
    uint d0 = 2 * pair;
    uint in_base = ((bl * 3) * H + h) * 64;               // qkv[b, l, 0, h, :]
    uint rope = row * 32 + pair;                          // cos/sin[b, l, h, pair]
    uint out_base = ((b * H + h) * L + l) * 64;           // out[b, h, l, :]
    float c = cos_t[rope];
    float s = sin_t[rope];

    for (uint which = 0; which < 2; ++which) {            // 0: q, 1: k
        uint base = in_base + which * H * 64;
        float x0 = float(qkv[base + d0]);
        float x1 = float(qkv[base + d0 + 1]);
        float r0 = float(T(x0 * c - x1 * s));             // torch: rotate in fp32, then .type_as(hidden)
        float r1 = float(T(x0 * s + x1 * c));
        float norm = metal::max(metal::sqrt(simd_sum(r0 * r0 + r1 * r1)), 1e-12f);
        const device float* gamma = which == 0 ? q_gamma : k_gamma;
        float g0 = gamma[h * 64 + d0] * scale[0];
        float g1 = gamma[h * 64 + d0 + 1] * scale[0];
        device T* out = which == 0 ? q_out : k_out;
        out[out_base + d0] = T(r0 / norm * g0);
        out[out_base + d0 + 1] = T(r1 / norm * g1);
    }
    uint v_base = in_base + 2 * H * 64;
    v_out[out_base + d0] = qkv[v_base + d0];
    v_out[out_base + d0 + 1] = qkv[v_base + d0 + 1];
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name='idol_rope_rmsnorm_qkv',
            input_names=['qkv', 'cos_t', 'sin_t', 'q_gamma', 'k_gamma', 'scale'],
            output_names=['q_out', 'k_out', 'v_out'],
            source=_SOURCE)
    return _kernel


def supported(qkv: mx.array) -> bool:
    """qkv [B, L, 3, H, D] in fp16 / bf16 with D = 64, on the GPU."""
    return (qkv.ndim == 5 and qkv.shape[-1] == HEAD_DIM and qkv.dtype in (mx.float16, mx.bfloat16)
            and mx.default_device() == mx.gpu)


def rope_rmsnorm_qkv(qkv: mx.array, cos: mx.array, sin: mx.array, q_gamma: mx.array, k_gamma: mx.array,
                     scale: float) -> Tuple[mx.array, mx.array, mx.array]:
    """qkv [B, L, 3, H, 64]; cos, sin [B or 1, L, H, 32] fp32; gammas [H, 64] fp32 ->
    q, k (rotated, then RMS-normalised times scale * gamma) and v, each [B, H, L, 64] in qkv's dtype."""
    B, L, _, H, D = qkv.shape
    cos = mx.broadcast_to(cos.astype(mx.float32), (B, L, H, D // 2))
    sin = mx.broadcast_to(sin.astype(mx.float32), (B, L, H, D // 2))
    shape = (B, H, L, D)
    return tuple(_get_kernel()(
        inputs=[qkv, cos, sin, q_gamma.astype(mx.float32), k_gamma.astype(mx.float32), mx.array([scale], mx.float32)],
        template=[('T', qkv.dtype), ('B', B), ('L', L), ('H', H)],
        grid=(B * L * H * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[shape] * 3, output_dtypes=[qkv.dtype] * 3))

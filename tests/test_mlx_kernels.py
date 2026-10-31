"""The fused RoPE + RMS-norm kernel against the op-by-op MLX path it replaces."""
import mlx.core as mx
import numpy as np
import pytest

from idol.mlx import kernels
from idol.mlx.layers import (MultiHeadRMSNorm, RePo3DRotaryEmbedding, RopeMultiHeadAttention, apply_rotary_emb,
                             cast_linears)


def _reference(qkv, cos, sin, q_norm, k_norm):
    q = q_norm(apply_rotary_emb(qkv[:, :, 0], (cos, sin)))
    k = k_norm(apply_rotary_emb(qkv[:, :, 1], (cos, sin)))
    return tuple(t.transpose(0, 2, 1, 3) for t in (q, k, qkv[:, :, 2]))


@pytest.mark.parametrize('dtype', [mx.float16, mx.bfloat16])
@pytest.mark.parametrize('B, L, H', [(1, 5, 2), (2, 37, 16)])
def test_fused_kernel_matches_the_reference(dtype, B, L, H):
    key = mx.random.split(mx.random.key(0), 5)
    qkv = (2 * mx.random.normal((B, L, 3, H, 64), key=key[0])).astype(dtype)
    angles = 3 * mx.random.normal((B, L, H, 32), key=key[1])
    cos, sin = mx.cos(angles), mx.sin(angles)
    q_norm, k_norm = MultiHeadRMSNorm(64, H), MultiHeadRMSNorm(64, H)
    q_norm.gamma = 1 + 0.1 * mx.random.normal((H, 64), key=key[2])
    k_norm.gamma = 1 + 0.1 * mx.random.normal((H, 64), key=key[3])
    expected = _reference(qkv, cos, sin, q_norm, k_norm)
    actual = kernels.rope_rmsnorm_qkv(qkv, cos, sin, q_norm.gamma, k_norm.gamma, q_norm.scale)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == dtype
        a, e = np.array(a.astype(mx.float32)), np.array(e.astype(mx.float32))
        assert np.abs(a - e).max() <= 2e-2 * np.abs(e).max()          # fp16/bf16 rounding of a few results
        assert np.mean(a != e) < 0.01


def test_attention_uses_the_kernel_in_fp16_and_agrees_with_fp32():
    C, H = 128, 2
    attn, rope = RopeMultiHeadAttention(C, H), RePo3DRotaryEmbedding(C, H, 64)
    x = mx.random.normal((2, 9, C), key=mx.random.key(1))
    expected = attn(x, rope(x))
    assert not kernels.supported(mx.zeros((2, 9, 3, H, 64)))     # fp32: op-by-op path
    cast_linears(attn, mx.float16)
    x16 = x.astype(mx.float16)
    assert kernels.supported(attn.qkv(x16).reshape(2, 9, 3, H, 64))
    actual = attn(x16, rope(x))
    assert np.allclose(np.array(actual.astype(mx.float32)), np.array(expected), atol=2e-2)


def test_unsupported_shapes_fall_back():
    assert not kernels.supported(mx.zeros((1, 4, 3, 2, 32), dtype=mx.float16))

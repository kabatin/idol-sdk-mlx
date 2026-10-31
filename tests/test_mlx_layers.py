"""Each MLX layer against the torch layer it ports, fp32 on both sides with random weights."""
import mlx.core as mx
import numpy as np
import pytest
import torch

from idol.mlx import layers as M
from idol.models import layers as T

from _parity import as_fp32_reference, assert_close, copy_weights, randomize_, to_mx

C, H, D = 128, 2, 64


def _x(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _pair(torch_module, mlx_module, seed=0):
    torch_module = as_fp32_reference(randomize_(torch_module, seed))
    return torch_module, copy_weights(mlx_module, torch_module)


def test_layernorm32_with_affine():
    t, m = _pair(T.LayerNorm32(C, eps=1e-6), M.LayerNorm32(C, eps=1e-6))
    x = _x(2, 7, C)
    assert_close(m(to_mx(x)), t(x))


def test_layernorm32_without_affine_keeps_the_input_dtype():
    m = M.LayerNorm32(C, affine=False, eps=1e-6)
    x = _x(2, 7, C)
    out = m(to_mx(x, mx.float16))
    assert out.dtype == mx.float16
    assert_close(out, T.LayerNorm32(C, elementwise_affine=False, eps=1e-6)(x), rtol=2e-3)


def test_multihead_rms_norm_is_l2_normalise_times_sqrt_d_times_per_head_gamma():
    t, m = _pair(T.MultiHeadRMSNorm(D, H), M.MultiHeadRMSNorm(D, H))
    x = _x(2, 5, H, D)
    assert_close(m(to_mx(x)), t(x))


@pytest.mark.parametrize('cls_t, cls_m, args', [
    (T.FeedForwardNet, M.FeedForwardNet, (C,)),
    (T.MLP, M.MLP, (5, C, C, 2)),
])
def test_mlps(cls_t, cls_m, args):
    t, m = _pair(cls_t(*args), cls_m(*args))
    x = _x(2, 3, args[0])
    assert_close(m(to_mx(x)), t(x))


@pytest.mark.parametrize('steps', [2, 10, 21, 64, 170, 171, 1000])
def test_torch_linspace_is_bit_exact(steps):
    for start, end in ((0.0, 10.0), (1.0, 16.0)):
        assert np.array_equal(M.torch_linspace(start, end, steps),
                              torch.linspace(start, end, steps, dtype=torch.float32).numpy())


def _assert_within_one_ulp(actual, expected):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    assert np.all(np.abs(actual.view(np.int32) - expected.view(np.int32)) <= 1)


@pytest.mark.parametrize('cls_t, cls_m, shape', [
    (T.PcdAbsolutePositionEmbedder, M.PcdAbsolutePositionEmbedder, (2, 9, 3)),
    (T.PcdAbsolutePositionEmbedderv2, M.PcdAbsolutePositionEmbedderv2, (11, 3)),
])
@pytest.mark.parametrize('channels', [C, 1024])
def test_point_position_embedders(cls_t, cls_m, shape, channels):
    """The frequencies agree to an ulp (torch's pow/linspace are not correctly rounded); given the same
    frequencies, the embeddings agree."""
    t, m = cls_t(channels), cls_m(channels)
    _assert_within_one_ulp(m._freqs, t.freqs.numpy())
    m._freqs = t.freqs.numpy()
    x = torch.rand(*shape, generator=torch.Generator().manual_seed(1))
    assert_close(m(to_mx(x)), t(x), atol=1e-5)


@pytest.mark.parametrize('kwargs', [{}, {'max_period': 1024, 'two_pi': True}])
def test_sinusoidal_embedder(kwargs):
    t, m = _pair(T.SinusoidalEmbedder(C, **kwargs), M.SinusoidalEmbedder(C, **kwargs))
    x = torch.tensor([3.0, 517.0])
    assert_close(m(to_mx(x)), t(x))


def test_repo_rotary_embedding_matches_the_complex_torch_version():
    t, m = _pair(T.RePo3DRotaryEmbedding(C, num_heads=H, head_dim=D), M.RePo3DRotaryEmbedding(C, num_heads=H, head_dim=D))
    x = _x(2, 6, C)
    q = _x(2, 6, H, D, seed=1)
    expected = T.apply_rotary_emb(q, t(x))            # complex path (real_valued=False)
    actual = M.apply_rotary_emb(to_mx(q), m(to_mx(x)))
    assert_close(actual, expected)


def test_rope_attention_rotates_before_the_qk_norm():
    t, m = _pair(T.RopeMultiHeadAttention(C, H), M.RopeMultiHeadAttention(C, H))
    rope_t, rope_m = _pair(T.RePo3DRotaryEmbedding(C, H, D), M.RePo3DRotaryEmbedding(C, H, D), seed=3)
    x = _x(2, 6, C)
    assert_close(m(to_mx(x), rope_m(to_mx(x))), t(x, rope_t(x)))


@pytest.mark.parametrize('cross', [False, True])
def test_multihead_attention(cross):
    kwargs = {'ctx_channels': 16, 'cross': True} if cross else {}
    t, m = _pair(T.MultiHeadAttention(C, H, **kwargs), M.MultiHeadAttention(C, H, **kwargs))
    x, ctx = _x(2, 6, C), _x(2, 9, 16, seed=1)
    if cross:
        assert_close(m(to_mx(x), to_mx(ctx)), t(x, ctx))
    else:
        assert_close(m(to_mx(x)), t(x))


@pytest.mark.parametrize('modulation', [True, False])
def test_unified_transformer_block(modulation):
    t, m = _pair(T.UnifiedTransformerBlock(C, H, modulation=modulation),
                 M.UnifiedTransformerBlock(C, H, modulation=modulation))
    rope_t, rope_m = _pair(T.RePo3DRotaryEmbedding(C, H, D), M.RePo3DRotaryEmbedding(C, H, D), seed=3)
    x, mod = _x(2, 6, C), _x(2, 6 * C, seed=2) * 0.1
    expected = t(x, mod if modulation else None, rope_t(x))
    actual = m(to_mx(x), to_mx(mod) if modulation else None, rope_m(to_mx(x)))
    assert_close(actual, expected)


def test_transformer_cross_block():
    t, m = _pair(T.TransformerCrossBlock(C, 16, H), M.TransformerCrossBlock(C, 16, H))
    x, ctx = _x(2, 6, C), _x(2, 9, 16, seed=1)
    assert_close(m(to_mx(x), to_mx(ctx)), t(x, ctx))


def test_modulated_cross_only_block():
    t, m = _pair(T.ModulatedTransformerCrossOnlyBlock(C, 16, H), M.ModulatedTransformerCrossOnlyBlock(C, 16, H))
    x, ctx, mod = _x(2, 6, C), _x(2, 9, 16, seed=1), _x(2, 6 * C, seed=2) * 0.1
    assert_close(m(to_mx(x), to_mx(mod), to_mx(ctx)), t(x, mod, ctx))


def test_fp16_blocks_stay_close_to_fp32():
    """The production dtype policy: fp16 Linear weights and activations in the blocks."""
    t, m = _pair(T.UnifiedTransformerBlock(C, H), M.UnifiedTransformerBlock(C, H))
    rope_t, rope_m = _pair(T.RePo3DRotaryEmbedding(C, H, D), M.RePo3DRotaryEmbedding(C, H, D), seed=3)
    M.cast_linears(m, mx.float16)
    x, mod = _x(2, 6, C), _x(2, 6 * C, seed=2) * 0.1
    actual = m(to_mx(x, mx.float16), to_mx(mod, mx.float16), rope_m(to_mx(x)))
    assert actual.dtype == mx.float16
    assert_close(actual, t(x, mod, rope_t(x)), rtol=1e-2)

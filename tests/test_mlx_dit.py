"""The MLX denoiser and sampler against torch, on a small random-weight configuration (fp32 both sides)."""
import mlx.core as mx
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from idol.mlx.dit import LatentSeqMMFlowModel as MDiT
from idol.mlx.sampler import sample_flow as m_sample_flow
from idol.mlx.sampler import time_grid
from idol.models.dit import LatentSeqMMFlowModel as TDiT
from idol.sampler import sample_flow as t_sample_flow

from _parity import as_fp32_reference, assert_close, copy_weights, randomize_, to_mx

SMALL = dict(q_token_length=64, in_channels=16, cam_channels=5, model_channels=128, cond_channels=32,
             cond2_channels=8, num_blocks=2, num_refiner_blocks=2, num_heads=2, num_head_channels=64)
L_COND = 7


def _models(dtype=mx.float32):
    t = as_fp32_reference(randomize_(TDiT(**SMALL), seed=1))
    m = copy_weights(MDiT(**SMALL, dtype=dtype), t)
    return t, m


def _inputs(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = TensorDict({'latent': torch.randn(batch, 64, 16, generator=g), 'camera': torch.randn(batch, 1, 5, generator=g)},
                   batch_size=batch)
    cond = TensorDict({'feature1': torch.randn(batch, L_COND, 32, generator=g),
                       'feature2': torch.randn(batch, L_COND, 8, generator=g)}, batch_size=batch)
    return x, cond


def test_forward_matches_torch():
    t, m = _models()
    x, cond = _inputs()
    tt = torch.tensor([830.0, 12.5])
    expected = t(x, tt, cond)
    v_latent, v_camera = m(to_mx(x['latent']), to_mx(x['camera']), to_mx(tt),
                           to_mx(cond['feature1']), to_mx(cond['feature2']))
    assert_close(v_latent, expected['latent'])
    assert_close(v_camera, expected['camera'])


def test_refined_condition_is_reusable_across_steps():
    """The condition refiner depends on neither the time nor the noise: refine once, denoise many times."""
    _, m = _models()
    x, cond = _inputs()
    f1, f2 = to_mx(cond['feature1']), to_mx(cond['feature2'])
    h_cond = m.refine_condition(f1, f2)
    for t in (999.0, 3.0):
        tt = mx.array([t, t])
        direct = m(to_mx(x['latent']), to_mx(x['camera']), tt, f1, f2)
        reused = m.denoise(to_mx(x['latent']), to_mx(x['camera']), tt, h_cond)
        for a, b in zip(direct, reused):
            assert_close(a, b, rtol=0, atol=0)


def test_fp16_forward_stays_close_to_fp32():
    t, m = _models(dtype=mx.float16)
    x, cond = _inputs()
    tt = torch.tensor([500.0, 500.0])
    expected = t(x, tt, cond)
    v_latent, _ = m(to_mx(x['latent']), to_mx(x['camera']), to_mx(tt), to_mx(cond['feature1']), to_mx(cond['feature2']))
    assert v_latent.dtype == mx.float32
    assert_close(v_latent, expected['latent'], rtol=2e-2)


def test_time_grid_matches_the_torch_sampler():
    grid = time_grid(50, 3.0)
    t_seq = np.linspace(1, 0, 51)
    assert np.array_equal(grid, 3.0 * t_seq / (1 + 2.0 * t_seq))


@pytest.mark.parametrize('cfg_strength', [0.0, 3.0])
def test_sampling_matches_torch(cfg_strength):
    t, m = _models()
    x, cond = _inputs(batch=1, seed=3)
    neg = cond.apply(torch.zeros_like)
    expected = t_sample_flow(t, x, cond=cond, neg_cond=neg, steps=3, rescale_t=3.0, cfg_strength=cfg_strength,
                             verbose=False, batch_cfg=False)
    latent, camera = m_sample_flow(m, to_mx(x['latent']), to_mx(x['camera']),
                                   cond=(to_mx(cond['feature1']), to_mx(cond['feature2'])),
                                   neg_cond=(to_mx(neg['feature1']), to_mx(neg['feature2'])),
                                   steps=3, rescale_t=3.0, cfg_strength=cfg_strength, verbose=False)
    assert_close(latent, expected['latent'], rtol=2e-4)
    assert_close(camera, expected['camera'], rtol=2e-4)

"""The MLX octree and Gaussian decoders against torch (small random-weight configs, fp32 both sides)."""
import mlx.core as mx
import numpy as np
import pytest
import torch

from idol.gaussian import Gaussian as TGaussian
from idol.mlx import decoder as M
from idol.mlx.gaussian import Gaussian as MGaussian
from idol.models import decoder as T

from _parity import as_fp32_reference, assert_close, copy_weights, randomize_, to_mx, to_np

REP = {'num_gaussians': 32, 'lr': {'_xyz': 1.0, '_features_dc': 1.0, '_opacity': 1.0, '_scaling': 1.0,
                                   '_rotation': 0.1},
       'perturbe_size': 1.5, '3d_filter_kernel_size': 0.0009, 'scaling_bias': 0.004, 'opacity_bias': 0.1}
OCTREE = dict(model_channels=128, cond_channels=16, num_blocks=2, num_heads=2)
GS = dict(in_channels=3, model_channels=128, cond_channels=16, num_blocks=2, num_heads=2, representation_config=REP)
U = 0.37   # the constant every torch.rand / rand_like returns in the exact-match tests


@pytest.fixture
def constant_torch_rand(monkeypatch):
    real_full, real_full_like = torch.full, torch.full_like
    monkeypatch.setattr(torch, 'rand', lambda *size, device=None, **_: real_full(
        tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else size, U, device=device))
    monkeypatch.setattr(torch, 'rand_like', lambda x, dtype=None, **_: real_full_like(x, U, dtype=dtype))


def _latent(seed=0, length=40):
    return torch.randn(1, length, 16, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize('seed', range(4))
def test_systematic_sample_counts_match_torch_exactly(constant_torch_rand, seed):
    g = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(1, 37, 8, generator=g) * 3, dim=-1)
    probs[0, 5] = 0          # a zero row falls back to uniform
    counts = torch.randint(0, 40, (1, 37), generator=g)
    expected = T.systematic_sample(probs, counts)
    actual = M.systematic_sample(probs[0].numpy(), counts[0].numpy(), np.full(37, U, dtype=np.float32))
    assert np.array_equal(actual, expected[0].numpy())
    assert actual.sum() == counts.sum().item()


def test_octree_forward_matches_torch():
    t = as_fp32_reference(randomize_(T.OctreeProbabilityFixedlenDecoder(**OCTREE), seed=2))
    m = copy_weights(M.OctreeProbabilityFixedlenDecoder(**OCTREE, dtype=mx.float32), t)
    x, level, cond = torch.rand(1, 13, 3), torch.tensor([8]), _latent()
    assert_close(m(to_mx(x), to_mx(level), to_mx(cond)), t(x, level, cond))


def test_octree_sampling_matches_torch(constant_torch_rand):
    t = as_fp32_reference(randomize_(T.OctreeProbabilityFixedlenDecoder(**OCTREE), seed=2))
    m = copy_weights(M.OctreeProbabilityFixedlenDecoder(**OCTREE, dtype=mx.float32), t)
    cond = _latent(seed=1)
    expected = T.OctreeProbabilityFixedlenDecoder.sample(t, cond, num_points=300, level=5)
    actual = M.sample_anchors(m, to_mx(cond), num_points=300, level=5, uniform=lambda shape: np.full(shape, U))
    assert actual.shape == (1, 300, 3)
    assert np.array_equal(to_np(actual), to_np(expected))


def test_octree_sampling_with_a_generator_is_seeded_and_inside_the_unit_cube():
    m = M.OctreeProbabilityFixedlenDecoder(**OCTREE, dtype=mx.float32)
    cond = to_mx(_latent())
    a = to_np(M.sample_anchors(m, cond, 200, 4, uniform=np.random.default_rng(5).random))
    b = to_np(M.sample_anchors(m, cond, 200, 4, uniform=np.random.default_rng(5).random))
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 1


def _gs_pair():
    t = as_fp32_reference(randomize_(T.GaussianFixedlenDecoder(**GS), seed=4))
    return t, copy_weights(M.GaussianFixedlenDecoder(**GS, dtype=mx.float32), t)


def test_gaussian_decoder_matches_torch():
    t, m = _gs_pair()
    points, cond = torch.rand(1, 20, 3), _latent(seed=2)
    features_t = t(points, cond)
    features_m = m(to_mx(points), to_mx(cond))
    assert_close(features_m, features_t)

    expected = t.to_representation(points, features_t)[0]
    actual = m.to_representation(to_mx(points), features_m)
    for name in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        assert_close(getattr(actual, name), getattr(expected, name), rtol=1e-4, atol=1e-6)


def test_gaussian_ply_matches_the_torch_writer(tmp_path):
    from plyfile import PlyData
    t, m = _gs_pair()
    points, cond = torch.rand(1, 20, 3), _latent(seed=2)
    t.to_representation(points, t(points, cond))[0].save_ply(str(tmp_path / 't.ply'))
    m.to_representation(to_mx(points), m(to_mx(points), to_mx(cond))).save_ply(str(tmp_path / 'm.ply'))
    vt, vm = (PlyData.read(str(tmp_path / f'{n}.ply'))['vertex'].data for n in 'tm')
    assert vt.dtype == vm.dtype and len(vt) == len(vm) == 20 * 32
    for name in vt.dtype.names:
        assert_close(vm[name], vt[name], rtol=1e-4, atol=1e-5)


def test_gaussian_activation_handles_saturated_values(tmp_path):
    n = 4
    g = MGaussian.from_raw(raw_xyz=mx.zeros((n, 3)), features_dc=mx.zeros((n, 1, 3)),
                           raw_scaling=mx.full((n, 3), -200.0), raw_rotation=mx.zeros((n, 4)),
                           raw_opacity=mx.full((n, 1), 200.0), scaling_bias=0.004, opacity_bias=0.1,
                           minimum_kernel_size=0.0009)
    g.save_ply(str(tmp_path / 'g.ply'))
    from plyfile import PlyData
    data = PlyData.read(str(tmp_path / 'g.ply'))['vertex'].data
    assert all(np.isfinite(data[name]).all() for name in data.dtype.names)
    assert len(g) == n

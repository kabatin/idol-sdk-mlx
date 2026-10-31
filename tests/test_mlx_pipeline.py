"""IdolMLXPipeline wiring, with small random components (no downloads)."""
import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

from idol.mlx.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
from idol.mlx.dit import LatentSeqMMFlowModel
from idol.mlx.pipeline import IdolMLXPipeline, IdolOutput

REP = {'num_gaussians': 32, 'lr': {'_xyz': 1.0, '_features_dc': 1.0, '_opacity': 1.0, '_scaling': 1.0,
                                   '_rotation': 0.1},
       'perturbe_size': 1.5, '3d_filter_kernel_size': 0.0009, 'scaling_bias': 0.004, 'opacity_bias': 0.1}


class FakeEncoder:
    """Image tokens that depend on the image and on the key, like the real encoder's."""
    def __call__(self, images, key=None):
        mean = mx.mean(images)
        f1 = mx.full((1, 7, 32), mean) + 0.1 * mx.random.normal((1, 7, 32), key=mx.random.key(0))
        f2 = 0.1 * mx.random.normal((1, 7, 8), key=key)
        return f1, f2


class FakeBackgroundRemover:
    def remove_background(self, image):
        alpha = np.zeros(image.size[::-1], dtype=np.uint8)
        alpha[10:-10, 15:-15] = 255
        rgba = image.copy()
        rgba.putalpha(Image.fromarray(alpha, mode='L'))
        return rgba


@pytest.fixture(scope='module')
def pipeline():
    def randomized(module, seed):
        from mlx.utils import tree_flatten, tree_unflatten
        params = tree_flatten(module.parameters())
        keys = mx.random.split(mx.random.key(seed), len(params))
        module.update(tree_unflatten([(k, (0.05 * mx.random.normal(v.shape, key=key)).astype(v.dtype))
                                      for (k, v), key in zip(params, keys)]))
        return module

    denoiser = randomized(LatentSeqMMFlowModel(q_token_length=64, in_channels=16, cam_channels=5, model_channels=128,
                                               cond_channels=32, cond2_channels=8, num_blocks=2, num_heads=2), 1)
    denoiser.pos_pe = mx.random.uniform(shape=(1, 64, 3), key=mx.random.key(2))
    decoder = randomized(OctreeProbabilityFixedlenDecoder(model_channels=128, cond_channels=16, num_blocks=2,
                                                          num_heads=2), 3)
    decoder_gs = randomized(GaussianFixedlenDecoder(in_channels=3, model_channels=128, cond_channels=16, num_blocks=2,
                                                    num_heads=2, representation_config=REP), 4)
    return IdolMLXPipeline(config={'image_size': 64, 'max_voxel_level': 4}, denoiser=denoiser, decoder=decoder,
                           decoder_gs=decoder_gs, background_remover=FakeBackgroundRemover(),
                           image_encoder=FakeEncoder(), compile=False)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((80, 120, 3)) * 255).astype(np.uint8))


def test_generates_gaussians_a_ply_and_a_camera(pipeline, image, tmp_path):
    output = pipeline(image, steps=2, num_gaussians=32 * 50)
    assert isinstance(output, IdolOutput)
    assert len(output.gaussian) == 32 * 50
    assert output.preprocessed_image.size == (64, 64) and output.preprocessed_image.mode == 'RGB'
    assert set(output.camera) == {'yaw', 'pitch', 'radius', 'fov'}
    output.gaussian.save_ply(str(tmp_path / 'object.ply'))
    vertex = PlyData.read(str(tmp_path / 'object.ply'))['vertex']
    assert vertex.count == 32 * 50
    assert all(np.isfinite(vertex[name]).all() for name in vertex.data.dtype.names)


def test_same_seed_same_object_new_seed_new_object(pipeline, image):
    a, b, c = (pipeline(image, steps=2, num_gaussians=640, seed=s) for s in (5, 5, 6))
    assert mx.array_equal(a.gaussian.xyz, b.gaussian.xyz)
    assert not mx.array_equal(a.gaussian.xyz, c.gaussian.xyz)


def test_compiled_sampling_matches_eager(pipeline, image):
    eager = pipeline(image, steps=2, num_gaussians=640, cfg_strength=0.0)
    pipeline.compile = True
    try:
        compiled = pipeline(image, steps=2, num_gaussians=640, cfg_strength=0.0)
    finally:
        pipeline.compile = False
    assert np.allclose(np.array(compiled.gaussian.xyz), np.array(eager.gaussian.xyz), atol=1e-4)


def test_accepts_a_path_and_the_crop_override(pipeline, image, tmp_path):
    path = tmp_path / 'in.png'
    image.save(path)
    assert pipeline.preprocess(str(path), mode='real', crop_margin=1.0).size == (64, 64)


@pytest.mark.parametrize('kwargs, message', [({'mode': 'photo'}, 'mode'), ({'steps': 0}, 'steps')])
def test_rejects_bad_arguments(pipeline, image, kwargs, message):
    with pytest.raises(ValueError, match=message):
        pipeline(image, **kwargs)


def test_lazy_exports():
    import idol.mlx
    assert idol.mlx.IdolMLXPipeline is IdolMLXPipeline
    with pytest.raises(AttributeError):
        idol.mlx.nothing_here


def test_restores_the_callers_cache_limit(pipeline, image):
    from idol.mlx import pipeline as P
    callers = 3 * 2 ** 30
    mx.set_cache_limit(callers)
    pipeline(image, steps=1, num_gaussians=640)
    assert mx.set_cache_limit(callers) == callers       # set_cache_limit returns the limit it replaces
    assert P.CACHE_LIMIT < callers


@pytest.mark.parametrize('seed', [-1, -(2 ** 63), 2 ** 64 + 5])
def test_any_integer_seed_works_and_is_reproducible(pipeline, image, seed):
    a, b = (pipeline(image, steps=1, num_gaussians=640, seed=seed) for _ in range(2))
    assert mx.array_equal(a.gaussian.xyz, b.gaussian.xyz)


def test_overlapping_calls_keep_the_cap_and_restore_the_callers_limit():
    """A enters, B enters, A leaves while B still runs: B must stay capped, and the caller's limit comes back."""
    import threading
    from idol.mlx import pipeline as P
    callers = 3 * 2 ** 30
    mx.set_cache_limit(callers)
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def a():
        with P._bounded_cache():
            a_in.set()
            b_in.wait()
        a_out.set()

    def b():
        a_in.wait()
        with P._bounded_cache():
            b_in.set()
            a_out.wait()
            limit = mx.set_cache_limit(P.CACHE_LIMIT)     # returns the limit in force
            mx.set_cache_limit(limit)
            seen['b after a left'] = limit

    threads = [threading.Thread(target=a), threading.Thread(target=b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen['b after a left'] == P.CACHE_LIMIT
    assert mx.set_cache_limit(callers) == callers


def test_warns_when_tf32_is_on(monkeypatch):
    from idol.mlx import pipeline as P
    monkeypatch.setattr(P, '_fp32_matmul_error', lambda: 1e-3)
    with pytest.warns(RuntimeWarning, match='TF32'):
        P._check_fp32_matmuls()
    monkeypatch.setattr(P, '_fp32_matmul_error', lambda: 1e-7)
    P._check_fp32_matmuls()     # no warning


def test_from_pretrained_loads_a_local_folder(tmp_path, monkeypatch):
    """The real loading code (config keys, constructors, safetensors names) with small random torch modules."""
    import json
    import torch
    from safetensors.torch import save_file
    from idol.mlx import pipeline as P
    from idol.models import GaussianFixedlenDecoder as TGS, LatentSeqMMFlowModel as TDiT
    from idol.models import OctreeProbabilityFixedlenDecoder as TOctree
    from idol.models.birefnet import BiRefNet as TBiRefNet
    from _parity import randomize_
    config = {
        'denoiser': {'args': dict(q_token_length=64, in_channels=16, cam_channels=5, model_channels=128,
                                  cond_channels=32, cond2_channels=8, num_blocks=2, num_heads=2)},
        'decoder': {'args': dict(model_channels=128, cond_channels=16, num_blocks=2, num_heads=2)},
        'decoder_gs': {'args': dict(in_channels=3, model_channels=128, cond_channels=16, num_blocks=2, num_heads=2,
                                    representation_config=REP)},
        'image_size': 64, 'max_voxel_level': 4,
        'image_encoders': {'dinov3': 'example/dinov3', 'flux2_vae': 'example/flux2'},
    }
    (tmp_path / 'config.json').write_text(json.dumps(config))
    modules = {'denoiser': TDiT(**config['denoiser']['args']), 'decoder': TOctree(**config['decoder']['args']),
               'decoder_gs': TGS(**config['decoder_gs']['args']), 'birefnet': TBiRefNet()}
    for name, module in modules.items():
        state = randomize_(module, seed=1).state_dict()
        save_file({k: v.contiguous() for k, v in state.items()}, str(tmp_path / f'{name}.safetensors'))
    checked, encoders = [], []
    monkeypatch.setattr(P, 'require_encoder_access', checked.append)
    monkeypatch.setattr(P.ImageEncoder, 'from_pretrained',
                        classmethod(lambda cls, *args: encoders.append(args) or FakeEncoder()))
    loaded = P.IdolMLXPipeline.from_pretrained(str(tmp_path), dinov3_dtype='fp32', compile=False)
    assert checked == [['example/dinov3', 'example/flux2']]
    assert encoders == [('example/dinov3', 'example/flux2', 'fp32')]
    assert loaded.denoiser.q_token_length == 64 and loaded.compile is False
    for name in ('denoiser', 'decoder', 'decoder_gs'):
        expected = modules[name].state_dict()['out_layer.bias' if name == 'denoiser' else 'out_proj.bias']
        actual = getattr(loaded, name)['out_layer' if name == 'denoiser' else 'out_proj']['bias']
        assert np.allclose(np.array(actual), expected.numpy())

"""The MLX denoiser and decoders with the released weights (speridlabs/idol, ungated, ~3 GB) against torch fp32.

Run with: pytest -m weights"""
import json
import os
import time

import mlx.core as mx
import numpy as np
import pytest
import torch

from idol.mlx.decoder import GaussianFixedlenDecoder as MGS
from idol.mlx.decoder import OctreeProbabilityFixedlenDecoder as MOctree
from idol.mlx.dit import LatentSeqMMFlowModel as MDiT
from idol.mlx.weights import load_torch_weights
from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
from safetensors.torch import load_file

from _parity import as_fp32_reference, assert_close, assert_rms_close, to_mx

# fp32: max error vs torch fp32; fp16: relative RMS error (torch's own fp16 run is at ~5e-3 on the Gaussian decoder)
FP16_RMS = 1e-2

pytestmark = pytest.mark.weights


@pytest.fixture(scope='module')
def weights_dir():
    from huggingface_hub import snapshot_download
    from idol.pipeline import WEIGHT_FILES
    return snapshot_download('speridlabs/idol', allow_patterns=WEIGHT_FILES)


@pytest.fixture(scope='module')
def config(weights_dir):
    with open(os.path.join(weights_dir, 'config.json')) as f:
        return json.load(f)


def _torch(cls, config, weights_dir, name):
    """fp32 before loading: the torch modules cast their blocks to fp16 in __init__, and loading into those would
    round the fp32 checkpoint (.float() afterwards cannot undo it)."""
    model = as_fp32_reference(cls(**config[name]['args']))
    model.load_state_dict(load_file(os.path.join(weights_dir, f'{name}.safetensors')), strict=True)
    return model


def _mlx(cls, config, weights_dir, name, dtype):
    return load_torch_weights(cls(**config[name]['args'], dtype=dtype), os.path.join(weights_dir, f'{name}.safetensors'))


def _condition(seed=0, length=4101):
    g = torch.Generator().manual_seed(seed)
    feature1 = torch.nn.functional.layer_norm(torch.randn(1, length, 1280, generator=g), (1280,))  # DINOv3: normed
    feature2 = torch.randn(1, length, 128, generator=g)
    feature2[:, :5] = 0                                                         # FLUX tokens: 5 zero pads
    return feature1, feature2


def test_denoiser(config, weights_dir):
    from tensordict import TensorDict
    g = torch.Generator().manual_seed(1)
    latent, camera = torch.randn(1, 8192, 16, generator=g), torch.randn(1, 1, 5, generator=g)
    feature1, feature2 = _condition()
    t = torch.tensor([700.0])

    reference = _torch(LatentSeqMMFlowModel, config, weights_dir, 'denoiser')
    with torch.no_grad():
        expected = reference(TensorDict({'latent': latent, 'camera': camera}, batch_size=1), t,
                             TensorDict({'feature1': feature1, 'feature2': feature2}, batch_size=1))
    del reference

    args = [to_mx(a) for a in (latent, camera, t, feature1, feature2)]
    for dtype in (mx.float32, mx.float16):
        model = _mlx(MDiT, config, weights_dir, 'denoiser', dtype)
        v_latent, v_camera = model(*args)
        mx.eval(v_latent, v_camera)
        check = (lambda a, b: assert_close(a, b, rtol=2e-3)) if dtype == mx.float32 else \
            (lambda a, b: assert_rms_close(a, b, FP16_RMS))
        check(v_latent, expected['latent'])
        check(v_camera, expected['camera'])
        if dtype == mx.float16:   # one guided step: batch 2, as the sampler runs it
            h_cond = model.refine_condition(mx.concatenate([args[3]] * 2), mx.concatenate([args[4]] * 2))
            batch = (mx.concatenate([args[0]] * 2), mx.concatenate([args[1]] * 2), mx.full((2,), 700.0))
            mx.eval(model.denoise(*batch, h_cond))
            start = time.perf_counter()
            mx.eval(model.denoise(*batch, h_cond))
            print(f'\nMLX fp16 guided denoiser step: {time.perf_counter() - start:.2f} s')
        del model


def test_octree_decoder(config, weights_dir):
    latent = torch.randn(1, 8192, 16, generator=torch.Generator().manual_seed(2))
    x = torch.rand(1, 4096, 3, generator=torch.Generator().manual_seed(3))
    level = torch.tensor([256])
    expected = _torch(OctreeProbabilityFixedlenDecoder, config, weights_dir, 'decoder')(x, level, latent)
    assert_close(_mlx(MOctree, config, weights_dir, 'decoder', mx.float32)(to_mx(x), to_mx(level), to_mx(latent)),
                 expected, rtol=1e-3)
    assert_rms_close(_mlx(MOctree, config, weights_dir, 'decoder', mx.float16)(to_mx(x), to_mx(level), to_mx(latent)),
                     expected, FP16_RMS)


def test_gaussian_decoder(config, weights_dir):
    latent = torch.randn(1, 8192, 16, generator=torch.Generator().manual_seed(2))
    points = torch.rand(1, 8192, 3, generator=torch.Generator().manual_seed(4))
    reference = _torch(GaussianFixedlenDecoder, config, weights_dir, 'decoder_gs')
    with torch.no_grad():
        features = reference(points, latent)
        expected = reference.to_representation(points, features)[0]
    for dtype in (mx.float32, mx.float16):
        model = _mlx(MGS, config, weights_dir, 'decoder_gs', dtype)
        actual_features = model(to_mx(points), to_mx(latent))
        actual = model.to_representation(to_mx(points), actual_features)
        pairs = ((actual_features, features), (actual.xyz, expected.xyz), (actual.opacity, expected.opacity))
        for a, b in pairs:
            if dtype == mx.float32:
                assert_close(a, b, rtol=2e-3)
            else:
                assert_rms_close(a, b, FP16_RMS)

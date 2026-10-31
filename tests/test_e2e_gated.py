"""End to end against torch with the released weights and the gated encoders: the same preprocessed image and the
same noise through both backends. Run with: pytest -m gated tests/test_e2e_gated.py (a few minutes on CPU)."""
import json
import os

import mlx.core as mx
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from _parity import assert_rms_close, to_mx

pytestmark = pytest.mark.gated


@pytest.fixture(scope='module')
def weights_dir():
    from huggingface_hub import snapshot_download
    from idol.pipeline import WEIGHT_FILES
    return snapshot_download('speridlabs/idol', allow_patterns=WEIGHT_FILES)


@pytest.fixture(scope='module')
def config(weights_dir):
    with open(os.path.join(weights_dir, 'config.json')) as f:
        return json.load(f)


@pytest.fixture(scope='module')
def prepared(weights_dir):
    from PIL import Image
    from idol.mlx.birefnet import BiRefNet
    from idol.preprocess import preprocess_image
    remover = BiRefNet.from_safetensors(os.path.join(weights_dir, 'birefnet.safetensors'))
    return preprocess_image(Image.open('demo/examples/fig-fox.webp'), background_remover=remover, crop_margin=1.2)


@pytest.fixture(scope='module')
def conditions(config, prepared, monkeypatch_module):
    """(torch features, MLX features) of the same image; the VAE posterior is taken at its mean on both sides."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from idol.conditioning import ImageEncoder as TorchEncoder
    from idol.mlx.conditioning import ImageEncoder as MLXEncoder
    encoders = config['image_encoders']
    pixels = np.asarray(prepared, dtype=np.float32) / 255.0

    monkeypatch_module.setattr(DiagonalGaussianDistribution, 'sample', lambda self, generator=None: self.mean)
    torch_encoder = TorchEncoder(encoders['dinov3'], encoders['flux2_vae'], dinov3_dtype='fp32', device='cpu')
    expected = torch_encoder(torch.tensor(pixels).permute(2, 0, 1)[None])
    del torch_encoder

    mlx_encoder = MLXEncoder.from_pretrained(encoders['dinov3'], encoders['flux2_vae'])
    images = mx.array(pixels)[None]
    pixels_norm = (images - mx.array([0.485, 0.456, 0.406])) / mx.array([0.229, 0.224, 0.225])
    from idol.mlx.layers import layer_norm
    feature1 = layer_norm(mlx_encoder.dinov3(pixels_norm).astype(mx.float32))
    mean_only = mx.zeros((1, 128, 128, 32))
    feature2 = mlx_encoder.flux2_vae.tokens(images, noise=mean_only)
    return expected, (feature1, feature2)


@pytest.fixture(scope='module')
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_conditioning_tokens_match_torch(conditions):
    expected, (feature1, feature2) = conditions
    assert feature1.shape == tuple(expected['feature1'].shape) == (1, 4101, 1280)
    assert feature2.shape == tuple(expected['feature2'].shape) == (1, 4101, 128)
    assert_rms_close(feature1, expected['feature1'], 5e-2)    # MLX DINOv3 in bf16 against torch fp32
    assert_rms_close(feature2, expected['feature2'], 5e-2)    # bf16 VAE on both sides


def test_two_guided_steps_match_torch(config, weights_dir, conditions):
    from safetensors.torch import load_file
    from idol.mlx.dit import LatentSeqMMFlowModel as MDiT
    from idol.mlx.sampler import sample_flow as m_sample_flow
    from idol.mlx.weights import load_torch_weights
    from idol.models import LatentSeqMMFlowModel
    from idol.sampler import sample_flow as t_sample_flow
    from _parity import as_fp32_reference

    expected_cond, mlx_cond = conditions
    g = torch.Generator().manual_seed(0)
    noise = TensorDict({'latent': torch.randn(1, 8192, 16, generator=g), 'camera': torch.randn(1, 1, 5, generator=g)},
                       batch_size=1)
    reference = as_fp32_reference(LatentSeqMMFlowModel(**config['denoiser']['args']))
    reference.load_state_dict(load_file(os.path.join(weights_dir, 'denoiser.safetensors')))
    expected = t_sample_flow(reference, noise, cond=expected_cond, neg_cond=expected_cond.apply(torch.zeros_like),
                             steps=2, verbose=False)
    del reference

    model = load_torch_weights(MDiT(**config['denoiser']['args']), os.path.join(weights_dir, 'denoiser.safetensors'))
    latent, camera = m_sample_flow(model, to_mx(noise['latent']), to_mx(noise['camera']), cond=mlx_cond,
                                   neg_cond=tuple(mx.zeros_like(c) for c in mlx_cond), steps=2, verbose=False,
                                   compile=True)
    assert_rms_close(latent, expected['latent'], 5e-2)
    assert_rms_close(camera, expected['camera'], 5e-2)

"""The MLX FLUX.2 VAE encoder against diffusers' AutoencoderKLFlux2 (small random config; real weights with -m gated)."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from idol.conditioning import ImageEncoder
from idol.mlx.flux2_vae import Flux2VAEEncoder

from _parity import assert_close, assert_rms_close, copy_weights, randomize_, to_mx

SMALL = dict(block_out_channels=(32, 64), layers_per_block=1, latent_channels=4, norm_num_groups=8)
TORCH_SMALL = dict(SMALL, down_block_types=('DownEncoderBlock2D',) * 2, up_block_types=('UpDecoderBlock2D',) * 2)


def _pair(dtype=mx.float32, seed=0):
    vae = randomize_(AutoencoderKLFlux2(**TORCH_SMALL), seed=seed).eval()
    m = copy_weights(Flux2VAEEncoder(**SMALL, dtype=dtype), vae, rename=Flux2VAEEncoder.weight_name)
    return vae, m


def _images(size=32, seed=0):
    return torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed))


def nhwc(t):
    return to_mx(t.permute(0, 2, 3, 1))


def test_moments_match_diffusers_fp32():
    vae, m = _pair()
    x = _images() * 2 - 1
    with torch.no_grad():
        dist = vae.encode(x).latent_dist
    mean, logvar = m.moments(nhwc(x))
    assert_close(mean, dist.mean.permute(0, 2, 3, 1))
    assert_close(logvar, dist.logvar.permute(0, 2, 3, 1))


def test_tokens_match_the_torch_conditioning_path(monkeypatch):
    """bf16 end to end, as idol.conditioning runs it, with the same sampling noise on both sides."""
    vae, m = _pair(dtype=mx.bfloat16, seed=1)
    vae = vae.to(torch.bfloat16)
    images = _images(seed=2)
    noise = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3))
    monkeypatch.setattr(DiagonalGaussianDistribution, 'sample',
                        lambda self, generator=None: self.mean + self.std * noise.to(self.mean.dtype))
    with torch.no_grad():
        expected = ImageEncoder._flux2_tokens(SimpleNamespace(flux2_vae=vae), images)
    actual = m.tokens(nhwc(images), noise=nhwc(noise))
    assert actual.shape == expected.shape == (1, 5 + 8 * 8, 16)
    assert actual.dtype == mx.float32
    assert np.all(np.array(actual[:, :5]) == 0)
    assert_rms_close(actual, expected, 2e-2)


def test_tokens_sample_with_a_seeded_key():
    _, m = _pair()
    images = nhwc(_images())
    a, b = m.tokens(images, key=mx.random.key(7)), m.tokens(images, key=mx.random.key(7))
    assert mx.array_equal(a, b)
    assert not mx.array_equal(a, m.tokens(images, key=mx.random.key(8)))


def test_weight_names_drop_the_decoder():
    assert Flux2VAEEncoder.weight_name('decoder.conv_in.weight') is None
    assert Flux2VAEEncoder.weight_name('post_quant_conv.bias') is None
    assert Flux2VAEEncoder.weight_name('encoder.conv_in.weight') == 'encoder.conv_in.weight'
    assert Flux2VAEEncoder.weight_name('bn.running_var') == 'bn.running_var'


@pytest.mark.gated
def test_real_weights_match_diffusers():
    vae = AutoencoderKLFlux2.from_pretrained('black-forest-labs/FLUX.2-dev', subfolder='vae').eval()
    m = Flux2VAEEncoder.from_pretrained('black-forest-labs/FLUX.2-dev', dtype=mx.bfloat16)
    images = _images(size=256, seed=4)
    with torch.no_grad():
        mean = vae.encode(images * 2 - 1).latent_dist.mean
    actual, _ = m.moments(nhwc(images * 2 - 1).astype(mx.bfloat16))
    assert_rms_close(actual, mean.permute(0, 2, 3, 1), 2e-2)

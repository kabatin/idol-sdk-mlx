"""idol.hub's access check (mocked Hugging Face) and the MLX image encoder's wiring (tiny random encoders)."""
import mlx.core as mx
import numpy as np
import pytest
from huggingface_hub.errors import GatedRepoError

from idol import hub


class _Response:
    status_code, headers, url, request = 403, {}, 'https://huggingface.co', None


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(hub, '_ACCESSIBLE_REPOS', set())
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)


def _patch(monkeypatch, denied=(), token='hf_x', network_error=False):
    import huggingface_hub

    def auth_check(repo):
        if network_error:
            raise OSError('offline')
        if repo in denied:
            raise GatedRepoError('gated', response=_Response())
    monkeypatch.setattr(huggingface_hub, 'auth_check', auth_check)
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: token)


def test_access_granted_is_cached(fresh_cache, monkeypatch):
    _patch(monkeypatch)
    hub.require_encoder_access(['a/b', 'c/d'])
    assert hub._ACCESSIBLE_REPOS == {'a/b', 'c/d'}


@pytest.mark.parametrize('token, reason', [(None, 'not logged in'), ('hf_x', 'does not have access')])
def test_denied_access_explains_what_to_do(fresh_cache, monkeypatch, token, reason):
    _patch(monkeypatch, denied={'c/d'}, token=token)
    with pytest.raises(hub.EncoderAccessError, match=reason) as error:
        hub.require_encoder_access(['a/b', 'c/d'])
    assert 'https://huggingface.co/c/d' in str(error.value) and 'a/b\n' not in str(error.value)


def test_network_problems_and_offline_mode_do_not_block(fresh_cache, monkeypatch):
    _patch(monkeypatch, network_error=True)
    hub.require_encoder_access(['a/b'])
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    _patch(monkeypatch, denied={'a/b'})
    hub.require_encoder_access(['a/b'])


def test_mlx_image_encoder_tokens():
    from idol.mlx.conditioning import ImageEncoder
    from idol.mlx.dinov3 import DINOv3ViT
    from idol.mlx.flux2_vae import Flux2VAEEncoder
    dinov3 = DINOv3ViT(hidden_size=64, num_hidden_layers=1, num_attention_heads=2, intermediate_size=128,
                       dtype=mx.float32)
    vae = Flux2VAEEncoder(block_out_channels=(32, 64), layers_per_block=1, latent_channels=4, norm_num_groups=8,
                          dtype=mx.float32)
    images = mx.random.uniform(shape=(1, 32, 32, 3), key=mx.random.key(0))
    feature1, feature2 = ImageEncoder(dinov3, vae)(images, key=mx.random.key(1))
    # 16 px DINOv3 patches: 2 x 2; this small VAE downsamples 2x, then 2 x 2 latent patches: 8 x 8
    # (the released VAE downsamples 8x, so both give 64 x 64 + 5 tokens for a 1024 image)
    assert feature1.shape == (1, 5 + 2 * 2, 64) and feature2.shape == (1, 5 + 8 * 8, 16)
    assert feature1.dtype == feature2.dtype == mx.float32
    f1 = np.array(feature1)
    assert np.allclose(f1.mean(-1), 0, atol=1e-4) and np.allclose(f1.std(-1), 1, atol=1e-2)   # layer-normed


def test_mlx_image_encoder_rejects_unknown_dtypes():
    from idol.mlx.conditioning import ImageEncoder
    with pytest.raises(ValueError, match='dinov3_dtype'):
        ImageEncoder.from_pretrained('a/b', 'c/d', dinov3_dtype='int8')

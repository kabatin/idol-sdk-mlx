"""The MLX DINOv3 ViT against transformers' DINOv3ViTModel (small random config; real weights with -m gated)."""
import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from idol.mlx.dinov3 import DINOv3ViT, rope_cos_sin

from _parity import assert_close, assert_rms_close, copy_weights, randomize_, to_mx

SMALL = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, use_gated_mlp=True,
             hidden_act='silu', num_register_tokens=4, patch_size=16, rope_theta=100.0, key_bias=False,
             query_bias=True, value_bias=True, layer_norm_eps=1e-5)


def _pair(dtype=mx.float32):
    config = DINOv3ViTConfig(**SMALL)
    t = randomize_(DINOv3ViTModel(config), seed=5).eval()
    m = copy_weights(DINOv3ViT(**{k: SMALL[k] for k in ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
                                                        'intermediate_size', 'num_register_tokens', 'patch_size',
                                                        'rope_theta', 'layer_norm_eps')}, dtype=dtype),
                     t, rename=DINOv3ViT.weight_name)
    return t, m


def _pixels(h=64, w=48, seed=0):
    return torch.randn(1, 3, h, w, generator=torch.Generator().manual_seed(seed))


def test_rope_matches_transformers():
    t, _ = _pair()
    pixels = _pixels()
    cos_t, sin_t = t.rope_embeddings(pixels)
    cos_m, sin_m = rope_cos_sin(4, 3, head_dim=32, theta=100.0)
    assert_close(cos_m, cos_t, rtol=1e-6, atol=1e-7)
    assert_close(sin_m, sin_t, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('size', [(64, 48), (32, 32)])
def test_forward_matches_transformers_fp32(size):
    t, m = _pair()
    pixels = _pixels(*size)
    with torch.no_grad():
        expected = t(pixel_values=pixels).last_hidden_state
    actual = m(to_mx(pixels.permute(0, 2, 3, 1)))
    assert actual.shape == (1, 5 + size[0] // 16 * size[1] // 16, 64)
    assert_close(actual, expected)


def test_forward_matches_transformers_bf16():
    t, m = _pair(dtype=mx.bfloat16)
    pixels = _pixels(seed=1)
    with torch.no_grad():
        expected = t.to(torch.bfloat16)(pixel_values=pixels.to(torch.bfloat16)).last_hidden_state
    actual = m(to_mx(pixels.permute(0, 2, 3, 1)))
    assert actual.dtype == mx.bfloat16
    assert_rms_close(actual, expected.float(), 2e-2)


def test_weight_names():
    assert DINOv3ViT.weight_name('model.layer.3.mlp.up_proj.weight') == 'layer.3.mlp.up_proj.weight'
    assert DINOv3ViT.weight_name('layer.3.mlp.up_proj.weight') == 'layer.3.mlp.up_proj.weight'
    assert DINOv3ViT.weight_name('embeddings.mask_token') is None


@pytest.mark.gated
def test_real_weights_match_transformers():
    from PIL import Image
    repo = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
    image = np.array(Image.open('demo/examples/fig-fox.webp').convert('RGB').resize((512, 512)), np.float32) / 255
    mean, std = np.array((0.485, 0.456, 0.406), np.float32), np.array((0.229, 0.224, 0.225), np.float32)
    pixels = torch.tensor((image - mean) / std).permute(2, 0, 1)[None]
    t = DINOv3ViTModel.from_pretrained(repo).eval()
    with torch.no_grad():
        expected = t(pixel_values=pixels).last_hidden_state
    del t
    for dtype, check in ((mx.float32, lambda a: assert_close(a, expected, rtol=2e-3)),
                         (mx.bfloat16, lambda a: assert_rms_close(a, expected, 3e-2))):
        check(DINOv3ViT.from_pretrained(repo, dtype=dtype)(to_mx(pixels.permute(0, 2, 3, 1))))


def test_rejects_unsupported_configs(tmp_path, monkeypatch):
    import json
    import huggingface_hub
    (tmp_path / 'config.json').write_text(json.dumps({'hidden_act': 'gelu', 'use_gated_mlp': True}))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda repo, name: str(tmp_path / 'config.json'))
    with pytest.raises(ValueError, match='only the SiLU'):
        DINOv3ViT.from_pretrained('example/repo')

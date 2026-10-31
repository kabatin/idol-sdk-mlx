"""The MLX BiRefNet (Swin-L backbone, ASPP-deformable decoder) against torch."""
import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image

from idol.mlx import birefnet as M
from idol.models import birefnet as T

from _parity import assert_close, copy_weights, randomize_, to_mx


def _nhwc(t):
    return to_mx(t.permute(0, 2, 3, 1))


@pytest.mark.parametrize('size', [24, 30])   # 30: padded to 36 windows
def test_swin_layer_with_shift_and_padding(size):
    t = randomize_(T._SwinBasicLayer(dim=32, depth=2, num_heads=2, window_size=12), seed=1).eval()
    m = copy_weights(M.SwinBasicLayer(dim=32, depth=2, num_heads=2, window_size=12), t)
    x = torch.randn(1, size * size, 32, generator=torch.Generator().manual_seed(0))
    expected = t(x, size, size)
    actual = m(to_mx(x), size, size)
    assert [a.shape[1:] if hasattr(a, 'shape') else a for a in actual] == \
           [tuple(e.shape[1:]) if hasattr(e, 'shape') else e for e in expected]
    assert_close(actual[0], expected[0])
    assert_close(actual[3], expected[3])


@pytest.fixture(scope='module')
def random_pair():
    t = randomize_(T.BiRefNet(), seed=3, std=0.02).eval()
    return t, copy_weights(M.BiRefNet(dtype=mx.float32), t)


def test_forward_matches_torch(random_pair):
    t, m = random_pair
    x = torch.randn(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = t(x)
    assert_close(m(_nhwc(x)), expected.permute(0, 2, 3, 1), rtol=1e-3, atol=1e-4)


def test_remove_background_matches_torch(random_pair):
    t, m = random_pair
    t.INPUT_SIZE = m.input_size = (256, 256)    # the 1024 default would be slow on random weights in torch
    image = Image.fromarray((np.random.default_rng(0).random((90, 140, 3)) * 255).astype(np.uint8))
    expected, actual = t.remove_background(image), m.remove_background(image)
    assert actual.mode == 'RGBA' and actual.size == image.size
    assert np.array_equal(np.array(actual)[..., :3], np.array(image))
    a, e = np.array(actual)[..., 3].astype(int), np.array(expected)[..., 3].astype(int)
    assert np.abs(a - e).max() <= 1


@pytest.mark.weights
def test_real_weights_on_example_images():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from idol.pipeline import WEIGHT_FILES
    import os
    path = os.path.join(snapshot_download('speridlabs/idol', allow_patterns=WEIGHT_FILES), 'birefnet.safetensors')
    t = T.BiRefNet()
    t.load_state_dict(load_file(path))
    t = t.float().eval()
    models = {dtype: M.BiRefNet.from_safetensors(path, dtype=dtype) for dtype in (mx.float32, mx.float16)}
    for name in ('obj-backpack', 'fig-fox', 'fur-egg-chair'):
        image = Image.open(f'demo/examples/{name}.webp').convert('RGB')
        with torch.no_grad():
            expected = np.array(t.remove_background(image))[..., 3] > 127
        for dtype, model in models.items():
            alpha = np.array(model.remove_background(image))[..., 3] > 127
            iou = (alpha & expected).sum() / max((alpha | expected).sum(), 1)
            assert iou > 0.99, (name, dtype, iou)

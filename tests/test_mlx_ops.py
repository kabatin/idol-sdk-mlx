"""MLX image ops against torch: bilinear resize (align_corners=True) and modulated deformable convolution."""
import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from idol.mlx import ops

from _parity import assert_close, to_mx


def _nhwc(t):
    return to_mx(t.permute(0, 2, 3, 1))


def _nchw(a):
    return torch.tensor(np.array(a.astype(mx.float32))).permute(0, 3, 1, 2)


@pytest.mark.parametrize('src, dst', [((8, 8), (16, 16)), ((16, 12), (8, 6)), ((7, 5), (13, 9)), ((1, 1), (6, 4)),
                                      ((32, 32), (1024, 1024)), ((1024, 1024), (731, 1203)), ((5, 5), (5, 5))])
def test_resize_bilinear_align_corners(src, dst):
    x = torch.randn(2, 3, *src, generator=torch.Generator().manual_seed(0))
    expected = F.interpolate(x, size=dst, mode='bilinear', align_corners=True)
    assert_close(_nchw(ops.resize_bilinear(_nhwc(x), dst)), expected, rtol=1e-5, atol=1e-6)


def test_resize_keeps_dtype():
    x = mx.ones((1, 4, 4, 2), dtype=mx.float16)
    assert ops.resize_bilinear(x, (8, 8)).dtype == mx.float16


@pytest.mark.parametrize('k, padding', [(1, 0), (3, 1), (7, 3)])
@pytest.mark.parametrize('offset_scale', [0.0, 1.7, 9.0])   # 9: many samples fall outside the image
def test_deform_conv2d_matches_torchvision(k, padding, offset_scale):
    g = torch.Generator().manual_seed(k)
    x = torch.randn(1, 6, 11, 9, generator=g)
    weight = torch.randn(5, 6, k, k, generator=g) * 0.2
    bias = torch.randn(5, generator=g)
    offset = torch.randn(1, 2 * k * k, 11, 9, generator=g) * offset_scale
    mask = torch.rand(1, k * k, 11, 9, generator=g) * 2
    expected = deform_conv2d(x, offset, weight, bias=bias, padding=padding, mask=mask)
    actual = ops.deform_conv2d(_nhwc(x), _nhwc(offset), to_mx(weight.permute(0, 2, 3, 1)), _nhwc(mask),
                               bias=to_mx(bias), padding=padding)
    assert_close(_nchw(actual), expected, rtol=1e-5, atol=1e-5)


def test_deform_conv2d_without_bias_and_in_fp16():
    g = torch.Generator().manual_seed(9)
    x, weight = torch.randn(1, 8, 10, 10, generator=g), torch.randn(4, 8, 3, 3, generator=g) * 0.2
    offset, mask = torch.randn(1, 18, 10, 10, generator=g), torch.rand(1, 9, 10, 10, generator=g)
    expected = deform_conv2d(x, offset, weight, padding=1, mask=mask)
    actual = ops.deform_conv2d(_nhwc(x).astype(mx.float16), _nhwc(offset).astype(mx.float16),
                               to_mx(weight.permute(0, 2, 3, 1), mx.float16), _nhwc(mask).astype(mx.float16), padding=1)
    assert actual.dtype == mx.float16
    assert_close(_nchw(actual), expected, rtol=1e-2)


def test_batch_norm_eval_matches_torch():
    bn = torch.nn.BatchNorm2d(6).eval()
    with torch.no_grad():
        for p in (bn.weight, bn.bias, bn.running_mean):
            p.copy_(torch.randn(6))
        bn.running_var.copy_(torch.rand(6) + 0.5)
    m = ops.BatchNorm2d(6)
    for name in ('weight', 'bias', 'running_mean', 'running_var'):
        m[name] = to_mx(getattr(bn, name))
    x = torch.randn(2, 6, 5, 4)
    assert_close(_nchw(m(_nhwc(x))), bn(x))

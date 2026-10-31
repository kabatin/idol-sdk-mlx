"""Image ops BiRefNet needs that MLX lacks, channels-last: F.interpolate(mode='bilinear', align_corners=True) to any
size, torchvision's modulated deformable convolution (DCNv2), and eval-mode BatchNorm."""
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

F32 = mx.float32


def _source_coords(size_in: int, size_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """align_corners=True: output i samples input i * (in - 1) / (out - 1), in fp32 as torch computes it.
    Returns the lower index, the upper index and the upper weight."""
    scale = np.float32(size_in - 1) / np.float32(size_out - 1) if size_out > 1 else np.float32(0)
    src = scale * np.arange(size_out, dtype=np.float32)
    low = np.minimum(np.floor(src).astype(np.int32), size_in - 1)
    high = np.minimum(low + 1, size_in - 1)
    return low, high, (src - low.astype(np.float32)).astype(np.float32)


def resize_bilinear(x: mx.array, size: Tuple[int, int]) -> mx.array:
    """x [B, H, W, C] -> [B, size[0], size[1], C], bilinear with align_corners=True (interpolated in fp32)."""
    B, H, W, C = x.shape
    if (H, W) == tuple(size):
        return x
    xf = x.astype(F32)
    low, high, w = _source_coords(H, size[0])
    w = mx.array(w)[None, :, None, None]
    xf = xf[:, mx.array(low)] * (1 - w) + xf[:, mx.array(high)] * w
    low, high, w = _source_coords(W, size[1])
    w = mx.array(w)[None, None, :, None]
    xf = xf[:, :, mx.array(low)] * (1 - w) + xf[:, :, mx.array(high)] * w
    return xf.astype(x.dtype)


def _bilinear_gather(flat: mx.array, y: mx.array, x: mx.array, H: int, W: int) -> mx.array:
    """torchvision's bilinear_interpolate: flat [B, H*W, C]; y, x [B, P] -> [B, P, C]. Samples outside
    (-1, H) x (-1, W) are 0, corners outside the image contribute 0."""
    inside = (y > -1) & (y < H) & (x > -1) & (x < W)
    y0, x0 = mx.floor(y), mx.floor(x)
    ly, lx = y - y0, x - x0
    out = None
    for dy, wy in ((0, 1 - ly), (1, ly)):
        for dx, wx in ((0, 1 - lx), (1, lx)):
            yy, xx = y0 + dy, x0 + dx
            valid = inside & (yy >= 0) & (yy <= H - 1) & (xx >= 0) & (xx <= W - 1)
            index = (mx.clip(yy, 0, H - 1) * W + mx.clip(xx, 0, W - 1)).astype(mx.int32)
            value = mx.take_along_axis(flat, index[..., None], axis=1)
            term = value * mx.where(valid, wy * wx, 0.0)[..., None]
            out = term if out is None else out + term
    return out


def deform_conv2d(x: mx.array, offset: mx.array, weight: mx.array, mask: mx.array, bias: Optional[mx.array] = None,
                  padding: int = 0) -> mx.array:
    """Modulated deformable convolution (torchvision.ops.deform_conv2d, stride 1, one offset group).

    x [B, H, W, Cin]; offset [B, H', W', 2K] as (dy, dx) per kernel tap (row-major taps); mask [B, H', W', K];
    weight [Cout, kh, kw, Cin] (MLX layout). Samples are interpolated in fp32, tap by tap; each tap's contribution
    is one matmul in the input dtype, summed over the taps in fp32 (like torchvision's single GEMM)."""
    B, H, W, C = x.shape
    Cout, kh, kw, _ = weight.shape
    Ho, Wo = H + 2 * padding - kh + 1, W + 2 * padding - kw + 1
    flat = x.reshape(B, H * W, C).astype(F32)
    base_y = mx.array(np.repeat(np.arange(Ho, dtype=np.float32) - padding, Wo))[None]
    base_x = mx.array(np.tile(np.arange(Wo, dtype=np.float32) - padding, Ho))[None]
    offset = offset.reshape(B, Ho * Wo, kh * kw, 2).astype(F32)
    mask = mask.reshape(B, Ho * Wo, kh * kw).astype(F32)
    out = None
    for i in range(kh):
        for j in range(kw):
            k = i * kw + j
            sampled = _bilinear_gather(flat, base_y + i + offset[:, :, k, 0], base_x + j + offset[:, :, k, 1], H, W)
            sampled = (sampled * mask[:, :, k, None]).astype(x.dtype)
            term = (sampled @ weight[:, i, j, :].T).astype(F32)
            out = term if out is None else out + term
    if bias is not None:
        out = out + bias.astype(F32)
    return out.astype(x.dtype).reshape(B, Ho, Wo, Cout)


class BatchNorm2d(nn.Module):
    """Eval-mode BatchNorm (running statistics), channels-last, computed in fp32 and cast back."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        scale = self.weight.astype(F32) * mx.rsqrt(self.running_var.astype(F32) + self.eps)
        shift = self.bias.astype(F32) - self.running_mean.astype(F32) * scale
        return (x.astype(F32) * scale + shift).astype(x.dtype)

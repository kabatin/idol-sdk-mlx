"""MLX port of idol/models/birefnet.py: BiRefNet with the Swin-L backbone (ZhengPeng7/BiRefNet), channels-last.
forward(x) returns the alpha map in [0, 1]; remove_background(image) returns the RGBA PIL image."""
import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from .ops import BatchNorm2d, deform_conv2d, resize_bilinear
from .weights import load_torch_weights

F32 = mx.float32


# -- Swin Transformer (Swin-Large preset) ---------------------------------------------

class SwinMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


def window_partition(x: mx.array, ws: int) -> mx.array:
    """[B, H, W, C] -> [B * H/ws * W/ws, ws * ws, C]."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(windows: mx.array, ws: int, H: int, W: int) -> mx.array:
    C = windows.shape[-1]
    B = windows.shape[0] // ((H // ws) * (W // ws))
    x = windows.reshape(B, H // ws, W // ws, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


def relative_position_index(ws: int) -> np.ndarray:
    coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing='ij')).reshape(2, -1)
    relative = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0) + (ws - 1)
    return (relative[..., 0] * (2 * ws - 1) + relative[..., 1]).reshape(-1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = mx.zeros(((2 * window_size - 1) ** 2, num_heads))
        self._index = relative_position_index(window_size)    # numpy: recomputed, not loaded
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B_, N, C = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B_, N, 3, H, C // H).transpose(2, 0, 3, 1, 4)
        attn = (qkv[0] * self.scale) @ qkv[1].swapaxes(-1, -2)
        bias = self.relative_position_bias_table[mx.array(self._index)].reshape(N, N, H).transpose(2, 0, 1)
        attn = attn + bias.astype(attn.dtype)[None]
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(B_ // nW, nW, H, N, N) + mask.astype(attn.dtype)[None, :, None]).reshape(-1, H, N, N)
        attn = mx.softmax(attn, axis=-1, precise=True)
        return self.proj((attn @ qkv[2]).transpose(0, 2, 1, 3).reshape(B_, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMlp(dim, int(dim * mlp_ratio))

    def __call__(self, x: mx.array, H: int, W: int, mask: mx.array) -> mx.array:
        B, L, C = x.shape
        ws, s = self.window_size, self.shift_size
        h = self.norm1(x).reshape(B, H, W, C)
        pad_r, pad_b = (ws - W % ws) % ws, (ws - H % ws) % ws
        h = mx.pad(h, [(0, 0), (0, pad_b), (0, pad_r), (0, 0)])
        Hp, Wp = H + pad_b, W + pad_r
        if s > 0:
            h = mx.roll(h, shift=(-s, -s), axis=(1, 2))
        h = self.attn(window_partition(h, ws), mask=mask if s > 0 else None)
        h = window_reverse(h, ws, Hp, Wp)
        if s > 0:
            h = mx.roll(h, shift=(s, s), axis=(1, 2))
        x = x + h[:, :H, :W].reshape(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def __call__(self, x: mx.array, H: int, W: int) -> mx.array:
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)
        x = mx.concatenate([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], axis=-1)
        return self.reduction(self.norm(x.reshape(B, -1, 4 * C)))


def shifted_window_mask(Hp: int, Wp: int, ws: int, shift: int) -> np.ndarray:
    """-100 between tokens of a window that come from different regions after the cyclic shift, else 0."""
    img = np.zeros((1, Hp, Wp, 1), dtype=np.float32)
    slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    for count, (h, w) in enumerate((h, w) for h in slices for w in slices):
        img[:, h, w, :] = count
    windows = img.reshape(1, Hp // ws, ws, Wp // ws, ws, 1).transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws)
    return np.where(windows[:, None, :] - windows[:, :, None] != 0, -100.0, 0.0).astype(np.float32)


class SwinBasicLayer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0,
                 downsample: bool = True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = [SwinBlock(dim, num_heads, window_size, 0 if i % 2 == 0 else window_size // 2, mlp_ratio)
                       for i in range(depth)]
        if downsample:
            self.downsample = PatchMerging(dim)

    def __call__(self, x: mx.array, H: int, W: int):
        ws = self.window_size
        mask = mx.array(shifted_window_mask(math.ceil(H / ws) * ws, math.ceil(W / ws) * ws, ws, self.shift_size))
        for block in self.blocks:
            x = block(x, H, W, mask)
        if 'downsample' in self:
            return x, H, W, self.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2
        return x, H, W, x, H, W


class SwinPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.proj(x))


class SwinLarge(nn.Module):
    """x [B, H, W, 3] -> four feature maps [B, H/4, W/4, 192] ... [B, H/32, W/32, 1536]."""
    def __init__(self):
        super().__init__()
        embed_dim, depths, num_heads = 192, (2, 2, 18, 2), (6, 12, 24, 48)
        self.num_features = [embed_dim * 2 ** i for i in range(4)]
        self.patch_embed = SwinPatchEmbed(4, 3, embed_dim)
        self.layers = [SwinBasicLayer(self.num_features[i], depths[i], num_heads[i], window_size=12,
                                      downsample=i < 3) for i in range(4)]
        for i, channels in enumerate(self.num_features):
            setattr(self, f'norm{i}', nn.LayerNorm(channels))

    def __call__(self, x: mx.array) -> Tuple[mx.array, ...]:
        x = self.patch_embed(x)
        B, Wh, Ww, C = x.shape
        x = x.reshape(B, Wh * Ww, C)
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            outs.append(getattr(self, f'norm{i}')(x_out).reshape(B, H, W, self.num_features[i]))
        return tuple(outs)


# -- ASPP-Deformable ---------------------------------------------------------------------

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                 bias: bool = False):
        super().__init__()
        self.padding = padding
        K = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, 2 * K, kernel_size, padding=padding)
        self.modulator_conv = nn.Conv2d(in_channels, K, kernel_size, padding=padding)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        modulator = 2.0 * mx.sigmoid(self.modulator_conv(x))
        bias = self.regular_conv.bias if 'bias' in self.regular_conv else None
        return deform_conv2d(x, self.offset_conv(x), self.regular_conv.weight, modulator, bias=bias,
                             padding=self.padding)


class ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels: int, planes: int, kernel_size: int, padding: int):
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size, padding=padding, bias=False)
        self.bn = BatchNorm2d(planes)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        out_channels = out_channels or in_channels
        inter = 256
        self.aspp1 = ASPPModuleDeformable(in_channels, inter, 1, padding=0)
        self.aspp_deforms = [ASPPModuleDeformable(in_channels, inter, k, padding=k // 2) for k in parallel_block_sizes]
        # [pool, conv, bn, relu] like the torch Sequential, so the weight names line up
        self.global_avg_pool = [nn.Identity(), nn.Conv2d(in_channels, inter, 1, bias=False), BatchNorm2d(inter),
                                nn.ReLU()]
        self.conv1 = nn.Conv2d(inter * (2 + len(parallel_block_sizes)), out_channels, 1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = self.aspp1(x)
        pooled = mx.mean(x.astype(F32), axis=(1, 2), keepdims=True).astype(x.dtype)
        _, conv, bn, relu = self.global_avg_pool
        x5 = mx.broadcast_to(relu(bn(conv(pooled))), (*x1.shape[:3], conv.weight.shape[0]))
        y = mx.concatenate([x1, *(m(x) for m in self.aspp_deforms), x5], axis=-1)
        return nn.relu(self.bn1(self.conv1(y)))


# -- Decoder blocks -------------------------------------------------------------------------

class BasicDecBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.bn_in = BatchNorm2d(inter_channels)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)
        self.bn_out = BatchNorm2d(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.bn_in(self.conv_in(x)))
        return self.bn_out(self.conv_out(self.dec_att(x)))


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv_out(self.conv1(x))


def image2patches(image: mx.array, ref: mx.array) -> mx.array:
    """'b (hg h) (wg w) c -> b h w (c hg wg)' with h, w the spatial size of ref."""
    B, H, W, C = image.shape
    h, w = ref.shape[1], ref.shape[2]
    hg, wg = H // h, W // w
    x = image.reshape(B, hg, h, wg, w, C).transpose(0, 2, 4, 5, 1, 3)
    return x.reshape(B, h, w, C * hg * wg)


def _gdt_branch(in_channels: int) -> List[nn.Module]:
    return [nn.Conv2d(in_channels, 16, 3, padding=1), BatchNorm2d(16), nn.ReLU()]


def _run(layers, x):
    for layer in layers:
        x = layer(x)
    return x


class BiRefNetDecoder(nn.Module):
    def __init__(self, channels=(3072, 1536, 768, 384)):
        super().__init__()
        c = channels
        self.ipt_blk5 = SimpleConvs(2 ** 10 * 3, c[0] // 8)
        self.ipt_blk4 = SimpleConvs(2 ** 8 * 3, c[0] // 8)
        self.ipt_blk3 = SimpleConvs(2 ** 6 * 3, c[1] // 8)
        self.ipt_blk2 = SimpleConvs(2 ** 4 * 3, c[2] // 8)
        self.ipt_blk1 = SimpleConvs(2 ** 0 * 3, c[3] // 8)
        self.decoder_block4 = BasicDecBlk(c[0] + c[0] // 8, c[1])
        self.decoder_block3 = BasicDecBlk(c[1] + c[0] // 8, c[2])
        self.decoder_block2 = BasicDecBlk(c[2] + c[1] // 8, c[3])
        self.decoder_block1 = BasicDecBlk(c[3] + c[2] // 8, c[3] // 2)
        self.conv_out1 = [nn.Conv2d(c[3] // 2 + c[3] // 8, 1, 1)]
        self.lateral_block4 = BasicLatBlk(c[1], c[1])
        self.lateral_block3 = BasicLatBlk(c[2], c[2])
        self.lateral_block2 = BasicLatBlk(c[3], c[3])
        self.gdt_convs_4, self.gdt_convs_3, self.gdt_convs_2 = (_gdt_branch(n) for n in (c[1], c[2], c[3]))
        self.gdt_convs_attn_4, self.gdt_convs_attn_3, self.gdt_convs_attn_2 = (
            [nn.Conv2d(16, 1, 1)] for _ in range(3))

    @staticmethod
    def _cat(*xs):
        return mx.concatenate(xs, axis=-1)

    def __call__(self, x, x1, x2, x3, x4):
        def gate(p, convs, attn):
            return p * mx.sigmoid(_run(attn, _run(convs, p)))

        def up(p, ref):
            return resize_bilinear(p, ref.shape[1:3])

        x4 = self._cat(x4, self.ipt_blk5(image2patches(x, x4)))
        p4 = gate(self.decoder_block4(x4), self.gdt_convs_4, self.gdt_convs_attn_4)
        _p3 = up(p4, x3) + self.lateral_block4(x3)

        _p3 = self._cat(_p3, self.ipt_blk4(image2patches(x, _p3)))
        p3 = gate(self.decoder_block3(_p3), self.gdt_convs_3, self.gdt_convs_attn_3)
        _p2 = up(p3, x2) + self.lateral_block3(x2)

        _p2 = self._cat(_p2, self.ipt_blk3(image2patches(x, _p2)))
        p2 = gate(self.decoder_block2(_p2), self.gdt_convs_2, self.gdt_convs_attn_2)
        _p1 = up(p2, x1) + self.lateral_block2(x1)

        _p1 = self._cat(_p1, self.ipt_blk2(image2patches(x, _p1)))
        _p1 = up(self.decoder_block1(_p1), x)
        _p1 = self._cat(_p1, self.ipt_blk1(image2patches(x, _p1)))
        return _run(self.conv_out1, _p1)


class BiRefNet(nn.Module):
    _CHANNELS = (3072, 1536, 768, 384)
    _NORM_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    _NORM_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def __init__(self, dtype=mx.float16, input_size: Tuple[int, int] = (1024, 1024)):
        super().__init__()
        self.dtype = dtype
        self.input_size = input_size
        self.bb = SwinLarge()
        cxt = sum(self._CHANNELS[1:])
        self.squeeze_module = [BasicDecBlk(self._CHANNELS[0] + cxt, self._CHANNELS[0])]
        self.decoder = BiRefNetDecoder(self._CHANNELS)
        self.update(tree_map(lambda p: p.astype(dtype), self.parameters()))

    @classmethod
    def from_safetensors(cls, path: str, dtype=mx.float16) -> 'BiRefNet':
        return load_torch_weights(cls(dtype=dtype), path)

    def _forward_enc(self, x: mx.array):
        x1, x2, x3, x4 = self.bb(x)
        H, W = x.shape[1:3]
        halves = self.bb(resize_bilinear(x, (H // 2, W // 2)))
        x1, x2, x3, x4 = (mx.concatenate([f, resize_bilinear(h, f.shape[1:3])], axis=-1)
                          for f, h in zip((x1, x2, x3, x4), halves))
        size = x4.shape[1:3]
        x4 = mx.concatenate([resize_bilinear(x1, size), resize_bilinear(x2, size), resize_bilinear(x3, size), x4],
                            axis=-1)
        return x1, x2, x3, x4

    def __call__(self, x: mx.array) -> mx.array:
        """x [B, H, W, 3] ImageNet-normalised -> alpha [B, H, W, 1] in [0, 1]."""
        x1, x2, x3, x4 = self._forward_enc(x)
        return mx.sigmoid(self.decoder(x, x1, x2, x3, _run(self.squeeze_module, x4)))

    def remove_background(self, image) -> 'Image.Image':
        from PIL import Image
        W, H = image.size
        pixels = mx.array(np.array(image, dtype=np.float32) / 255.0)[None]
        pixels = resize_bilinear(pixels, self.input_size)
        pixels = ((pixels - mx.array(self._NORM_MEAN)) / mx.array(self._NORM_STD)).astype(self.dtype)
        alpha = resize_bilinear(self(pixels).astype(F32), (H, W))[0, ..., 0]
        rgba = image.copy()
        rgba.putalpha(Image.fromarray((np.clip(np.array(alpha), 0, 1) * 255).astype(np.uint8), mode='L'))
        return rgba

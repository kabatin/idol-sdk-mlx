"""MLX ports of idol/models/layers.py: same parameter names, same dtype flow as the torch layers.

The torch blocks keep their Linear weights in fp16 (``convert_module_to_f16``) and compute LayerNorm and the RMS
norms in fp32; ``cast_linears`` reproduces the first, the norms cast explicitly. RoPE uses (cos, sin) pairs."""
import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

F32 = mx.float32


def cast_linears(module: nn.Module, dtype) -> nn.Module:
    """``convert_module_to_f16``: every Linear (weight and bias) under ``module`` to ``dtype``."""
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            for key in ('weight', 'bias'):
                if key in sub:
                    sub[key] = sub[key].astype(dtype)
    return module


def scaled_dot_product_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """q [B, L, H, D], k and v [B, S, H, D] -> [B, L, H, D], softmax(q k^T / sqrt(D)) v."""
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
    return out.transpose(0, 2, 1, 3)


# -- Norms and MLPs -------------------------------------------------------------

class LayerNorm32(nn.Module):
    """LayerNorm computed in fp32 and cast back to the input dtype."""
    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        weight, bias = (self.weight, self.bias) if self.affine else (None, None)
        return mx.fast.layer_norm(x.astype(F32), weight, bias, self.eps).astype(x.dtype)


def layer_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """F.layer_norm without affine parameters."""
    return mx.fast.layer_norm(x, None, None, eps)


class MultiHeadRMSNorm(nn.Module):
    """x [..., H, D] -> L2-normalised over D, times sqrt(D) and a per-head gamma [H, D] (computed in fp32)."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(F32)
        norm = mx.sqrt(mx.sum(xf * xf, axis=-1, keepdims=True))
        return (xf / mx.maximum(norm, 1e-12) * self.gamma * self.scale).astype(x.dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = [nn.Linear(channels, hidden), nn.GELU(approx='precise'), nn.Linear(hidden, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.mlp:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, channels: int, inner_channels: int, channels_out: int, mlp_layer_num: int = 2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers += [nn.Linear(channels if i == 0 else inner_channels, inner_channels), nn.GELU(approx='precise')]
        self.mlp = layers + [nn.Linear(inner_channels, channels_out)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.mlp:
            x = layer(x)
        return x


# -- Positional and scalar embeddings ----------------------------------------------

def torch_linspace(start: float, end: float, steps: int) -> np.ndarray:
    """torch.linspace in fp32, bit for bit: the step is computed in fp32 and each half is a fused multiply-add from
    its own end (emulated in float64, which rounds once like an FMA)."""
    step = np.float64(np.float32(np.float32(end) - np.float32(start)) / np.float32(max(steps - 1, 1)))
    idx = np.arange(steps, dtype=np.float64)
    low = np.float64(np.float32(start)) + step * idx
    high = np.float64(np.float32(end)) - step * (steps - 1 - idx)
    return np.where(idx < steps // 2, low, high).astype(np.float32)


def _powers_of_two(exponents_f32: np.ndarray) -> np.ndarray:
    """2 ** exponents, correctly rounded to fp32. torch's pow is not correctly rounded (it is off by up to an ulp,
    differently on CPU and CUDA) and no other library reproduces it; the exponents themselves match torch exactly."""
    return np.power(2.0, exponents_f32.astype(np.float64)).astype(np.float32)


def _pad_channels(embed: mx.array, channels: int) -> mx.array:
    missing = channels - embed.shape[-1]
    if missing <= 0:
        return embed
    return mx.concatenate([embed, mx.zeros((*embed.shape[:-1], missing), dtype=embed.dtype)], axis=-1)


class PcdAbsolutePositionEmbedder(nn.Module):
    """Sinusoidal embedding of 3D positions in [0, 1] (the denoiser's latent token positions)."""
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 16):
        super().__init__()
        self.channels = channels
        freq_dim = channels // in_channels // 2
        res_dim = max(0, freq_dim - max_res)
        exponents = np.concatenate([np.arange(max_res, dtype=np.float32),   # the fp32 arithmetic of the torch layer
                                    np.arange(res_dim, dtype=np.float32) / np.float32(max(res_dim, 1))
                                    * np.float32(max_res)])
        self._freqs = _powers_of_two(exponents[:freq_dim])

    def __call__(self, x: mx.array) -> mx.array:
        *dims, _ = x.shape
        angles = x.reshape(-1, 1).astype(F32) * mx.array(self._freqs)[None] * 2 * math.pi
        embed = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        return _pad_channels(embed.reshape(*dims, -1), self.channels)


class PcdAbsolutePositionEmbedderv2(nn.Module):
    """Sinusoidal embedding of 3D points in [0, 1] (the decoders' query points), x [N, 3] -> [N, channels]."""
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        freq_dim = channels // in_channels // 2
        self._freqs = _powers_of_two(torch_linspace(0.0, float(max_res), freq_dim))

    def __call__(self, x: mx.array) -> mx.array:
        angles = x[..., None] * mx.array(self._freqs).astype(x.dtype) * math.pi
        embed = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        return _pad_channels(embed.reshape(x.shape[0], -1), self.channels)


class SinusoidalEmbedder(nn.Module):
    """Embeds a scalar per batch element (diffusion time, octree level) into a vector."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000,
                 two_pi: bool = False):
        super().__init__()
        self.mlp = [nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)]
        half = frequency_embedding_size // 2
        self._freqs = np.exp(np.float32(-np.log(max_period)) * np.arange(half, dtype=np.float32) / half)
        self._freqs = self._freqs.astype(np.float32)
        self.two_pi = two_pi

    def __call__(self, t: mx.array) -> mx.array:
        args = t[:, None].astype(F32) * mx.array(self._freqs)[None]
        if self.two_pi:
            args = args * 2 * math.pi
        x = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        for layer in self.mlp:
            x = layer(x)
        return x


# -- RoPE ---------------------------------------------------------------------------

def apply_rotary_emb(hidden: mx.array, freqs: Tuple[mx.array, mx.array]) -> mx.array:
    """hidden [B, L, H, D] rotated pairwise (interleaved pairs) by (cos, sin) [B, L, H, D/2], in fp32."""
    cos, sin = freqs
    x = hidden.astype(F32).reshape(*hidden.shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return out.reshape(hidden.shape).astype(hidden.dtype)


class RePo3DRotaryEmbedding(nn.Module):
    """Learned rotary position: each token's features predict a 3D offset per head, turned into RoPE angles.
    Returns (cos, sin) [B, L, H, head_dim / 2] in fp32."""
    def __init__(self, model_channels: int, num_heads: int, head_dim: int, repo_hidden_ratio: float = 0.125,
                 max_freq: float = 16.0):
        super().__init__()
        self.num_heads = num_heads
        hidden = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, hidden, bias=False)
        self.content_map = nn.Linear(model_channels, hidden, bias=False)
        self.final_map = nn.Linear(hidden, 3 * num_heads, bias=False)
        dim_0 = dim_1 = 2 * (head_dim // 6)
        dims = [dim_0, dim_1, head_dim - dim_0 - dim_1]
        self.freqs_0, self.freqs_1, self.freqs_2 = (mx.linspace(1.0, float(max_freq), d // 2) for d in dims)

    def __call__(self, hidden: mx.array) -> Tuple[mx.array, mx.array]:
        h = self.norm(hidden.astype(F32))
        out = self.final_map(nn.silu(self.gate_map(h)) * self.content_map(h))
        B, L, _ = out.shape
        delta = out.reshape(B, L, self.num_heads, 3)
        # clamp_mul(x, f) = x * tanh(f) + x * (f - tanh(f)) is x * f at inference
        angles = mx.concatenate([delta[..., i:i + 1] * f * math.pi
                                 for i, f in enumerate((self.freqs_0, self.freqs_1, self.freqs_2))], axis=-1)
        return mx.cos(angles), mx.sin(angles)


# -- Attention -------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Self or cross attention with RMS-normalised queries and keys (the decoders)."""
    def __init__(self, channels: int, num_heads: int, ctx_channels: Optional[int] = None, cross: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.cross = cross
        head_dim = channels // num_heads
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels, channels * 2)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array, context: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        H = self.num_heads
        if self.cross:
            q = self.to_q(x).reshape(B, L, H, -1)
            kv = self.to_kv(context).reshape(B, context.shape[1], 2, H, -1)
            k, v = kv[:, :, 0], kv[:, :, 1]
        else:
            qkv = self.to_qkv(x).reshape(B, L, 3, H, -1)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        h = scaled_dot_product_attention(self.q_rms_norm(q), self.k_rms_norm(k), v)
        return self.to_out(h.reshape(B, L, -1))


class RopeMultiHeadAttention(nn.Module):
    """Self attention with rotary embeddings, then RMS-normalised queries and keys (the denoiser)."""
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array, rope_emb: Tuple[mx.array, mx.array]) -> mx.array:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q = self.q_norm(apply_rotary_emb(qkv[:, :, 0], rope_emb))
        k = self.k_norm(apply_rotary_emb(qkv[:, :, 1], rope_emb))
        return self.out(scaled_dot_product_attention(q, k, qkv[:, :, 2]).reshape(B, L, C))


# -- Transformer blocks ------------------------------------------------------------------

def _modulation(mod: mx.array):
    """[B, 6C] -> six [B, 1, C] (shift, scale, gate for attention, then for the MLP)."""
    return mx.split(mod[:, None, :], 6, axis=-1)


class UnifiedTransformerBlock(nn.Module):
    """Denoiser block: RoPE self attention + FFN, modulated by the diffusion time (shared adaLN plus this block's
    shift table); modulation=False is the condition refiner, with plain affine LayerNorms."""
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0, modulation: bool = True):
        super().__init__()
        self.modulation = modulation
        self.norm1 = LayerNorm32(channels, eps=1e-6, affine=not modulation)
        self.norm2 = LayerNorm32(channels, eps=1e-6, affine=not modulation)
        self.attn = RopeMultiHeadAttention(channels, num_heads=num_heads)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if modulation:
            self.shift_table = mx.zeros((1, 6 * channels))

    def __call__(self, x: mx.array, mod: Optional[mx.array], rotary_emb) -> mx.array:
        if not self.modulation:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            return x + self.mlp(self.norm2(x))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _modulation(
            mod + self.shift_table.astype(mod.dtype))
        h = self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa, rope_emb=rotary_emb)
        x = x + h * gate_msa
        h = self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        return x + h * gate_mlp


class TransformerCrossBlock(nn.Module):
    """Gaussian decoder block: self attention + cross attention to the latent + FFN."""
    def __init__(self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, eps=1e-6, affine=False)
        self.norm2 = LayerNorm32(channels, eps=1e-6, affine=True)
        self.norm3 = LayerNorm32(channels, eps=1e-6, affine=False)
        self.self_attn = MultiHeadAttention(channels, num_heads=num_heads)
        self.cross_attn = MultiHeadAttention(channels, num_heads=num_heads, ctx_channels=ctx_channels, cross=True)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    """Octree decoder block: cross attention to the latent + FFN, modulated by the octree level."""
    def __init__(self, channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, eps=1e-6, affine=False)
        self.norm2 = LayerNorm32(channels, eps=1e-6, affine=False)
        self.cross_attn = MultiHeadAttention(channels, num_heads=num_heads, ctx_channels=ctx_channels, cross=True)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def __call__(self, x: mx.array, mod: mx.array, context: mx.array) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _modulation(mod)
        x = x + self.cross_attn(self.norm1(x) * (1 + scale_msa) + shift_msa, context) * gate_msa
        return x + self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp) * gate_mlp

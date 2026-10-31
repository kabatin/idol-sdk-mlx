"""DINOv3 ViT (transformers' DINOv3ViTModel, eval mode) in MLX: patch conv, [CLS, registers, patches], 2D RoPE on
the patch tokens, LayerScale, SwiGLU MLP, final LayerNorm. Returns last_hidden_state [B, 1 + R + HW/256, C]."""
import json
import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from .weights import load_torch_weights


def rope_cos_sin(patches_h: int, patches_w: int, head_dim: int, theta: float) -> Tuple[mx.array, mx.array]:
    """cos, sin [HW, head_dim] in fp32: patch centres in [-1, 1] as (y, x), angles laid out [y, x, y, x] x D/4."""
    inv_freq = (1.0 / np.power(np.float32(theta), np.arange(0, 1, 4 / head_dim, dtype=np.float32))).astype(np.float32)
    ys = np.arange(0.5, patches_h, dtype=np.float32) / np.float32(patches_h)
    xs = np.arange(0.5, patches_w, dtype=np.float32) / np.float32(patches_w)
    coords = np.stack(np.meshgrid(ys, xs, indexing='ij'), axis=-1).reshape(-1, 2) * np.float32(2) - np.float32(1)
    angles = (np.float32(2 * math.pi) * coords[:, :, None] * inv_freq[None, None, :]).reshape(len(coords), -1)
    angles = mx.array(np.tile(angles, 2))
    return mx.cos(angles), mx.sin(angles)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, N, C = x.shape
        H = self.num_heads
        q, k, v = (proj(x).reshape(B, N, H, -1).transpose(0, 2, 1, 3) for proj in (self.q_proj, self.k_proj, self.v_proj))
        prefix = N - cos.shape[0]          # CLS + registers: not rotated
        q, k = (mx.concatenate([t[:, :, :prefix], t[:, :, prefix:] * cos + _rotate_half(t[:, :, prefix:]) * sin],
                               axis=2) for t in (q, k))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, N, C))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lambda1 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.lambda1


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden)
        self.up_proj = nn.Linear(dim, hidden)
        self.down_proj = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden: int, eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attention = Attention(dim, num_heads)
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = GatedMLP(dim, hidden)
        self.layer_scale2 = LayerScale(dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.layer_scale1(self.attention(self.norm1(x), cos, sin))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class PatchEmbeddings(nn.Module):
    def __init__(self, dim: int, patch_size: int, num_register_tokens: int):
        super().__init__()
        self.cls_token = mx.zeros((1, 1, dim))
        self.register_tokens = mx.zeros((1, num_register_tokens, dim))
        self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, pixels: mx.array) -> mx.array:
        patches = self.patch_embeddings(pixels)
        B, h, w, C = patches.shape
        return mx.concatenate([mx.broadcast_to(self.cls_token, (B, 1, C)),
                               mx.broadcast_to(self.register_tokens, (B, *self.register_tokens.shape[1:])),
                               patches.reshape(B, h * w, C)], axis=1)


class DINOv3ViT(nn.Module):
    def __init__(self, hidden_size: int = 1280, num_hidden_layers: int = 32, num_attention_heads: int = 20,
                 intermediate_size: int = 5120, num_register_tokens: int = 4, patch_size: int = 16,
                 rope_theta: float = 100.0, layer_norm_eps: float = 1e-5, dtype=mx.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.head_dim = hidden_size // num_attention_heads
        self.rope_theta = rope_theta
        self.embeddings = PatchEmbeddings(hidden_size, patch_size, num_register_tokens)
        self.layer = [Layer(hidden_size, num_attention_heads, intermediate_size, layer_norm_eps)
                      for _ in range(num_hidden_layers)]
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.update(tree_map(lambda p: p.astype(dtype), self.parameters()))

    @staticmethod
    def weight_name(name: str) -> Optional[str]:
        """Checkpoint (or in-memory 'model.layer.*') name -> this module's; the mask token is training-only."""
        if name == 'embeddings.mask_token':
            return None
        return name[len('model.'):] if name.startswith('model.layer.') else name

    @classmethod
    def from_pretrained(cls, repo: str, dtype=mx.bfloat16) -> 'DINOv3ViT':
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo, 'config.json')) as f:
            c = json.load(f)
        if (not c.get('use_gated_mlp') or c.get('key_bias', False) or not c.get('query_bias', True)
                or c.get('hidden_act', 'silu') != 'silu'):
            raise ValueError(f'{repo}: only the SiLU gated-MLP, no-key-bias DINOv3 layout is supported')
        model = cls(hidden_size=c['hidden_size'], num_hidden_layers=c['num_hidden_layers'],
                    num_attention_heads=c['num_attention_heads'], intermediate_size=c['intermediate_size'],
                    num_register_tokens=c['num_register_tokens'], patch_size=c['patch_size'],
                    rope_theta=c['rope_theta'], layer_norm_eps=c['layer_norm_eps'], dtype=dtype)
        return load_torch_weights(model, hf_hub_download(repo, 'model.safetensors'), rename=cls.weight_name)

    def __call__(self, pixels: mx.array) -> mx.array:
        """pixels [B, H, W, 3], ImageNet-normalised -> last_hidden_state [B, 1 + R + HW / p^2, C] in the module dtype."""
        pixels = pixels.astype(self.dtype)
        x = self.embeddings(pixels)
        cos, sin = rope_cos_sin(pixels.shape[1] // self.patch_size, pixels.shape[2] // self.patch_size,
                                self.head_dim, self.rope_theta)
        cos, sin = cos.astype(self.dtype), sin.astype(self.dtype)
        for layer in self.layer:
            x = layer(x, cos, sin)
        return self.norm(x)

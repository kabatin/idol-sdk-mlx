"""MLX port of idol/models/decoder.py: the octree decoder places anchor points, the Gaussian decoder turns each
anchor into 32 Gaussians. The octree descent (data-dependent shapes, a few thousand cells) runs its resampling in
numpy; the transformer passes run in MLX."""
from types import MappingProxyType
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gaussian import Gaussian
from .layers import (F32, ModulatedTransformerCrossOnlyBlock, PcdAbsolutePositionEmbedderv2, SinusoidalEmbedder,
                     TransformerCrossBlock, cast_linears, layer_norm)

# child k of a cell sits at offset (k & 1, (k >> 1) & 1, k >> 2), the order of the torch decoder
CHILD_OFFSET = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=np.int64)
_CDF_MAX = np.float32(1.0 - 1e-12)   # what torch's clamp(max=1 - 1e-12) is in fp32 (1.0)


class OctreeProbabilityFixedlenDecoder(nn.Module):
    def __init__(self, model_channels: int, cond_channels: int, num_blocks: int, num_heads: int,
                 mlp_ratio: float = 4.0, dtype=mx.float16):
        super().__init__()
        C = model_channels
        self.dtype = dtype
        self.in_proj = nn.Linear(3, C)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(channels=C, in_channels=3)
        self.input_layer = nn.Linear(C, C)
        self.l_embedder = SinusoidalEmbedder(C, max_period=1024, two_pi=True)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(C, 6 * C)]
        self.blocks = [ModulatedTransformerCrossOnlyBlock(C, ctx_channels=cond_channels, num_heads=num_heads,
                                                          mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        self.out_proj = nn.Linear(C, 2 ** 3)
        for block in self.blocks:
            cast_linears(block, dtype)

    def __call__(self, x: mx.array, level: mx.array, cond: mx.array) -> mx.array:
        """x: cell centres [B, L, 3] in [0, 1]; level: the cells' octree resolution [B]; cond: latent [B, Lc, C].
        Returns the logits [B, L, 8] of how each cell's points split among its 8 children."""
        B, L, _ = x.shape
        h = self.input_layer(self.in_proj(x) + self.pos_embedder(x.reshape(-1, 3)).reshape(B, L, -1))
        level_emb = self.adaLN_modulation[1](nn.silu(self.l_embedder(level.astype(F32)))).astype(self.dtype)
        out, context = h.astype(self.dtype), cond.astype(self.dtype)
        for block in self.blocks:
            out = block(out, level_emb, context)
        return self.out_proj(layer_norm(out.astype(F32)))


def systematic_sample(probs: np.ndarray, counts: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """Split counts[r] points among the P bins of probs[r] by systematic resampling, one uniform offset per row
    (torch.rand in the torch decoder) -> [R, P] counts. Same fp32 arithmetic and bin edges as torch."""
    probs = np.maximum(probs.astype(np.float32), 0)
    R, P = probs.shape
    row_sums = probs.sum(axis=1, keepdims=True, dtype=np.float32)
    probs = np.where(row_sums == 0, np.float32(1.0 / P), probs / np.maximum(row_sums, np.float32(1)))
    cdf = np.minimum(np.cumsum(probs, axis=1, dtype=np.float32), _CDF_MAX)

    counts = counts.astype(np.int64)
    rows = np.repeat(np.arange(R), counts)                       # one entry per point
    k = np.arange(rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
    n = counts[rows].astype(np.float32)
    us = np.minimum(uniform.astype(np.float32)[rows] / n + k.astype(np.float32) / n, _CDF_MAX)
    # one searchsorted over all rows: shift row r by 2r (exact in float64, rows can't overlap)
    shift = 2.0 * np.arange(R)
    flat_cdf = (cdf.astype(np.float64) + shift[:, None]).ravel()
    idx = np.searchsorted(flat_cdf, us.astype(np.float64) + shift[rows], side='left') - rows * P
    return np.bincount(rows * P + np.minimum(idx, P - 1), minlength=R * P).reshape(R, P)


def sample_anchors(model: OctreeProbabilityFixedlenDecoder, cond: mx.array, num_points: int, level: int,
                   uniform: Callable = np.random.random) -> mx.array:
    """Place num_points anchor points [1, num_points, 3] in [0, 1]^3, descending the octree to ``level``.
    uniform(shape) draws the random numbers (a seeded numpy Generator's ``random``)."""
    if cond.shape[0] != 1:
        raise ValueError('the MLX octree decoder samples one object at a time')
    coords = np.zeros((1, 3), dtype=np.int64)
    counts = np.array([num_points], dtype=np.int64)
    for l in range(1, level + 1):
        parent = (coords.astype(np.float32) + np.float32(0.5)) / np.float32(1 << (l - 1))
        logits = model(mx.array(parent)[None], mx.array([1 << l]), cond)
        probs = np.array(mx.softmax(logits.astype(F32), axis=-1))[0]
        sampled = systematic_sample(probs, counts, np.asarray(uniform((len(counts),)), dtype=np.float32)).ravel()
        children = (coords[:, None, :] * 2 + CHILD_OFFSET[None]).reshape(-1, 3)
        keep = sampled > 0
        coords, counts = children[keep], sampled[keep]
    points = np.repeat(coords, counts, axis=0).astype(np.float32)
    jitter = np.asarray(uniform(points.shape), dtype=np.float32)    # uniformly inside each leaf cell
    return mx.array((points + jitter) / np.float32(1 << level))[None]


class GaussianFixedlenDecoder(nn.Module):
    def __init__(self, in_channels: int, model_channels: int, cond_channels: int, num_blocks: int, num_heads: int,
                 representation_config: dict, mlp_ratio: float = 4.0, dtype=mx.float16):
        super().__init__()
        self.dtype = dtype
        # a mapping proxy, not a dict: MLX modules treat dict attributes as parameter containers
        self.rep_config = MappingProxyType({**representation_config,
                                            'lr': MappingProxyType(dict(representation_config['lr']))})
        per_anchor = representation_config['num_gaussians']
        # per anchor: offset (3), colour (3), scale (3), rotation (4), opacity (1), offset scale (1), x 32 Gaussians
        layout = (('_xyz', (3,)), ('_features_dc', (1, 3)), ('_scaling', (3,)), ('_rotation', (4,)),
                  ('_opacity', (1,)), ('_offset_scale', (1,)))
        ranges, start = {}, 0
        for name, shape in layout:
            size = per_anchor * int(np.prod(shape))
            ranges[name] = (start, start + size, shape)
            start += size
        self.ranges = MappingProxyType(ranges)

        C = model_channels
        self.in_proj = nn.Linear(in_channels, C)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(channels=C, in_channels=3)
        self.input_layer = nn.Linear(C, C)
        self.blocks = [TransformerCrossBlock(C, ctx_channels=cond_channels, num_heads=num_heads, mlp_ratio=mlp_ratio)
                       for _ in range(num_blocks)]
        self.out_proj = nn.Linear(C, start)
        self.points_offset_perturbation = mx.zeros((per_anchor, 3))
        self.base_offset_scale = mx.zeros(())
        for block in self.blocks:
            cast_linears(block, dtype)

    def __call__(self, points: mx.array, cond: mx.array) -> mx.array:
        """points: anchors [B, L, 3]; cond: latent [B, Lc, C] -> per-anchor outputs [B, L, 32 x 15]."""
        B, L, C = points.shape
        h = self.input_layer(self.in_proj(points) + self.pos_embedder(points.reshape(-1, C)).reshape(B, L, -1))
        out, context = h.astype(self.dtype), cond.astype(self.dtype)
        for block in self.blocks:
            out = block(out, context)
        return self.out_proj(layer_norm(out.astype(F32)))

    def _slice(self, features: mx.array, name: str) -> mx.array:
        start, end, shape = self.ranges[name]
        return features[:, :, start:end].reshape(features.shape[0], -1, self.rep_config['num_gaussians'], *shape)

    def to_representation(self, points: mx.array, features: mx.array) -> Gaussian:
        """Anchors [1, L, 3] and their features [1, L, 480] -> the L x 32 Gaussians."""
        config, lr = self.rep_config, self.rep_config['lr']
        offset_scale = mx.logaddexp(self._slice(features, '_offset_scale') + self.base_offset_scale, 0.0)
        offset = self._slice(features, '_xyz') * lr['_xyz'] + self.points_offset_perturbation
        offset = mx.tanh(offset) * 0.5 * config['perturbe_size'] * offset_scale

        def raw(name):
            value = self._slice(features, name)[0]
            return value.reshape(-1, *value.shape[2:]) * lr[name]

        return Gaussian.from_raw(
            raw_xyz=(offset[0] + points[0, :, None, :]).reshape(-1, 3),
            features_dc=raw('_features_dc'), raw_scaling=raw('_scaling'), raw_rotation=raw('_rotation'),
            raw_opacity=raw('_opacity'), scaling_bias=config['scaling_bias'], opacity_bias=config['opacity_bias'],
            minimum_kernel_size=config['3d_filter_kernel_size'])


def decode_gaussians(decoder: OctreeProbabilityFixedlenDecoder, decoder_gs: GaussianFixedlenDecoder,
                     latent: mx.array, num_gaussians: int, level: int, uniform: Callable = np.random.random
                     ) -> Gaussian:
    """Latent [1, 8192, 16] -> Gaussians: num_gaussians / 32 anchors, 32 Gaussians each."""
    per_anchor = decoder_gs.rep_config['num_gaussians']
    anchors = sample_anchors(decoder, latent, num_points=max(1, int(num_gaussians) // per_anchor), level=level,
                             uniform=uniform)
    return decoder_gs.to_representation(anchors, decoder_gs(anchors, latent))

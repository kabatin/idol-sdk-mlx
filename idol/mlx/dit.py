"""MLX port of idol/models/dit.py: the rectified-flow transformer over the object latent (8192 tokens) and the input
camera (1 token), conditioned on the image tokens; joint attention over [latent, image, camera] with learned RoPE.

The image tokens go through the condition refiner without the time or the noise, so ``refine_condition`` runs once
per image and ``denoise`` once per sampling step."""
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .layers import (F32, MLP, PcdAbsolutePositionEmbedder, RePo3DRotaryEmbedding, SinusoidalEmbedder,
                     UnifiedTransformerBlock, cast_linears, layer_norm)


class LatentSeqMMFlowModel(nn.Module):
    def __init__(self, q_token_length: int, in_channels: int, cam_channels: int, model_channels: int,
                 cond_channels: int, cond2_channels: int, num_blocks: int, num_refiner_blocks: int = 2,
                 num_heads: int = 16, num_head_channels: int = 64, mlp_ratio: float = 4, dtype=mx.float16):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.dtype = dtype   # of the transformer blocks; embeddings, norms and output layers stay fp32
        C = model_channels

        def blocks(count, modulation=True):
            return [UnifiedTransformerBlock(C, num_heads=num_heads, mlp_ratio=mlp_ratio, modulation=modulation)
                    for _ in range(count)]

        def rope_layers(count):
            return [RePo3DRotaryEmbedding(C, num_heads=num_heads, head_dim=num_head_channels) for _ in range(count)]

        self.t_embedder = SinusoidalEmbedder(C)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(C, 6 * C)]
        self.input_layer = nn.Linear(in_channels, C)
        self.cond_embedder = nn.Linear(cond_channels, C)
        self.cond_embedder2 = nn.Linear(cond2_channels, C)
        self.pos_pe = mx.zeros((1, q_token_length, 3))
        self.pos_embedder = PcdAbsolutePositionEmbedder(C)
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = MLP(cam_channels, C, C, mlp_layer_num=num_refiner_blocks)
        self.blocks = blocks(num_blocks)
        self.shift_table = mx.zeros((1, 2, C))
        self.out_layer = nn.Linear(C, in_channels)
        self.cam_out_layer = nn.Linear(C, cam_channels)
        for block in self.blocks + self.noise_refiner + self.context_refiner:
            cast_linears(block, dtype)

    def refine_condition(self, feature1: mx.array, feature2: mx.array) -> mx.array:
        """feature1 [B, L_c, 1280] (DINOv3), feature2 [B, L_c, 128] (FLUX.2 VAE) -> refined tokens [B, L_c, C]."""
        h = (self.cond_embedder(feature1) + self.cond_embedder2(feature2)).astype(self.dtype)
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h = block(h, None, repo(h))
        return h

    def denoise(self, latent: mx.array, camera: mx.array, t: mx.array, h_cond: mx.array) -> Tuple[mx.array, mx.array]:
        """latent [B, 8192, 16], camera [B, 1, 5], t [B] in [0, 1000], h_cond from refine_condition ->
        the velocity (latent, camera)."""
        dt = self.dtype
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation[1](nn.silu(t_emb)).astype(dt)

        h_x = self.input_layer(latent).astype(dt) + self.pos_embedder(self.pos_pe).astype(dt)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, t_mod, repo(h_x))
        h_cam = self.cam_refiner(camera).astype(dt)

        h = mx.concatenate([h_x, h_cond, h_cam], axis=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, t_mod, repo(h))

        shift, scale = mx.split(self.shift_table + t_emb[:, None, :], 2, axis=1)
        h_x = layer_norm(h[:, :latent.shape[1]].astype(F32)) * (1 + scale) + shift
        h_cam = layer_norm(h[:, -camera.shape[1]:].astype(F32)) * (1 + scale) + shift
        return self.out_layer(h_x), self.cam_out_layer(h_cam)

    def __call__(self, latent, camera, t, feature1, feature2) -> Tuple[mx.array, mx.array]:
        return self.denoise(latent, camera, t, self.refine_condition(feature1, feature2))

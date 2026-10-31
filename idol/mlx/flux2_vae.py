"""The encoder half of the FLUX.2 VAE (diffusers AutoencoderKLFlux2) in MLX, channels-last, and the image tokens
Idol conditions on (idol/conditioning.py: sample, 2x2 patchify, batch-norm statistics, 5 zero tokens in front)."""
import json
from typing import Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .weights import load_torch_weights

F32 = mx.float32
TOKEN_PADDING = 5   # zero tokens in front, so the VAE tokens line up with DINOv3's CLS + 4 register tokens


class GroupNorm32(nn.Module):
    """GroupNorm over contiguous channel groups (PyTorch's), computed in fp32 like torch's bf16 kernels do."""
    def __init__(self, num_groups: int, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        g = x.astype(F32).reshape(B, H * W, self.num_groups, C // self.num_groups)
        mean = mx.mean(g, axis=(1, 3), keepdims=True)
        var = mx.var(g, axis=(1, 3), keepdims=True)
        g = ((g - mean) * mx.rsqrt(var + self.eps)).reshape(B, H, W, C)
        return (g * self.weight.astype(F32) + self.bias.astype(F32)).astype(x.dtype)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.norm1 = GroupNorm32(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = GroupNorm32(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return (self.conv_shortcut(x) if 'conv_shortcut' in self else x) + h


class Downsample2D(nn.Module):
    """Zero pad one pixel right and bottom, then a stride-2 3x3 conv without padding."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)]))


class AttentionBlock(nn.Module):
    """The mid block's single-head self attention over all pixels, with a residual."""
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.group_norm = GroupNorm32(groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        h = self.group_norm(x).reshape(B, 1, H * W, C)
        out = mx.fast.scaled_dot_product_attention(self.to_q(h), self.to_k(h), self.to_v(h), scale=C ** -0.5)
        return x + self.to_out[0](out).reshape(B, H, W, C)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int, groups: int, downsample: bool):
        super().__init__()
        self.resnets = [ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, groups)
                        for i in range(layers)]
        if downsample:
            self.downsamplers = [Downsample2D(out_channels)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        return self.downsamplers[0](x) if 'downsamplers' in self else x


class MidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int, attention: bool):
        super().__init__()
        self.resnets = [ResnetBlock2D(channels, channels, groups) for _ in range(2)]
        self.attentions = [AttentionBlock(channels, groups)] if attention else []

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attention in self.attentions:
            x = attention(x)
        return self.resnets[1](x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, block_out_channels: Sequence[int], layers_per_block: int,
                 latent_channels: int, groups: int, mid_attention: bool):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        channels = [block_out_channels[0], *block_out_channels]
        self.down_blocks = [DownEncoderBlock2D(channels[i], channels[i + 1], layers_per_block, groups,
                                               downsample=i < len(block_out_channels) - 1)
                            for i in range(len(block_out_channels))]
        self.mid_block = MidBlock2D(block_out_channels[-1], groups, mid_attention)
        self.conv_norm_out = GroupNorm32(groups, block_out_channels[-1])
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * latent_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(self.mid_block(x))))


class BatchNormStatistics(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))


class Flux2VAEEncoder(nn.Module):
    """Images (NHWC) -> the posterior moments, or the FLUX.2 image tokens [B, 5 + H/16 * W/16, 4 * latent]."""
    def __init__(self, in_channels: int = 3, block_out_channels: Sequence[int] = (128, 256, 512, 512),
                 layers_per_block: int = 2, latent_channels: int = 32, norm_num_groups: int = 32,
                 mid_block_add_attention: bool = True, batch_norm_eps: float = 1e-4, dtype=mx.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.batch_norm_eps = batch_norm_eps
        self.encoder = Encoder(in_channels, block_out_channels, layers_per_block, latent_channels, norm_num_groups,
                               mid_block_add_attention)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.bn = BatchNormStatistics(4 * latent_channels)
        # every tensor in the module dtype, the BN statistics too (diffusers casts them on load as well)
        self.update(nn.utils.tree_map(lambda p: p.astype(dtype), self.parameters()))

    @staticmethod
    def weight_name(name: str) -> Optional[str]:
        """diffusers checkpoint name -> this module's (None: the decoder half, not needed)."""
        return None if name.startswith(('decoder.', 'post_quant_conv.')) else name

    @classmethod
    def from_pretrained(cls, repo: str, subfolder: str = 'vae', dtype=mx.bfloat16) -> 'Flux2VAEEncoder':
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo, f'{subfolder}/config.json')) as f:
            config = json.load(f)
        model = cls(in_channels=config['in_channels'], block_out_channels=config['block_out_channels'],
                    layers_per_block=config['layers_per_block'], latent_channels=config['latent_channels'],
                    norm_num_groups=config['norm_num_groups'],
                    mid_block_add_attention=config.get('mid_block_add_attention', True),
                    batch_norm_eps=config['batch_norm_eps'], dtype=dtype)
        return load_torch_weights(model, hf_hub_download(repo, f'{subfolder}/diffusion_pytorch_model.safetensors'),
                                  rename=cls.weight_name)

    def moments(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """x [B, H, W, 3] in [-1, 1] -> mean and log-variance [B, H/8, W/8, latent] (logvar clamped to [-30, 20])."""
        h = self.quant_conv(self.encoder(x.astype(self.dtype)))
        mean, logvar = mx.split(h, 2, axis=-1)
        return mean, mx.clip(logvar, -30.0, 20.0)

    def tokens(self, images: mx.array, noise: Optional[mx.array] = None,
               key: Optional[mx.array] = None) -> mx.array:
        """images [B, H, W, 3] in [0, 1] -> fp32 tokens [B, 5 + H/16 * W/16, 4 * latent] (idol.conditioning).
        The latent is sampled from the posterior: noise [B, H/8, W/8, latent] if given, else drawn with key."""
        mean, logvar = self.moments(images.astype(self.dtype) * 2 - 1)
        if noise is None:
            noise = mx.random.normal(mean.shape, key=key)
        latents = mean + mx.exp(0.5 * logvar) * noise.astype(self.dtype)
        B, H, W, C = latents.shape
        # 2x2 patchify, channel = c * 4 + dy * 2 + dx (the order of the FLUX.2 pipeline)
        latents = latents.reshape(B, H // 2, 2, W // 2, 2, C).transpose(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(B, H // 2, W // 2, 4 * C)
        std = mx.sqrt(self.bn.running_var + self.batch_norm_eps)
        tokens = ((latents - self.bn.running_mean) / std).astype(F32).reshape(B, -1, 4 * C)
        return mx.concatenate([mx.zeros((B, TOKEN_PADDING, 4 * C)), tokens], axis=1)

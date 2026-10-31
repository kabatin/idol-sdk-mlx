"""The image tokens the denoiser is conditioned on (idol/conditioning.py): DINOv3 tokens, layer-normed, and the
FLUX.2 VAE tokens; both [B, 4101, C] for a 1024 x 1024 image."""
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from .dinov3 import DINOv3ViT
from .flux2_vae import Flux2VAEEncoder
from .layers import layer_norm

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
DTYPES = {'fp32': mx.float32, 'bf16': mx.bfloat16, 'fp16': mx.float16}


class ImageEncoder:
    def __init__(self, dinov3: DINOv3ViT, flux2_vae: Flux2VAEEncoder):
        self.dinov3 = dinov3
        self.flux2_vae = flux2_vae

    @classmethod
    def from_pretrained(cls, dinov3_repo: str, flux2_vae_repo: str, dinov3_dtype: str = 'bf16') -> 'ImageEncoder':
        if dinov3_dtype not in DTYPES:
            raise ValueError(f'dinov3_dtype must be one of {sorted(DTYPES)}, got {dinov3_dtype!r}')
        return cls(DINOv3ViT.from_pretrained(dinov3_repo, dtype=DTYPES[dinov3_dtype]),
                   Flux2VAEEncoder.from_pretrained(flux2_vae_repo, dtype=mx.bfloat16))

    def __call__(self, images: mx.array, key: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """images [B, H, W, 3] in [0, 1] -> (DINOv3 tokens, FLUX.2 VAE tokens), fp32. key: the VAE posterior sample."""
        pixels = (images.astype(mx.float32) - mx.array(IMAGENET_MEAN)) / mx.array(IMAGENET_STD)
        feature1 = layer_norm(self.dinov3(pixels).astype(mx.float32))
        feature2 = self.flux2_vae.tokens(images, key=key)
        return feature1, feature2

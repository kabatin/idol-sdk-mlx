"""MLX backend for Apple Silicon: the whole Idol pipeline without PyTorch.

    from idol.mlx import IdolMLXPipeline
    pipeline = IdolMLXPipeline.from_pretrained("speridlabs/idol")
    pipeline("photo.jpg", mode="real").gaussian.save_ply("photo.ply")
"""
import os

# Precision defaults, read by MLX at the first matmul / convolution, unless the user set them:
# - On M5, MLX computes fp32 matmuls in TF32 (~1e-3 relative error). The torch model keeps its embeddings, RoPE maps
#   and output layers in full fp32; so does this backend.
# - MLX picks Winograd for some 3x3 convolutions; in bf16 it doubles the FLUX.2 VAE's error against fp32
#   (2.8e-2 vs 1.3e-2 relative RMS; torch bf16: 0.95e-2) for ~0.8 s saved per image.
os.environ.setdefault('MLX_ENABLE_TF32', '0')
os.environ.setdefault('MLX_CONV_WINOGRAD', '0')

__all__ = ['IdolMLXPipeline', 'IdolOutput']


def __getattr__(name):
    if name in __all__:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

"""MLX backend for Apple Silicon: the whole Idol pipeline without PyTorch."""
import os

# On M5, MLX computes fp32 matmuls in TF32 unless told otherwise (read at the first matmul). The torch model keeps
# its embeddings, RoPE maps and output layers in full fp32; so does this backend unless MLX_ENABLE_TF32 is set.
os.environ.setdefault('MLX_ENABLE_TF32', '0')

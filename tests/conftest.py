import os

os.environ.setdefault('MLX_ENABLE_TF32', '0')  # before mlx runs its first matmul: exact fp32 parity tests

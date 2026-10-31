import os

# before MLX runs its first matmul / convolution: the precision defaults of idol.mlx (see idol/mlx/__init__.py)
os.environ.setdefault('MLX_ENABLE_TF32', '0')
os.environ.setdefault('MLX_CONV_WINOGRAD', '0')

"""The Gaussians as MLX arrays (activated values), written with the shared PLY writer (idol/ply.py)."""
import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..ply import EPS, write_gaussian_ply


def _inverse_sigmoid(x):
    return mx.log(x / (1 - x))


@dataclass(frozen=True)
class Gaussian:
    """xyz [N, 3], features_dc [N, 1, 3] (degree-0 SH colour), scaling [N, 3], rotation [N, 4] unit quaternions
    (w, x, y, z), opacity [N, 1]. The object is z-up, inside [-0.5, 0.5]^3."""
    xyz: mx.array
    features_dc: mx.array
    scaling: mx.array
    rotation: mx.array
    opacity: mx.array

    @classmethod
    def from_raw(cls, raw_xyz, features_dc, raw_scaling, raw_rotation, raw_opacity, scaling_bias: float,
                 opacity_bias: float, minimum_kernel_size: float) -> 'Gaussian':
        """The activations of idol.gaussian.Gaussian: positions shifted into [-0.5, 0.5]^3, softplus scales with a
        minimum kernel size, normalised rotations, sigmoid opacities (biases applied in the raw space)."""
        scaling_offset = scaling_bias + math.log(-math.expm1(-scaling_bias))      # inverse softplus
        opacity_offset = math.log(opacity_bias / (1 - opacity_bias))              # inverse sigmoid
        scaling = mx.logaddexp(raw_scaling + scaling_offset, 0.0)                  # softplus
        rotation = raw_rotation + mx.array([1.0, 0.0, 0.0, 0.0])
        norm = mx.sqrt(mx.sum(rotation * rotation, axis=-1, keepdims=True))
        return cls(xyz=raw_xyz - 0.5, features_dc=features_dc,
                   scaling=mx.sqrt(scaling * scaling + minimum_kernel_size ** 2),
                   rotation=rotation / mx.maximum(norm, 1e-12),
                   opacity=mx.sigmoid(raw_opacity + opacity_offset))

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def save_ply(self, path: str) -> None:
        """Write the standard 3D Gaussian Splatting PLY (z-up object turned to the y-up convention viewers use)."""
        def host(a):
            return np.array(a.astype(mx.float32))
        write_gaussian_ply(
            path,
            xyz=host(self.xyz),
            f_dc=host(self.features_dc.reshape(len(self), 3)),
            opacity_logit=host(_inverse_sigmoid(mx.clip(self.opacity, EPS, 1 - EPS))),
            log_scale=host(mx.log(mx.maximum(self.scaling, EPS))),
            rotation=host(self.rotation),
        )

"""The standard 3D Gaussian Splatting PLY, written from numpy arrays (shared by the torch and MLX backends)."""
import numpy as np
from plyfile import PlyData, PlyElement

PLY_FIELDS = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
              'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
# Clamp before the inverse activations: a saturated opacity (exactly 1.0 after the sigmoid) or a zero scale would
# otherwise be written as +/-inf, which some PLY viewers reject.
EPS = 1e-6


def quaternion_multiply(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions; a is one quaternion, b is (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def write_gaussian_ply(path, xyz, f_dc, opacity_logit, log_scale, rotation):
    """xyz [N, 3] (z-up), f_dc [N, 3], opacity_logit [N, 1], log_scale [N, 3], rotation [N, 4] (w, x, y, z).

    The z-up object is turned to the y-up convention viewers use: +90 degrees about x, (x, y, z) -> (x, -z, y)."""
    xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    rotation = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), rotation)
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, f_dc, opacity_logit, log_scale, rotation), axis=1).astype(np.float32)
    elements = np.empty(attributes.shape[0], dtype=[(name, 'f4') for name in PLY_FIELDS])
    for column, name in enumerate(PLY_FIELDS):
        elements[name] = attributes[:, column]
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

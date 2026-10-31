import torch
import torch.nn.functional as F

from .ply import EPS, quaternion_multiply, write_gaussian_ply  # noqa: F401 (quaternion_multiply: re-exported)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class Gaussian:
    """Built from the Gaussian decoder's raw outputs; the activated values are what renderers and PLY files use.

    xyz [N, 3] positions, features_dc [N, 1, 3] colour (degree-0 spherical harmonics), scaling [N, 3],
    rotation [N, 4] unit quaternions (w, x, y, z), opacity [N, 1]. The object is z-up, inside [-0.5, 0.5]^3."""

    def __init__(self, raw_xyz, features_dc, raw_scaling, raw_rotation, raw_opacity,
                 scaling_bias: float, opacity_bias: float, minimum_kernel_size: float):
        device = raw_xyz.device
        aabb = torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        scaling_offset = inverse_softplus(torch.tensor(scaling_bias)).to(device)
        opacity_offset = inverse_sigmoid(torch.tensor(opacity_bias)).to(device)
        rotation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        self.xyz = raw_xyz * aabb[None, 3:] + aabb[None, :3]
        self.features_dc = features_dc
        scaling = F.softplus(raw_scaling + scaling_offset)
        self.scaling = torch.sqrt(torch.square(scaling) + minimum_kernel_size ** 2)
        self.rotation = F.normalize(raw_rotation + rotation_offset[None, :])
        self.opacity = torch.sigmoid(raw_opacity + opacity_offset)

    def save_ply(self, path):
        """Write the standard 3D Gaussian Splatting PLY (z-up object turned to the y-up convention viewers use)."""
        write_gaussian_ply(
            path,
            xyz=self.xyz.detach().cpu().numpy(),
            f_dc=self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            opacity_logit=inverse_sigmoid(self.opacity.clamp(EPS, 1 - EPS)).detach().cpu().numpy(),
            log_scale=torch.log(self.scaling.clamp_min(EPS)).detach().cpu().numpy(),
            rotation=self.rotation.detach().cpu().numpy(),
        )

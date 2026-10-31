"""The backend-independent pieces: the PLY writer, the camera decoding and the torch-free imports."""
import hashlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from idol.camera import decode_camera
from idol.gaussian import Gaussian

# sha256 of the PLY written by Gaussian.save_ply before the writer moved to idol/ply.py
LEGACY_PLY_SHA256 = '5c1f2065a1de5a6f98ebcfbc1623856da1cf3d45f6d03978bc83cfc493afc295'


def _legacy_gaussian():
    g = torch.Generator().manual_seed(0)
    n = 64
    return Gaussian(raw_xyz=torch.rand(n, 3, generator=g), features_dc=torch.randn(n, 1, 3, generator=g),
                    raw_scaling=torch.randn(n, 3, generator=g), raw_rotation=torch.randn(n, 4, generator=g) * 0.1,
                    raw_opacity=torch.randn(n, 1, generator=g) * 3, scaling_bias=0.004, opacity_bias=0.1,
                    minimum_kernel_size=0.0009)


def test_save_ply_is_byte_identical_to_the_legacy_writer(tmp_path):
    path = tmp_path / 'g.ply'
    _legacy_gaussian().save_ply(str(path))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LEGACY_PLY_SHA256


@pytest.mark.parametrize('latent, expected', [
    ([0.3, -0.5, 0.8, 0.4, 0.2],
     {'yaw': 2.60117309257816, 'pitch': 0.9409567982783891, 'radius': 2.4999999627470975, 'fov': 9.147842791271882}),
    ([0.0, 1.0, 0.0, 0.0, 0.3],
     {'yaw': 0.0, 'pitch': 0.0, 'radius': 1000.0, 'fov': 0.034377468042563135}),
    ([0.1, 0.2, 0.3, 0.0, 0.5],
     {'yaw': 0.4636476090008061, 'pitch': 0.9302740388525044, 'radius': 1000.0, 'fov': 0.057295774738434745}),
])
def test_decode_camera_matches_the_legacy_values(latent, expected):
    camera = decode_camera(np.array(latent, dtype=np.float32))
    assert camera.keys() == expected.keys()
    for key, value in expected.items():
        assert camera[key] == pytest.approx(value, rel=1e-6)


def test_decode_camera_falls_back_to_the_front_view_for_a_zero_direction():
    camera = decode_camera(np.array([0.0, 0.0, 0.0, 0.5, 0.1], dtype=np.float32))
    assert camera['yaw'] == pytest.approx(0.0)
    assert camera['pitch'] == pytest.approx(np.pi / 2)


@pytest.mark.parametrize('module', ['idol', 'idol.ply', 'idol.camera', 'idol.hub', 'idol.preprocess',
                                    'idol.sample', 'idol.mlx', 'idol.mlx.pipeline'])
def test_shared_modules_import_without_torch(module):
    heavy = ('torch', 'tensordict', 'transformers', 'diffusers', 'torchvision')
    code = (f'import sys, {module}; loaded = [m for m in {heavy!r} if m in sys.modules]; '
            f'assert not loaded, loaded')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_package_still_exposes_the_torch_api():
    import idol
    assert idol.IdolPipeline.__name__ == 'IdolPipeline'
    assert idol.Gaussian is Gaussian
    with pytest.raises(AttributeError):
        idol.does_not_exist


def test_submodules_are_reachable_as_attributes_after_a_bare_import():
    code = 'import idol; assert idol.pipeline.IdolPipeline and idol.gaussian.Gaussian and idol.camera.decode_camera'
    subprocess.run([sys.executable, '-c', code], check=True)


def test_decode_camera_still_accepts_torch_tensors():
    latent = [0.3, -0.5, 0.8, 0.4, 0.2]
    expected = decode_camera(np.array(latent, dtype=np.float32))
    for tensor in (torch.tensor(latent), torch.tensor(latent, dtype=torch.bfloat16),
                   torch.tensor(latent, dtype=torch.float16)):
        camera = decode_camera(tensor)
        for key in expected:
            assert camera[key] == pytest.approx(expected[key], rel=1e-2)

"""The CLI: backend selection and the files it writes (with a stub pipeline)."""
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from idol import sample


@pytest.mark.parametrize('system, machine, has_mlx, expected', [
    ('Darwin', 'arm64', True, 'mlx'), ('Darwin', 'arm64', False, 'torch'),
    ('Darwin', 'x86_64', True, 'torch'), ('Linux', 'x86_64', True, 'torch'),
])
def test_auto_backend(monkeypatch, system, machine, has_mlx, expected):
    monkeypatch.setattr(sample.platform, 'system', lambda: system)
    monkeypatch.setattr(sample.platform, 'machine', lambda: machine)
    monkeypatch.setattr(sample.importlib.util, 'find_spec', lambda name: object() if has_mlx else None)
    assert sample.resolve_backend('auto') == expected


def test_explicit_and_unknown_backends():
    assert sample.resolve_backend('torch') == 'torch'
    assert sample.resolve_backend('mlx') == 'mlx'
    with pytest.raises(ValueError):
        sample.resolve_backend('cuda')


class StubGaussian:
    def save_ply(self, path):
        open(path, 'w').write('ply')


def test_main_writes_the_outputs_for_a_folder(monkeypatch, tmp_path):
    calls = []

    def stub_pipeline(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(gaussian=StubGaussian(), preprocessed_image=Image.new('RGB', (4, 4)),
                               camera={'yaw': 0.0, 'pitch': 0.0, 'radius': 2.0, 'fov': 30.0})

    loaded = {}
    monkeypatch.setattr(sample, 'load_pipeline', lambda backend, weights, **kw: loaded.update(backend=backend, **kw)
                        or stub_pipeline)
    images = tmp_path / 'in'
    images.mkdir()
    for name in ('a.png', 'b.webp', 'notes.txt'):
        (images / name).write_bytes(b'')
    sample.main(['--image', str(images), '--out', str(tmp_path / 'out'), '--backend', 'mlx', '--steps', '3',
                 '--no_optimize'])
    assert loaded == {'backend': 'mlx', 'dinov3_dtype': 'bf16', 'optimize': False}
    assert [c[0].rsplit('/', 1)[1] for c in calls] == ['a.png', 'b.webp']
    assert calls[0][1]['steps'] == 3
    for name in ('a', 'b'):
        assert (tmp_path / 'out' / f'{name}.ply').read_text() == 'ply'
        assert json.loads((tmp_path / 'out' / f'{name}_camera.json').read_text())['fov'] == 30.0


def test_main_explains_missing_encoder_access(monkeypatch, tmp_path):
    def denied(*args, **kwargs):
        raise sample.EncoderAccessError('no access')
    monkeypatch.setattr(sample, 'load_pipeline', denied)
    with pytest.raises(SystemExit, match='no access'):
        sample.main(['--image', str(tmp_path / 'x.png'), '--out', str(tmp_path / 'out')])

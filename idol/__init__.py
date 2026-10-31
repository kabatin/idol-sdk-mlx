"""Idol: a single image to 3D Gaussians.

The torch pipeline is imported on first use, so the MLX backend (``idol.mlx``) runs without importing torch."""
__version__ = '0.1.0'

_LAZY = {'Gaussian': '.gaussian', 'IdolOutput': '.pipeline', 'IdolPipeline': '.pipeline'}
__all__ = list(_LAZY)


def __getattr__(name):
    import importlib
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    try:   # a submodule (idol.pipeline, idol.gaussian, ...), as a plain package import would expose it
        return importlib.import_module(f'.{name}', __name__)
    except ModuleNotFoundError as error:
        if error.name != f'{__name__}.{name}':
            raise
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

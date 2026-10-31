"""Load PyTorch state dicts (safetensors files or tensors) into the MLX modules.

The MLX modules use the same parameter names as the torch modules they port, so the mapping is one to one; the
loader transposes conv weights (OIHW -> OHWI) and casts each tensor to the dtype the MLX module chose for it."""
from typing import Callable, Dict, Iterable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# stored in the torch checkpoints but recomputed (or unused) at inference
SKIP_SUFFIXES = ('num_batches_tracked', 'relative_position_index')


def read_safetensors(path: str) -> Dict[str, mx.array]:
    return mx.load(path)


def load_torch_weights(module: nn.Module, weights: Union[str, Dict[str, mx.array]],
                       rename: Optional[Callable[[str], Optional[str]]] = None,
                       skip: Iterable[str] = SKIP_SUFFIXES, strict: bool = True) -> nn.Module:
    """weights: a safetensors path or name -> array in the torch layout. rename maps a torch name to the MLX
    name (None drops the tensor). strict: every MLX parameter must be loaded and every tensor used."""
    if isinstance(weights, str):
        weights = read_safetensors(weights)
    skip = tuple(skip)
    params = dict(tree_flatten(module.parameters()))
    loaded = {}
    for name, value in weights.items():
        if name.endswith(skip):
            continue
        target = rename(name) if rename else name
        if target is None:
            continue
        if target not in params:
            if strict:
                raise KeyError(f'{type(module).__name__}: unexpected weight {name!r}')
            continue
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        if value.shape != params[target].shape:
            raise ValueError(f'{type(module).__name__}: {name!r} has shape {value.shape}, '
                             f'expected {params[target].shape}')
        loaded[target] = value.astype(params[target].dtype)
    missing = sorted(set(params) - set(loaded))
    if strict and missing:
        raise KeyError(f'{type(module).__name__}: missing weights {missing[:8]}{" ..." if len(missing) > 8 else ""}')
    module.load_weights(list(loaded.items()), strict=False)
    mx.eval(module.parameters())
    return module

"""Helpers for comparing the MLX modules with the torch modules they port."""
import mlx.core as mx
import numpy as np
import torch

from idol.mlx.weights import load_torch_weights


def randomize_(module: torch.nn.Module, seed: int = 0, std: float = 0.05) -> torch.nn.Module:
    """Give every float parameter and buffer a random value (zero-initialised tables would hide porting bugs).
    Norm weights and RMS gammas stay around 1, RoPE frequencies around their initial values."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if not tensor.is_floating_point():
                continue
            noise = torch.randn(tensor.shape, generator=g, dtype=torch.float32)
            leaf = name.rsplit('.', 1)[-1]
            if leaf == 'gamma' or ('norm' in name and leaf == 'weight'):
                value = 1 + 0.1 * noise
            elif leaf.startswith('freqs_'):
                value = tensor.float() * (1 + 0.05 * noise)
            elif leaf in ('pos_pe', 'running_mean'):
                value = torch.rand(tensor.shape, generator=g)
            elif leaf == 'running_var':
                value = 0.5 + torch.rand(tensor.shape, generator=g)
            else:
                value = std * noise
            tensor.copy_(value.to(tensor.dtype))
    return module


def as_fp32_reference(module: torch.nn.Module) -> torch.nn.Module:
    """The DiT and decoders cast their blocks to fp16 and keep `self.dtype = float16`: undo both."""
    module = module.float()
    for sub in module.modules():
        if getattr(sub, 'dtype', None) == torch.float16 and not isinstance(getattr(type(sub), 'dtype', None), property):
            sub.dtype = torch.float32
    return module.eval()


def copy_weights(mlx_module, torch_module: torch.nn.Module, **kwargs):
    weights = {k: mx.array(v.detach().float().cpu().numpy()) if v.is_floating_point() else mx.array(v.cpu().numpy())
               for k, v in torch_module.state_dict().items()}
    load_torch_weights(mlx_module, weights, **kwargs)
    return mlx_module


def to_mx(t: torch.Tensor, dtype=mx.float32):
    return mx.array(t.detach().float().cpu().numpy()).astype(dtype)


def to_np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().float().cpu().numpy()
    return np.array(a.astype(mx.float32))


def assert_close(actual, expected, rtol=1e-4, atol=1e-5):
    actual, expected = to_np(actual), to_np(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    scale = max(float(np.abs(expected).max()), 1e-6)
    err = float(np.abs(actual - expected).max())
    assert err <= atol + rtol * scale, f'max abs error {err:.3e} (scale {scale:.3e})'

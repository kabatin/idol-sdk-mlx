"""Euler sampling of the rectified flow with classifier-free guidance (idol/sampler.py, batched guidance)."""
from typing import Tuple

import mlx.core as mx
import numpy as np
from tqdm import tqdm


def time_grid(steps: int, rescale_t: float) -> np.ndarray:
    """steps + 1 times from 1 (noise) to 0 (data), shifted towards t = 1 by rescale_t."""
    t_seq = np.linspace(1, 0, steps + 1)
    return rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)


def sample_flow(model, latent: mx.array, camera: mx.array, cond: Tuple[mx.array, mx.array],
                neg_cond: Tuple[mx.array, mx.array], steps: int = 50, rescale_t: float = 3.0,
                cfg_strength: float = 3.0, verbose: bool = True,
                compile: bool = False) -> Tuple[mx.array, mx.array]:
    """Integrate the noise (latent [B, 8192, 16], camera [B, 1, 5]) to data in ``steps`` Euler steps.

    cond / neg_cond: (DINOv3 tokens, FLUX.2 VAE tokens). Guided velocity = (1 + w) v(cond) - w v(neg_cond); both
    run as one batch, and the condition tokens are refined once for all steps. compile: mx.compile the step."""
    guided = cfg_strength > 0
    if guided:
        h_cond = model.refine_condition(*(mx.concatenate([c, n]) for c, n in zip(cond, neg_cond)))
    else:
        h_cond = model.refine_condition(*cond)
    denoise = mx.compile(model.denoise) if compile else model.denoise
    n = latent.shape[0]
    t_seq = time_grid(steps, rescale_t)
    for i in tqdm(range(steps), desc='Sampling', disable=not verbose):
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        if guided:
            both = denoise(mx.concatenate([latent, latent]), mx.concatenate([camera, camera]),
                                 mx.full((2 * n,), 1000 * t, dtype=mx.float32), h_cond)
            v_latent, v_camera = ((1 + cfg_strength) * v[:n] - cfg_strength * v[n:] for v in both)
        else:
            v_latent, v_camera = denoise(latent, camera, mx.full((n,), 1000 * t, dtype=mx.float32), h_cond)
        latent = latent - v_latent * (t - t_prev)
        camera = camera - v_camera * (t - t_prev)
        mx.eval(latent, camera)
    return latent, camera

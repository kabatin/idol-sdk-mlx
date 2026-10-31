"""IdolMLXPipeline: one image -> 3D Gaussians on Apple Silicon, without PyTorch. Same call and outputs as
idol.IdolPipeline; same seed, same object within this backend (MLX's random numbers differ from torch's, so the
objects differ from the CUDA backend's)."""
import json
import os
from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from ..camera import decode_camera
from ..hub import require_encoder_access
from ..preprocess import CROP_MARGIN, preprocess_image
from .birefnet import BiRefNet
from .conditioning import ImageEncoder
from .decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder, decode_gaussians
from .dit import LatentSeqMMFlowModel
from .gaussian import Gaussian
from .sampler import sample_flow
from .weights import load_torch_weights

DEFAULT_REPO = 'speridlabs/idol'
WEIGHT_FILES = ['config.json', 'denoiser.safetensors', 'decoder.safetensors', 'decoder_gs.safetensors',
                'birefnet.safetensors']


@dataclass
class IdolOutput:
    gaussian: Gaussian               # the object; .save_ply(path) writes a standard 3DGS PLY
    preprocessed_image: Image.Image  # the 1024 x 1024 image the model saw (background removed, cropped)
    camera: dict                     # predicted input camera: yaw, pitch (radians), radius, fov (degrees)


class IdolMLXPipeline:
    """compile (default on): the denoiser step is mx.compile'd (fused elementwise ops, ~10% faster sampling)."""
    def __init__(self, config: dict, denoiser: LatentSeqMMFlowModel, decoder: OctreeProbabilityFixedlenDecoder,
                 decoder_gs: GaussianFixedlenDecoder, background_remover: BiRefNet, image_encoder: ImageEncoder,
                 compile: bool = True):
        self.config = config
        self.denoiser = denoiser
        self.decoder = decoder
        self.decoder_gs = decoder_gs
        self.background_remover = background_remover
        self.image_encoder = image_encoder
        self.compile = compile

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = DEFAULT_REPO, dinov3_dtype: str = 'bf16',
                        compile: bool = True) -> 'IdolMLXPipeline':
        if os.path.isdir(repo_or_dir):
            weights_dir = repo_or_dir
        else:
            from huggingface_hub import hf_hub_download, snapshot_download
            # check the gated image encoders before downloading the ~3 GB of weights
            with open(hf_hub_download(repo_or_dir, 'config.json')) as f:
                encoders = json.load(f)['image_encoders']
            require_encoder_access([encoders['dinov3'], encoders['flux2_vae']])
            weights_dir = snapshot_download(repo_or_dir, allow_patterns=WEIGHT_FILES)
        with open(os.path.join(weights_dir, 'config.json')) as f:
            config = json.load(f)
        encoders = config['image_encoders']
        require_encoder_access([encoders['dinov3'], encoders['flux2_vae']])

        def load(model, name):
            return load_torch_weights(model, os.path.join(weights_dir, f'{name}.safetensors'))

        return cls(
            config=config,
            denoiser=load(LatentSeqMMFlowModel(**config['denoiser']['args']), 'denoiser'),
            decoder=load(OctreeProbabilityFixedlenDecoder(**config['decoder']['args']), 'decoder'),
            decoder_gs=load(GaussianFixedlenDecoder(**config['decoder_gs']['args']), 'decoder_gs'),
            background_remover=BiRefNet.from_safetensors(os.path.join(weights_dir, 'birefnet.safetensors')),
            image_encoder=ImageEncoder.from_pretrained(encoders['dinov3'], encoders['flux2_vae'], dinov3_dtype),
            compile=compile,
        )

    def preprocess(self, image: Union[str, Image.Image], mode: str = 'synthetic',
                   crop_margin: Optional[float] = None) -> Image.Image:
        """mode 'synthetic' (renders, generated images of one object: crop 1.2) or 'real' (photos: crop 1.7);
        crop_margin overrides the mode's value."""
        if mode not in CROP_MARGIN:
            raise ValueError(f'mode must be one of {sorted(CROP_MARGIN)}, got {mode!r}')
        if isinstance(image, str):
            image = Image.open(image)
        margin = crop_margin if crop_margin is not None else CROP_MARGIN[mode]
        return preprocess_image(image, background_remover=self.background_remover,
                                image_size=self.config['image_size'], crop_margin=margin)

    def __call__(self, image: Union[str, Image.Image], mode: str = 'synthetic', seed: int = 42, steps: int = 50,
                 cfg_strength: float = 3.0, rescale_t: float = 3.0, num_gaussians: int = 262144,
                 crop_margin: Optional[float] = None, verbose: bool = False) -> IdolOutput:
        """Generate the 3D Gaussians of the object in ``image``.

        mode / crop_margin: see ``preprocess``. num_gaussians: 32,768 to 1,048,576, rounded down to a multiple of 32."""
        if steps < 1:
            raise ValueError(f'steps must be at least 1, got {steps}')
        prepared = self.preprocess(image, mode=mode, crop_margin=crop_margin)
        pixels = mx.array(np.asarray(prepared, dtype=np.float32) / 255.0)[None]

        key_vae, key_latent, key_camera = mx.random.split(mx.random.key(seed), 3)
        cond = self.image_encoder(pixels, key=key_vae)
        denoiser = self.denoiser
        latent = mx.random.normal((1, denoiser.q_token_length, denoiser.in_channels), key=key_latent)
        camera = mx.random.normal((1, 1, denoiser.cam_channels), key=key_camera)
        neg_cond = tuple(mx.zeros_like(c) for c in cond)
        latent, camera = sample_flow(denoiser, latent, camera, cond=cond, neg_cond=neg_cond, steps=steps,
                                     rescale_t=rescale_t, cfg_strength=cfg_strength, verbose=verbose,
                                     compile=self.compile)
        gaussian = self.decode(latent, num_gaussians, seed=seed)
        return IdolOutput(gaussian=gaussian, preprocessed_image=prepared,
                          camera=decode_camera(np.array(camera[0, 0])))

    def decode(self, latent: mx.array, num_gaussians: int = 262144, seed: int = 42) -> Gaussian:
        """Latent [1, 8192, 16] -> Gaussians: the octree decoder places num_gaussians / 32 anchor points, the
        Gaussian decoder turns each anchor into 32 Gaussians."""
        gaussian = decode_gaussians(self.decoder, self.decoder_gs, latent, num_gaussians,
                                    level=self.config['max_voxel_level'], uniform=np.random.default_rng(seed).random)
        mx.eval(gaussian.xyz, gaussian.features_dc, gaussian.scaling, gaussian.rotation, gaussian.opacity)
        return gaussian

"""Generate 3D Gaussians (.ply) from an image or a folder of images."""
import argparse
import importlib.util
import json
import os
import platform

from .hub import EncoderAccessError

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
BACKENDS = ('auto', 'torch', 'mlx')


def resolve_backend(backend: str) -> str:
    """'auto': MLX on Apple Silicon when mlx is installed, torch otherwise."""
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}, got {backend!r}')
    if backend != 'auto':
        return backend
    apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    return 'mlx' if apple_silicon and importlib.util.find_spec('mlx') is not None else 'torch'


def load_pipeline(backend: str, weights: str, dinov3_dtype: str = 'bf16', optimize: bool = True):
    if resolve_backend(backend) == 'mlx':
        from .mlx import IdolMLXPipeline
        return IdolMLXPipeline.from_pretrained(weights, dinov3_dtype=dinov3_dtype, compile=optimize)
    from .pipeline import IdolPipeline
    return IdolPipeline.from_pretrained(weights, dinov3_dtype=dinov3_dtype, optimize=optimize)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, help='an image, or a folder of images')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic',
                        help="'synthetic': renders and generated images of one object (crop 1.2); "
                             "'real': photos (crop 1.7)")
    parser.add_argument('--crop_margin', type=float, default=None,
                        help='override the crop: square side = object bounding box x this')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg_strength', type=float, default=3.0, help='classifier-free guidance strength')
    parser.add_argument('--rescale_t', type=float, default=3.0, help='time-step shift towards high noise')
    parser.add_argument('--num_gaussians', type=int, default=262144,
                        help='32,768 to 1,048,576, a multiple of 32')
    parser.add_argument('--weights', default='speridlabs/idol', help='Hugging Face repo or local folder')
    parser.add_argument('--dinov3_dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--backend', choices=BACKENDS, default='auto',
                        help="'mlx': Apple Silicon (no PyTorch); 'torch': CUDA or CPU; 'auto': MLX on Apple Silicon "
                             "when mlx is installed, torch otherwise")
    parser.add_argument('--no_optimize', dest='optimize', action='store_false',
                        help='turn off the faster sampling (torch: torch.compile, batched guidance, cuDNN attention, '
                             'TF32, on by default on GPU; mlx: mx.compile of the denoiser step)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.isdir(args.image):
        images = sorted(os.path.join(args.image, name) for name in os.listdir(args.image)
                        if name.lower().endswith(IMAGE_EXTENSIONS))
    else:
        images = [args.image]
    os.makedirs(args.out, exist_ok=True)

    try:
        pipeline = load_pipeline(args.backend, args.weights, dinov3_dtype=args.dinov3_dtype, optimize=args.optimize)
    except EncoderAccessError as error:
        raise SystemExit(f'error: {error}')
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        output = pipeline(path, mode=args.mode, crop_margin=args.crop_margin, seed=args.seed, steps=args.steps,
                          cfg_strength=args.cfg_strength, rescale_t=args.rescale_t,
                          num_gaussians=args.num_gaussians, verbose=True)
        output.gaussian.save_ply(os.path.join(args.out, f'{name}.ply'))
        output.preprocessed_image.save(os.path.join(args.out, f'{name}_input.png'))
        with open(os.path.join(args.out, f'{name}_camera.json'), 'w') as f:
            json.dump(output.camera, f, indent=2)
        print(f'{name}: {args.out}/{name}.ply')


if __name__ == '__main__':
    main()

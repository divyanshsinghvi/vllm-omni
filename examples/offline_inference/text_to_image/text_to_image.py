# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with vllm-omni.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Diffusion model name or local path.")
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
    parser.add_argument("--negative_prompt", default=None, help="Negative text prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        dest="guidance_scale",
        help="Alias for --guidance_scale.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="generated_image.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
    )

    # Filter out None values to let the model use its defaults
    generate_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "generator": generator,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_outputs_per_prompt": args.num_images_per_prompt,
    }
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    images = omni.generate(**generate_kwargs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "generated_image"
    if args.num_images_per_prompt <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AMD Micro-World Text-to-World (T2W) example.

Generates action-controlled video from a text prompt with keyboard/mouse
inputs.  The user controls a first-person camera via WASD + mouse.

Prerequisites
─────────────
1. Download the Micro-World transformer:
       huggingface-cli download amd/Micro-World-T2W --local-dir ./Micro-World-T2W

2. Download the base Wan2.1-T2V-1.3B model (provides tokenizer, text
   encoder, and VAE):
       huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

Usage
─────
    # Walk forward through a forest (default)
    python text_to_world.py \
        --transformer-path ./Micro-World-T2W/transformer \
        --base-model-path ./Wan2.1-T2V-1.3B

    # Strafe left while looking right
    python text_to_world.py \
        --transformer-path ./Micro-World-T2W/transformer \
        --base-model-path ./Wan2.1-T2V-1.3B \
        --prompt "Exploring an ancient temple in first person perspective" \
        --actions "strafe_left+look_right"

    # Custom action sequence via JSON file
    python text_to_world.py \
        --transformer-path ./Micro-World-T2W/transformer \
        --base-model-path ./Wan2.1-T2V-1.3B \
        --action-file my_actions.json

Action format
─────────────
Keyboard: 7-dim binary vector per raw frame [W, S, A, D, Space, Shift, Ctrl]
Mouse:    2-dim float vector per raw frame   [mouse_y, mouse_x]

The number of raw action frames equals num_frames (one per output frame).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# ── Preset action sequences ─────────────────────────────────────────────

_ACTION_PRESETS = {
    "walk_forward": {"keyboard": [1, 0, 0, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
    "walk_backward": {"keyboard": [0, 1, 0, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
    "strafe_left": {"keyboard": [0, 0, 1, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
    "strafe_right": {"keyboard": [0, 0, 0, 1, 0, 0, 0], "mouse": [0.0, 0.0]},
    "jump_forward": {"keyboard": [1, 0, 0, 0, 1, 0, 0], "mouse": [0.0, 0.0]},
    "sprint_forward": {"keyboard": [1, 0, 0, 0, 0, 1, 0], "mouse": [0.0, 0.0]},
    "look_left": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [0.0, -3.0]},
    "look_right": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [0.0, 3.0]},
    "look_up": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [-3.0, 0.0]},
    "look_down": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [3.0, 0.0]},
    "stand_still": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
}


def build_actions(preset_name: str, num_frames: int) -> tuple[list, list]:
    """Build action lists from a preset name (supports '+' combos)."""
    parts = preset_name.split("+")
    keyboard = [0] * 7
    mouse = [0.0, 0.0]
    for part in parts:
        part = part.strip()
        if part not in _ACTION_PRESETS:
            raise ValueError(f"Unknown action preset '{part}'. Available: {', '.join(sorted(_ACTION_PRESETS.keys()))}")
        p = _ACTION_PRESETS[part]
        keyboard = [max(a, b) for a, b in zip(keyboard, p["keyboard"])]
        mouse = [a + b for a, b in zip(mouse, p["mouse"])]

    keyboard_actions = [keyboard] * num_frames
    mouse_actions = [mouse] * num_frames
    return keyboard_actions, mouse_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate action-controlled video with AMD Micro-World T2W.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available action presets: {', '.join(sorted(_ACTION_PRESETS.keys()))}",
    )
    parser.add_argument(
        "--transformer-path",
        required=True,
        help="Path to Micro-World-T2W/transformer directory.",
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help="Path to base Wan2.1-T2V-1.3B model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Running along a cliffside path in a tropical island in first person perspective, "
        "with turquoise waters crashing against the rocks below.",
    )
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument(
        "--actions",
        default="walk_forward",
        help="Action preset name or combo (e.g. 'walk_forward+look_right').",
    )
    parser.add_argument(
        "--action-file",
        default=None,
        help="JSON file with {mouse_actions, keyboard_actions} lists.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=352, help="Video height.")
    parser.add_argument("--width", type=int, default=640, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of output video frames (4n+1).")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--flow-shift", type=int, default=3, help="Scheduler flow shift.")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS.")
    parser.add_argument("--output", type=str, default="micro_world_t2w_output.mp4", help="Output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16

    # Build or load actions
    if args.action_file:
        with open(args.action_file) as f:
            action_data = json.load(f)
        keyboard_actions = action_data["keyboard_actions"]
        mouse_actions = action_data["mouse_actions"]
    else:
        keyboard_actions, mouse_actions = build_actions(args.actions, args.num_frames)

    # Convert to tensors [1, num_frames, dim]
    keyboard_tensor = torch.tensor(keyboard_actions, dtype=dtype, device=device).unsqueeze(0)
    mouse_tensor = torch.tensor(mouse_actions, dtype=dtype, device=device).unsqueeze(0)

    # ── Load model components ────────────────────────────────────────────

    # For loading original-format Wan2.1 base model components, we use the
    # Micro-World repo's own loaders which handle the .pth weight format.
    import sys

    # Try to use Micro-World's native loaders for the base model
    micro_world_repo = str(Path(__file__).resolve().parents[3] / "Micro-World")
    if not Path(micro_world_repo).exists():
        # Fallback: try /workspace/Micro-World
        micro_world_repo = "/workspace/Micro-World"
    if Path(micro_world_repo).exists():
        sys.path.insert(0, micro_world_repo)

    from microworld.models import AutoencoderKLWan, AutoTokenizer, WanActionControlNetModel, WanT5EncoderModel
    from microworld.pipeline.pipeline_wan_action_t2w import WanActionT2WPipeline
    from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(micro_world_repo) / "config" / "wan2.1" / "wan_civitai.yaml")

    print("Loading transformer...")
    transformer = WanActionControlNetModel.from_pretrained(
        args.transformer_path, low_cpu_mem_usage=False, torch_dtype=dtype
    )

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        str(Path(args.base_model_path) / config["vae_kwargs"]["vae_subpath"]),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.base_model_path) / config["text_encoder_kwargs"]["tokenizer_subpath"]),
    )

    print("Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(Path(args.base_model_path) / config["text_encoder_kwargs"]["text_encoder_subpath"]),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    ).eval()

    print("Setting up pipeline...")
    scheduler = FlowUniPCMultistepScheduler(shift=1)
    pipeline = WanActionT2WPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    ).to(device)

    # ── Generate ─────────────────────────────────────────────────────────

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"\n{'=' * 60}")
    print("Micro-World T2W Generation:")
    print(f"  Prompt:     {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Actions:    {args.actions}")
    print(f"  Video:      {args.width}x{args.height}, {args.num_frames} frames @ {args.fps} fps")
    print(f"  Steps:      {args.num_inference_steps}, guidance: {args.guidance_scale}")
    print(f"  Seed:       {args.seed}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    with torch.no_grad():
        result = pipeline(
            args.prompt,
            mouse_actions=mouse_tensor,
            keyboard_actions=keyboard_tensor,
            negative_prompt=args.negative_prompt or None,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            shift=args.flow_shift,
            with_ui=False,
        ).videos
    generation_time = time.perf_counter() - generation_start

    print(f"\nGeneration time: {generation_time:.2f}s")
    print(f"Output shape: {result.shape}")

    # ── Save video ───────────────────────────────────────────────────────

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video

    # result shape: (batch, channels, frames, height, width) as numpy
    video = result
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if video.ndim == 5:
        video = video[0]  # remove batch → (C, F, H, W)
    if video.shape[0] in (3, 4):
        video = np.transpose(video, (1, 2, 3, 0))  # → (F, H, W, C)

    export_to_video(video, str(output_path), fps=args.fps)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()

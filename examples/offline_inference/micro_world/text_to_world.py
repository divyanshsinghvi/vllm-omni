# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AMD Micro-World Text-to-World (T2W) example.

Generates action-controlled video from a text prompt with keyboard/mouse inputs.

Setup
─────
    huggingface-cli download amd/Micro-World-T2W --local-dir ./Micro-World-T2W
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./Wan2.1-Base

    mkdir -p ./Micro-World-T2W-combined
    ln -s $(pwd)/Micro-World-T2W/transformer   ./Micro-World-T2W-combined/transformer
    ln -s $(pwd)/Wan2.1-Base/tokenizer         ./Micro-World-T2W-combined/tokenizer
    ln -s $(pwd)/Wan2.1-Base/text_encoder      ./Micro-World-T2W-combined/text_encoder
    ln -s $(pwd)/Wan2.1-Base/vae               ./Micro-World-T2W-combined/vae

Usage
─────
    python text_to_world.py --model ./Micro-World-T2W-combined
    python text_to_world.py --model ./Micro-World-T2W-combined --actions "strafe_left+look_right"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

_ACTION_PRESETS = {
    "walk_forward": {"keyboard": [1, 0, 0, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
    "walk_backward": {"keyboard": [0, 1, 0, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
    "strafe_left": {"keyboard": [0, 0, 1, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
    "strafe_right": {"keyboard": [0, 0, 0, 1, 0, 0, 0], "mouse": [0.0, 0.0]},
    "jump_forward": {"keyboard": [1, 0, 0, 0, 1, 0, 0], "mouse": [0.0, 0.0]},
    "sprint_forward": {"keyboard": [1, 0, 0, 0, 0, 1, 0], "mouse": [0.0, 0.0]},
    "look_left": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [0.0, -3.0]},
    "look_right": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [0.0, 3.0]},
    "stand_still": {"keyboard": [0, 0, 0, 0, 0, 0, 0], "mouse": [0.0, 0.0]},
}


def build_actions(preset_name: str, num_frames: int) -> tuple[list, list]:
    """Build action lists from a preset name (supports '+' combos like 'walk_forward+look_right')."""
    keyboard = [0] * 7
    mouse = [0.0, 0.0]
    for part in preset_name.split("+"):
        part = part.strip()
        if part not in _ACTION_PRESETS:
            raise ValueError(f"Unknown preset '{part}'. Available: {sorted(_ACTION_PRESETS.keys())}")
        p = _ACTION_PRESETS[part]
        keyboard = [max(a, b) for a, b in zip(keyboard, p["keyboard"])]
        mouse = [a + b for a, b in zip(mouse, p["mouse"])]
    return [keyboard] * num_frames, [mouse] * num_frames


def parse_reference_action_list(action_list: list) -> tuple[list, list]:
    """Parse Micro-World reference action_list format.

    Format: ``[[end_frame, "w s a d shift ctrl _ mouse_y mouse_x"], ..., "space_frames"]``
    e.g.  ``[[20, "1 0 0 0 0 0 0 0 0"], [40, "0 1 0 0 0 0 0 0 5"], "30 60"]``

    Returns: ``(keyboard_actions, mouse_actions)`` as per-frame lists.
    """
    keyboard = [[0, 0, 0, 0, 0, 0, 0]]
    mouse = [[0.0, 0.0]]
    space_frames = set(map(int, action_list[-1].split())) if action_list[-1] else set()

    for i in range(len(action_list) - 1):
        end_frame, action = action_list[i]
        w, s, a, d, shift, ctrl, _unused, my, mx = map(float, action.split())
        start_frame = 1 if i == 0 else action_list[i - 1][0] + 1
        for frame in range(start_frame, int(end_frame) + 1):
            keyboard.append([int(w), int(s), int(a), int(d), int(frame in space_frames), int(shift), int(ctrl)])
            mouse.append([float(my), float(mx)])
    return keyboard, mouse


def parse_args():
    parser = argparse.ArgumentParser(description="AMD Micro-World T2W generation.")
    parser.add_argument("--model", required=True, help="Combined model directory path.")
    parser.add_argument(
        "--prompt", default="Running along a cliffside path in a tropical island in first person perspective."
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--actions", default="walk_forward", help="Action preset (e.g. 'walk_forward+look_right').")
    parser.add_argument("--action-file", default=None, help="JSON file with {mouse_actions, keyboard_actions}.")
    parser.add_argument(
        "--action-list",
        default=None,
        help=(
            "Micro-World reference action_list as JSON string, e.g. "
            '\'[[20, "1 0 0 0 0 0 0 0 0"], [40, "0 1 0 0 0 0 0 0 5"], "30 60"]\''
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=str, default="micro_world_t2w_output.mp4")
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=3000,
        help="Stage init timeout in seconds (raise for slow filesystems).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.action_file:
        with open(args.action_file) as f:
            action_data = json.load(f)
        keyboard_actions = action_data["keyboard_actions"]
        mouse_actions = action_data["mouse_actions"]
    elif args.action_list:
        keyboard_actions, mouse_actions = parse_reference_action_list(json.loads(args.action_list))
    else:
        keyboard_actions, mouse_actions = build_actions(args.actions, args.num_frames)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    stage_config = str(
        Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_t2w.yaml"
    )
    omni = Omni(
        model=args.model,
        stage_configs_path=stage_config,
        flow_shift=args.flow_shift,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.stage_init_timeout,
    )

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    print(f"Generating {args.num_frames} frames at {args.width}x{args.height} with actions={args.actions}")
    t0 = time.perf_counter()
    outputs = omni.generate(
        prompt_dict,
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            extra_args={"mouse_actions": mouse_actions, "keyboard_actions": keyboard_actions},
        ),
    )
    print(f"Generation time: {time.perf_counter() - t0:.2f}s")

    # Extract frames
    first = outputs[0] if isinstance(outputs, list) else outputs
    if hasattr(first, "is_pipeline_output") and first.is_pipeline_output:
        first = first.request_output
    frames = first.images[0] if hasattr(first, "images") and first.images else None
    if frames is None:
        raise ValueError("No video frames in output.")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0].permute(1, 2, 3, 0) if video.shape[1] in (3, 4) else video[0]
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        video = video.float().numpy()
    elif isinstance(frames, np.ndarray):
        video = frames[0] if frames.ndim == 5 else frames
    else:
        video = frames

    export_to_video(video, str(output_path), fps=args.fps)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

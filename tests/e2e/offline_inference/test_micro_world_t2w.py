import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["amd/Micro-World-T2W"]


def test_micro_world_t2w_action_module():
    """Test ActionModule produces correct output shapes."""
    from vllm_omni.diffusion.models.micro_world.action_module import ActionModule

    # T2W variant (flatten_spatial=True)
    am_t2w = ActionModule(flatten_spatial=True)
    mouse = torch.randn(1, 81, 2)
    keyboard = torch.randn(1, 81, 7)
    grid_sizes = (21, 30, 45)
    out = am_t2w(mouse, keyboard, grid_sizes)
    assert out.shape == (1, 21 * 30 * 45, 1536)

    # I2W variant (flatten_spatial=False)
    am_i2w = ActionModule(flatten_spatial=False)
    out = am_i2w(mouse, keyboard, grid_sizes)
    assert out.shape == (1, 21, 1536)


def test_micro_world_t2w_registry():
    """Test both pipelines are registered in all registries."""
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert "MicroWorldT2WPipeline" in _DIFFUSION_MODELS
    assert "MicroWorldI2WPipeline" in _DIFFUSION_MODELS
    assert "MicroWorldT2WPipeline" in _DIFFUSION_POST_PROCESS_FUNCS
    assert "MicroWorldI2WPipeline" in _DIFFUSION_POST_PROCESS_FUNCS
    assert "MicroWorldT2WPipeline" in _DIFFUSION_PRE_PROCESS_FUNCS
    assert "MicroWorldI2WPipeline" in _DIFFUSION_PRE_PROCESS_FUNCS


@pytest.mark.parametrize("model_name", models)
def test_micro_world_t2w_generation(model_name: str):
    """E2E test: generate action-controlled video via Omni entrypoint."""
    m = Omni(
        model=model_name,
        flow_shift=3.0,
    )

    height = 352
    width = 640
    num_frames = 17
    # 17 latent frames * temporal_ratio(4) + 1 = 69 raw action frames
    num_action_frames = 69
    # Walk forward: W=1, rest=0
    keyboard_actions = [[1, 0, 0, 0, 0, 0, 0]] * num_action_frames
    mouse_actions = [[0.0, 0.0]] * num_action_frames

    outputs = m.generate(
        prompts="A first person view walking through a forest",
        sampling_params_list=OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=2,
            guidance_scale=3.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            extra_args={
                "mouse_actions": mouse_actions,
                "keyboard_actions": keyboard_actions,
            },
        ),
    )
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    frames = req_out.images[0]

    assert frames is not None
    assert hasattr(frames, "shape")
    assert frames.shape[1] == num_frames
    assert frames.shape[2] == height
    assert frames.shape[3] == width

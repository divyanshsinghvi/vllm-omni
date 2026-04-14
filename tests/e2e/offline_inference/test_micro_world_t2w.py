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

models = [os.environ.get("MICRO_WORLD_T2W_MODEL", "amd/Micro-World-T2W")]


# ── Unit tests ───────────────────────────────────────────────────────────


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


# ── Correlation / regression tests ───────────────────────────────────────


def test_action_module_sensitivity():
    """Different actions must produce different outputs (action signal is not ignored)."""
    from vllm_omni.diffusion.models.micro_world.action_module import ActionModule

    torch.manual_seed(0)
    am = ActionModule(flatten_spatial=True)
    am.eval()

    grid_sizes = (5, 22, 40)

    # Walk forward: W=1
    mouse_walk = torch.zeros(1, 69, 2)
    keyboard_walk = torch.zeros(1, 69, 7)
    keyboard_walk[0, :, 0] = 1.0

    # Stand still: all zeros
    mouse_still = torch.zeros(1, 69, 2)
    keyboard_still = torch.zeros(1, 69, 7)

    with torch.no_grad():
        out_walk = am(mouse_walk, keyboard_walk, grid_sizes)
        out_still = am(mouse_still, keyboard_still, grid_sizes)

    mse = ((out_walk - out_still) ** 2).mean().item()
    assert mse > 1e-6, f"Walk and still outputs are too similar (MSE={mse:.8f}). Actions may be silently ignored."


def test_action_module_mouse_sensitivity():
    """Different mouse inputs must produce different outputs."""
    from vllm_omni.diffusion.models.micro_world.action_module import ActionModule

    torch.manual_seed(0)
    am = ActionModule(flatten_spatial=True)
    am.eval()

    grid_sizes = (5, 22, 40)
    keyboard = torch.zeros(1, 69, 7)

    mouse_left = torch.zeros(1, 69, 2)
    mouse_left[0, :, 0] = -5.0  # look left

    mouse_right = torch.zeros(1, 69, 2)
    mouse_right[0, :, 0] = 5.0  # look right

    with torch.no_grad():
        out_left = am(mouse_left, keyboard, grid_sizes)
        out_right = am(mouse_right, keyboard, grid_sizes)

    mse = ((out_left - out_right) ** 2).mean().item()
    assert mse > 1e-6, (
        f"Left and right mouse outputs are too similar (MSE={mse:.8f}). Mouse actions may be silently ignored."
    )


# ── E2E generation tests ─────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", models)
def test_micro_world_t2w_generation(model_name: str):
    """E2E test: generate action-controlled video via Omni entrypoint."""
    stage_config = str(
        Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_t2w.yaml"
    )
    m = Omni(
        model=model_name,
        stage_configs_path=stage_config,
        flow_shift=3.0,
        stage_init_timeout=3000,
        init_timeout=3000,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_micro_world_t2w_action_changes_output():
    """Regression: different actions must produce different generated videos.

    Runs the pipeline twice with the same seed and prompt but different
    keyboard actions (walk-forward vs stand-still).  If the MSE between
    the two outputs is below a threshold, the action path is broken.

    Golden reference (seed=42, 10 steps, 352x640, 17 frames):
        walk  mean=0.1749  std=0.1279
        still mean=0.1843  std=0.1535
        MSE(walk, still) = 0.0126
    """
    base_model = os.environ.get("WAN21_T2V_PATH", "pretrained_models/Wan2.1-T2V-1.3B")
    t2w_path = os.environ.get("MICRO_WORLD_T2W_PATH", "pretrained_models/Micro-World-T2W/transformer")
    micro_world_repo = os.environ.get("MICRO_WORLD_REPO", "/workspace/Micro-World")

    for path in [base_model, t2w_path, micro_world_repo]:
        if not os.path.exists(path):
            pytest.skip(f"Required path not found: {path}")

    sys.path.insert(0, micro_world_repo)
    from microworld.models import AutoencoderKLWan, AutoTokenizer, WanActionControlNetModel, WanT5EncoderModel
    from microworld.pipeline.pipeline_wan_action_t2w import WanActionT2WPipeline
    from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from microworld.utils.utils import parse_action_list
    from omegaconf import OmegaConf

    device = "cuda"
    dtype = torch.bfloat16
    config = OmegaConf.load(os.path.join(micro_world_repo, "config/wan2.1/wan_civitai.yaml"))

    transformer = WanActionControlNetModel.from_pretrained(t2w_path, low_cpu_mem_usage=False, torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base_model, config["vae_kwargs"]["vae_subpath"]),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_model, config["text_encoder_kwargs"]["tokenizer_subpath"])
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base_model, config["text_encoder_kwargs"]["text_encoder_subpath"]),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    ).eval()

    scheduler = FlowUniPCMultistepScheduler(shift=1)
    pipeline = WanActionT2WPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler
    ).to(device)

    prompt = "A first person view in a forest"

    # Walk forward
    kb_walk, ms_walk = parse_action_list([[16, "1 0 0 0 0 0 0 0 0"], ""])
    kb_walk = kb_walk[None].to(device=device, dtype=dtype)
    ms_walk = ms_walk[None].to(device=device, dtype=dtype)

    # Stand still
    kb_still, ms_still = parse_action_list([[16, "0 0 0 0 0 0 0 0 0"], ""])
    kb_still = kb_still[None].to(device=device, dtype=dtype)
    ms_still = ms_still[None].to(device=device, dtype=dtype)

    with torch.no_grad():
        result_walk = pipeline(
            prompt,
            mouse_actions=ms_walk,
            keyboard_actions=kb_walk,
            num_frames=17,
            height=352,
            width=640,
            generator=torch.Generator(device=device).manual_seed(42),
            guidance_scale=3.0,
            num_inference_steps=10,
            shift=3,
            with_ui=False,
        ).videos

        result_still = pipeline(
            prompt,
            mouse_actions=ms_still,
            keyboard_actions=kb_still,
            num_frames=17,
            height=352,
            width=640,
            generator=torch.Generator(device=device).manual_seed(42),
            guidance_scale=3.0,
            num_inference_steps=10,
            shift=3,
            with_ui=False,
        ).videos

    walk_t = torch.from_numpy(result_walk) if not isinstance(result_walk, torch.Tensor) else result_walk
    still_t = torch.from_numpy(result_still) if not isinstance(result_still, torch.Tensor) else result_still

    mse = ((walk_t.float() - still_t.float()) ** 2).mean().item()

    # Actions must produce meaningfully different outputs
    assert mse > 0.001, (
        f"Walk and still videos are too similar (MSE={mse:.6f}). The action injection path may be broken."
    )

    # Sanity: outputs should be valid images
    for name, t in [("walk", walk_t), ("still", still_t)]:
        assert t.min() >= 0.0, f"{name} has negative pixels"
        assert t.max() <= 1.0, f"{name} exceeds 1.0"
        assert not torch.isnan(t).any(), f"{name} contains NaN"
        assert t.std() > 0.01, f"{name} appears constant"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_micro_world_t2w_golden_statistics():
    """Regression: output pixel statistics must match golden reference.

    Golden reference (seed=42, 10 steps, 352x640, 17 frames, bfloat16):
        walk  mean≈0.1749  std≈0.1279
        still mean≈0.1843  std≈0.1535
    Tolerance is set wide (±0.05) to allow for minor numerical
    differences across GPU architectures.
    """
    base_model = os.environ.get("WAN21_T2V_PATH", "pretrained_models/Wan2.1-T2V-1.3B")
    t2w_path = os.environ.get("MICRO_WORLD_T2W_PATH", "pretrained_models/Micro-World-T2W/transformer")
    micro_world_repo = os.environ.get("MICRO_WORLD_REPO", "/workspace/Micro-World")

    for path in [base_model, t2w_path, micro_world_repo]:
        if not os.path.exists(path):
            pytest.skip(f"Required path not found: {path}")

    sys.path.insert(0, micro_world_repo)
    from microworld.models import AutoencoderKLWan, AutoTokenizer, WanActionControlNetModel, WanT5EncoderModel
    from microworld.pipeline.pipeline_wan_action_t2w import WanActionT2WPipeline
    from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from microworld.utils.utils import parse_action_list
    from omegaconf import OmegaConf

    device = "cuda"
    dtype = torch.bfloat16
    config = OmegaConf.load(os.path.join(micro_world_repo, "config/wan2.1/wan_civitai.yaml"))

    transformer = WanActionControlNetModel.from_pretrained(t2w_path, low_cpu_mem_usage=False, torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(base_model, config["vae_kwargs"]["vae_subpath"]),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_model, config["text_encoder_kwargs"]["tokenizer_subpath"])
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base_model, config["text_encoder_kwargs"]["text_encoder_subpath"]),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    ).eval()

    scheduler = FlowUniPCMultistepScheduler(shift=1)
    pipeline = WanActionT2WPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler
    ).to(device)

    kb, ms = parse_action_list([[16, "1 0 0 0 0 0 0 0 0"], ""])
    kb = kb[None].to(device=device, dtype=dtype)
    ms = ms[None].to(device=device, dtype=dtype)

    with torch.no_grad():
        result = pipeline(
            "A first person view in a forest",
            mouse_actions=ms,
            keyboard_actions=kb,
            num_frames=17,
            height=352,
            width=640,
            generator=torch.Generator(device=device).manual_seed(42),
            guidance_scale=3.0,
            num_inference_steps=10,
            shift=3,
            with_ui=False,
        ).videos

    t = torch.from_numpy(result) if not isinstance(result, torch.Tensor) else result

    # Golden values (tolerance ±0.05 for cross-GPU reproducibility)
    golden_mean = 0.1749
    golden_std = 0.1279
    tolerance = 0.05

    actual_mean = t.mean().item()
    actual_std = t.std().item()

    assert abs(actual_mean - golden_mean) < tolerance, (
        f"Output mean {actual_mean:.4f} deviates from golden {golden_mean:.4f} by more than {tolerance}"
    )
    assert abs(actual_std - golden_std) < tolerance, (
        f"Output std {actual_std:.4f} deviates from golden {golden_std:.4f} by more than {tolerance}"
    )

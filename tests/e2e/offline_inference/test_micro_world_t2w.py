import os
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


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


def test_micro_world_t2w_weight_mapping():
    """Test weight name mapping from Wan2.1 original to vllm-omni format."""
    safetensors = pytest.importorskip("safetensors.torch")

    model_path = os.environ.get(
        "MICRO_WORLD_T2W_PATH",
        "pretrained_models/Micro-World-T2W/transformer/diffusion_pytorch_model.safetensors",
    )
    if not os.path.exists(model_path):
        pytest.skip(f"Model weights not found at {model_path}")

    weights = safetensors.load_file(model_path)

    # Simulate the full mapping pipeline
    weight_name_remapping = [
        ("head.head.", "proj_out."),
        ("head.modulation", "output_scale_shift_prepare.scale_shift_table"),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
    ]

    stacked_params = [
        (".attn1.to_qkv", ".attn1.to_q", "q"),
        (".attn1.to_qkv", ".attn1.to_k", "k"),
        (".attn1.to_qkv", ".attn1.to_v", "v"),
    ]

    def map_name(name):
        for old, new in weight_name_remapping:
            if old in name:
                name = name.replace(old, new)
                break
        name = name.replace(".self_attn.q.", ".attn1.to_q.")
        name = name.replace(".self_attn.k.", ".attn1.to_k.")
        name = name.replace(".self_attn.v.", ".attn1.to_v.")
        name = name.replace(".self_attn.o.", ".attn1.to_out.")
        name = name.replace(".self_attn.norm_q.", ".attn1.norm_q.")
        name = name.replace(".self_attn.norm_k.", ".attn1.norm_k.")
        name = name.replace(".cross_attn.q.", ".attn2.to_q.")
        name = name.replace(".cross_attn.k.", ".attn2.to_k.")
        name = name.replace(".cross_attn.v.", ".attn2.to_v.")
        name = name.replace(".cross_attn.o.", ".attn2.to_out.")
        name = name.replace(".cross_attn.norm_q.", ".attn2.norm_q.")
        name = name.replace(".cross_attn.norm_k.", ".attn2.norm_k.")
        if ".modulation" in name and "action_preprocess" not in name:
            name = name.replace(".modulation", ".scale_shift_table")
        if ".ffn.0." in name:
            name = name.replace(".ffn.0.", ".ffn.net_0.")
        elif ".ffn.2." in name:
            name = name.replace(".ffn.2.", ".ffn.net_2.")
        for param_name, weight_name, _ in stacked_params:
            if weight_name in name:
                name = name.replace(weight_name, param_name)
                break
        return name

    mapped_keys = {map_name(k) for k in weights}

    # 1278 HF keys should map to 1098 unique keys (180 merged via QKV fusion)
    assert len(weights) == 1278
    assert len(mapped_keys) == 1098

    # No unfused Q keys should remain
    unfused_q = [k for k in mapped_keys if ".attn1.to_q." in k]
    assert len(unfused_q) == 0, f"Unfused Q keys found: {unfused_q}"

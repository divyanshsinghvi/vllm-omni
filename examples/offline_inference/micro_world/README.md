# AMD Micro-World (T2W)

[AMD Micro-World](https://github.com/AMD-AGI/Micro-World) is a lightweight
action-controlled interactive world model built on Wan2.1. Users drive a
first-person camera through video frames via keyboard (W/A/S/D/Space/Shift/Ctrl)
and mouse inputs, with the same diffusion model that generated the scene also
rendering the keypress HUD overlays into the output.

## Setup

Install dependencies:
```
uv pip install -e .
```

Download the action-conditioned LoRA + transformer (AMD) and the base Wan2.1
weights (Wan-AI):
```
huggingface-cli download amd/Micro-World-T2W --local-dir ./Micro-World-T2W
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./Wan2.1-Base
```

The pipeline expects a single combined directory that mixes the AMD-trained
transformer with Wan2.1's tokenizer, text encoder, and VAE. The transformer
config and a Kohya-style LoRA come from `Micro-World-T2W`; everything else
comes from the Wan2.1 release:
```
mkdir -p ./Micro-World-T2W-combined
ln -s $(pwd)/Micro-World-T2W/transformer   ./Micro-World-T2W-combined/transformer
ln -s $(pwd)/Micro-World-T2W/lora_diffusion_pytorch_model.safetensors \
                                           ./Micro-World-T2W-combined/lora_diffusion_pytorch_model.safetensors
ln -s $(pwd)/Wan2.1-Base/tokenizer         ./Micro-World-T2W-combined/tokenizer
ln -s $(pwd)/Wan2.1-Base/text_encoder      ./Micro-World-T2W-combined/text_encoder
ln -s $(pwd)/Wan2.1-Base/vae               ./Micro-World-T2W-combined/vae
```

The LoRA is merged into the transformer at pipeline init (look for
`LoRA merge: 306 layers merged, 0 skipped` in the log).

## Run

The example defaults reproduce the AMD reference run (same prompt, action_list,
seed, steps, guidance, flow_shift):
```
python examples/offline_inference/micro_world/text_to_world.py \
  --model ./Micro-World-T2W-combined \
  --output micro_world_t2w_output.mp4
```

Override prompt/actions/seed:
```
python examples/offline_inference/micro_world/text_to_world.py \
  --model ./Micro-World-T2W-combined \
  --prompt "Exploring an ancient jungle ruin in first person perspective." \
  --actions "strafe_left+look_right" --seed 42
```

Custom action list (Micro-World reference format —
`[[end_frame, "w s a d shift ctrl _ mouse_y mouse_x"], ..., "space_frames"]`):
```
python examples/offline_inference/micro_world/text_to_world.py \
  --model ./Micro-World-T2W-combined \
  --action-list '[[20, "1 0 0 0 0 0 0 0 0"], [40, "0 1 0 0 0 0 0 0 5"], "30 60"]'
```

## Implementation Overview

Micro-World runs as a single-stage video diffusion pipeline.

- **Stage 0** (`micro_world_t2w`) takes a text prompt + per-frame
  mouse/keyboard inputs and emits an mp4. Internally:
    1. T5 text encoder produces prompt embeddings (re-padded to `text_len=512`
       with zeros so cross-attention K/V count matches the reference).
    2. `ActionModule` encodes mouse (2-dim) + keyboard (7-dim) per-frame inputs
       into action features via sliding-window-grouped MLPs with sinusoidal
       positional encoding on the keyboard stream.
    3. `MicroWorldControlNetTransformer` runs the diffusion denoising loop:
       at every 2nd block the parallel action branch produces a zero-init
       skip connection that's added back into the main hidden state
       (ControlNet pattern).
    4. The Wan2.1 VAE decodes the final latent into `(num_frames, 352, 640)`
       RGB, written to mp4 at 15 fps.

Key components live in `vllm_omni/diffusion/models/micro_world/`:

- `action_module.py` — `ActionModule`: shared mouse/keyboard preprocessor
  used by both T2W and I2W.
- `micro_world_transformer.py` — `MicroWorldControlNetTransformer` (T2W,
  1.3B) and `MicroWorldAdaLNTransformer` (I2W, 14B). T2W uses a parallel
  ControlNet branch; I2W folds action features into the timestep
  modulation via AdaLN.
- `pipeline_micro_world_t2w.py` — `MicroWorldT2WPipeline`. Hosts the
  scheduler (`FlowUniPCMultistepScheduler`, `flow_shift=3`), VAE, T5
  encoder, and the Kohya LoRA merger.

The transformer extends Wan2.1's block layout (30 layers, dim=1536, ffn=8960,
12 heads, qk_norm via RMSNorm, cross_attn_norm). Wan2.1 → vllm-omni name
remapping is handled in `MicroWorldControlNetTransformer.load_weights`
(`self_attn.q/k/v` → fused `attn1.to_qkv`, `cross_attn.*` → separate
`attn2.to_q/k/v`, `modulation` → `scale_shift_table`, etc.).

Action conditioning is `extra_args={"mouse_actions": [...], "keyboard_actions": [...]}`
on the sampling params.

# AMD Micro-World

[AMD Micro-World](https://github.com/AMD-AGI/Micro-World) is a lightweight
action-controlled interactive world model built on Wan2.1. Users drive a
first-person camera through video frames via keyboard (W/A/S/D/Space/Shift/Ctrl)
and mouse inputs. Two variants are supported:

- **T2W** (`MicroWorldT2WPipeline`, 1.3B): text → action-controlled video.
- **I2W** (`MicroWorldI2WPipeline`, 14B): image + text → action-controlled video.

The keyboard/cursor HUD overlay shown in the AMD reference videos is rendered
as cv2 post-processing on top of the model output (not by the diffusion model
itself); pass `--draw-hud` to enable it.

## Why a custom `model_index.json`?

Both pipelines need a hand-written `model_index.json` in the combined model
directory. The `_class_name` field tells vllm-omni which pipeline class to
instantiate. The Wan-AI base repos ship `model_index.json` with
`_class_name: "WanPipeline"` (T2V base) or `"WanImageToVideoPipeline"` (I2V
base) — if you symlink those into the combined dir, vllm routes to the
diffusers stock pipeline instead of `MicroWorld{T2W,I2W}Pipeline`, the
custom `load_weights` / Kohya LoRA merger never runs, and weights are
silently skipped. So always write a fresh `model_index.json` with the right
`_class_name`.

## T2W — Text-to-World (1.3B)

### Setup

Install dependencies:
```
uv pip install -e .
```

Download the action-conditioned LoRA + transformer (AMD) and the base Wan2.1
T2V weights (Wan-AI):
```
huggingface-cli download amd/Micro-World-T2W --local-dir ./Micro-World-T2W
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./Wan2.1-T2V-Base
```

Build the combined directory (transformer + LoRA from AMD, the rest from
Wan-AI; `model_index.json` is hand-written with the Micro-World pipeline class):
```
mkdir -p ./Micro-World-T2W-combined
ln -s $(pwd)/Micro-World-T2W/transformer   ./Micro-World-T2W-combined/transformer
ln -s $(pwd)/Micro-World-T2W/lora_diffusion_pytorch_model.safetensors \
                                           ./Micro-World-T2W-combined/lora_diffusion_pytorch_model.safetensors
ln -s $(pwd)/Wan2.1-T2V-Base/tokenizer     ./Micro-World-T2W-combined/tokenizer
ln -s $(pwd)/Wan2.1-T2V-Base/text_encoder  ./Micro-World-T2W-combined/text_encoder
ln -s $(pwd)/Wan2.1-T2V-Base/vae           ./Micro-World-T2W-combined/vae

cat > ./Micro-World-T2W-combined/model_index.json << 'EOF'
{
  "_class_name": "MicroWorldT2WPipeline",
  "_diffusers_version": "0.34.0",
  "scheduler": ["diffusers", "UniPCMultistepScheduler"],
  "text_encoder": ["transformers", "UMT5EncoderModel"],
  "tokenizer": ["transformers", "T5TokenizerFast"],
  "transformer": ["diffusers", "MicroWorldControlNetTransformer"],
  "vae": ["diffusers", "AutoencoderKLWan"]
}
EOF
```

The LoRA is merged into the transformer at pipeline init (look for
`LoRA merge: 306 layers merged, 0 skipped` in the log).

### Run

The example defaults reproduce the AMD reference run (same prompt, action_list,
seed=43, 30 steps, guidance=3.0, flow_shift=3.0):
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

Add `--draw-hud` to overlay the WASD + mouse-cursor HUD onto each output frame
(post-processing only; requires `opencv-python`).

## I2W — Image-to-World (14B)

### Setup

Download the I2W transformer + LoRA (AMD) and the base Wan2.1 I2V weights
(Wan-AI). The I2W transformer is ~36GB, so consider scoping the I2V base
download to only the subdirectories the pipeline needs:
```
huggingface-cli download amd/Micro-World-I2W --local-dir ./Micro-World-I2W
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir ./Wan2.1-I2V-Base \
  --include "tokenizer/*" "text_encoder/*" "image_processor/*" "image_encoder/*" "vae/*"
```

Build the combined directory (note the extra `image_processor` and
`image_encoder` symlinks compared to T2W):
```
mkdir -p ./Micro-World-I2W-combined
ln -s $(pwd)/Micro-World-I2W/transformer   ./Micro-World-I2W-combined/transformer
ln -s $(pwd)/Micro-World-I2W/lora_diffusion_pytorch_model.safetensors \
                                           ./Micro-World-I2W-combined/lora_diffusion_pytorch_model.safetensors
ln -s $(pwd)/Wan2.1-I2V-Base/tokenizer        ./Micro-World-I2W-combined/tokenizer
ln -s $(pwd)/Wan2.1-I2V-Base/text_encoder     ./Micro-World-I2W-combined/text_encoder
ln -s $(pwd)/Wan2.1-I2V-Base/image_processor  ./Micro-World-I2W-combined/image_processor
ln -s $(pwd)/Wan2.1-I2V-Base/image_encoder    ./Micro-World-I2W-combined/image_encoder
ln -s $(pwd)/Wan2.1-I2V-Base/vae              ./Micro-World-I2W-combined/vae

cat > ./Micro-World-I2W-combined/model_index.json << 'EOF'
{
  "_class_name": "MicroWorldI2WPipeline",
  "_diffusers_version": "0.34.0",
  "scheduler": ["diffusers", "UniPCMultistepScheduler"],
  "text_encoder": ["transformers", "UMT5EncoderModel"],
  "tokenizer": ["transformers", "T5TokenizerFast"],
  "image_encoder": ["transformers", "CLIPVisionModel"],
  "image_processor": ["transformers", "CLIPImageProcessor"],
  "transformer": ["diffusers", "MicroWorldAdaLNTransformer"],
  "vae": ["diffusers", "AutoencoderKLWan"]
}
EOF
```

I2W's LoRA covers 488 sites (vs. T2W's 306 — the extras are the I2V image
cross-attention `cross_attn_k_img/v_img` and the `img_emb` MLP). Look for
`LoRA merge: 488 layers merged, 0 skipped` in the log to confirm.

### Run

The example defaults reproduce the AMD reference run for I2W
(`asset/street_night.jpg` + walk-down action sequence, seed=43, 30 steps,
guidance=6.0, flow_shift=3.0):
```
python examples/offline_inference/micro_world/image_to_world.py \
  --model ./Micro-World-I2W-combined \
  --image ./Micro-World/asset/street_night.jpg \
  --output micro_world_i2w_output.mp4
```

Override image / prompt / actions:
```
python examples/offline_inference/micro_world/image_to_world.py \
  --model ./Micro-World-I2W-combined \
  --image my_scene.jpg \
  --prompt "First-person walk through a forest path." \
  --action-list '[[40, "1 0 0 0 0 0 0 0 0"], [80, "0 0 1 0 0 0 0 0 0"], "40"]'
```

`--draw-hud` works the same as in T2W.

## Implementation Overview

Both pipelines are single-stage video diffusion. T2W uses ControlNet-style
parallel action blocks; I2W uses AdaLN action modulation.

### T2W (`MicroWorldT2WPipeline`)

1. T5 text encoder produces prompt embeddings (re-padded to `text_len=512`
   with zeros so cross-attention K/V count matches the reference).
2. `ActionModule` encodes mouse (2-dim) + keyboard (7-dim) per-frame inputs
   into action features via sliding-window-grouped MLPs with sinusoidal
   positional encoding on the keyboard stream.
3. `MicroWorldControlNetTransformer` runs the diffusion denoising loop:
   at every 2nd block the parallel action branch produces a zero-init
   skip connection that's added back into the main hidden state.
4. The Wan2.1 VAE decodes the final latent into `(num_frames, 352, 640)`
   RGB, written to mp4 at 15 fps.

### I2W (`MicroWorldI2WPipeline`)

Same flow with two additions:

1. CLIP image encoder + image embedder MLP project the input image into the
   transformer's hidden dim. The patch_embed input is 36-channel (16 noise +
   16 image latent + 4 mask), not 16.
2. Each transformer block has an extra image-K/V cross-attention path
   (`add_k_proj` / `add_v_proj` / `norm_added_k`) that attends from the
   spatial-temporal hidden state to the projected CLIP image embedding,
   alongside the regular text cross-attention.
3. `MicroWorldAdaLNTransformer` injects action features into each block's
   timestep modulation (AdaLN) instead of using a parallel ControlNet
   branch — better suited for the 14B model.

Key components live in `vllm_omni/diffusion/models/micro_world/`:

- `action_module.py` — `ActionModule`: shared mouse/keyboard preprocessor.
- `micro_world_transformer.py` — `MicroWorldControlNetTransformer` (T2W) and
  `MicroWorldAdaLNTransformer` (I2W). Both subclass `WanTransformer3DModel`
  with custom `load_weights` that handle Wan2.1 → diffusers name remapping
  (`self_attn.q/k/v` → fused `attn1.to_qkv`, `cross_attn.k_img/v_img` →
  `attn2.add_k_proj/add_v_proj`, `img_emb.proj.{1,3}` →
  `condition_embedder.image_embedder.ff.net.{0.proj,2}`, `modulation` →
  `scale_shift_table`, etc.).
- `pipeline_micro_world_t2w.py` — `MicroWorldT2WPipeline`. Hosts the
  scheduler (`FlowUniPCMultistepScheduler`, `flow_shift=3`), VAE, T5
  encoder, and the Kohya LoRA merger (306 layers).
- `pipeline_micro_world_i2w.py` — `MicroWorldI2WPipeline`. Adds CLIP image
  encoder/processor and an extended LoRA merger (488 layers including
  `cross_attn_{k,v}_img` and `img_emb_proj_{1,3}`).

Action conditioning is `extra_args={"mouse_actions": [...], "keyboard_actions": [...]}`
on the sampling params.

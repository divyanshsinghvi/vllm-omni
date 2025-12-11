# Text to Image Generation

This example demonstrates how to generate images using `vllm-omni` with various diffusion models.

## Local CLI Usage

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee.png
```

### Arguments

- `--model`: The model name or local path (default: `Qwen/Qwen-Image`).
- `--prompt`: The text prompt for generation.
- `--negative_prompt`: The negative text prompt (optional).
- `--guidance_scale`: The classifier-free guidance scale (optional). Alias: `--cfg_scale`.
- `--height`: Height of the generated image (default: 1024).
- `--width`: Width of the generated image (default: 1024).
- `--seed`: Random seed for reproducibility.
- `--output`: Path to save the output image.

### Examples

**Qwen-Image:**
```bash
python text_to_image.py --model Qwen/Qwen-Image --prompt "a beautiful sunset" --guidance_scale 4.0
```

**Wan2.1:**
```bash
python text_to_image.py --model Wan-AI/Wan2.1-T2I-1.3B --prompt "a cat in a box" --guidance_scale 5.0
```

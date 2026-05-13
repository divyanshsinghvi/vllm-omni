#!/usr/bin/env bash
# Build the Micro-World T2W and I2W "combined" model dirs (AMD transformer/LoRA
# + Wan2.1 base components + custom model_index.json) so the regression test
# can point MICRO_WORLD_{T2W,I2W}_MODEL at them.
#
# Usage:
#   ROOT=/some/cache/dir tests/e2e/accuracy/micro_world/setup_combined.sh
#   # exports MICRO_WORLD_T2W_MODEL and MICRO_WORLD_I2W_MODEL paths to stdout
#
# Safe to re-run: existing symlinks/files are left alone.
set -euo pipefail

ROOT="${ROOT:-${HF_HOME:-$HOME/.cache/huggingface}/micro_world}"
mkdir -p "$ROOT"

T2W_AMD="$ROOT/Micro-World-T2W"
T2W_BASE="$ROOT/Wan2.1-T2V-Base"
T2W_COMBINED="$ROOT/Micro-World-T2W-combined"

I2W_AMD="$ROOT/Micro-World-I2W"
I2W_BASE="$ROOT/Wan2.1-I2V-Base"
I2W_COMBINED="$ROOT/Micro-World-I2W-combined"

# --- T2W -----------------------------------------------------------------
[ -d "$T2W_AMD/transformer" ] || huggingface-cli download amd/Micro-World-T2W --local-dir "$T2W_AMD"
[ -d "$T2W_BASE/tokenizer" ] || huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local-dir "$T2W_BASE" --include "tokenizer/*" "text_encoder/*" "vae/*"

mkdir -p "$T2W_COMBINED"
ln -sfn "$T2W_AMD/transformer" "$T2W_COMBINED/transformer"
ln -sfn "$T2W_AMD/lora_diffusion_pytorch_model.safetensors" \
        "$T2W_COMBINED/lora_diffusion_pytorch_model.safetensors"
ln -sfn "$T2W_BASE/tokenizer"    "$T2W_COMBINED/tokenizer"
ln -sfn "$T2W_BASE/text_encoder" "$T2W_COMBINED/text_encoder"
ln -sfn "$T2W_BASE/vae"          "$T2W_COMBINED/vae"
cat > "$T2W_COMBINED/model_index.json" <<'EOF'
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

# --- I2W -----------------------------------------------------------------
[ -d "$I2W_AMD/transformer" ] || huggingface-cli download amd/Micro-World-I2W --local-dir "$I2W_AMD"
[ -d "$I2W_BASE/tokenizer" ] || huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers \
    --local-dir "$I2W_BASE" --include "tokenizer/*" "text_encoder/*" "image_processor/*" "image_encoder/*" "vae/*"

mkdir -p "$I2W_COMBINED"
ln -sfn "$I2W_AMD/transformer" "$I2W_COMBINED/transformer"
ln -sfn "$I2W_AMD/lora_diffusion_pytorch_model.safetensors" \
        "$I2W_COMBINED/lora_diffusion_pytorch_model.safetensors"
ln -sfn "$I2W_BASE/tokenizer"       "$I2W_COMBINED/tokenizer"
ln -sfn "$I2W_BASE/text_encoder"    "$I2W_COMBINED/text_encoder"
ln -sfn "$I2W_BASE/image_processor" "$I2W_COMBINED/image_processor"
ln -sfn "$I2W_BASE/image_encoder"   "$I2W_COMBINED/image_encoder"
ln -sfn "$I2W_BASE/vae"             "$I2W_COMBINED/vae"
cat > "$I2W_COMBINED/model_index.json" <<'EOF'
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

echo "MICRO_WORLD_T2W_MODEL=$T2W_COMBINED"
echo "MICRO_WORLD_I2W_MODEL=$I2W_COMBINED"

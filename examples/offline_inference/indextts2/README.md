<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

# IndexTTS2 Text-to-Speech

IndexTTS2 is a two-stage TTS model that generates high-quality speech from text using a GPT-based autoregressive model followed by a speech-to-mel diffusion model.

## Requirements

```bash
pip install bigvgan==2.4.1 descript-audio-codec==1.0.0 funasr==1.3.1 \
    sentencepiece==0.2.1 tn einops==0.8.2 torchaudio==2.10.0 soundfile==0.13.1
```

## Model Setup

Download the pretrained model:

```bash
# Clone or download IndexTTS2 weights to pretrained_models/IndexTTS2/
# Required files: bpe.model, config.yaml, gpt.pth, s2mel.pth,
#                 wav2vec2bert_stats.pt, feat1.pt, feat2.pt
```

## Usage

```bash
python examples/offline_inference/indextts2/verify_e2e_indextts2.py \
    --model pretrained_models/IndexTTS2 \
    --audio-path prompt.wav \
    --prompt "Hello, this is a test of the IndexTTS2 text to speech system."
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `pretrained_models/IndexTTS2` | Path to model directory |
| `--stage-config` | `vllm_omni/model_executor/stage_configs/indextts2.yaml` | Stage config path |
| `--prompt` | `"Hello, this is a test..."` | Text to synthesize |
| `--audio-path` | `prompt.wav` | Speaker reference audio |
| `--emo-path` | Same as `--audio-path` | Emotion reference audio |
| `--emo-weight` | `0.5` | Emotion weight (0.0 to 1.0) |

## Architecture

The model runs as a two-stage pipeline:

1. **Stage 0 (GPT)**: Takes text tokens + speaker conditioning, generates mel codes autoregressively
2. **Stage 1 (S2Mel)**: Takes mel codes + reference audio features, generates mel spectrogram via diffusion, then vocodes to audio using BigVGAN

## External Dependencies

| Component | Package | Purpose |
|---|---|---|
| BigVGAN vocoder | `bigvgan` | Mel-to-waveform synthesis |
| Vector quantization | `descript-audio-codec` | Codebook quantization in length regulator |
| CAMPPlus speaker encoder | `funasr` | Speaker style embedding extraction |
| BPE tokenizer | `sentencepiece` | Text tokenization |
| Text normalization | `tn` | Chinese/English text normalization |
| Semantic encoder | `transformers` (w2v-bert-2.0) | Speaker conditioning features |
| Semantic codec | `amphion/MaskGCT` (HuggingFace) | Quantized semantic features |

## Setup

Install dependencies:
```
uv pip install onnxruntime==1.23.2
uv pip install x-transformers==2.12.2
```

Download the model snapshot:
```
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

Add `config.json` in `pretrained_models/Fun-CosyVoice3-0.5B/`:
```json
{
    "model_type": "cosyvoice3",
    "architectures": [
        "CosyVoice3Model"
    ]
}
```

Run the offline verification script:
```
python examples/offline_inference/text_to_speech/verify_e2e_cosyvoice.py \
  --model pretrained_models/Fun-CosyVoice3-0.5B \
  --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN
```

## Implementation Overview

CosyVoice3 runs as a 2-stage Omni pipeline:
- Stage 0 (text_speech_lm) converts text + prompt audio to speech tokens.
- Stage 1 (chunk_aware_flow_matching) converts speech tokens + prompt features to audio.

Key components in `vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py`:
- `CosyVoice3MultiModalProcessor` builds the multimodal inputs:
  - Tokenizes `prompt` and `prompt_text`.
  - Extracts speech tokens and mel features from the prompt audio.
  - Extracts a speaker embedding.
- `CosyVoice3Model` implements both stages:
  - Stage 0 uses `CosyVoice3LM` and outputs speech tokens + conditioning features.
  - Stage 1 runs the flow model (DiT-based CFM) and HiFiGAN to synthesize waveform.

Stage wiring is configured in `vllm_omni/model_executor/stage_configs/cosyvoice3.yaml`:
- Stage 0 emits latent speech tokens .
- Stage 1 consumes them via `custom_process_input_func` and outputs audio.

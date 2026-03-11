# Speech API

vLLM-Omni provides an OpenAI-compatible API for text-to-speech (TTS) generation using Qwen3-TTS models.

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

```bash
# CustomVoice model (predefined speakers)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

### Generate Speech

**Using curl:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav
```

**Using Python:**

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Using OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="vivian",
    input="Hello, how are you?",
)

response.stream_to_file("output.wav")
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | The text to synthesize into speech |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `voice` | string | "vivian" | Speaker name (e.g., vivian, ryan, aiden) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

#### vLLM-Omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | string | "CustomVoice" | TTS task type: CustomVoice, VoiceDesign, or Base |
| `language` | string | "Auto" | Language (see supported languages below) |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | integer | 2048 | Maximum tokens to generate |
| `initial_codec_chunk_frames` | integer | null | Initial chunk size for reduced TTFA (overrides stage config) |

**Supported languages:** Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

#### Voice Clone Parameters (Base task)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_audio` | string | null | Reference audio (URL or base64 data URL) |
| `ref_text` | string | null | Transcript of reference audio |
| `x_vector_only_mode` | bool | null | Use speaker embedding only (no ICL) |

### Response Format

Returns binary audio data with appropriate `Content-Type` header (e.g., `audio/wav`).

### Voices Endpoint

```
GET /v1/audio/voices
```

Lists available voices for the loaded model.

```json
{
    "voices": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
}
```

## Examples

### CustomVoice with Style Instruction

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav
```

### VoiceDesign (Natural Language Voice Description)

```bash
# Start server with VoiceDesign model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello world",
        "task_type": "VoiceDesign",
        "instructions": "A warm, friendly female voice with a gentle tone"
    }' --output designed.wav
```

### Base (Voice Cloning)

```bash
# Start server with Base model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice",
        "task_type": "Base",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Original transcript of the reference audio"
    }' --output cloned.wav
```

## Batch Speech Generation

The batch endpoint synthesizes multiple texts in a single request, returning all results as JSON with base64-encoded audio.

### Endpoint

```
POST /v1/audio/speech/batch
Content-Type: application/json
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | array | **required** | List of items to synthesize (1–32) |
| `model` | string | server's model | Model to use |
| `voice` | string | null | Default voice for all items |
| `response_format` | string | "wav" | Default audio format for all items |
| `speed` | float | 1.0 | Default playback speed (0.25–4.0) |
| `task_type` | string | null | Default TTS task type |
| `language` | string | null | Default language |
| `instructions` | string | null | Default voice style instructions |
| `ref_audio` | string | null | Default reference audio (Base task) |
| `ref_text` | string | null | Default reference transcript (Base task) |
| `max_new_tokens` | integer | null | Default max tokens |

Each item in the `items` array requires only `input` (the text). All other fields are optional and override the batch-level defaults when set:

| Field | Type | Description |
|-------|------|-------------|
| `input` | string | **required** — text to synthesize |
| `voice` | string | Override voice for this item |
| `response_format` | string | Override format for this item |
| `speed` | float | Override speed for this item |
| `task_type` | string | Override task type |
| `language` | string | Override language |
| `instructions` | string | Override instructions |
| `ref_audio` | string | Override reference audio |
| `ref_text` | string | Override reference transcript |
| `max_new_tokens` | integer | Override max tokens |

### Response Format

```json
{
    "id": "speech-batch-abc123",
    "results": [
        {
            "index": 0,
            "status": "success",
            "audio_data": "<base64-encoded audio>",
            "media_type": "audio/wav"
        },
        {
            "index": 1,
            "status": "error",
            "error": "Input text cannot be empty"
        }
    ],
    "total": 2,
    "succeeded": 1,
    "failed": 1
}
```

### Examples

**Basic batch with shared defaults:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech/batch \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"input": "Hello, how are you?"},
            {"input": "Goodbye, see you later!"}
        ],
        "voice": "vivian",
        "language": "English"
    }'
```

**Per-item overrides (different voices and formats):**

```bash
curl -X POST http://localhost:8091/v1/audio/speech/batch \
    -H "Content-Type: application/json" \
    -d '{
        "items": [
            {"input": "Hello!", "voice": "vivian", "response_format": "mp3"},
            {"input": "你好！", "voice": "ryan", "language": "Chinese"}
        ],
        "response_format": "wav"
    }'
```

**Decoding the response in Python:**

```python
import base64
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech/batch",
    json={
        "items": [
            {"input": "First sentence."},
            {"input": "Second sentence."},
        ],
        "voice": "vivian",
    },
    timeout=300.0,
)

for result in response.json()["results"]:
    if result["status"] == "success":
        audio_bytes = base64.b64decode(result["audio_data"])
        with open(f"output_{result['index']}.wav", "wb") as f:
            f.write(audio_bytes)
```

### Configuration

The batch endpoint has two configurable limits, passed as engine kwargs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tts_batch_max_items` | 32 | Maximum number of items per batch request |
| `tts_batch_concurrency` | 8 | Maximum items synthesized concurrently within a batch |

Items beyond the concurrency limit are queued and processed as in-flight items complete.

## Supported Models

| Model | Task Type | Description |
|-------|-----------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Base | Smaller/faster variant for voice cloning |

## Error Responses

### 400 Bad Request

Invalid parameters:

```json
{
    "error": {
        "message": "Input text cannot be empty",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    }
}
```

### 404 Not Found

Model not found:

```json
{
    "error": {
        "message": "The model `xxx` does not exist.",
        "type": "NotFoundError",
        "param": "model",
        "code": 404
    }
}
```

## Troubleshooting

### "TTS model did not produce audio output"

Ensure you're using the correct model variant for your task type:
- CustomVoice task → CustomVoice model
- VoiceDesign task → VoiceDesign model
- Base task → Base model

### Server Not Running

```bash
# Check if server is responding
curl http://localhost:8091/v1/audio/voices
```

### Out of Memory

If you encounter OOM errors:
1. Use smaller model variant: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
2. Reduce `--gpu-memory-utilization`

### Unsupported Speaker

Use `/v1/audio/voices` to list available voices for the loaded model.

## Development

Enable debug logging:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --uvicorn-log-level debug
```

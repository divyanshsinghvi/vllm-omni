# Qwen-Image Offline Inference

This folder provides several entrypoints for experimenting with `Qwen/Qwen-Image` using vLLM-Omni:

- `web_demo.py`: lightweight Gradio UI for interactive prompt/seed/CFG exploration.

## Basic Usage

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    prompt = "a cup of coffee on the table"
    images = omni.generate(prompt)
    images[0].save("coffee.png")
```

## Local CLI Usage

Check `text_to_image.py` in `examples/offline_inference/text_to_image` for more details.

> ℹ️ Qwen-Image currently publishes best-effort presets at `1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, and `1056x1584`. Adjust `--height/--width` accordingly for the most reliable outcomes.

## Web UI Demo

Launch the gradio demo:

```bash
python gradio_demo.py --port 7862
```

Then open `http://localhost:7862/` on your local browser to interact with the web UI.

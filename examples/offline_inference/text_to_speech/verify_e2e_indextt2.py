import argparse
import os

import soundfile as sf
from vllm import SamplingParams

# Use Omni entrypoint directly
from vllm_omni.entrypoints.omni import Omni


def run_e2e():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pretrained_models/IndexTTS2/")
    parser.add_argument("--stage-config", type=str, default="vllm_omni/model_executor/stage_configs/indextts2.yaml")
    parser.add_argument("--prompt", type=str, default="Hello, this is a test of the Index TTS system capability.")
    parser.add_argument("--audio-path", type=str, default="prompt.wav")
    parser.add_argument("--emo-path", type=str, default=None, help="Path to emotion audio (defaults to audio-path)")
    parser.add_argument("--emo-weight", type=float, default=0.5, help="Emotion weight (0.0 to 1.0)")
    args = parser.parse_args()

    # Use audio_path for emo_path if not specified
    if args.emo_path is None:
        args.emo_path = args.audio_path

    # Ensure stage config exists
    if not os.path.exists(args.stage_config):
        raise Exception(f"{args.stage_config} does not exist!")

    print(f"Initializing IndexTTS E2E with model={args.model}")

    # Validate audio paths
    if args.audio_path and not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if args.emo_path and not os.path.exists(args.emo_path):
        raise FileNotFoundError(f"Emotion audio file not found: {args.emo_path}")

    print(f"Using speaker audio: {args.audio_path}")
    print(f"Using emotion audio: {args.emo_path}")
    print(f"Emotion weight: {args.emo_weight}")

    # Initialize Omni
    # This spins up the engine(s) based on the stage config
    omni = Omni(model=args.model, stage_configs_path=args.stage_config, skip_tokenizer_init=True)

    print("Model initialized. Preparing inputs...")

    # Pass audio paths via runtime_additional_information (following Qwen3TTS pattern)
    # Audio is loaded by the model, not pre-loaded here
    prompts = {
        "prompt": args.prompt,
        "runtime_additional_information": {
            "audio_path": [args.audio_path],
            "emo_audio_path": [args.emo_path],
            "emo_weight": [args.emo_weight],
        },
    }

    print(f"Generating for prompt: {args.prompt}")

    # Build SamplingParams for each stage (GPT, S2Mel, Vocoder)
    gpt_sampling = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        max_tokens=1024,
        detokenize=False,
    )
    s2mel_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096,
        detokenize=False,
    )
    vocoder_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096,
        detokenize=True,
    )

    sampling_params_list = [gpt_sampling, s2mel_sampling, vocoder_sampling]

    # Generate (Omni orchestrator requires a per-stage SamplingParams list)
    outputs = omni.generate(prompts, sampling_params_list=sampling_params_list[:2])

    # Verify outputs
    print(f"Received {len(outputs)} outputs.")
    for i, stage_outputs in enumerate(outputs):
        request_outputs = stage_outputs.request_output or []
        for j, output in enumerate(request_outputs):
            mm = output.outputs[0].multimodal_output
            audio = mm["audio"]
            sample_rate = 22050
            print(f"Audio shape: {audio.shape}")

            audio_numpy = audio.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            out_path = f"output_{i}_{j}.wav"
            sf.write(out_path, audio_numpy, samplerate=sample_rate, format="WAV")
            print(f"Saved audio to {out_path}")


if __name__ == "__main__":
    run_e2e()

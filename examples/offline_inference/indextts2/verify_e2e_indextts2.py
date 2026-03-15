# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end verification script for IndexTTS2 via vLLM Omni.

Generates speech from text using the IndexTTS2 model with voice cloning.

Usage:
    python verify_e2e_indextts2.py \
        --model pretrained_models/IndexTTS2 \
        --audio-path prompt.wav \
        --prompt "Hello, this is a test of the IndexTTS2 system."
"""

import argparse
import os

import soundfile as sf
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni


def run_e2e():
    parser = argparse.ArgumentParser(description="IndexTTS2 E2E verification")
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained_models/IndexTTS2",
        help="Path to IndexTTS2 model directory.",
    )
    parser.add_argument(
        "--stage-config",
        type=str,
        default="vllm_omni/model_executor/stage_configs/indextts2.yaml",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, this is a test of the IndexTTS2 text to speech system.",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="prompt.wav",
        help="Path to reference audio for voice cloning.",
    )
    parser.add_argument(
        "--emo-path",
        type=str,
        default=None,
        help="Path to emotion audio (defaults to audio-path).",
    )
    parser.add_argument(
        "--emo-weight",
        type=float,
        default=0.5,
        help="Emotion weight (0.0 to 1.0).",
    )
    args = parser.parse_args()

    # Use audio_path for emo_path if not specified
    if args.emo_path is None:
        args.emo_path = args.audio_path

    # Validate paths
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model directory not found: {args.model}")
    if not os.path.exists(args.stage_config):
        raise FileNotFoundError(f"Stage config not found: {args.stage_config}")
    if args.audio_path and not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if args.emo_path and not os.path.exists(args.emo_path):
        raise FileNotFoundError(f"Emotion audio file not found: {args.emo_path}")

    print(f"Initializing IndexTTS2 E2E with model={args.model}")
    print(f"Using speaker audio: {args.audio_path}")
    print(f"Using emotion audio: {args.emo_path}")
    print(f"Emotion weight: {args.emo_weight}")

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        log_stats=True,
    )

    print("Model initialized. Preparing inputs...")

    prompts = {
        "prompt": args.prompt,
        "additional_information": {
            "audio_path": [args.audio_path],
            "emo_audio_path": [args.emo_path],
            "emo_weight": [args.emo_weight],
        },
    }

    print(f"Generating for prompt: {args.prompt}")

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

    outputs = omni.generate(prompts, sampling_params_list=sampling_params_list[:2])

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

    omni.close()
    print("Done!")


if __name__ == "__main__":
    run_e2e()

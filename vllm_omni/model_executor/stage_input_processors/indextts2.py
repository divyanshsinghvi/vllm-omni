import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def gpt2s2mel(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt | list = None,
    requires_multimodal_data: bool = False,
):
    """Convert GPT stage output to S2Mel stage input.

    Extracts latent representations from the GPT stage and formats them
    as input for the S2Mel (semantic-to-mel) stage.

    Args:
        stage_list: List of stage objects containing outputs
        engine_input_source: List of source stage IDs to pull data from
        prompt: Original prompt(s), used to preserve multi_modal_data if needed
        requires_multimodal_data: Whether to pass through multimodal data

    Returns:
        List of OmniTokensPrompt objects formatted for S2Mel stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    gpt_outputs = stage_list[source_stage_id].engine_outputs

    # Handle prompt as list
    if prompt is not None and not isinstance(prompt, list):
        prompt = [prompt]

    # Extract multi_modal_data if needed
    multi_modal_data = {}
    if prompt is not None and requires_multimodal_data:
        multi_modal_data = {
            gpt_output.request_id: p.get("multi_modal_data", None) for gpt_output, p in zip(gpt_outputs, prompt)
        }

    s2mel_inputs = []
    for i, gpt_output in enumerate(gpt_outputs):
        output = gpt_output.outputs[0]
        mm_output = output.multimodal_output

        # Extract latent tensors from GPT output
        latent = mm_output["latent"]
        speech_conditioning_latent = mm_output.get("speech_conditioning_latent", None)

        # Ensure tensors are detached and on correct device
        if isinstance(latent, torch.Tensor):
            latent = latent.clone().detach()
        if isinstance(speech_conditioning_latent, torch.Tensor):
            speech_conditioning_latent = speech_conditioning_latent.clone().detach()

        additional_information = {
            "latent": latent,
            "speech_conditioning_latent": speech_conditioning_latent,
            "latent_shape": list(latent.shape) if isinstance(latent, torch.Tensor) else None,
        }

        s2mel_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # Dummy token for S2Mel stage
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data.get(gpt_output.request_id, None)
                    if requires_multimodal_data and multi_modal_data
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    logger.debug(f"gpt2s2mel: Converted {len(gpt_outputs)} GPT outputs to S2Mel inputs")
    return s2mel_inputs


def s2mel2vocoder(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt | list = None,
    requires_multimodal_data: bool = False,
):
    """Convert S2Mel stage output to Vocoder stage input.

    Extracts mel-spectrogram from the S2Mel stage and formats it
    as input for the vocoder (mel-to-waveform) stage.

    Args:
        stage_list: List of stage objects containing outputs
        engine_input_source: List of source stage IDs to pull data from
        prompt: Original prompt(s), used to preserve multi_modal_data if needed
        requires_multimodal_data: Whether to pass through multimodal data

    Returns:
        List of OmniTokensPrompt objects formatted for Vocoder stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    s2mel_outputs = stage_list[source_stage_id].engine_outputs

    vocoder_inputs = []
    for s2mel_output in s2mel_outputs:
        output = s2mel_output.outputs[0]
        mm_output = output.multimodal_output

        # Extract mel-spectrogram from S2Mel output
        mel = mm_output.get("mel", None)

        if isinstance(mel, torch.Tensor):
            mel = mel.clone().detach()

        additional_information = {
            "mel": mel,
            "mel_shape": list(mel.shape) if isinstance(mel, torch.Tensor) else None,
        }

        vocoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # Dummy token for Vocoder stage
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    logger.debug(f"s2mel2vocoder: Converted {len(s2mel_outputs)} S2Mel outputs to Vocoder inputs")
    return vocoder_inputs

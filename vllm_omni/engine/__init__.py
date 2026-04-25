"""
Engine components for vLLM-Omni.
"""

from typing import Any

import msgspec
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)

from vllm_omni.data_entry_keys import OmniPayloadStruct


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for additional
    information payloads, enabling direct transfer of pre-computed data
    between pipeline stages.

    Note: prompt_embeds is inherited from EngineCoreRequest
    (torch.Tensor | None). PromptEmbedsPayload should be decoded to
    torch.Tensor before constructing this request.
    """

    additional_information: OmniPayloadStruct | None = None


class OmniEngineCoreOutput(EngineCoreOutput):
    pooling_output: dict[str, Any] | None = None
    # Finished flag for streaming input segment
    is_segment_finished: bool | None = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None


class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []

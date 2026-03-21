"""Structured payload types for inter-stage communication.

Adding a new model?
~~~~~~~~~~~~~~~~~~~
Every key you put into the inter-stage payload (``additional_information``,
``multimodal_output``, ``pooling_output``) **must** use the nested
``OmniPayload`` TypedDict structure.  For each category, every known
qualifier is an explicit field so misspellings are caught statically.

Categories
    hidden_states  – intermediate / output hidden-state tensors
    embed          – embedding tensors (prefill, decode, special tokens)
    ids            – token-ID sequences
    codes          – codec / audio code tensors
    meta           – scalar metadata, control flags, shapes

This module provides:
- Structured ``TypedDict`` types for static type checking (``OmniPayload``)
"""

from __future__ import annotations

from typing import TypedDict

import torch

# ── Structured payload types ──
# These are TypedDicts (plain dicts at runtime, zero overhead) that give
# static type checking and IDE autocomplete for inter-stage payloads.
# Every field is optional (total=False) because each stage only populates
# the subset it needs.


class HiddenStates(TypedDict, total=False):
    output: torch.Tensor
    trailing_text: torch.Tensor
    last: torch.Tensor
    layers: dict[int, torch.Tensor]


class Embeddings(TypedDict, total=False):
    prefill: torch.Tensor
    decode: torch.Tensor
    cached_decode: torch.Tensor
    tts_bos: torch.Tensor
    tts_eos: torch.Tensor
    tts_pad: torch.Tensor
    tts_pad_projected: torch.Tensor
    voice: torch.Tensor
    speech_feat: torch.Tensor
    thinker_reply: torch.Tensor


class Codes(TypedDict, total=False):
    audio: torch.Tensor
    ref: torch.Tensor


class Ids(TypedDict, total=False):
    all: list[int]
    prompt: list[int]
    output: list[int]
    speech_token: list[int]
    prior_image: list[int]


class OmniPayloadMeta(TypedDict, total=False):
    finished: bool
    left_context_size: int
    override_keys: list[tuple[str, str]]
    num_processed_tokens: int
    next_stage_prompt_len: int
    ar_width: int
    eol_token_id: int
    visual_token_start_id: int
    visual_token_end_id: int
    gen_token_mask: torch.Tensor
    omni_task: list[str]
    height: int
    width: int
    decode_flag: bool
    codec_streaming: bool
    ref_code_len: int
    talker_prefill_offset: int


class OmniPayload(TypedDict, total=False):
    hidden_states: HiddenStates
    embed: Embeddings
    ids: Ids
    codes: Codes
    meta: OmniPayloadMeta
    generated_len: int
    model_outputs: list[torch.Tensor]
    mtp_inputs: tuple[torch.Tensor, torch.Tensor]

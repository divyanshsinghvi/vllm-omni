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
- ``serialize_payload`` / ``deserialize_payload`` for transport across
  process boundaries via ``AdditionalInformationPayload``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload

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
    latent: torch.Tensor
    generated_len: int
    model_outputs: list[torch.Tensor]
    mtp_inputs: tuple[torch.Tensor, torch.Tensor]


# ── Keys whose values are nested dicts (TypedDict sub-categories) ──
_NESTED_KEYS = frozenset({"hidden_states", "embed", "ids", "codes", "meta"})

# ── dtype helpers ──
_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def _dtype_to_name(dtype: torch.dtype) -> str:
    return _DTYPE_TO_NAME.get(dtype, str(dtype).replace("torch.", ""))


def _serialize_tensor(t: torch.Tensor) -> AdditionalInformationEntry:
    from vllm_omni.engine import AdditionalInformationEntry

    t_cpu = t.detach().to("cpu").contiguous()
    return AdditionalInformationEntry(
        tensor_data=t_cpu.numpy().tobytes(),
        tensor_shape=list(t_cpu.shape),
        tensor_dtype=_dtype_to_name(t_cpu.dtype),
    )


def _deserialize_tensor(entry: AdditionalInformationEntry) -> torch.Tensor:
    dt = np.dtype(entry.tensor_dtype or "float32")
    arr = np.frombuffer(entry.tensor_data, dtype=dt)  # type: ignore[arg-type]
    arr = arr.reshape(entry.tensor_shape)
    return torch.from_numpy(arr.copy())


def serialize_payload(
    payload: OmniPayload,
) -> AdditionalInformationPayload | None:
    """Serialize an ``OmniPayload`` for EngineCore transport.

    Nested sub-dicts are flattened to dotted keys (e.g.
    ``hidden_states.output``) so the wire format stays flat.
    ``hidden_states.layers`` is expanded to ``hidden_states.layer_N``.
    """
    from vllm_omni.engine import (
        AdditionalInformationEntry,
        AdditionalInformationPayload,
    )

    entries: dict[str, AdditionalInformationEntry] = {}

    for key, value in payload.items():
        if key in _NESTED_KEYS and isinstance(value, dict):
            for qual, val in value.items():
                # Special-case: layers dict keyed by int
                if qual == "layers" and key == "hidden_states" and isinstance(val, dict):
                    for layer_idx, tensor in val.items():
                        entries[f"hidden_states.layer_{layer_idx}"] = _serialize_tensor(tensor)
                elif isinstance(val, torch.Tensor):
                    entries[f"{key}.{qual}"] = _serialize_tensor(val)
                elif isinstance(val, list):
                    entries[f"{key}.{qual}"] = AdditionalInformationEntry(list_data=val)
                else:
                    # Scalars (bool, int, etc.) — wrap in a single-element list
                    entries[f"{key}.{qual}"] = AdditionalInformationEntry(list_data=[val])
        elif isinstance(value, torch.Tensor):
            entries[key] = _serialize_tensor(value)
        elif isinstance(value, list):
            entries[key] = AdditionalInformationEntry(list_data=value)

    return AdditionalInformationPayload(entries=entries) if entries else None


def deserialize_payload(
    wire: AdditionalInformationPayload,
) -> OmniPayload:
    """Deserialize an ``AdditionalInformationPayload`` back to ``OmniPayload``.

    Dotted keys are unflattened into nested dicts.
    ``hidden_states.layer_N`` keys are collected into ``hidden_states.layers``.
    """
    result: dict[str, Any] = {}

    for key, entry in wire.entries.items():
        # Reconstruct value
        if entry.tensor_data is not None:
            val: Any = _deserialize_tensor(entry)
        elif entry.list_data is not None:
            val = entry.list_data
        else:
            continue

        # Dotted key → nested dict
        if "." in key:
            type_key, qualifier = key.split(".", 1)
            sub = result.setdefault(type_key, {})
            # hidden_states.layer_N → layers[N]
            if type_key == "hidden_states" and qualifier.startswith("layer_"):
                layers = sub.setdefault("layers", {})
                layer_idx = int(qualifier[len("layer_") :])
                layers[layer_idx] = val
            else:
                # Unwrap single-element scalar lists from meta
                if type_key == "meta" and isinstance(val, list) and len(val) == 1:
                    val = val[0]
                sub[qualifier] = val
        else:
            result[key] = val

    return result  # type: ignore[return-value]

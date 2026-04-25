# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured payload types for inter-stage communication.

Categories under ``OmniPayload``:
    hidden_states  – intermediate / output hidden-state tensors
    embed          – embedding tensors (prefill, decode, special tokens)
    ids            – token-ID sequences
    codes          – codec / audio code tensors
    meta           – scalar metadata, control flags, shapes
"""

from __future__ import annotations

from typing import Any, TypedDict

import msgspec
import numpy as np
import torch


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
    finished: torch.Tensor
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
    speaker: Any
    language: Any
    request_id: str


# ── msgspec.Struct mirror of the TypedDicts (runtime-validated) ──


class _StructBase(msgspec.Struct, omit_defaults=True, kw_only=True, forbid_unknown_fields=True):
    pass


class HiddenStatesStruct(_StructBase):
    output: torch.Tensor | None = None
    output_shape: list[int] | None = None
    trailing_text: torch.Tensor | None = None
    last: torch.Tensor | None = None
    layers: dict[int, torch.Tensor] | None = None


class EmbeddingsStruct(_StructBase):
    prefill: torch.Tensor | None = None
    prefill_shape: list[int] | None = None
    decode: torch.Tensor | None = None
    cached_decode: torch.Tensor | None = None
    tts_bos: torch.Tensor | None = None
    tts_eos: torch.Tensor | None = None
    tts_pad: torch.Tensor | None = None
    tts_pad_projected: torch.Tensor | None = None
    voice: torch.Tensor | None = None
    speech_feat: torch.Tensor | None = None
    thinker_reply: torch.Tensor | None = None


class CodesStruct(_StructBase):
    audio: torch.Tensor | None = None
    ref: torch.Tensor | None = None


class IdsStruct(_StructBase):
    all: list[int] | None = None
    prompt: list[int] | None = None
    output: list[int] | None = None
    speech_token: list[int] | None = None
    prior_image: list[int] | None = None


class MetaStruct(_StructBase):
    finished: torch.Tensor | None = None
    left_context_size: int | None = None
    override_keys: list[tuple[str, str]] | None = None
    num_processed_tokens: int | None = None
    next_stage_prompt_len: int | None = None
    ar_width: int | None = None
    eol_token_id: int | None = None
    visual_token_start_id: int | None = None
    visual_token_end_id: int | None = None
    gen_token_mask: torch.Tensor | None = None
    omni_task: list[str] | None = None
    height: int | None = None
    width: int | None = None
    decode_flag: bool | None = None
    codec_streaming: bool | None = None
    ref_code_len: int | None = None
    talker_prefill_offset: int | None = None
    codec_chunk_frames: int | None = None
    codec_left_context_frames: int | None = None
    code_flat_numel: int | None = None
    omni_final_stage_id: int | None = None


class OmniPayloadStruct(_StructBase):
    hidden: torch.Tensor | None = None
    hidden_states: HiddenStatesStruct | None = None
    embed: EmbeddingsStruct | None = None
    ids: IdsStruct | None = None
    codes: CodesStruct | None = None
    meta: MetaStruct | None = None
    latent: torch.Tensor | None = None
    generated_len: int | None = None
    model_outputs: list[torch.Tensor] | None = None
    mtp_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    speaker: Any = None
    language: Any = None
    request_id: str | None = None
    past_key_values: list[int] | None = None
    kv_metadata: dict[str, Any] | None = None


_NESTED_STRUCTS: dict[str, type[_StructBase]] = {
    "hidden_states": HiddenStatesStruct,
    "embed": EmbeddingsStruct,
    "ids": IdsStruct,
    "codes": CodesStruct,
    "meta": MetaStruct,
}


_TENSOR_MARKER = "__tensor__"


def _msgspec_dec_hook(typ: type, obj: Any) -> Any:
    """Bridge non-msgspec types when decoding bytes/dicts into Structs."""
    if typ is torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict) and obj.get(_TENSOR_MARKER):
            arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
            arr = arr.reshape(obj["shape"])
            return torch.from_numpy(arr.copy())
        raise TypeError(f"cannot decode {type(obj).__name__} into torch.Tensor")
    raise NotImplementedError(f"no decoder for {typ}")


def _msgspec_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return {
            _TENSOR_MARKER: True,
            "data": obj.detach().cpu().contiguous().numpy().tobytes(),
            "shape": list(obj.shape),
            "dtype": _dtype_to_name(obj.dtype),
        }
    raise NotImplementedError(f"no encoder for {type(obj).__name__}")


_OMNI_PAYLOAD_ENCODER = msgspec.msgpack.Encoder(enc_hook=_msgspec_enc_hook)
_OMNI_PAYLOAD_DECODER = msgspec.msgpack.Decoder(OmniPayloadStruct, dec_hook=_msgspec_dec_hook)


def encode_payload(struct: OmniPayloadStruct) -> bytes:
    """Encode ``OmniPayloadStruct`` to msgpack bytes for cross-process transport."""
    return _OMNI_PAYLOAD_ENCODER.encode(struct)


def decode_payload(data: bytes) -> OmniPayloadStruct:
    """Decode msgpack bytes back to ``OmniPayloadStruct``, validating the schema."""
    return _OMNI_PAYLOAD_DECODER.decode(data)


def to_struct(payload: dict[str, Any]) -> OmniPayloadStruct:
    """Convert a payload dict into ``OmniPayloadStruct``, validating types.

    Raises ``msgspec.ValidationError`` on:
      * unknown top-level keys (typos, legacy flat keys)
      * unknown sub-keys under any nested category
      * type mismatches (e.g., ``meta.left_context_size`` not an ``int``)
    """
    return msgspec.convert(payload, OmniPayloadStruct, dec_hook=_msgspec_dec_hook)


def validate_payload(payload: dict[str, Any] | None, *, context: str = "payload") -> None:
    """Validate a payload matches the ``OmniPayload`` schema, raising on drift.

    Wraps :func:`to_struct` and re-raises ``msgspec.ValidationError`` with
    the call-site ``context`` prepended.  ``None`` is allowed (treated as
    "no payload to check").
    """
    if payload is None:
        return
    try:
        to_struct(payload)
    except msgspec.ValidationError as exc:
        raise msgspec.ValidationError(f"{context}: {exc}") from exc


def to_dict(struct: OmniPayloadStruct) -> dict[str, Any]:
    """Convert ``OmniPayloadStruct`` to a plain dict, dropping ``None`` fields."""
    out: dict[str, Any] = {}
    for field in OmniPayloadStruct.__struct_fields__:
        value = getattr(struct, field)
        if value is None:
            continue
        if isinstance(value, _StructBase):
            sub: dict[str, Any] = {}
            for sk in value.__struct_fields__:
                sv = getattr(value, sk)
                if sv is not None:
                    sub[sk] = sv
            if sub:
                out[field] = sub
        else:
            out[field] = value
    return out


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

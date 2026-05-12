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

from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload


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
    speech_token: torch.Tensor
    embedding: torch.Tensor
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
    stream_finished: torch.Tensor
    req_id: list[str]
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
    decode_token_start: int | None = None
    decode_token_end: int | None = None
    cached_decode: torch.Tensor | None = None
    tts_bos: torch.Tensor | None = None
    tts_eos: torch.Tensor | None = None
    tts_pad: torch.Tensor | None = None
    tts_pad_projected: torch.Tensor | None = None
    voice: torch.Tensor | None = None
    speech_feat: torch.Tensor | None = None
    speech_token: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
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
    stream_finished: torch.Tensor | None = None
    req_id: list[str] | None = None
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


class VoiceClonePromptStruct(_StructBase):
    """Inline speaker embedding payload used by qwen3_tts x-vector clone."""

    ref_spk_embedding: list[float] | None = None


class Qwen3TTSInputStruct(_StructBase):
    task_type: str | list[str] | None = None
    x_vector_only_mode: bool | list[bool] | None = None
    voice_clone_prompt: list[VoiceClonePromptStruct] | None = None
    non_streaming_mode: bool | list[bool] | None = None
    # ``instruct`` (qwen3_tts wire-name); the top-level ``OmniInputStruct.instruction``
    # is the cross-model alias for the same intent.
    instruct: str | list[str] | None = None
    ref_ids: list[int] | None = None


class MossTTSInputStruct(_StructBase):
    mode: str | list[str] | None = None
    prompt_audio_path: str | list[str] | None = None
    prompt_audio_array: list[Any] | None = None  # [[wav_list, sr]] shape
    max_new_frames: int | list[int] | None = None
    seed: int | list[int] | None = None
    text_temperature: float | list[float] | None = None
    text_top_p: float | list[float] | None = None
    text_top_k: int | list[int] | None = None
    audio_temperature: float | list[float] | None = None
    audio_top_p: float | list[float] | None = None
    audio_top_k: int | list[int] | None = None
    audio_repetition_penalty: float | list[float] | None = None


class MingTTSInputStruct(_StructBase):
    ming_task: str | None = None
    prompt: str | None = None
    use_zero_spk_emb: bool | None = None
    cfg: float | None = None
    sigma: float | None = None
    temperature: float | None = None
    spk_emb: list[float] | None = None  # user-input list shape (JSON)
    spk_emb_tensor: torch.Tensor | None = None  # stage-processed tensor shape
    max_text_length: int | None = None
    max_steps: int | None = None
    max_decode_steps: int | None = None  # alias used by serving_speech
    use_static_cache: bool | None = None
    stream_decode: bool | None = None
    prompt_wav_lat: torch.Tensor | None = None
    prompt_wav_emb: torch.Tensor | None = None


class FishSpeechInputStruct(_StructBase):
    fish_structured_voice_clone: bool | None = None
    ref_audio_wav: torch.Tensor | None = None
    ref_audio_sr: int | None = None


class VoxCPMInputStruct(_StructBase):
    cfg_value: float | list[float] | None = None
    inference_timesteps: int | list[int] | None = None
    min_len: int | list[int] | None = None
    max_len: int | list[int] | None = None
    retry_badcase: bool | list[bool] | None = None
    retry_badcase_max_times: int | list[int] | None = None
    retry_badcase_ratio_threshold: float | list[float] | None = None
    streaming_prefix_len: int | list[int] | None = None
    prompt_wav_path: str | list[str] | None = None
    latent_audio_feat: torch.Tensor | None = None  # stage-runtime carry-over


class VoxCPM2InputStruct(_StructBase):
    text_token_ids: list[list[int]] | None = None
    prompt_audio: list[Any] | None = None  # [[ref_audio, ref_sr]] when paired with prompt_text


class OmniVoiceInputStruct(_StructBase):
    ref_audio_tokens: torch.Tensor | None = None


class OmniInputStruct(_StructBase):
    text: str | list[str] | None = None
    speaker: str | list[str] | None = None
    language: str | list[str] | None = None
    instruction: str | list[str] | None = None
    max_new_tokens: int | list[int] | None = None
    voice: str | list[str] | None = None
    voice_name: str | list[str] | None = None
    voice_created_at: int | list[int] | None = None
    ref_audio: list[Any] | None = None  # [[wav, sr]] list-of-pair shape used by TTS producers
    ref_text: str | list[str] | None = None
    prompt_text: str | list[str] | None = None
    initial_codec_chunk_frames: int | None = None
    global_request_id: str | list[str] | None = None
    _is_dummy: bool | None = None
    _omni_req_id: str | None = None
    qwen3_tts: Qwen3TTSInputStruct | None = None
    moss: MossTTSInputStruct | None = None
    ming: MingTTSInputStruct | None = None
    fish_speech: FishSpeechInputStruct | None = None
    voxcpm: VoxCPMInputStruct | None = None
    voxcpm2: VoxCPM2InputStruct | None = None
    omnivoice: OmniVoiceInputStruct | None = None


class OmniPayloadStruct(_StructBase):
    """Stage-to-stage payload. Stage processors only forward stage *outputs*
    (codes/meta/embed/hidden_states/ids); user input controls live on
    ``OmniInputStruct`` and never round-trip through stage payloads."""

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
    # Forwarded input context: stage processors that need user-supplied
    # speaker/language for downstream stages set these. Other input controls
    # (text, ref_audio, per-model params) are *not* forwarded.
    speaker: list[str] | str | None = None
    language: list[str] | str | None = None
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
    """Bridge non-msgspec types when encoding Structs to msgpack bytes."""
    if isinstance(obj, torch.Tensor):
        t_cpu = obj.detach().to("cpu").contiguous()
        return {
            _TENSOR_MARKER: True,
            "data": t_cpu.numpy().tobytes(),
            "dtype": _dtype_to_name(t_cpu.dtype),
            "shape": list(t_cpu.shape),
        }
    raise NotImplementedError(f"no encoder for {type(obj).__name__}")


_T = OmniPayloadStruct | OmniInputStruct


def encode_struct(struct: _T) -> bytes:
    """Encode an ``OmniPayloadStruct`` / ``OmniInputStruct`` to msgpack bytes.

    Tensors are encoded inline via :func:`_msgspec_enc_hook` (zero-copy on the
    bytes payload; the dtype/shape come along as metadata).
    """
    return msgspec.msgpack.encode(struct, enc_hook=_msgspec_enc_hook)


def decode_struct(data: bytes, typ: type[_T]) -> _T:
    """Decode msgpack bytes back into ``typ`` (``OmniPayloadStruct`` etc)."""
    return msgspec.msgpack.decode(data, type=typ, dec_hook=_msgspec_dec_hook)


def to_input_struct(payload: dict[str, Any]) -> OmniInputStruct:
    """Convert a payload dict into ``OmniInputStruct``, validating types."""
    return msgspec.convert(payload, OmniInputStruct, dec_hook=_msgspec_dec_hook)


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


def _struct_to_dict_recursive(val: Any) -> Any:
    """Recursively expand nested ``_StructBase`` values into plain dicts.

    Tensors, primitives, and dicts pass through unchanged. Lists are walked
    element-wise so ``list[StructBase]`` (e.g. ``ref_audio``) is also expanded.
    """
    if isinstance(val, _StructBase):
        out: dict[str, Any] = {}
        for sk in val.__struct_fields__:
            sv = getattr(val, sk)
            if sv is None:
                continue
            out[sk] = _struct_to_dict_recursive(sv)
        return out
    if isinstance(val, list):
        return [_struct_to_dict_recursive(v) for v in val]
    return val


def to_dict(struct: OmniPayloadStruct | OmniInputStruct | _StructBase) -> dict[str, Any]:
    """Convert any ``_StructBase`` to a plain dict, dropping ``None`` fields.

    Nested sub-structs are expanded recursively so the result is dict-of-dicts
    all the way down.
    """
    return _struct_to_dict_recursive(struct) or {}


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


# ── Keys whose values are nested dicts (TypedDict sub-categories) ──
_NESTED_KEYS = frozenset({"hidden_states", "embed", "ids", "codes", "meta"})
_INPUT_NESTED_KEYS = frozenset({"qwen3_tts", "moss", "ming", "fish_speech", "voxcpm", "voxcpm2", "omnivoice"})


def flatten_payload(
    payload: dict[str, Any] | OmniPayloadStruct | OmniInputStruct,
    nested_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Flatten a nested payload (dict or struct) to dotted keys.

    Sub-fields under ``nested_keys`` (default: stage-payload keys
    ``_NESTED_KEYS``; pass ``_INPUT_NESTED_KEYS`` for ``OmniInputStruct``)
    are expanded one level: ``{"codes": {"audio": tensor}}`` →
    ``{"codes.audio": tensor}``. ``hidden_states["layers"]`` is expanded to
    ``hidden_states.layer_N``. Other top-level values are kept as-is.
    """
    if isinstance(payload, OmniInputStruct) and nested_keys is None:
        nested_keys = _INPUT_NESTED_KEYS
    if nested_keys is None:
        nested_keys = _NESTED_KEYS
    if isinstance(payload, _StructBase):
        payload = to_dict(payload)
    if not payload:
        return {}
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if key in nested_keys and isinstance(value, dict):
            for qual, val in value.items():
                if qual == "layers" and key == "hidden_states" and isinstance(val, dict):
                    for layer_idx, tensor in val.items():
                        flat[f"hidden_states.layer_{layer_idx}"] = tensor
                else:
                    flat[f"{key}.{qual}"] = val
        else:
            flat[key] = value
    return flat


def unflatten_payload(flat: dict[str, Any]) -> dict[str, Any]:
    """Unflatten dotted keys back to nested dicts.

    Reverse of :func:`flatten_payload`.
    ``hidden_states.layer_N`` keys are collected into ``hidden_states.layers``.
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if "." in key:
            type_key, qualifier = key.split(".", 1)
            sub = result.setdefault(type_key, {})
            if type_key == "hidden_states" and qualifier.startswith("layer_"):
                layers = sub.setdefault("layers", {})
                layer_idx = int(qualifier[len("layer_") :])
                layers[layer_idx] = value
            else:
                sub[qualifier] = value
        else:
            result[key] = value
    return result


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
    payload: OmniPayload | OmniPayloadStruct | OmniInputStruct,
) -> AdditionalInformationPayload | None:
    """Serialize an ``OmniPayload`` for EngineCore transport.

    Uses :func:`flatten_payload` to produce dotted keys, then converts
    each value to an ``AdditionalInformationEntry``.
    """
    from vllm_omni.engine import (
        AdditionalInformationEntry,
        AdditionalInformationPayload,
    )

    flat = flatten_payload(payload)
    entries: dict[str, AdditionalInformationEntry] = {}

    for key, value in flat.items():
        if isinstance(value, torch.Tensor):
            entries[key] = _serialize_tensor(value)
        elif isinstance(value, list):
            entries[key] = AdditionalInformationEntry(list_data=value)
        elif value is not None:
            entries[key] = AdditionalInformationEntry(scalar_data=value)

    return AdditionalInformationPayload(entries=entries) if entries else None


def deserialize_payload(
    wire: AdditionalInformationPayload,
) -> OmniPayload:
    """Deserialize an ``AdditionalInformationPayload`` back to ``OmniPayload``.

    Decodes entries to tensors/lists, then uses :func:`unflatten_payload`
    to reconstruct the nested structure.
    """
    flat: dict[str, Any] = {}

    for key, entry in wire.entries.items():
        if entry.tensor_data is not None:
            flat[key] = _deserialize_tensor(entry)
        elif entry.list_data is not None:
            flat[key] = entry.list_data
        elif entry.scalar_data is not None:
            flat[key] = entry.scalar_data

    return unflatten_payload(flat)  # type: ignore[return-value]

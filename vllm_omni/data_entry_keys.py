"""Canonical data-entry key names for inter-stage communication.

Adding a new model?
~~~~~~~~~~~~~~~~~~~
Every key you put into the inter-stage payload (``additional_information``,
``multimodal_output``, ``pooling_output``) **must** follow this format:

    {type}.{qualifier}          e.g.  "hidden_states.output"

Allowed types
    hidden_states  – intermediate / output hidden-state tensors
    embed          – embedding tensors (prefill, decode, special tokens)
    ids            – token-ID sequences
    codes          – codec / audio code tensors
    meta           – scalar metadata, control flags, shapes

Rules
    1. Pick from the existing keys below whenever your concept matches.
       Two models that transfer the same semantic concept MUST use the
       same key (e.g. all models use ``"codes.audio"`` for audio codes,
       not a model-specific name).
    2. If no existing key fits, create a new ``{type}.{qualifier}`` key.
       Keep the qualifier short and descriptive (snake_case).
    3. Never use bare names like ``"hidden"`` or ``"finished"`` — always
       use the dotted form.  ``normalize_keys()`` will catch old-style
       names at runtime and log a deprecation warning.

Existing keys (reference)
    hidden_states.layer_0          hidden_states.layer_{N}
    hidden_states.output           hidden_states.trailing_text
    hidden_states.last
    embed.prefill                  embed.decode
    embed.cached_decode            embed.tts_bos / tts_eos / tts_pad
    embed.tts_pad_projected        embed.voice
    embed.speech_feat              embed.thinker_reply
    ids.all        ids.prompt      ids.output
    ids.speech_token               ids.prior_image
    codes.audio                    codes.ref
    meta.finished                  meta.left_context_size
    meta.override_keys             meta.num_processed_tokens
    meta.next_stage_prompt_len     meta.ar_width
    meta.eol_token_id              meta.visual_token_start_id
    meta.visual_token_end_id       meta.gen_token_mask
    meta.generated_len             meta.omni_task
    meta.height                    meta.width
    meta.decode_flag               meta.codec_streaming
    meta.ref_code_len              meta.talker_prefill_offset

This module provides:
- Structured ``TypedDict`` types for static type checking (``OmniPayload``)
- ``unflatten_payload`` / ``flatten_payload``: convert between flat and nested
- ``_LEGACY_KEY_MAP``: old key name → canonical key name
- ``normalize_keys(payload)``: translate any legacy keys in a dict
- ``validate_key(key)``: check a key follows the naming convention
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import torch

logger = logging.getLogger(__name__)

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
    override_keys: list[str]
    num_processed_tokens: int
    next_stage_prompt_len: int
    ar_width: int
    eol_token_id: int
    visual_token_start_id: int
    visual_token_end_id: int
    gen_token_mask: torch.Tensor
    generated_len: int
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


# ── Legacy key mapping (old name → canonical name) ──
_LEGACY_KEY_MAP: dict[str, str] = {
    # hidden states
    "0": "hidden_states.layer_0",
    "hidden": "hidden_states.layer_0",
    "24": "hidden_states.layer_24",
    "thinker_hidden_states": "hidden_states.output",
    "thinker_result": "hidden_states.output",
    "latent": "hidden_states.output",
    "trailing_text_hidden": "hidden_states.trailing_text",
    "tailing_text_hidden": "hidden_states.trailing_text",
    "last_talker_hidden": "hidden_states.last",
    # embeddings
    "thinker_prefill_embeddings": "embed.prefill",
    "prompt_embeds": "embed.prefill",
    "talker_prompt_embeds": "embed.prefill",
    "thinker_decode_embeddings": "embed.decode",
    "decode_output_prompt_embeds": "embed.decode",
    "cached_thinker_decode_embeddings": "embed.cached_decode",
    "tts_bos_embed": "embed.tts_bos",
    "tts_eos_embed": "embed.tts_eos",
    "tts_pad_embed": "embed.tts_pad",
    "tts_pad_embed_projected": "embed.tts_pad_projected",
    "embedding": "embed.voice",
    "speech_feat": "embed.speech_feat",
    "thinker_reply_part": "embed.thinker_reply",
    # token ids
    "thinker_sequences": "ids.all",
    "thinker_input_ids": "ids.prompt",
    "prompt_token_ids": "ids.prompt",
    "prefix_ids": "ids.prompt",
    "thinker_output_token_ids": "ids.output",
    "speech_token": "ids.speech_token",
    "prior_token_image_ids": "ids.prior_image",
    # codes
    "code_predictor_codes": "codes.audio",
    "audio_codes": "codes.audio",
    "ref_code": "codes.ref",
    # metadata
    "finished": "meta.finished",
    "finished_flag": "meta.finished",
    "left_context_size": "meta.left_context_size",
    "override_keys": "meta.override_keys",
    "num_processed_tokens": "meta.num_processed_tokens",
    "next_stage_prompt_len": "meta.next_stage_prompt_len",
    "ar_width": "meta.ar_width",
    "eol_token_id": "meta.eol_token_id",
    "visual_token_start_id": "meta.visual_token_start_id",
    "visual_token_end_id": "meta.visual_token_end_id",
    "gen_token_mask": "meta.gen_token_mask",
    "generated_len": "meta.generated_len",
    "omni_task": "meta.omni_task",
    "height": "meta.height",
    "width": "meta.width",
    "decode_flag": "meta.decode_flag",
    "codec_streaming": "meta.codec_streaming",
    "ref_code_len": "meta.ref_code_len",
    "talker_prefill_offset": "meta.talker_prefill_offset",
}


_VALID_TYPES = frozenset({"hidden_states", "embed", "ids", "codes", "meta"})

_LAYER_RE = re.compile(r"^hidden_states\.layer_(\d+)$")


def validate_key(key: str) -> bool:
    """Return *True* if *key* follows the ``{type}.{qualifier}`` convention.

    >>> validate_key("codes.audio")
    True
    >>> validate_key("finished")
    False
    """
    parts = key.split(".", 1)
    return len(parts) == 2 and parts[0] in _VALID_TYPES and len(parts[1]) > 0


def normalize_keys(payload: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
    """Return a new dict with legacy key names replaced by canonical names.

    Keys already using canonical names are passed through unchanged.
    Unknown keys (not legacy and not canonical) are warned about if they
    don't follow the ``{type}.{qualifier}`` naming convention.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = _LEGACY_KEY_MAP.get(key)
        if canonical is not None:
            if warn:
                logger.warning(
                    "Deprecated data-entry key %r – use %r instead",
                    key,
                    canonical,
                )
            out[canonical] = value
        else:
            if warn and not validate_key(key):
                logger.warning(
                    "Data-entry key %r does not follow the "
                    "'{type}.{qualifier}' convention – see "
                    "vllm_omni/data_entry_keys.py for guidance",
                    key,
                )
            out[key] = value
    return out


def unflatten_payload(flat: dict[str, Any]) -> OmniPayload:
    """Convert a flat ``{type}.{qualifier}`` dict into a nested ``OmniPayload``.

    Legacy key names are normalized first.  Keys that don't match the
    ``{type}.{qualifier}`` convention are silently skipped.

    The special pattern ``hidden_states.layer_N`` is grouped under
    ``hidden_states["layers"][N]``.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        # Normalize legacy keys
        canonical = _LEGACY_KEY_MAP.get(key, key)

        # Handle hidden_states.layer_N → layers dict
        m = _LAYER_RE.match(canonical)
        if m:
            hs = nested.setdefault("hidden_states", {})
            layers = hs.setdefault("layers", {})
            layers[int(m.group(1))] = value
            continue

        parts = canonical.split(".", 1)
        if len(parts) == 2 and parts[0] in _VALID_TYPES:
            sub = nested.setdefault(parts[0], {})
            sub[parts[1]] = value
        # Non-conforming keys are silently dropped from the typed view.
        # They remain accessible in the original flat dict.

    return nested  # type: ignore[return-value]


def flatten_payload(nested: OmniPayload) -> dict[str, Any]:
    """Convert a nested ``OmniPayload`` back to a flat ``{type}.{qualifier}`` dict.

    The ``hidden_states["layers"]`` sub-dict is expanded to individual
    ``hidden_states.layer_N`` keys.
    """
    flat: dict[str, Any] = {}
    for type_key in ("hidden_states", "embed", "ids", "codes", "meta"):
        sub = nested.get(type_key)  # type: ignore[literal-required]
        if sub is None:
            continue
        for qual, val in sub.items():
            if qual == "layers" and type_key == "hidden_states":
                if isinstance(val, dict):
                    for layer_idx, tensor in val.items():
                        flat[f"hidden_states.layer_{layer_idx}"] = tensor
            else:
                flat[f"{type_key}.{qual}"] = val
    return flat

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Shared TTS utility functions for speaker and language extraction.

These utilities are model-agnostic and can be used by any TTS model stage
processor (qwen3_omni, qwen2_5_omni, qwen3_tts, etc.).
"""

from typing import Any

from vllm_omni.data_entry_keys import OmniInputStruct, OmniPayloadStruct
from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload

# =============================================================================
# Read from ``OmniPayloadStruct.input`` — used by producers migrated to emit
# ``OmniPayloadStruct(input=OmniInputStruct(...))``. The legacy extractors
# below still read top-level keys for unmigrated producers.
# =============================================================================


def _input_value(entries: dict[str, AdditionalInformationEntry] | None, field: str) -> Any:
    """Read ``input.<field>`` from a wire ``entries`` dict.

    The wire flattens ``OmniInputStruct`` one level under ``input.*`` keys.
    """
    if not entries:
        return None
    entry = entries.get(f"input.{field}")
    if entry is None:
        return None
    return entry.list_data if entry.list_data is not None else entry.scalar_data


def _input_struct(prompt: dict[str, Any] | list[dict[str, Any]] | None, index: int) -> OmniInputStruct | None:
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if not isinstance(p, dict):
        return None
    add_info = p.get("additional_information")
    if isinstance(add_info, OmniPayloadStruct):
        return add_info.input
    return None


def input_speaker_from_request(request: Any) -> str | None:
    add_info: AdditionalInformationPayload | None = getattr(request, "additional_information", None)
    if add_info is None:
        return None
    val = _input_value(add_info.entries, "speaker")
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, str) and val.strip():
        return val.lower().strip()
    return None


def input_language_from_request(request: Any) -> list[str] | None:
    add_info: AdditionalInformationPayload | None = getattr(request, "additional_information", None)
    if add_info is None:
        return None
    val = _input_value(add_info.entries, "language")
    if isinstance(val, list) and val:
        return val
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return None


def input_speaker_from_prompt(prompt: dict[str, Any] | list[dict[str, Any]] | None, index: int = 0) -> list[str] | None:
    inp = _input_struct(prompt, index)
    if inp is None or inp.speaker is None:
        return None
    return inp.speaker if isinstance(inp.speaker, list) else [inp.speaker]


def input_language_from_prompt(
    prompt: dict[str, Any] | list[dict[str, Any]] | None, index: int = 0
) -> list[str] | None:
    inp = _input_struct(prompt, index)
    if inp is None or inp.language is None:
        return None
    return inp.language if isinstance(inp.language, list) else [inp.language]


# =============================================================================
# Speaker helpers
# =============================================================================


def extract_speaker_from_runtime_info(
    runtime_additional_information: list[dict[str, Any]] | None,
) -> str | None:
    """Extract speaker from per-request runtime info dicts.

    Iterates through the list of per-request info dicts and returns the first
    non-empty speaker string found, normalized to lowercase.

    Args:
        runtime_additional_information: List of per-request additional info
            dicts, as passed to the model's forward() method.

    Returns:
        The speaker string (lowercase, stripped), or None if not present.
    """
    if not runtime_additional_information:
        return None
    for info in runtime_additional_information:
        vt = info.get("speaker")
        if vt is None:
            continue
        if isinstance(vt, (list, tuple)) and len(vt) > 0:
            vt = vt[0]
        if isinstance(vt, str) and vt.strip():
            return vt.lower().strip()
        if vt is not None:
            return str(vt).lower().strip()
    return None


def extract_speaker_from_request(request: Any) -> str | None:
    """Extract speaker from a request's additional_information field.

    Reads from the structured ``additional_information.entries["speaker"]``
    field used by the engine serialization layer.

    Args:
        request: An OmniEngineCoreRequest (or compatible object) with an
            ``additional_information`` attribute.

    Returns:
        The speaker string (lowercase, stripped), or None if not present.
    """
    additional_information = getattr(request, "additional_information", None)
    if additional_information is None:
        return None
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("speaker")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        val = list_data[0]
        return val.lower().strip() if isinstance(val, str) else str(val).lower().strip()
    return None


def extract_speaker_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract speaker from a prompt's additional_information dict.

    Used in non-async stage processors where the prompt is an
    OmniTokensPrompt / TextPrompt dict (or a list of them).

    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The speaker as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    speaker = add_info.get("speaker")
    if isinstance(speaker, list) and speaker:
        return speaker
    return None


# =============================================================================
# Language helpers
# =============================================================================


def extract_language_from_runtime_info(
    runtime_additional_information: list[dict[str, Any]] | None,
) -> str | None:
    """Extract language from per-request runtime info dicts.
    Args:
        runtime_additional_information: List of per-request additional info
            dicts, as passed to the model's forward() method.

    Returns:
        The language string (e.g. "Chinese", "English", "Auto"), or None.
    """
    if not runtime_additional_information:
        return None
    for info in runtime_additional_information:
        lang = info.get("language")
        if lang is None:
            continue
        if isinstance(lang, (list, tuple)) and len(lang) > 0:
            return lang
        if isinstance(lang, str) and lang.strip():
            return [lang.strip()]
    return None


def extract_language_from_request(request: Any) -> str | None:
    """Extract language from a request's additional_information field.

    Args:
        request: An OmniEngineCoreRequest (or compatible object) with an
            ``additional_information`` attribute.

    Returns:
        The language string, or None if not present.
    """
    additional_information = getattr(request, "additional_information", None)
    if additional_information is None:
        return None
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("language")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        return list_data
    return None


def extract_language_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract language from a prompt's additional_information dict.
    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The language as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    language = add_info.get("language")
    if isinstance(language, list) and language:
        return language
    return None

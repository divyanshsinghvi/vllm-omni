# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, to_struct

logger = init_logger(__name__)


def serialize_additional_information(
    raw_info: dict[str, Any] | OmniPayloadStruct | None,
    *,
    log_prefix: str | None = None,
) -> OmniPayloadStruct | None:
    """Convert dict-form ``OmniPayload`` into ``OmniPayloadStruct`` for cross-process transport."""
    if raw_info is None:
        return None
    if isinstance(raw_info, OmniPayloadStruct):
        return raw_info
    return to_struct(raw_info)


def deserialize_additional_information(payload: dict | OmniPayloadStruct | None) -> dict:
    """Convert an ``OmniPayloadStruct`` back into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return to_dict(payload)

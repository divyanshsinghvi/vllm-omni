"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    OmniInputStruct,
    OmniPayloadStruct,
    deserialize_payload,
    serialize_payload,
)
from vllm_omni.engine import AdditionalInformationPayload

logger = init_logger(__name__)


def serialize_additional_information(
    raw_info: dict[str, Any] | OmniPayloadStruct | OmniInputStruct | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    """Serialize omni request metadata for EngineCore transport.

    Delegates to ``serialize_payload`` which understands ``OmniPayload``
    (TypedDict / ``OmniPayloadStruct``) for stage payloads and
    ``OmniInputStruct`` for input-side request controls.
    """
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info

    return serialize_payload(raw_info)


def deserialize_additional_information(
    payload: dict | AdditionalInformationPayload | None,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return deserialize_payload(payload)  # type: ignore[return-value]

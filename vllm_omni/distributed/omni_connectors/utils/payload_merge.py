# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared depth-2 chunk-payload merge for accumulating connector chunks.

Used by both ``OmniChunkTransferAdapter._update_request_payload`` and
``OmniConnectorModelRunnerMixin._accumulate_payload`` so the two transports
agree on how nested payload dicts combine across chunks.
"""

from collections.abc import Iterable
from typing import Any

import torch

from vllm_omni.worker.payload_span import (
    get_tensor_span,
    merge_tensor_spans,
)

# Span groups: each entry is (tensor_key, start_key, end_key) within the
# nested sub-dict (e.g. inside ``embed``). When all three are present in
# both origin and incoming, the tensor is merged via ``merge_tensor_spans``
# instead of naive ``torch.cat``.
_SPAN_GROUPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "embed": (
        ("decode", "decode_token_start", "decode_token_end"),
        ("cached_decode", "cached_decode_token_start", "cached_decode_token_end"),
    ),
}


def _normalize_override_keys(raw: Any) -> set[tuple[str, str] | str]:
    out: set[tuple[str, str] | str] = set()
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return out
    for k in raw:
        if isinstance(k, list):
            out.add(tuple(k))  # type: ignore[arg-type]
        elif isinstance(k, tuple):
            out.add(k)
        elif isinstance(k, str):
            out.add(k)
    return out


def _merge_value(origin_val: Any, incoming_val: Any) -> Any:
    if isinstance(incoming_val, torch.Tensor) and isinstance(origin_val, torch.Tensor):
        return torch.cat([origin_val, incoming_val], dim=0)
    if isinstance(incoming_val, list) and isinstance(origin_val, list):
        return origin_val + incoming_val
    return incoming_val


def _merge_subdict(
    origin_sub: dict[str, Any],
    incoming_sub: dict[str, Any],
    *,
    type_key: str,
    override_keys: set[tuple[str, str] | str],
) -> dict[str, Any]:
    merged = dict(origin_sub)
    span_groups = _SPAN_GROUPS.get(type_key, ())
    span_handled: set[str] = set()

    for tensor_key, start_key, end_key in span_groups:
        if tensor_key not in incoming_sub:
            continue
        if (type_key, tensor_key) in override_keys or tensor_key in override_keys:
            continue
        merged_span = merge_tensor_spans(
            get_tensor_span(origin_sub, tensor_key=tensor_key, start_key=start_key, end_key=end_key),
            get_tensor_span(incoming_sub, tensor_key=tensor_key, start_key=start_key, end_key=end_key),
        )
        if merged_span is None:
            continue
        tensor, start, end = merged_span
        merged[tensor_key] = tensor
        merged[start_key] = start
        merged[end_key] = end
        span_handled |= {tensor_key, start_key, end_key}

    for sub_key, sub_val in incoming_sub.items():
        if sub_key in span_handled:
            continue
        # `meta.finished` is always taken from incoming (terminal-state signal,
        # not a value to accumulate).
        if type_key == "meta" and sub_key == "finished":
            merged[sub_key] = sub_val
            continue
        if (type_key, sub_key) in override_keys or sub_key in override_keys:
            merged[sub_key] = sub_val
            continue
        if sub_key in origin_sub:
            merged[sub_key] = _merge_value(origin_sub[sub_key], sub_val)
        else:
            merged[sub_key] = sub_val

    return merged


def merge_chunk_payloads(origin: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge ``incoming`` chunk payload into ``origin`` (depth-2).

    Rules:
      * For each top-level key whose value is a dict, recurse one level:
        - tensors are concatenated along dim=0,
        - lists are extended,
        - scalars/None: incoming wins,
        - ``embed.decode`` / ``embed.cached_decode``: span-aware merge when
          the matching ``*_token_start``/``*_token_end`` fields are present
          in both origin and incoming; otherwise naive cat.
        - ``meta.finished`` is always taken from incoming.
        - Sub-keys listed in ``incoming.meta.override_keys`` are replaced
          (no merge). Entries may be either ``(type_key, sub_key)`` tuples
          or bare ``sub_key`` strings.
      * Top-level non-dict keys follow the tensor/list/scalar rules above.

    Returns a shallow-copy dict; sub-dicts are also copied. Tensors and lists
    are not deep-copied (callers should treat the result as read-mostly).
    """
    override_meta = incoming.get("meta", {}) if isinstance(incoming.get("meta"), dict) else {}
    override_keys = _normalize_override_keys(override_meta.get("override_keys", ()))

    merged: dict[str, Any] = dict(origin)

    for type_key, incoming_val in incoming.items():
        if isinstance(incoming_val, dict):
            origin_sub = origin.get(type_key)
            origin_sub = origin_sub if isinstance(origin_sub, dict) else {}
            merged[type_key] = _merge_subdict(origin_sub, incoming_val, type_key=type_key, override_keys=override_keys)
            continue
        if type_key in override_keys:
            merged[type_key] = incoming_val
            continue
        if type_key in origin:
            merged[type_key] = _merge_value(origin[type_key], incoming_val)
        else:
            merged[type_key] = incoming_val

    return merged

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.distributed.omni_connectors.utils.payload_merge import merge_chunk_payloads

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _t(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def test_depth2_tensor_concat_under_meta_embed_codes():
    origin = {
        "embed": {"decode": _t([[1.0, 2.0]])},
        "codes": {"audio": torch.tensor([1, 2], dtype=torch.int64)},
        "meta": {"left_context_size": 4},
    }
    incoming = {
        "embed": {"decode": _t([[3.0, 4.0]])},
        "codes": {"audio": torch.tensor([3, 4], dtype=torch.int64)},
        "meta": {"left_context_size": 5, "finished": torch.tensor(False)},
    }

    merged = merge_chunk_payloads(origin, incoming)

    assert torch.equal(merged["embed"]["decode"], _t([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(merged["codes"]["audio"], torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    # `left_context_size` is a scalar and not in override_keys, so incoming wins.
    assert merged["meta"]["left_context_size"] == 5
    # `finished` is always taken from incoming.
    assert bool(merged["meta"]["finished"]) is False


def test_meta_finished_taken_from_incoming():
    origin = {"meta": {"finished": torch.tensor(False)}}
    incoming = {"meta": {"finished": torch.tensor(True)}}
    merged = merge_chunk_payloads(origin, incoming)
    assert bool(merged["meta"]["finished"]) is True


def test_override_keys_replace_instead_of_merging():
    origin = {
        "embed": {"prefill": _t([[1.0]])},
        "meta": {},
    }
    incoming = {
        "embed": {"prefill": _t([[9.0, 9.0]])},
        "meta": {"override_keys": [["embed", "prefill"]]},
    }
    merged = merge_chunk_payloads(origin, incoming)
    # prefill is replaced wholesale, not concatenated.
    assert torch.equal(merged["embed"]["prefill"], _t([[9.0, 9.0]]))


def test_span_aware_merge_adjacent():
    origin = {
        "embed": {
            "decode": _t([[1.0], [2.0]]),
            "decode_token_start": 10,
            "decode_token_end": 12,
        }
    }
    incoming = {
        "embed": {
            "decode": _t([[3.0], [4.0]]),
            "decode_token_start": 12,
            "decode_token_end": 14,
        }
    }
    merged = merge_chunk_payloads(origin, incoming)
    assert torch.equal(merged["embed"]["decode"], _t([[1.0], [2.0], [3.0], [4.0]]))
    assert merged["embed"]["decode_token_start"] == 10
    assert merged["embed"]["decode_token_end"] == 14


def test_span_aware_merge_partial_overlap_trims_incoming():
    origin = {
        "embed": {
            "decode": _t([[1.0], [2.0], [3.0]]),
            "decode_token_start": 10,
            "decode_token_end": 13,
        }
    }
    incoming = {
        "embed": {
            "decode": _t([[3.5], [4.0], [5.0]]),
            "decode_token_start": 12,
            "decode_token_end": 15,
        }
    }
    merged = merge_chunk_payloads(origin, incoming)
    # Overlap = 13 - 12 = 1, so first row of incoming is trimmed.
    assert torch.equal(merged["embed"]["decode"], _t([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    assert merged["embed"]["decode_token_start"] == 10
    assert merged["embed"]["decode_token_end"] == 15


def test_span_aware_falls_back_to_naive_cat_without_span_metadata():
    origin = {"embed": {"decode": _t([[1.0]])}}
    incoming = {"embed": {"decode": _t([[2.0]])}}
    merged = merge_chunk_payloads(origin, incoming)
    assert torch.equal(merged["embed"]["decode"], _t([[1.0], [2.0]]))


def test_list_extension_inside_subdict():
    origin = {"meta": {"some_list": [1, 2]}}
    incoming = {"meta": {"some_list": [3, 4]}}
    merged = merge_chunk_payloads(origin, incoming)
    assert merged["meta"]["some_list"] == [1, 2, 3, 4]


def test_first_chunk_returns_dict_unchanged():
    incoming = {"embed": {"decode": _t([[1.0]])}}
    merged = merge_chunk_payloads({}, incoming)
    assert torch.equal(merged["embed"]["decode"], _t([[1.0]]))


def test_parity_chunk_adapter_vs_mixin_accumulator():
    """Both call sites should produce identical merged output for the same
    chunk pair. This pins the contract that the two transports agree.
    """
    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
        OmniChunkTransferAdapter,
    )

    chunk1 = {
        "embed": {
            "decode": _t([[1.0], [2.0]]),
            "decode_token_start": 0,
            "decode_token_end": 2,
        },
        "codes": {"audio": torch.tensor([1], dtype=torch.int64)},
        "meta": {"finished": torch.tensor(False)},
    }
    chunk2 = {
        "embed": {
            "decode": _t([[3.0]]),
            "decode_token_start": 2,
            "decode_token_end": 3,
        },
        "codes": {"audio": torch.tensor([2], dtype=torch.int64)},
        "meta": {"finished": torch.tensor(True)},
    }

    # Adapter path
    adapter = OmniChunkTransferAdapter.__new__(OmniChunkTransferAdapter)
    adapter.request_payload = {}
    adapter._update_request_payload("r1", dict(chunk1))
    adapter_merged = adapter._update_request_payload("r1", dict(chunk2))

    # Mixin path: emulate by calling merge_chunk_payloads directly the same
    # way `_accumulate_payload` does (post-refactor it's a thin wrapper).
    mixin_merged = merge_chunk_payloads(dict(chunk1), dict(chunk2))

    assert torch.equal(adapter_merged["embed"]["decode"], mixin_merged["embed"]["decode"])
    assert adapter_merged["embed"]["decode_token_start"] == mixin_merged["embed"]["decode_token_start"]
    assert adapter_merged["embed"]["decode_token_end"] == mixin_merged["embed"]["decode_token_end"]
    assert torch.equal(adapter_merged["codes"]["audio"], mixin_merged["codes"]["audio"])
    assert bool(adapter_merged["meta"]["finished"]) == bool(mixin_merged["meta"]["finished"])

"""Tests for data_entry_keys: TypedDict payload structure."""

import torch

from vllm_omni.data_entry_keys import OmniPayload


class TestOmniPayload:
    def test_nested_payload_structure(self):
        """Verify OmniPayload can be constructed with nested dicts."""
        payload: OmniPayload = {
            "hidden_states": {"output": torch.tensor([1.0])},
            "embed": {"prefill": torch.tensor([2.0])},
            "codes": {"audio": torch.tensor([3.0])},
            "ids": {"all": [1, 2, 3]},
            "meta": {"finished": True},
        }
        assert torch.equal(payload["hidden_states"]["output"], torch.tensor([1.0]))
        assert torch.equal(payload["embed"]["prefill"], torch.tensor([2.0]))
        assert torch.equal(payload["codes"]["audio"], torch.tensor([3.0]))
        assert payload["ids"]["all"] == [1, 2, 3]
        assert payload["meta"]["finished"] is True

    def test_partial_payload(self):
        """OmniPayload fields are all optional (total=False)."""
        payload: OmniPayload = {"meta": {"finished": True}}
        assert payload["meta"]["finished"] is True

    def test_empty_payload(self):
        payload: OmniPayload = {}
        assert len(payload) == 0

"""Tests for data_entry_keys: TypedDict types and flatten."""

import torch

from vllm_omni.data_entry_keys import flatten_payload


class TestFlattenPayload:
    def test_basic_flatten(self):
        nested = {
            "hidden_states": {"output": torch.tensor([1.0])},
            "embed": {"prefill": torch.tensor([2.0])},
            "codes": {"audio": torch.tensor([3.0])},
            "ids": {"all": [1, 2, 3]},
            "meta": {"finished": True},
        }
        flat = flatten_payload(nested)
        assert torch.equal(flat["hidden_states.output"], torch.tensor([1.0]))
        assert torch.equal(flat["embed.prefill"], torch.tensor([2.0]))
        assert torch.equal(flat["codes.audio"], torch.tensor([3.0]))
        assert flat["ids.all"] == [1, 2, 3]
        assert flat["meta.finished"] is True

    def test_layer_keys(self):
        nested = {
            "hidden_states": {
                "output": torch.tensor([3.0]),
                "layers": {
                    0: torch.tensor([1.0]),
                    24: torch.tensor([2.0]),
                },
            },
        }
        flat = flatten_payload(nested)
        assert torch.equal(flat["hidden_states.layer_0"], torch.tensor([1.0]))
        assert torch.equal(flat["hidden_states.layer_24"], torch.tensor([2.0]))
        assert torch.equal(flat["hidden_states.output"], torch.tensor([3.0]))

    def test_empty_dict(self):
        assert flatten_payload({}) == {}

"""Tests for data_entry_keys: TypedDict types, flatten/unflatten, normalize."""

import torch

from vllm_omni.data_entry_keys import (
    flatten_payload,
    normalize_keys,
    unflatten_payload,
    validate_key,
)


class TestValidateKey:
    def test_valid_keys(self):
        assert validate_key("codes.audio")
        assert validate_key("hidden_states.output")
        assert validate_key("meta.finished")
        assert validate_key("embed.prefill")
        assert validate_key("ids.all")

    def test_invalid_keys(self):
        assert not validate_key("finished")
        assert not validate_key("code_predictor_codes")
        assert not validate_key("unknown.type.extra")
        assert not validate_key("")
        assert not validate_key("hidden_states.")


class TestNormalizeKeys:
    def test_legacy_keys_mapped(self):
        payload = {"finished": True, "code_predictor_codes": [1, 2]}
        result = normalize_keys(payload, warn=False)
        assert "meta.finished" in result
        assert "codes.audio" in result
        assert "finished" not in result

    def test_canonical_keys_passthrough(self):
        payload = {"meta.finished": True, "codes.audio": [1, 2]}
        result = normalize_keys(payload, warn=False)
        assert result == payload

    def test_new_legacy_keys(self):
        payload = {"decode_flag": True, "codec_streaming": False}
        result = normalize_keys(payload, warn=False)
        assert result == {"meta.decode_flag": True, "meta.codec_streaming": False}


class TestUnflattenPayload:
    def test_basic_unflatten(self):
        flat = {
            "hidden_states.output": torch.tensor([1.0]),
            "embed.prefill": torch.tensor([2.0]),
            "codes.audio": torch.tensor([3.0]),
            "ids.all": [1, 2, 3],
            "meta.finished": True,
        }
        nested = unflatten_payload(flat)
        assert torch.equal(nested["hidden_states"]["output"], torch.tensor([1.0]))
        assert torch.equal(nested["embed"]["prefill"], torch.tensor([2.0]))
        assert torch.equal(nested["codes"]["audio"], torch.tensor([3.0]))
        assert nested["ids"]["all"] == [1, 2, 3]
        assert nested["meta"]["finished"] is True

    def test_layer_keys(self):
        flat = {
            "hidden_states.layer_0": torch.tensor([1.0]),
            "hidden_states.layer_24": torch.tensor([2.0]),
            "hidden_states.output": torch.tensor([3.0]),
        }
        nested = unflatten_payload(flat)
        hs = nested["hidden_states"]
        assert torch.equal(hs["layers"][0], torch.tensor([1.0]))
        assert torch.equal(hs["layers"][24], torch.tensor([2.0]))
        assert torch.equal(hs["output"], torch.tensor([3.0]))

    def test_legacy_keys_normalized(self):
        flat = {"finished": True, "code_predictor_codes": torch.tensor([1.0])}
        nested = unflatten_payload(flat)
        assert nested["meta"]["finished"] is True
        assert torch.equal(nested["codes"]["audio"], torch.tensor([1.0]))

    def test_empty_dict(self):
        assert unflatten_payload({}) == {}

    def test_non_conforming_keys_dropped(self):
        flat = {"some_random_key": 42, "codes.audio": torch.tensor([1.0])}
        nested = unflatten_payload(flat)
        assert "some_random_key" not in nested
        assert torch.equal(nested["codes"]["audio"], torch.tensor([1.0]))


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


class TestRoundTrip:
    def test_flat_round_trip(self):
        flat = {
            "hidden_states.output": torch.tensor([1.0]),
            "hidden_states.last": torch.tensor([2.0]),
            "embed.prefill": torch.tensor([3.0]),
            "embed.tts_bos": torch.tensor([4.0]),
            "codes.audio": torch.tensor([5.0]),
            "ids.all": [1, 2, 3],
            "ids.prompt": [1, 2],
            "meta.finished": True,
            "meta.num_processed_tokens": 42,
        }
        result = flatten_payload(unflatten_payload(flat))
        assert set(result.keys()) == set(flat.keys())
        for k in flat:
            if isinstance(flat[k], torch.Tensor):
                assert torch.equal(result[k], flat[k])
            else:
                assert result[k] == flat[k]

    def test_flat_round_trip_with_layers(self):
        flat = {
            "hidden_states.layer_0": torch.tensor([1.0]),
            "hidden_states.layer_24": torch.tensor([2.0]),
            "hidden_states.output": torch.tensor([3.0]),
        }
        result = flatten_payload(unflatten_payload(flat))
        assert set(result.keys()) == set(flat.keys())
        for k in flat:
            assert torch.equal(result[k], flat[k])

    def test_nested_round_trip(self):
        nested = {
            "hidden_states": {
                "output": torch.tensor([1.0]),
                "layers": {0: torch.tensor([2.0])},
            },
            "embed": {"prefill": torch.tensor([3.0])},
            "meta": {"finished": True},
        }
        result = unflatten_payload(flatten_payload(nested))
        assert torch.equal(
            result["hidden_states"]["output"],
            nested["hidden_states"]["output"],
        )
        assert torch.equal(
            result["hidden_states"]["layers"][0],
            nested["hidden_states"]["layers"][0],
        )
        assert torch.equal(
            result["embed"]["prefill"],
            nested["embed"]["prefill"],
        )
        assert result["meta"]["finished"] is True

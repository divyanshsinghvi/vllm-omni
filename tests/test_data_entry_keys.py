"""Tests for data_entry_keys."""

import msgspec
import pytest
import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayloadStruct,
    to_dict,
    to_struct,
)


class TestOmniPayloadStruct:
    """Runtime-validated mirror of OmniPayload (msgspec.Struct)."""

    def test_to_struct_validates_dict(self):
        d = {"meta": {"left_context_size": 25, "finished": torch.tensor(False)}}
        s = to_struct(d)
        assert s.meta.left_context_size == 25

    def test_to_struct_rejects_legacy_flat_top_level(self):
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            to_struct({"code_predictor_codes": torch.zeros(3, 8)})

    def test_to_struct_rejects_legacy_flat_meta_field(self):
        # `left_context_size` at top level (legacy) instead of under `meta`
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            to_struct({"left_context_size": 25})

    def test_to_struct_rejects_typo_in_subkey(self):
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            to_struct({"meta": {"finisheed": True}})

    def test_to_struct_rejects_wrong_type(self):
        with pytest.raises(msgspec.ValidationError, match="Expected"):
            to_struct({"meta": {"left_context_size": "not_an_int"}})

    def test_round_trip_dict_struct_dict(self):
        original = {
            "meta": {"left_context_size": 7, "finished": torch.tensor(True)},
            "codes": {"audio": torch.zeros(2, 8)},
            "hidden_states": {"output": torch.zeros(4, 16)},
        }
        s = to_struct(original)
        d = to_dict(s)
        assert sorted(d.keys()) == sorted(original.keys())
        for top, sub in original.items():
            assert sorted(d[top].keys()) == sorted(sub.keys())

    def test_to_dict_drops_unset_fields(self):
        s = OmniPayloadStruct(meta=MetaStruct(left_context_size=10))
        d = to_dict(s)
        assert d == {"meta": {"left_context_size": 10}}

    def test_struct_with_all_categories(self):
        d = {
            "hidden_states": {"output": torch.zeros(1)},
            "embed": {"prefill": torch.zeros(1), "tts_bos": torch.zeros(1)},
            "ids": {"all": [1, 2], "prompt": [1]},
            "codes": {"audio": torch.zeros(1)},
            "meta": {"left_context_size": 3, "num_processed_tokens": 7},
        }
        s = to_struct(d)
        assert isinstance(s.hidden_states, HiddenStatesStruct)
        assert isinstance(s.embed, EmbeddingsStruct)
        assert isinstance(s.ids, IdsStruct)
        assert isinstance(s.codes, CodesStruct)
        assert isinstance(s.meta, MetaStruct)
        assert s.ids.all == [1, 2]
        assert s.meta.num_processed_tokens == 7


class TestValidatePayload:
    def test_raises_on_unknown_top_level(self):
        from vllm_omni.data_entry_keys import validate_payload

        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            validate_payload({"code_predictor_codes": torch.zeros(3, 8)}, context="test_boundary")

    def test_raises_on_unknown_sub_key(self):
        from vllm_omni.data_entry_keys import validate_payload

        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            validate_payload({"meta": {"finisheed": True}})

    def test_none_is_ok(self):
        from vllm_omni.data_entry_keys import validate_payload

        validate_payload(None)  # should not raise

    def test_valid_payload_passes(self):
        from vllm_omni.data_entry_keys import validate_payload

        validate_payload({"meta": {"left_context_size": 5}})

    def test_context_in_error_message(self):
        from vllm_omni.data_entry_keys import validate_payload

        with pytest.raises(msgspec.ValidationError, match="my_call_site"):
            validate_payload({"bad": 1}, context="my_call_site")


class TestNativeMsgspecEncoding:
    """Phase 6 scaffolding: native msgspec encode/decode for OmniPayloadStruct."""

    def test_encode_decode_round_trip_tensor(self):
        from vllm_omni.data_entry_keys import decode_payload, encode_payload

        original = OmniPayloadStruct(
            codes=CodesStruct(audio=torch.tensor([1, 2, 3, 4], dtype=torch.long)),
            meta=MetaStruct(left_context_size=5, finished=torch.tensor(True)),
        )
        wire = encode_payload(original)
        assert isinstance(wire, bytes)
        restored = decode_payload(wire)
        assert isinstance(restored, OmniPayloadStruct)
        assert torch.equal(restored.codes.audio, original.codes.audio)
        assert restored.meta.left_context_size == 5
        assert bool(restored.meta.finished.item()) is True

    def test_encode_decode_round_trip_dtypes(self):
        from vllm_omni.data_entry_keys import decode_payload, encode_payload

        # bfloat16 excluded: numpy() doesn't support it; callers must cast before serializing.
        for dtype in (torch.float32, torch.float16, torch.int64, torch.bool):
            original = OmniPayloadStruct(codes=CodesStruct(audio=torch.tensor([1, 0, 1], dtype=dtype)))
            restored = decode_payload(encode_payload(original))
            assert restored.codes.audio.dtype == dtype, f"dtype mismatch for {dtype}"

    def test_encode_decode_preserves_shape(self):
        from vllm_omni.data_entry_keys import decode_payload, encode_payload

        t = torch.randn(3, 4, 5)
        original = OmniPayloadStruct(hidden_states=HiddenStatesStruct(output=t))
        restored = decode_payload(encode_payload(original))
        assert restored.hidden_states.output.shape == (3, 4, 5)
        assert torch.allclose(restored.hidden_states.output, t)

    def test_encode_decode_speaker_language(self):
        from vllm_omni.data_entry_keys import decode_payload, encode_payload

        original = OmniPayloadStruct(speaker="ethan", language="en")
        restored = decode_payload(encode_payload(original))
        assert restored.speaker == "ethan"
        assert restored.language == "en"

    def test_decode_rejects_unknown_field(self):
        from vllm_omni.data_entry_keys import _OMNI_PAYLOAD_ENCODER, decode_payload

        # Manually craft msgpack with unknown top-level field
        bad_dict = {"code_predictor_codes": [1, 2, 3]}
        wire = _OMNI_PAYLOAD_ENCODER.encode(bad_dict)
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            decode_payload(wire)

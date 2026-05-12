"""Behavior tests for ``input_*`` extractors that read from ``OmniInputStruct``."""

from types import SimpleNamespace

from vllm_omni.data_entry_keys import OmniInputStruct, OmniPayloadStruct, serialize_payload
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    input_language_from_prompt,
    input_language_from_request,
    input_speaker_from_prompt,
    input_speaker_from_request,
)


def _build_request(input_struct: OmniInputStruct):
    """Round-trip a payload through the wire and wrap it as a request mock."""
    wire = serialize_payload(OmniPayloadStruct(input=input_struct))
    return SimpleNamespace(additional_information=wire)


def test_extractors_read_from_wire_round_trip():
    request = _build_request(OmniInputStruct(speaker=["Alice"], language=["en"]))
    assert input_speaker_from_request(request) == "alice"
    assert input_language_from_request(request) == ["en"]


def test_extractors_return_none_when_input_missing():
    request = _build_request(OmniInputStruct(instruction="be polite"))
    assert input_speaker_from_request(request) is None
    assert input_language_from_request(request) is None


def test_prompt_extractors_read_struct_directly():
    prompt = {"additional_information": OmniPayloadStruct(input=OmniInputStruct(speaker="bob"))}
    assert input_speaker_from_prompt(prompt) == ["bob"]
    assert input_language_from_prompt(prompt) is None

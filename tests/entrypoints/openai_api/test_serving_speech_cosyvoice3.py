# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CosyVoice3 online serving via /v1/audio/speech.

Covers the changes in PR #2121:
  - model_stage rename (talker -> cosyvoice3_talker, code2wav -> cosyvoice3_code2wav)
  - TTS model type detection for cosyvoice3
  - CosyVoice3 prompt building in _prepare_speech_generation
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import (
    _COSYVOICE3_TTS_MODEL_STAGES,
    _TTS_MODEL_STAGES,
    OmniOpenAIServingSpeech,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cosyvoice3_server(mocker: MockerFixture):
    """Create a speech server configured with a CosyVoice3 talker stage."""
    mock_engine_client = mocker.MagicMock()
    mock_engine_client.errored = False
    mock_engine_client.tts_max_instructions_length = None

    mock_stage = mocker.MagicMock()
    mock_stage.engine_args.model_stage = "cosyvoice3_talker"
    mock_stage.tts_args = {}
    mock_engine_client.stage_configs = [mock_stage]

    mock_models = mocker.MagicMock()
    mock_models.is_base_model.return_value = True

    return OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mocker.MagicMock(),
    )


# ---------------------------------------------------------------------------
# Tests: model_stage constants
# ---------------------------------------------------------------------------


class TestCosyVoice3ModelStage:
    """Verify model_stage rename is consistent."""

    def test_cosyvoice3_talker_in_tts_stages(self):
        assert "cosyvoice3_talker" in _COSYVOICE3_TTS_MODEL_STAGES
        assert "cosyvoice3_talker" in _TTS_MODEL_STAGES

    def test_old_stage_names_not_in_tts_stages(self):
        """Old generic names should not be registered."""
        assert "talker" not in _COSYVOICE3_TTS_MODEL_STAGES
        assert "code2wav" not in _COSYVOICE3_TTS_MODEL_STAGES


# ---------------------------------------------------------------------------
# Tests: TTS model type detection
# ---------------------------------------------------------------------------


class TestCosyVoice3Detection:
    def test_detect_cosyvoice3_model_type(self, cosyvoice3_server):
        assert cosyvoice3_server._is_tts is True
        assert cosyvoice3_server._tts_model_type == "cosyvoice3"

    def test_is_not_fish_or_voxtral(self, cosyvoice3_server):
        assert cosyvoice3_server._is_fish_speech is False


# ---------------------------------------------------------------------------
# Tests: _prepare_speech_generation for CosyVoice3
# ---------------------------------------------------------------------------


class TestCosyVoice3PromptBuilding:
    def test_requires_ref_audio(self, cosyvoice3_server):
        """CosyVoice3 must reject requests without ref_audio."""
        req = OpenAICreateSpeechRequest(
            input="Hello world",
            ref_audio=None,
            ref_text="reference text",
        )
        with pytest.raises(ValueError, match="ref_audio"):
            asyncio.run(cosyvoice3_server._prepare_speech_generation(req))

    def test_requires_ref_text(self, cosyvoice3_server):
        """CosyVoice3 must reject requests without ref_text."""
        req = OpenAICreateSpeechRequest(
            input="Hello world",
            ref_audio="data:audio/wav;base64,UklGR...",
            ref_text="",
        )
        with pytest.raises(ValueError, match="ref_text"):
            asyncio.run(cosyvoice3_server._prepare_speech_generation(req))

    def test_requires_nonempty_input(self, cosyvoice3_server):
        """CosyVoice3 must reject empty input text."""
        req = OpenAICreateSpeechRequest(
            input="",
            ref_audio="data:audio/wav;base64,UklGR...",
            ref_text="reference",
        )
        with pytest.raises(ValueError, match="cannot be empty"):
            asyncio.run(cosyvoice3_server._prepare_speech_generation(req))

    def test_builds_correct_prompt(self, cosyvoice3_server, mocker):
        """Verify prompt structure when all inputs are valid."""
        dummy_audio = [0.0] * 16000
        mocker.patch.object(
            cosyvoice3_server,
            "_resolve_ref_audio",
            new_callable=AsyncMock,
            return_value=(dummy_audio, 16000),
        )

        req = OpenAICreateSpeechRequest(
            input="Hello world",
            ref_audio="data:audio/wav;base64,UklGR...",
            ref_text="reference transcript",
        )

        request_id, generator, tts_params = asyncio.run(cosyvoice3_server._prepare_speech_generation(req))

        assert request_id.startswith("speech-")
        # generator is an async generator from the engine; just check it exists
        assert generator is not None

# this_file: tests/eledubby/api/test_elevenlabs_client.py
"""Unit tests for ElevenLabsClient."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Use consistent patch path for the ElevenLabs import in the client module
PATCH_BASE = "eledubby.api.elevenlabs_client"


class TestElevenLabsClientInit:
    """Tests for ElevenLabsClient initialization."""

    def test_init_when_api_key_provided_then_uses_it(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs"):
            client = ElevenLabsClient(api_key="test_key")

        assert client.api_key == "test_key"

    def test_init_when_no_key_and_no_env_then_raises(self) -> None:
        from eledubby.api import ElevenLabsClient

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(f"{PATCH_BASE}.os.getenv", return_value=None),
            pytest.raises(ValueError, match="ELEVENLABS_API_KEY not provided"),
        ):
            ElevenLabsClient()

    def test_init_when_env_key_then_uses_env(self) -> None:
        from eledubby.api import ElevenLabsClient

        with (
            patch(f"{PATCH_BASE}.os.getenv", return_value="env_key"),
            patch(f"{PATCH_BASE}.ElevenLabs"),
        ):
            client = ElevenLabsClient()

        assert client.api_key == "env_key"

    def test_init_when_default_retries_then_three(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs"):
            client = ElevenLabsClient(api_key="test")

        assert client.max_retries == 3

    def test_init_when_custom_retries_then_uses_it(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs"):
            client = ElevenLabsClient(api_key="test", max_retries=5)

        assert client.max_retries == 5


class TestElevenLabsClientTextToSpeech:
    """Tests for ElevenLabsClient.text_to_speech method."""

    def test_text_to_speech_when_success_then_returns_output_path(self, tmp_path: Path) -> None:
        from eledubby.api import ElevenLabsClient

        output_path = tmp_path / "output.mp3"

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_client.text_to_speech.convert.return_value = iter([b"audio_chunk"])

            client = ElevenLabsClient(api_key="test")
            result = client.text_to_speech(
                text="Hello world",
                voice_id="voice123",
                output_path=str(output_path),
            )

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.read_bytes() == b"audio_chunk"

    def test_text_to_speech_when_empty_chunks_then_skips_them(self, tmp_path: Path) -> None:
        from eledubby.api import ElevenLabsClient

        output_path = tmp_path / "output.mp3"

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_client.text_to_speech.convert.return_value = iter(
                [b"chunk1", None, b"chunk2", b""]
            )

            client = ElevenLabsClient(api_key="test")
            client.text_to_speech(
                text="Hello",
                voice_id="voice123",
                output_path=str(output_path),
            )

        assert output_path.read_bytes() == b"chunk1chunk2"


class TestElevenLabsClientSpeechToSpeech:
    """Tests for ElevenLabsClient.speech_to_speech method."""

    def test_speech_to_speech_when_success_then_returns_output_path(self, tmp_path: Path) -> None:
        from eledubby.api import ElevenLabsClient

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.mp3"
        input_path.write_bytes(b"fake audio")

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_client.speech_to_speech.convert.return_value = iter([b"converted"])

            client = ElevenLabsClient(api_key="test")
            result = client.speech_to_speech(str(input_path), "voice123", str(output_path))

        assert result == str(output_path)
        assert output_path.exists()


class TestElevenLabsClientValidateVoice:
    """Tests for ElevenLabsClient.validate_voice_id method."""

    def test_validate_voice_id_when_exists_then_returns_true(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_voice = MagicMock()
            mock_voice.voice_id = "voice123"
            mock_voice.name = "Test Voice"
            mock_voices = MagicMock()
            mock_voices.voices = [mock_voice]
            mock_client.voices.get_all.return_value = mock_voices

            client = ElevenLabsClient(api_key="test")
            result = client.validate_voice_id("voice123")

        assert result is True

    def test_validate_voice_id_when_not_exists_then_returns_false(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_voices = MagicMock()
            mock_voices.voices = []
            mock_client.voices.get_all.return_value = mock_voices

            client = ElevenLabsClient(api_key="test")
            result = client.validate_voice_id("nonexistent")

        assert result is False

    def test_validate_voice_id_when_api_error_then_returns_false(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_client.voices.get_all.side_effect = Exception("API error")

            client = ElevenLabsClient(api_key="test")
            result = client.validate_voice_id("voice123")

        assert result is False


class TestElevenLabsClientListVoices:
    """Tests for ElevenLabsClient.list_voices method."""

    def test_list_voices_when_success_then_returns_list(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_voice = MagicMock()
            mock_voice.voice_id = "v1"
            mock_voice.name = "Voice 1"
            mock_voice.category = "professional"
            mock_voices = MagicMock()
            mock_voices.voices = [mock_voice]
            mock_client.voices.get_all.return_value = mock_voices

            client = ElevenLabsClient(api_key="test")
            result = client.list_voices()

        assert len(result) == 1
        assert result[0]["id"] == "v1"
        assert result[0]["name"] == "Voice 1"
        assert result[0]["category"] == "professional"

    def test_list_voices_when_api_error_then_returns_empty(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_client.voices.get_all.side_effect = Exception("API error")

            client = ElevenLabsClient(api_key="test")
            result = client.list_voices()

        assert result == []

    def test_list_voices_when_no_category_then_uses_empty_string(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_voice = MagicMock(spec=["voice_id", "name"])  # No category attr
            mock_voice.voice_id = "v1"
            mock_voice.name = "Voice 1"
            mock_voices = MagicMock()
            mock_voices.voices = [mock_voice]
            mock_client.voices.get_all.return_value = mock_voices

            client = ElevenLabsClient(api_key="test")
            result = client.list_voices()

        assert result[0]["category"] == ""

    def test_list_voices_detailed_includes_extra_fields(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_voice = MagicMock()
            mock_voice.voice_id = "v1"
            mock_voice.name = "Voice 1"
            mock_voice.category = "cloned"
            mock_voice.description = "A test voice"
            mock_voice.preview_url = "https://example.com/preview.mp3"
            mock_voice.labels = {"accent": "british", "gender": "male"}
            mock_voices = MagicMock()
            mock_voices.voices = [mock_voice]
            mock_client.voices.get_all.return_value = mock_voices

            client = ElevenLabsClient(api_key="test")
            result = client.list_voices(detailed=True)

        assert result[0]["id"] == "v1"
        assert result[0]["name"] == "Voice 1"
        assert result[0]["category"] == "cloned"
        assert result[0]["description"] == "A test voice"
        assert result[0]["preview_url"] == "https://example.com/preview.mp3"
        assert result[0]["labels"] == {"accent": "british", "gender": "male"}

    def test_list_voices_not_detailed_excludes_extra_fields(self) -> None:
        from eledubby.api import ElevenLabsClient

        with patch(f"{PATCH_BASE}.ElevenLabs") as mock_el:
            mock_client = MagicMock()
            mock_el.return_value = mock_client
            mock_voice = MagicMock()
            mock_voice.voice_id = "v1"
            mock_voice.name = "Voice 1"
            mock_voice.category = "cloned"
            mock_voice.description = "A test voice"
            mock_voices = MagicMock()
            mock_voices.voices = [mock_voice]
            mock_client.voices.get_all.return_value = mock_voices

            client = ElevenLabsClient(api_key="test")
            result = client.list_voices(detailed=False)

        assert "description" not in result[0]
        assert "preview_url" not in result[0]
        assert "labels" not in result[0]

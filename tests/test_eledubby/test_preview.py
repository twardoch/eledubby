# this_file: tests/test_eledubby/test_preview.py
"""Tests for preview command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPreviewCommand:
    """Tests for the preview CLI command."""

    def test_preview_no_voice_lists_voices(self, capsys) -> None:
        """Test preview without voice argument lists available voices."""
        with patch("eledubby.eledubby.ElevenLabsClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_voices.return_value = [
                {"id": "voice1", "name": "Voice One", "category": "premade"},
                {"id": "voice2", "name": "Voice Two", "category": ""},
            ]

            from eledubby.eledubby import preview

            preview(voice=None)

            captured = capsys.readouterr()
            assert "Available voices" in captured.out
            assert "voice1" in captured.out
            assert "Voice One" in captured.out

    def test_preview_generates_audio(self, tmp_path: Path) -> None:
        """Test preview generates audio file."""
        output_path = tmp_path / "test_preview.mp3"

        with patch("eledubby.eledubby.ElevenLabsClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.validate_voice_id.return_value = True
            mock_client.text_to_speech.return_value = str(output_path)

            from eledubby.eledubby import preview

            preview(voice="test_voice", output=str(output_path))

            mock_client.text_to_speech.assert_called_once()
            call_kwargs = mock_client.text_to_speech.call_args[1]
            assert call_kwargs["voice_id"] == "test_voice"
            assert call_kwargs["output_path"] == str(output_path)

    def test_preview_with_custom_text(self, tmp_path: Path) -> None:
        """Test preview with custom text."""
        output_path = tmp_path / "test_preview.mp3"
        custom_text = "Custom preview text for testing"

        with patch("eledubby.eledubby.ElevenLabsClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.validate_voice_id.return_value = True
            mock_client.text_to_speech.return_value = str(output_path)

            from eledubby.eledubby import preview

            preview(voice="test_voice", text=custom_text, output=str(output_path))

            call_kwargs = mock_client.text_to_speech.call_args[1]
            assert call_kwargs["text"] == custom_text

    def test_preview_invalid_voice_exits(self, capsys) -> None:
        """Test preview with invalid voice exits with error."""
        with patch("eledubby.eledubby.ElevenLabsClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.validate_voice_id.return_value = False

            from eledubby.eledubby import preview

            with pytest.raises(SystemExit) as exc_info:
                preview(voice="invalid_voice")

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "not found" in captured.out

    def test_preview_no_api_key_exits(self, capsys) -> None:
        """Test preview without API key exits with error."""
        with patch("eledubby.eledubby.ElevenLabsClient") as MockClient:
            MockClient.side_effect = ValueError("ELEVENLABS_API_KEY not provided")

            from eledubby.eledubby import preview

            with pytest.raises(SystemExit) as exc_info:
                preview(voice="test_voice")

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Missing ElevenLabs API key" in captured.out

    def test_preview_with_play_calls_ffplay(self, tmp_path: Path) -> None:
        """Test preview with play option calls ffplay."""
        output_path = tmp_path / "test_preview.mp3"

        with (
            patch("eledubby.eledubby.ElevenLabsClient") as MockClient,
            patch("eledubby.eledubby.subprocess.run") as mock_run,
        ):
            mock_client = MockClient.return_value
            mock_client.validate_voice_id.return_value = True
            mock_client.text_to_speech.return_value = str(output_path)
            mock_run.return_value = MagicMock(returncode=0)

            from eledubby.eledubby import preview

            preview(voice="test_voice", output=str(output_path), play=True)

            # Check ffplay was called
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "ffplay" in cmd
            assert str(output_path) in cmd

    def test_preview_default_output_path(self) -> None:
        """Test preview generates default output filename."""
        with (
            patch("eledubby.eledubby.ElevenLabsClient") as MockClient,
            patch("eledubby.eledubby.Path.cwd") as mock_cwd,
        ):
            mock_client = MockClient.return_value
            mock_client.validate_voice_id.return_value = True
            mock_client.text_to_speech.return_value = "/tmp/preview.mp3"
            mock_cwd.return_value = Path("/tmp")

            from eledubby.eledubby import preview

            preview(voice="my_voice_id")

            call_kwargs = mock_client.text_to_speech.call_args[1]
            assert "preview_my_voice_id.mp3" in call_kwargs["output_path"]

# this_file: tests/test_eledubby/test_voices.py
"""Tests for voices CLI command."""

import json
from unittest.mock import MagicMock, patch

import pytest

from eledubby.eledubby import voices


class TestVoicesCommand:
    """Tests for the voices CLI command."""

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_csv_output(self, mock_client_class: MagicMock, capsys) -> None:
        """Default output is CSV format."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = [
            {"id": "v1", "name": "Voice 1", "category": "cloned"},
            {"id": "v2", "name": "Voice 2", "category": "premade"},
        ]
        mock_client_class.return_value = mock_client

        voices(api_key="test_key")

        captured = capsys.readouterr()
        # CSV uses \r\n line endings, normalize them
        output = captured.out.replace("\r\n", "\n").strip()
        lines = output.split("\n")
        assert lines[0] == "id,name,category"
        assert lines[1] == "v1,Voice 1,cloned"
        assert lines[2] == "v2,Voice 2,premade"

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_json_output(self, mock_client_class: MagicMock, capsys) -> None:
        """With --json flag, output is JSON format."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = [
            {"id": "v1", "name": "Voice 1", "category": "cloned"},
        ]
        mock_client_class.return_value = mock_client

        voices(api_key="test_key", json=True)

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert len(result) == 1
        assert result[0]["id"] == "v1"
        assert result[0]["name"] == "Voice 1"

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_detailed_csv_includes_extra_columns(
        self, mock_client_class: MagicMock, capsys
    ) -> None:
        """Detailed CSV includes description, preview_url, labels columns."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = [
            {
                "id": "v1",
                "name": "Voice 1",
                "category": "cloned",
                "description": "Test desc",
                "preview_url": "https://example.com/preview.mp3",
                "labels": {"accent": "british"},
            },
        ]
        mock_client_class.return_value = mock_client

        voices(api_key="test_key", detailed=True)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "description" in lines[0]
        assert "preview_url" in lines[0]
        assert "labels" in lines[0]
        assert "Test desc" in lines[1]
        assert "accent=british" in lines[1]

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_detailed_json_includes_extra_fields(
        self, mock_client_class: MagicMock, capsys
    ) -> None:
        """Detailed JSON includes extra fields."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = [
            {
                "id": "v1",
                "name": "Voice 1",
                "category": "cloned",
                "description": "Test desc",
                "preview_url": "https://example.com/preview.mp3",
                "labels": {"accent": "british", "gender": "male"},
            },
        ]
        mock_client_class.return_value = mock_client

        voices(api_key="test_key", json=True, detailed=True)

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result[0]["description"] == "Test desc"
        assert result[0]["labels"]["accent"] == "british"

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_passes_detailed_to_client(self, mock_client_class: MagicMock) -> None:
        """Detailed flag is passed to list_voices."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = []
        mock_client_class.return_value = mock_client

        with pytest.raises(SystemExit):  # No voices found exits with 0
            voices(api_key="test_key", detailed=True)

        mock_client.list_voices.assert_called_once_with(detailed=True)

    def test_voices_exits_when_no_api_key(self) -> None:
        """Exits with error when no API key provided."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("eledubby.eledubby.ElevenLabsClient") as mock_client_class,
        ):
            mock_client_class.side_effect = ValueError("ELEVENLABS_API_KEY not provided")

            with pytest.raises(SystemExit) as exc_info:
                voices()

            assert exc_info.value.code == 1

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_exits_zero_when_no_voices_found(self, mock_client_class: MagicMock) -> None:
        """Exits with code 0 when no voices found (not an error)."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = []
        mock_client_class.return_value = mock_client

        with pytest.raises(SystemExit) as exc_info:
            voices(api_key="test_key")

        assert exc_info.value.code == 0

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_voices_csv_escapes_special_characters(
        self, mock_client_class: MagicMock, capsys
    ) -> None:
        """CSV properly escapes special characters in values."""
        mock_client = MagicMock()
        mock_client.list_voices.return_value = [
            {"id": "v1", "name": 'Voice "with" quotes', "category": "cloned"},
        ]
        mock_client_class.return_value = mock_client

        voices(api_key="test_key")

        captured = capsys.readouterr()
        # CSV escapes quotes by doubling them
        assert '"Voice ""with"" quotes"' in captured.out

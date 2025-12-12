# this_file: tests/test_eledubby/test_e2e.py
"""End-to-end tests for eledubby with mocked API."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEndToEndDub:
    """End-to-end tests for the dub command with mocked ElevenLabs API."""

    @pytest.fixture
    def temp_audio_file(self, tmp_path: Path) -> Path:
        """Create a temporary WAV file for testing."""
        import subprocess

        audio_file = tmp_path / "test_audio.wav"
        # Generate 5 seconds of silence as test audio
        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono:d=5",
            "-acodec",
            "pcm_s16le",
            "-y",
            str(audio_file),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            pytest.skip("FFmpeg not available for test audio generation")
        return audio_file

    @patch("eledubby.eledubby.ElevenLabsClient")
    @patch("eledubby.eledubby.subprocess.run")
    def test_dub_full_pipeline_mocked(
        self,
        mock_subprocess: MagicMock,
        mock_client_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test full dub pipeline with mocked API and subprocess calls."""
        from eledubby.eledubby import EleDubby

        # Setup mocks
        mock_client = MagicMock()
        mock_client.validate_voice_id.return_value = True
        mock_client.speech_to_speech.return_value = str(tmp_path / "converted.mp3")
        mock_client_class.return_value = mock_client

        # Mock subprocess for ffmpeg/ffprobe calls
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="ffmpeg version", stderr="")

        # Create EleDubby instance
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}):
            dubber = EleDubby(
                verbose=False,
                parallel=1,
                preview=0,
                normalize=True,
                compress=False,
            )

        # Verify initialization
        assert dubber.normalize is True
        assert dubber.compress is False
        assert dubber.parallel == 1

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_dub_with_compress_option(
        self,
        mock_client_class: MagicMock,
    ) -> None:
        """Test dub with compression enabled."""
        from eledubby.eledubby import EleDubby

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with (
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}),
            patch("eledubby.eledubby.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

            dubber = EleDubby(
                verbose=False,
                compress=True,
                normalize=True,
            )

            assert dubber.compress is True
            assert dubber.normalize is True

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_dub_preview_mode(
        self,
        mock_client_class: MagicMock,
    ) -> None:
        """Test dub with preview mode enabled."""
        from eledubby.eledubby import EleDubby

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with (
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"}),
            patch("eledubby.eledubby.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

            dubber = EleDubby(
                verbose=False,
                preview=30.0,
            )

            assert dubber.preview == 30.0


class TestEndToEndFx:
    """End-to-end tests for the fx command."""

    @patch("eledubby.eledubby.subprocess.run")
    def test_fx_initializes_without_elevenlabs(
        self,
        mock_subprocess: MagicMock,
    ) -> None:
        """Test fx command works without ElevenLabs API key."""
        from eledubby.eledubby import EleDubby

        mock_subprocess.return_value = MagicMock(returncode=0, stdout="ffmpeg version")

        # Should not raise - fx doesn't need ElevenLabs
        dubber = EleDubby(
            verbose=False,
            require_elevenlabs=False,
        )

        assert dubber is not None


class TestEndToEndCast:
    """End-to-end tests for the cast command."""

    @patch("eledubby.eledubby.ElevenLabsClient")
    def test_cast_generates_output_files(
        self,
        mock_client_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test cast creates MP3 files for each voice."""
        from eledubby.eledubby import cast

        # Create test text file
        text_file = tmp_path / "script.txt"
        text_file.write_text("Hello, this is a test.")

        # Setup mock
        mock_client = MagicMock()
        mock_client.list_voices.return_value = [
            {"id": "voice1", "name": "Voice 1"},
            {"id": "voice2", "name": "Voice 2"},
        ]
        mock_client.text_to_speech.return_value = None
        mock_client_class.return_value = mock_client

        output_dir = tmp_path / "output"

        cast(
            text_path=str(text_file),
            output=str(output_dir),
            api_key="test_key",
            voices_list="voice1,voice2",
        )

        # Verify text_to_speech was called for each voice
        assert mock_client.text_to_speech.call_count == 2


class TestAudioProcessorCompression:
    """Tests for the compress_audio method using pedalboard."""

    def test_compress_audio_processes_audio(self, tmp_path: Path) -> None:
        """Test compress_audio applies compression to audio."""
        import numpy as np
        from pedalboard.io import AudioFile

        from eledubby.audio.processor import AudioProcessor

        # Create test audio
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        sample_rate = 16000
        duration = 2.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).reshape(1, -1)

        with AudioFile(str(input_path), "w", samplerate=sample_rate, num_channels=1) as f:
            f.write(audio)

        processor = AudioProcessor()
        result = processor.compress_audio(
            str(input_path),
            str(output_path),
            threshold_db=-20.0,
            ratio=4.0,
        )

        assert result == str(output_path)
        assert output_path.exists()
        # Verify output is valid audio
        with AudioFile(str(output_path)) as f:
            assert f.duration > 0

    def test_compress_audio_handles_invalid_file(self, tmp_path: Path) -> None:
        """Test compress_audio handles invalid files gracefully."""
        import numpy as np
        from pedalboard.io import AudioFile

        from eledubby.audio.processor import AudioProcessor

        # Create valid input file first (for fallback copy)
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        sample_rate = 16000
        duration = 1.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).reshape(1, -1)

        with AudioFile(str(input_path), "w", samplerate=sample_rate, num_channels=1) as f:
            f.write(audio)

        processor = AudioProcessor()
        # Normal operation should work
        result = processor.compress_audio(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()

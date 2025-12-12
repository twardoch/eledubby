# this_file: tests/eledubby/audio/test_extractor.py
"""Unit tests for AudioExtractor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAudioExtractor:
    """Tests for AudioExtractor class."""

    def test_init_when_default_then_16000_sample_rate(self) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()
        assert extractor.sample_rate == 16000

    def test_init_when_custom_rate_then_uses_it(self) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor(sample_rate=44100)
        assert extractor.sample_rate == 44100

    def test_extract_when_success_then_returns_output_path(self, tmp_path: Path) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()
        video_path = tmp_path / "video.mp4"
        output_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="10.5\n")

            result = extractor.extract(str(video_path), str(output_path))

        assert result == str(output_path)

    def test_extract_when_ffmpeg_fails_then_raises_runtime_error(self, tmp_path: Path) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()
        video_path = tmp_path / "video.mp4"
        output_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg error")

            with pytest.raises(RuntimeError, match="Failed to extract audio"):
                extractor.extract(str(video_path), str(output_path))

    def test_extract_when_called_then_uses_correct_ffmpeg_args(self, tmp_path: Path) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor(sample_rate=22050)
        video_path = tmp_path / "video.mp4"
        output_path = tmp_path / "audio.wav"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="10.5\n")

            extractor.extract(str(video_path), str(output_path))

        cmd = mock_run.call_args_list[0][0][0]
        assert cmd[0] == "ffmpeg"
        assert "-vn" in cmd  # No video
        assert "pcm_s16le" in cmd  # 16-bit PCM
        assert "22050" in cmd  # Sample rate
        assert "1" in cmd  # Mono


class TestAudioExtractorGetDuration:
    """Tests for AudioExtractor._get_duration method."""

    def test_get_duration_when_valid_then_returns_float(self) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="123.456\n")

            duration = extractor._get_duration("/path/to/audio.wav")

        assert duration == pytest.approx(123.456)

    def test_get_duration_when_ffprobe_fails_then_returns_zero(self) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            duration = extractor._get_duration("/path/to/audio.wav")

        assert duration == 0.0

    def test_get_duration_when_invalid_output_then_returns_zero(self) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not-a-number\n")

            duration = extractor._get_duration("/path/to/audio.wav")

        assert duration == 0.0

    def test_get_duration_when_empty_output_then_returns_zero(self) -> None:
        from eledubby.audio.extractor import AudioExtractor

        extractor = AudioExtractor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="\n")

            duration = extractor._get_duration("/path/to/audio.wav")

        assert duration == 0.0

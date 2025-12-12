# this_file: tests/eledubby/audio/test_processor.py
"""Unit tests for AudioProcessor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAudioProcessorMeasureDuration:
    """Tests for AudioProcessor.measure_duration method."""

    def test_measure_duration_when_valid_audio_then_returns_duration(self) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="12.345\n")

            duration = processor.measure_duration("/path/to/audio.wav")

        assert duration == pytest.approx(12.345)

    def test_measure_duration_when_ffprobe_fails_then_returns_zero(self) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            duration = processor.measure_duration("/path/to/audio.wav")

        assert duration == 0.0

    def test_measure_duration_when_invalid_output_then_returns_zero(self) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not a number\n")

            duration = processor.measure_duration("/path/to/audio.wav")

        assert duration == 0.0


class TestAudioProcessorAdjustDuration:
    """Tests for AudioProcessor.adjust_duration method."""

    def test_adjust_duration_when_within_tolerance_then_copies_file(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch.object(processor, "measure_duration", return_value=10.0):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = processor.adjust_duration(str(input_path), 10.03, str(output_path))

        assert result == str(output_path)
        # Should call cp, not ffmpeg
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "cp"

    def test_adjust_duration_when_needs_padding_then_pads(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch.object(processor, "measure_duration", return_value=8.0):
            with patch.object(processor, "_pad_audio", return_value=str(output_path)) as mock_pad:
                result = processor.adjust_duration(str(input_path), 10.0, str(output_path))

        mock_pad.assert_called_once_with(str(input_path), pytest.approx(2.0), str(output_path))
        assert result == str(output_path)

    def test_adjust_duration_when_needs_trimming_then_trims(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch.object(processor, "measure_duration", return_value=12.0):
            with patch.object(processor, "_trim_audio", return_value=str(output_path)) as mock_trim:
                result = processor.adjust_duration(str(input_path), 10.0, str(output_path))

        mock_trim.assert_called_once_with(str(input_path), 10.0, str(output_path))
        assert result == str(output_path)


class TestAudioProcessorPadAudio:
    """Tests for AudioProcessor._pad_audio method."""

    def test_pad_audio_when_called_then_generates_silence_and_concatenates(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("os.remove"):  # Prevent cleanup error
                result = processor._pad_audio(str(input_path), 2.0, str(output_path))

        assert result == str(output_path)
        # Should call ffmpeg twice: once for silence, once for concat
        assert mock_run.call_count == 2


class TestAudioProcessorTrimAudio:
    """Tests for AudioProcessor._trim_audio method."""

    def test_trim_audio_when_called_then_uses_ffmpeg_with_duration(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = processor._trim_audio(str(input_path), 8.5, str(output_path))

        assert result == str(output_path)
        cmd = mock_run.call_args[0][0]
        assert "-t" in cmd
        assert "8.5" in cmd


class TestAudioProcessorNormalizeAudio:
    """Tests for AudioProcessor.normalize_audio method."""

    def test_normalize_audio_when_success_then_returns_output_path(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = processor.normalize_audio(str(input_path), str(output_path))

        assert result == str(output_path)
        cmd = mock_run.call_args[0][0]
        assert "loudnorm" in str(cmd)

    def test_normalize_audio_when_fails_then_copies_original(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch("subprocess.run") as mock_run:
            # First call fails (normalize), second succeeds (copy)
            mock_run.side_effect = [
                MagicMock(returncode=1, stderr="error"),
                MagicMock(returncode=0),
            ]

            result = processor.normalize_audio(str(input_path), str(output_path))

        assert result == str(output_path)
        assert mock_run.call_count == 2

    def test_normalize_audio_when_custom_target_then_uses_it(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            processor.normalize_audio(str(input_path), str(output_path), target_db=-16.0)

        cmd = mock_run.call_args[0][0]
        assert "I=-16.0" in str(cmd)

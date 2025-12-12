# this_file: tests/test_eledubby/audio/test_processor.py
"""Unit tests for AudioProcessor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pedalboard.io import AudioFile


def create_test_wav(path: Path, duration: float = 1.0, sample_rate: int = 16000) -> None:
    """Create a test WAV file with a simple tone."""
    num_samples = int(duration * sample_rate)
    # Generate a simple sine wave
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    audio = audio.reshape(1, -1)  # Shape: (channels, samples)

    with AudioFile(str(path), "w", samplerate=sample_rate, num_channels=1) as f:
        f.write(audio)


class TestAudioProcessorMeasureDuration:
    """Tests for AudioProcessor.measure_duration method."""

    def test_measure_duration_when_valid_audio_then_returns_duration(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        audio_path = tmp_path / "test.wav"
        create_test_wav(audio_path, duration=5.0)

        duration = processor.measure_duration(str(audio_path))

        assert duration == pytest.approx(5.0, abs=0.01)

    def test_measure_duration_when_file_not_exists_then_returns_zero(self) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        duration = processor.measure_duration("/nonexistent/path/audio.wav")

        assert duration == 0.0


class TestAudioProcessorAdjustDuration:
    """Tests for AudioProcessor.adjust_duration method."""

    def test_adjust_duration_when_within_tolerance_then_copies_file(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=10.0)

        result = processor.adjust_duration(str(input_path), 10.03, str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        # Check duration is approximately the same
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(10.0, abs=0.1)

    def test_adjust_duration_when_needs_padding_then_pads(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=8.0)

        result = processor.adjust_duration(str(input_path), 10.0, str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(10.0, abs=0.01)

    def test_adjust_duration_when_needs_trimming_then_trims(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=12.0)

        result = processor.adjust_duration(str(input_path), 10.0, str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(10.0, abs=0.01)


class TestAudioProcessorPadAudio:
    """Tests for AudioProcessor._pad_audio method."""

    def test_pad_audio_when_called_then_adds_silence(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=5.0)

        result = processor._pad_audio(str(input_path), 2.0, str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(7.0, abs=0.01)


class TestAudioProcessorTrimAudio:
    """Tests for AudioProcessor._trim_audio method."""

    def test_trim_audio_when_called_then_trims_to_duration(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=10.0)

        result = processor._trim_audio(str(input_path), 8.5, str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(8.5, abs=0.01)


class TestAudioProcessorCompressAudio:
    """Tests for AudioProcessor.compress_audio method."""

    def test_compress_audio_when_success_then_returns_output_path(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.compress_audio(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        # Verify output is valid audio
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(2.0, abs=0.1)

    def test_compress_audio_when_custom_params_then_applies_them(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.compress_audio(
            str(input_path),
            str(output_path),
            threshold_db=-15.0,
            ratio=6.0,
            attack_ms=10.0,
            release_ms=100.0,
        )

        assert result == str(output_path)
        assert output_path.exists()


class TestAudioProcessorNormalizeAudio:
    """Tests for AudioProcessor.normalize_audio method."""

    def test_normalize_audio_when_success_then_returns_output_path(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.normalize_audio(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        # Verify output is valid audio
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(2.0, abs=0.1)

    def test_normalize_audio_when_custom_target_then_uses_it(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.normalize_audio(str(input_path), str(output_path), target_db=-16.0)

        assert result == str(output_path)
        assert output_path.exists()


class TestAudioProcessorReduceNoise:
    """Tests for AudioProcessor noise reduction methods."""

    def test_reduce_noise_when_success_then_returns_output_path(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.reduce_noise(str(input_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        # Verify output is valid audio
        with AudioFile(str(output_path)) as f:
            assert f.duration == pytest.approx(2.0, abs=0.1)

    def test_reduce_noise_when_custom_params_then_uses_them(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.reduce_noise(
            str(input_path), str(output_path), noise_reduction_db=18.0, noise_floor_db=-50.0
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_reduce_noise_advanced_gate_method(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.reduce_noise_advanced(
            str(input_path), str(output_path), method="gate", strength=0.7
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_reduce_noise_advanced_expander_method(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        create_test_wav(input_path, duration=2.0)

        result = processor.reduce_noise_advanced(
            str(input_path), str(output_path), method="expander", strength=0.5
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_reduce_noise_advanced_clamps_strength(self, tmp_path: Path) -> None:
        from eledubby.audio.processor import AudioProcessor

        processor = AudioProcessor()
        input_path = tmp_path / "input.wav"
        output_path1 = tmp_path / "output1.wav"
        output_path2 = tmp_path / "output2.wav"
        create_test_wav(input_path, duration=2.0)

        # Test with out-of-range strength values - should clamp and not error
        result1 = processor.reduce_noise_advanced(str(input_path), str(output_path1), strength=1.5)
        result2 = processor.reduce_noise_advanced(str(input_path), str(output_path2), strength=-0.5)

        assert result1 == str(output_path1)
        assert result2 == str(output_path2)
        assert output_path1.exists()
        assert output_path2.exists()

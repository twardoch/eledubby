# this_file: tests/test_eledubby/audio/test_segmenter.py
"""Unit tests for AudioSegmenter."""

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


class TestAudioSegmenter:
    """Tests for AudioSegmenter class."""

    def test_segment_when_single_segment_then_creates_one_file(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        create_test_wav(input_audio, duration=10.0)
        output_dir = tmp_path / "segments"

        paths = segmenter.segment(str(input_audio), [(0.0, 10.0)], str(output_dir))

        assert len(paths) == 1
        assert "segment_0000.wav" in paths[0]
        assert Path(paths[0]).exists()
        # Verify the segment has correct duration
        with AudioFile(paths[0]) as f:
            assert f.duration == pytest.approx(10.0, abs=0.01)

    def test_segment_when_multiple_segments_then_creates_multiple_files(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        create_test_wav(input_audio, duration=30.0)
        output_dir = tmp_path / "segments"
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

        paths = segmenter.segment(str(input_audio), segments, str(output_dir))

        assert len(paths) == 3
        assert "segment_0000.wav" in paths[0]
        assert "segment_0001.wav" in paths[1]
        assert "segment_0002.wav" in paths[2]
        for path in paths:
            assert Path(path).exists()
            with AudioFile(path) as f:
                assert f.duration == pytest.approx(10.0, abs=0.01)

    def test_segment_when_creates_output_directory(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        create_test_wav(input_audio, duration=10.0)
        output_dir = tmp_path / "new_dir" / "segments"

        segmenter.segment(str(input_audio), [(0.0, 10.0)], str(output_dir))

        assert output_dir.exists()

    def test_segment_when_extracts_middle_section(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        create_test_wav(input_audio, duration=30.0)
        output_dir = tmp_path / "segments"

        paths = segmenter.segment(str(input_audio), [(5.0, 15.0)], str(output_dir))

        assert len(paths) == 1
        with AudioFile(paths[0]) as f:
            # Should be 10 seconds (15.0 - 5.0)
            assert f.duration == pytest.approx(10.0, abs=0.01)


class TestAudioSegmenterConcatenate:
    """Tests for AudioSegmenter.concatenate method."""

    def test_concatenate_when_multiple_files_then_combines_them(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        seg1 = tmp_path / "seg1.wav"
        seg2 = tmp_path / "seg2.wav"
        create_test_wav(seg1, duration=5.0)
        create_test_wav(seg2, duration=5.0)
        output = tmp_path / "output.wav"

        result = segmenter.concatenate([str(seg1), str(seg2)], str(output))

        assert result == str(output)
        assert output.exists()
        with AudioFile(str(output)) as f:
            # Combined duration should be 10 seconds
            assert f.duration == pytest.approx(10.0, abs=0.01)

    def test_concatenate_when_single_file_then_still_works(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        seg1 = tmp_path / "seg1.wav"
        create_test_wav(seg1, duration=5.0)
        output = tmp_path / "output.wav"

        result = segmenter.concatenate([str(seg1)], str(output))

        assert result == str(output)
        assert output.exists()
        with AudioFile(str(output)) as f:
            assert f.duration == pytest.approx(5.0, abs=0.01)

    def test_concatenate_when_empty_list_then_raises_error(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        output = tmp_path / "output.wav"

        with pytest.raises(ValueError, match="No segments to concatenate"):
            segmenter.concatenate([], str(output))

    def test_concatenate_when_three_files_then_combines_all(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        seg1 = tmp_path / "seg1.wav"
        seg2 = tmp_path / "seg2.wav"
        seg3 = tmp_path / "seg3.wav"
        create_test_wav(seg1, duration=3.0)
        create_test_wav(seg2, duration=4.0)
        create_test_wav(seg3, duration=5.0)
        output = tmp_path / "output.wav"

        result = segmenter.concatenate([str(seg1), str(seg2), str(seg3)], str(output))

        assert result == str(output)
        with AudioFile(str(output)) as f:
            # Combined duration should be 12 seconds
            assert f.duration == pytest.approx(12.0, abs=0.01)

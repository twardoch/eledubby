# this_file: tests/eledubby/audio/test_segmenter.py
"""Unit tests for AudioSegmenter."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAudioSegmenter:
    """Tests for AudioSegmenter class."""

    def test_segment_when_single_segment_then_creates_one_file(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"fake wav data")
        output_dir = tmp_path / "segments"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            paths = segmenter.segment(
                str(input_audio), [(0.0, 10.0)], str(output_dir)
            )

        assert len(paths) == 1
        assert "segment_0000.wav" in paths[0]
        assert mock_run.call_count == 1

    def test_segment_when_multiple_segments_then_creates_multiple_files(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"fake wav data")
        output_dir = tmp_path / "segments"
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            paths = segmenter.segment(str(input_audio), segments, str(output_dir))

        assert len(paths) == 3
        assert "segment_0000.wav" in paths[0]
        assert "segment_0001.wav" in paths[1]
        assert "segment_0002.wav" in paths[2]
        assert mock_run.call_count == 3

    def test_segment_when_creates_output_directory(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"fake wav data")
        output_dir = tmp_path / "new_dir" / "segments"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            segmenter.segment(str(input_audio), [(0.0, 10.0)], str(output_dir))

        assert output_dir.exists()

    def test_segment_when_ffmpeg_fails_then_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"fake wav data")
        output_dir = tmp_path / "segments"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg error")

            with pytest.raises(RuntimeError, match="Segmentation failed"):
                segmenter.segment(str(input_audio), [(0.0, 10.0)], str(output_dir))

    def test_segment_when_calls_ffmpeg_with_correct_args(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        input_audio = tmp_path / "input.wav"
        input_audio.write_bytes(b"fake wav data")
        output_dir = tmp_path / "segments"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            segmenter.segment(str(input_audio), [(5.0, 15.0)], str(output_dir))

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "-ss" in cmd
        assert "5.0" in cmd  # start time
        assert "-t" in cmd
        assert "10.0" in cmd  # duration


class TestAudioSegmenterConcatenate:
    """Tests for AudioSegmenter.concatenate method."""

    def test_concatenate_when_multiple_files_then_creates_concat_file(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        seg1 = tmp_path / "seg1.wav"
        seg2 = tmp_path / "seg2.wav"
        seg1.write_bytes(b"fake1")
        seg2.write_bytes(b"fake2")
        output = tmp_path / "output.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = segmenter.concatenate([str(seg1), str(seg2)], str(output))

        assert result == str(output)
        # Concat file should be cleaned up
        assert not (tmp_path / "output.wav.txt").exists()

    def test_concatenate_when_ffmpeg_fails_then_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        seg1 = tmp_path / "seg1.wav"
        seg1.write_bytes(b"fake1")
        output = tmp_path / "output.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="concat error")

            with pytest.raises(RuntimeError, match="Concatenation failed"):
                segmenter.concatenate([str(seg1)], str(output))

    def test_concatenate_when_single_file_then_still_works(self, tmp_path: Path) -> None:
        from eledubby.audio.segmenter import AudioSegmenter

        segmenter = AudioSegmenter()
        seg1 = tmp_path / "seg1.wav"
        seg1.write_bytes(b"fake1")
        output = tmp_path / "output.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = segmenter.concatenate([str(seg1)], str(output))

        assert result == str(output)

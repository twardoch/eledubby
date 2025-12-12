# this_file: tests/eledubby/audio/test_analyzer.py
"""Unit tests for SilenceAnalyzer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestSilenceAnalyzer:
    """Tests for SilenceAnalyzer class."""

    def test_init_when_default_then_minus_40_db(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()
        assert analyzer.silence_threshold_db == -40

    def test_init_when_custom_threshold_then_uses_it(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer(silence_threshold_db=-30)
        assert analyzer.silence_threshold_db == -30

    def test_calculate_silence_scores_when_silent_audio_then_high_scores(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer(silence_threshold_db=-40)
        # Create silent audio (near zero amplitude)
        sample_rate = 16000
        duration = 1.0
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)

        scores = analyzer._calculate_silence_scores(audio_data, sample_rate)

        assert len(scores) > 0
        # All scores should be 1.0 (maximum silence) for silent audio
        for _time, score in scores:
            assert score == 1.0

    def test_calculate_silence_scores_when_loud_audio_then_low_scores(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer(silence_threshold_db=-40)
        sample_rate = 16000
        duration = 1.0
        # Create loud audio (full amplitude sine wave)
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        scores = analyzer._calculate_silence_scores(audio_data, sample_rate)

        assert len(scores) > 0
        # All scores should be 0 for loud audio
        for _time, score in scores:
            assert score == 0.0

    def test_calculate_silence_scores_when_mixed_audio_then_varied_scores(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer(silence_threshold_db=-40)
        sample_rate = 16000
        # Create audio: 0.5s silence, 0.5s loud
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), dtype=np.float32)
        loud = np.sin(2 * np.pi * 440 * t)
        audio_data = np.concatenate([silence, loud])

        scores = analyzer._calculate_silence_scores(audio_data, sample_rate)

        # Should have both high and low scores
        high_scores = [s for _, s in scores if s > 0.5]
        low_scores = [s for _, s in scores if s == 0.0]
        assert len(high_scores) > 0
        assert len(low_scores) > 0

    def test_find_optimal_segments_when_short_audio_then_single_segment(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()
        # Short audio, shorter than min_duration
        silence_scores = [(i * 0.05, 0.5) for i in range(100)]  # 5s of scores
        total_duration = 5.0

        segments = analyzer._find_optimal_segments(
            silence_scores,
            sample_rate=16000,
            min_duration=10.0,
            max_duration=20.0,
            total_duration=total_duration,
        )

        assert len(segments) == 1
        assert segments[0] == (0.0, 5.0)

    def test_find_optimal_segments_when_long_audio_then_multiple_segments(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()
        # 45 seconds of audio with silence scores
        silence_scores = [(i * 0.05, 0.8 if i % 300 < 10 else 0.1) for i in range(900)]
        total_duration = 45.0

        segments = analyzer._find_optimal_segments(
            silence_scores,
            sample_rate=16000,
            min_duration=10.0,
            max_duration=20.0,
            total_duration=total_duration,
        )

        # Should have 2-4 segments for 45s audio with 10-20s segments
        assert 2 <= len(segments) <= 5
        # First segment should start at 0
        assert segments[0][0] == 0.0
        # Last segment should end at total duration
        assert segments[-1][1] == total_duration
        # Segments should be contiguous
        for i in range(len(segments) - 1):
            assert segments[i][1] == segments[i + 1][0]

    def test_find_optimal_segments_when_prefers_silent_points_then_splits_at_silence(
        self,
    ) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()
        # Create scores with clear silence at 15s
        silence_scores = []
        for i in range(600):  # 30s
            time = i * 0.05
            if 14.5 <= time <= 15.5:
                score = 1.0  # Very silent
            else:
                score = 0.0  # Loud
            silence_scores.append((time, score))

        segments = analyzer._find_optimal_segments(
            silence_scores,
            sample_rate=16000,
            min_duration=10.0,
            max_duration=20.0,
            total_duration=30.0,
        )

        # Should split near 15s (the silent point)
        assert len(segments) >= 2
        # First segment should end near 15s
        assert 14.0 <= segments[0][1] <= 16.0

    def test_analyze_when_wav_file_then_returns_segments(self, tmp_path: Path) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer
        from scipy.io import wavfile

        analyzer = SilenceAnalyzer()

        # Create a test WAV file
        sample_rate = 16000
        duration = 25.0  # Long enough to need splitting
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        # Add silence in the middle
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        audio[int(12 * sample_rate) : int(13 * sample_rate)] = 0  # Silence at 12-13s

        wav_path = tmp_path / "test.wav"
        wavfile.write(str(wav_path), sample_rate, (audio * 32767).astype(np.int16))

        segments = analyzer.analyze(str(wav_path), min_duration=10.0, max_duration=20.0)

        assert len(segments) >= 1
        assert segments[0][0] == 0.0
        assert segments[-1][1] == pytest.approx(duration, abs=0.1)


class TestSilenceAnalyzerEdgeCases:
    """Edge case tests for SilenceAnalyzer."""

    def test_analyze_when_empty_scores_then_returns_full_segment(self) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()
        segments = analyzer._find_optimal_segments(
            silence_scores=[],
            sample_rate=16000,
            min_duration=10.0,
            max_duration=20.0,
            total_duration=5.0,
        )

        # Should return single segment covering all audio
        assert len(segments) == 1
        assert segments[0] == (0.0, 5.0)

    def test_calculate_silence_scores_when_all_zeros_normalized_then_handles_gracefully(
        self,
    ) -> None:
        from eledubby.audio.analyzer import SilenceAnalyzer

        analyzer = SilenceAnalyzer()
        # Audio that's exactly zero (edge case for normalization)
        audio_data = np.zeros(16000, dtype=np.float32)

        scores = analyzer._calculate_silence_scores(audio_data, 16000)

        # Should not crash and return valid scores
        assert len(scores) > 0

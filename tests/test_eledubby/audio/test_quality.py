# this_file: tests/test_eledubby/audio/test_quality.py
"""Tests for audio quality assessment module."""

import numpy as np
import pytest
from scipy.io import wavfile

from eledubby.audio.quality import AudioQualityChecker, QualityReport


class TestAudioQualityChecker:
    """Tests for AudioQualityChecker."""

    @pytest.fixture
    def checker(self):
        """Create quality checker instance."""
        return AudioQualityChecker()

    @pytest.fixture
    def clean_audio_file(self, tmp_path):
        """Create a clean test audio file."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Clean sine wave at -12dB
        audio = 0.25 * np.sin(2 * np.pi * 440 * t)
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_path = tmp_path / "clean.wav"
        wavfile.write(str(audio_path), sample_rate, audio_int16)
        return str(audio_path)

    @pytest.fixture
    def clipped_audio_file(self, tmp_path):
        """Create a clipped test audio file."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Loud sine wave that clips
        audio = 1.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_path = tmp_path / "clipped.wav"
        wavfile.write(str(audio_path), sample_rate, audio_int16)
        return str(audio_path)

    @pytest.fixture
    def silent_audio_file(self, tmp_path):
        """Create a mostly silent test audio file."""
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        # Mostly silence with brief sound
        audio = np.zeros(samples)
        audio[samples // 4 : samples // 4 + 1000] = 0.1 * np.sin(np.linspace(0, 100, 1000))
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_path = tmp_path / "silent.wav"
        wavfile.write(str(audio_path), sample_rate, audio_int16)
        return str(audio_path)

    @pytest.fixture
    def dc_offset_audio_file(self, tmp_path):
        """Create audio file with DC offset."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Sine wave with DC offset
        audio = 0.25 * np.sin(2 * np.pi * 440 * t) + 0.1
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_path = tmp_path / "dc_offset.wav"
        wavfile.write(str(audio_path), sample_rate, audio_int16)
        return str(audio_path)

    def test_analyze_clean_audio(self, checker, clean_audio_file):
        """Test analyzing clean audio passes all checks."""
        report = checker.analyze(clean_audio_file)

        assert isinstance(report, QualityReport)
        assert report.passed is True
        assert len(report.issues) == 0
        assert report.sample_rate == 16000
        assert report.channels == 1
        assert report.duration > 1.9
        assert report.clipping_ratio < 0.001

    def test_analyze_clipped_audio(self, checker, clipped_audio_file):
        """Test analyzing clipped audio detects clipping."""
        report = checker.analyze(clipped_audio_file)

        assert report.passed is False
        assert report.clipping_ratio > 0.001
        assert any("clipping" in issue.lower() for issue in report.issues)

    def test_analyze_silent_audio(self, checker, silent_audio_file):
        """Test analyzing mostly silent audio detects high silence ratio."""
        report = checker.analyze(silent_audio_file)

        assert report.silence_ratio > 0.5
        assert any("silence" in issue.lower() for issue in report.issues)

    def test_analyze_dc_offset(self, checker, dc_offset_audio_file):
        """Test analyzing audio with DC offset detects the issue."""
        report = checker.analyze(dc_offset_audio_file)

        assert abs(report.dc_offset) > 0.05
        assert any("dc offset" in issue.lower() for issue in report.issues)

    def test_report_to_dict(self, checker, clean_audio_file):
        """Test QualityReport.to_dict() method."""
        report = checker.analyze(clean_audio_file)
        report_dict = report.to_dict()

        assert "duration" in report_dict
        assert "sample_rate" in report_dict
        assert "peak_db" in report_dict
        assert "rms_db" in report_dict
        assert "issues" in report_dict
        assert "passed" in report_dict

    def test_compare_files(self, checker, clean_audio_file, clipped_audio_file):
        """Test comparing two audio files."""
        result = checker.compare(clean_audio_file, clipped_audio_file)

        assert "original" in result
        assert "processed" in result
        assert "delta" in result
        assert "original_passed" in result
        assert "processed_passed" in result

        # Clean audio should pass, clipped should fail
        assert result["original_passed"] is True
        assert result["processed_passed"] is False

    def test_stereo_audio(self, checker, tmp_path):
        """Test analyzing stereo audio."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Stereo audio
        left = 0.25 * np.sin(2 * np.pi * 440 * t)
        right = 0.25 * np.sin(2 * np.pi * 550 * t)
        audio = np.column_stack([left, right])
        audio_int16 = (audio * 32767).astype(np.int16)

        audio_path = tmp_path / "stereo.wav"
        wavfile.write(str(audio_path), sample_rate, audio_int16)

        report = checker.analyze(str(audio_path))

        assert report.channels == 2
        assert report.passed is True

    def test_custom_thresholds(self, tmp_path):
        """Test using custom quality thresholds."""
        # Create audio with moderate clipping (5% of samples)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Add some clipping to a portion
        clip_start = len(audio) // 4
        clip_end = clip_start + len(audio) // 20
        audio[clip_start:clip_end] = 1.0

        audio_int16 = (audio * 32767).astype(np.int16)
        audio_path = tmp_path / "moderate_clip.wav"
        wavfile.write(str(audio_path), sample_rate, audio_int16)

        # Default checker should detect clipping
        default = AudioQualityChecker()
        default_report = default.analyze(str(audio_path))
        assert any("clipping" in issue.lower() for issue in default_report.issues)

        # Very permissive checker should not
        permissive = AudioQualityChecker(
            min_snr_db=5.0,
            max_clipping_ratio=0.1,  # Allow 10% clipping
            max_dc_offset=0.5,
        )
        permissive_report = permissive.analyze(str(audio_path))
        # Only check that sample clipping ratio issue is not reported
        assert not any("clipping detected" in issue.lower() for issue in permissive_report.issues)

    def test_metrics_calculation(self, checker, clean_audio_file):
        """Test that all metrics are calculated correctly."""
        report = checker.analyze(clean_audio_file)

        # Peak should be reasonable for -12dB sine wave
        assert -20 < report.peak_db < 0

        # RMS should be less than peak
        assert report.rms_db < report.peak_db

        # Crest factor for sine wave should be around 1.4 (sqrt(2))
        assert 1.0 < report.crest_factor < 2.0

        # Dynamic range should be positive
        assert report.dynamic_range_db > 0

        # DC offset should be near zero for clean audio
        assert abs(report.dc_offset) < 0.01

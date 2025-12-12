# this_file: src/eledubby/audio/quality.py
"""Audio quality assessment module."""

import subprocess
from dataclasses import dataclass

import numpy as np
from loguru import logger
from pedalboard.io import AudioFile


@dataclass
class QualityReport:
    """Audio quality assessment report."""

    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    peak_db: float
    rms_db: float
    snr_db: float | None
    silence_ratio: float
    clipping_ratio: float
    dc_offset: float
    crest_factor: float
    dynamic_range_db: float
    loudness_lufs: float | None
    issues: list[str]
    passed: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "peak_db": self.peak_db,
            "rms_db": self.rms_db,
            "snr_db": self.snr_db,
            "silence_ratio": self.silence_ratio,
            "clipping_ratio": self.clipping_ratio,
            "dc_offset": self.dc_offset,
            "crest_factor": self.crest_factor,
            "dynamic_range_db": self.dynamic_range_db,
            "loudness_lufs": self.loudness_lufs,
            "issues": self.issues,
            "passed": self.passed,
        }


class AudioQualityChecker:
    """Assess audio quality metrics."""

    # Quality thresholds
    MIN_SNR_DB = 15.0  # Lowered for better real-world compatibility
    MAX_CLIPPING_RATIO = 0.001  # 0.1% samples
    MAX_DC_OFFSET = 0.01
    MIN_DYNAMIC_RANGE_DB = 6.0
    MAX_SILENCE_RATIO = 0.5  # 50% silence is suspicious
    CLIPPING_THRESHOLD = 0.99  # Near full scale

    def __init__(
        self,
        min_snr_db: float = MIN_SNR_DB,
        max_clipping_ratio: float = MAX_CLIPPING_RATIO,
        max_dc_offset: float = MAX_DC_OFFSET,
    ):
        """Initialize quality checker.

        Args:
            min_snr_db: Minimum acceptable SNR in dB
            max_clipping_ratio: Maximum acceptable clipping ratio
            max_dc_offset: Maximum acceptable DC offset
        """
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio
        self.max_dc_offset = max_dc_offset

    def analyze(self, audio_path: str) -> QualityReport:
        """Analyze audio file and generate quality report.

        Args:
            audio_path: Path to audio file

        Returns:
            QualityReport with quality metrics and assessment
        """
        # Load audio using pedalboard (returns float32 in [-1, 1])
        with AudioFile(audio_path) as f:
            sample_rate = f.samplerate
            channels = f.num_channels
            audio_data = f.read(f.frames)
            duration = f.duration

        # pedalboard always returns float32, assume 32-bit depth
        # (actual file bit depth not directly exposed, but normalized)
        bit_depth = 32

        # Normalize to float64 for precision (pedalboard returns channels x samples)
        audio_float = audio_data.astype(np.float64)

        # Handle stereo (pedalboard returns channels x samples)
        if audio_float.ndim > 1 and audio_float.shape[0] > 1:
            audio_mono = np.mean(audio_float, axis=0)
        elif audio_float.ndim > 1:
            audio_mono = audio_float[0]
        else:
            audio_mono = audio_float

        # Calculate metrics
        peak = np.max(np.abs(audio_mono))
        peak_db = 20 * np.log10(peak) if peak > 0 else -120

        rms = np.sqrt(np.mean(audio_mono**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -120

        # Crest factor (peak to RMS ratio)
        crest_factor = peak / rms if rms > 0 else 0

        # DC offset
        dc_offset = np.mean(audio_mono)

        # Clipping detection
        clipping_samples = np.sum(np.abs(audio_mono) >= self.CLIPPING_THRESHOLD)
        clipping_ratio = clipping_samples / len(audio_mono)

        # Silence detection (below -60dB)
        silence_threshold = 10 ** (-60 / 20)
        silence_samples = np.sum(np.abs(audio_mono) < silence_threshold)
        silence_ratio = silence_samples / len(audio_mono)

        # Dynamic range (difference between peak and noise floor)
        # Estimate noise floor from quietest 10% of non-silent samples
        non_silent = audio_mono[np.abs(audio_mono) >= silence_threshold]
        if len(non_silent) > 0:
            sorted_levels = np.sort(np.abs(non_silent))
            noise_floor = np.mean(sorted_levels[: len(sorted_levels) // 10])
            noise_floor_db = 20 * np.log10(noise_floor) if noise_floor > 0 else -120
            dynamic_range_db = peak_db - noise_floor_db
            snr_db = rms_db - noise_floor_db
        else:
            dynamic_range_db = 0
            snr_db = None

        # Try to get LUFS using ffmpeg
        loudness_lufs = self._get_loudness_lufs(audio_path)

        # Assess quality issues
        issues = []

        if snr_db is not None and snr_db < self.min_snr_db:
            issues.append(f"Low SNR: {snr_db:.1f}dB (min: {self.min_snr_db}dB)")

        if clipping_ratio > self.max_clipping_ratio:
            issues.append(
                f"Clipping detected: {clipping_ratio * 100:.2f}% samples "
                f"(max: {self.max_clipping_ratio * 100:.2f}%)"
            )

        if abs(dc_offset) > self.max_dc_offset:
            issues.append(f"DC offset: {dc_offset:.4f} (max: {self.max_dc_offset})")

        if dynamic_range_db < self.MIN_DYNAMIC_RANGE_DB:
            issues.append(
                f"Low dynamic range: {dynamic_range_db:.1f}dB (min: {self.MIN_DYNAMIC_RANGE_DB}dB)"
            )

        if silence_ratio > self.MAX_SILENCE_RATIO:
            issues.append(
                f"High silence ratio: {silence_ratio * 100:.1f}% "
                f"(max: {self.MAX_SILENCE_RATIO * 100:.0f}%)"
            )

        if peak_db > -0.1:
            issues.append(f"Peak too high (potential clipping): {peak_db:.1f}dB")

        passed = len(issues) == 0

        return QualityReport(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            peak_db=peak_db,
            rms_db=rms_db,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
            clipping_ratio=clipping_ratio,
            dc_offset=dc_offset,
            crest_factor=crest_factor,
            dynamic_range_db=dynamic_range_db,
            loudness_lufs=loudness_lufs,
            issues=issues,
            passed=passed,
        )

    def _get_loudness_lufs(self, audio_path: str) -> float | None:
        """Get integrated loudness in LUFS using ffmpeg.

        Args:
            audio_path: Path to audio file

        Returns:
            Loudness in LUFS or None if measurement failed
        """
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-af",
            "loudnorm=I=-24:print_format=json",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            # Parse LUFS from output
            import re

            match = re.search(r'"input_i"\s*:\s*"(-?\d+\.?\d*)"', result.stderr)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.debug(f"Could not measure LUFS: {e}")

        return None

    def compare(self, original_path: str, processed_path: str) -> dict:
        """Compare original and processed audio quality.

        Args:
            original_path: Path to original audio
            processed_path: Path to processed audio

        Returns:
            Comparison report with delta metrics
        """
        original = self.analyze(original_path)
        processed = self.analyze(processed_path)

        return {
            "original": original.to_dict(),
            "processed": processed.to_dict(),
            "delta": {
                "peak_db": processed.peak_db - original.peak_db,
                "rms_db": processed.rms_db - original.rms_db,
                "dynamic_range_db": processed.dynamic_range_db - original.dynamic_range_db,
                "silence_ratio": processed.silence_ratio - original.silence_ratio,
                "clipping_ratio": processed.clipping_ratio - original.clipping_ratio,
            },
            "original_passed": original.passed,
            "processed_passed": processed.passed,
        }

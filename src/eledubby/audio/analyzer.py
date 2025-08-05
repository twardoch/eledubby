# this_file: audio/analyzer.py
"""Audio analysis module for silence detection."""

import numpy as np
from loguru import logger
from scipy.io import wavfile


class SilenceAnalyzer:
    """Analyzes audio for silence detection and optimal split points."""

    def __init__(self, silence_threshold_db: float = -40):
        """Initialize silence analyzer.

        Args:
            silence_threshold_db: Silence threshold in dB below max
        """
        self.silence_threshold_db = silence_threshold_db

    def analyze(
        self, audio_path: str, min_duration: float = 10.0, max_duration: float = 20.0
    ) -> list[tuple[float, float]]:
        """Analyze audio and find optimal segment boundaries.

        Args:
            audio_path: Path to audio file
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds

        Returns:
            List of (start_time, end_time) tuples for segments
        """
        # Load audio
        sample_rate, audio_data = wavfile.read(audio_path)
        audio_data = audio_data.astype(np.float32)

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        logger.debug(f"Loaded audio: {len(audio_data) / sample_rate:.2f}s at {sample_rate}Hz")

        # Find silence points
        silence_scores = self._calculate_silence_scores(audio_data, sample_rate)

        # Find optimal split points
        segments = self._find_optimal_segments(
            silence_scores, sample_rate, min_duration, max_duration, len(audio_data) / sample_rate
        )

        logger.info(f"Found {len(segments)} segments")
        return segments

    def _calculate_silence_scores(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> list[tuple[float, float]]:
        """Calculate silence scores throughout the audio.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate

        Returns:
            List of (time, score) tuples
        """
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = int(0.05 * sample_rate)  # 50ms hop

        scores = []

        # Convert threshold from dB to linear
        threshold_linear = 10 ** (self.silence_threshold_db / 20)

        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i : i + window_size]

            # Calculate RMS energy
            rms = np.sqrt(np.mean(window**2))

            # Calculate silence score (0-1, higher is more silent)
            silence_score = 1.0 - rms / threshold_linear if rms < threshold_linear else 0.0

            # Add duration bonus for longer silent regions
            time_position = i / sample_rate
            scores.append((time_position, silence_score))

        return scores

    def _find_optimal_segments(
        self,
        silence_scores: list[tuple[float, float]],
        sample_rate: int,  # noqa: ARG002
        min_duration: float,
        max_duration: float,
        total_duration: float,
    ) -> list[tuple[float, float]]:
        """Find optimal segment boundaries based on silence scores.

        Args:
            silence_scores: List of (time, score) tuples
            sample_rate: Audio sample rate
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration
            total_duration: Total audio duration

        Returns:
            List of (start_time, end_time) tuples
        """
        segments = []
        current_start = 0.0

        while current_start < total_duration:
            # Find the best split point in the window
            window_start = current_start + min_duration
            window_end = min(current_start + max_duration, total_duration)

            if window_start >= total_duration:
                # Last segment is too short, extend previous
                if segments:
                    segments[-1] = (segments[-1][0], total_duration)
                else:
                    segments.append((0.0, total_duration))
                break

            # Find highest scoring silence point in window
            best_score = -1
            best_time = window_end

            for time, score in silence_scores:
                if window_start <= time <= window_end:
                    # Add position weight (prefer middle of window)
                    position_weight = 1.0 - abs(time - (window_start + window_end) / 2) / (
                        max_duration / 2
                    )
                    weighted_score = score * 0.7 + position_weight * 0.3

                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_time = time

            # Create segment
            segments.append((current_start, best_time))
            current_start = best_time

        return segments

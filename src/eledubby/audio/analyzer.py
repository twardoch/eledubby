# this_file: audio/analyzer.py
"""Audio analysis module for silence detection."""

import numpy as np
from loguru import logger
from pedalboard.io import AudioFile


class SilenceAnalyzer:
    """Analyzes audio for silence detection and optimal split points."""

    # Maximum file size in bytes to load fully into memory (50MB)
    MAX_FULL_LOAD_SIZE = 50 * 1024 * 1024
    # Chunk size for streaming analysis (10 seconds at 48kHz stereo 16-bit)
    CHUNK_DURATION_SECONDS = 10.0

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

        Uses streaming analysis for large files to minimize memory usage.

        Args:
            audio_path: Path to audio file
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds

        Returns:
            List of (start_time, end_time) tuples for segments
        """
        import os

        file_size = os.path.getsize(audio_path)

        if file_size <= self.MAX_FULL_LOAD_SIZE:
            # Small file - load fully for faster processing
            return self._analyze_full(audio_path, min_duration, max_duration)
        else:
            # Large file - use streaming analysis via pedalboard
            logger.info(f"Large file ({file_size / 1024 / 1024:.1f}MB), using streaming analysis")
            return self._analyze_streaming(audio_path, min_duration, max_duration)

    def _analyze_full(
        self, audio_path: str, min_duration: float, max_duration: float
    ) -> list[tuple[float, float]]:
        """Analyze audio by loading fully into memory (for smaller files)."""
        # Load audio using pedalboard
        with AudioFile(audio_path) as f:
            sample_rate = f.samplerate
            audio_data = f.read(f.frames)

        # Convert to mono if stereo (pedalboard returns channels x samples)
        if audio_data.ndim > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        elif audio_data.ndim > 1:
            audio_data = audio_data[0]

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

    def _analyze_streaming(
        self, audio_path: str, min_duration: float, max_duration: float
    ) -> list[tuple[float, float]]:
        """Analyze audio in chunks to minimize memory usage (for large files)."""
        # Get file info using pedalboard
        with AudioFile(audio_path) as f:
            sample_rate = f.samplerate
            total_duration = f.duration

        logger.debug(f"Streaming analysis: {total_duration:.2f}s at {sample_rate}Hz")

        # Calculate chunk size
        chunk_frames = int(self.CHUNK_DURATION_SECONDS * sample_rate)
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = int(0.05 * sample_rate)  # 50ms hop

        # Convert threshold from dB to linear
        threshold_linear = 10 ** (self.silence_threshold_db / 20)

        silence_scores = []
        global_max = 0.0

        # First pass: find global max for normalization
        with AudioFile(audio_path) as f:
            while f.tell() < f.frames:
                chunk = f.read(chunk_frames)
                if chunk.size == 0:
                    break
                # Handle stereo by taking mean (pedalboard returns channels x samples)
                if chunk.ndim > 1 and chunk.shape[0] > 1:
                    chunk = np.mean(chunk, axis=0)
                elif chunk.ndim > 1:
                    chunk = chunk[0]
                chunk_max = np.max(np.abs(chunk))
                if chunk_max > global_max:
                    global_max = chunk_max

        if global_max == 0:
            global_max = 1.0

        # Second pass: calculate silence scores
        current_frame = 0
        with AudioFile(audio_path) as f:
            while f.tell() < f.frames:
                chunk = f.read(chunk_frames)
                if chunk.size == 0:
                    break

                # Handle stereo by taking mean (pedalboard returns channels x samples)
                if chunk.ndim > 1 and chunk.shape[0] > 1:
                    chunk = np.mean(chunk, axis=0)
                elif chunk.ndim > 1:
                    chunk = chunk[0]

                chunk_len = len(chunk)

                # Normalize
                chunk = chunk / global_max

                # Calculate silence scores for this chunk
                for i in range(0, chunk_len - window_size, hop_size):
                    window = chunk[i : i + window_size]
                    rms = np.sqrt(np.mean(window**2))

                    silence_score = 1.0 - rms / threshold_linear if rms < threshold_linear else 0.0
                    time_position = (current_frame + i) / sample_rate
                    silence_scores.append((time_position, silence_score))

                current_frame += chunk_len

        # Find optimal split points
        segments = self._find_optimal_segments(
            silence_scores, sample_rate, min_duration, max_duration, total_duration
        )

        logger.info(f"Found {len(segments)} segments (streaming)")
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

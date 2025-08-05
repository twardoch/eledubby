# this_file: audio/extractor.py
"""Audio extraction module using ffmpeg."""

import subprocess

from loguru import logger


class AudioExtractor:
    """Handles audio extraction from video files."""

    def __init__(self, sample_rate: int = 16000):
        """Initialize audio extractor.

        Args:
            sample_rate: Target sample rate for extracted audio
        """
        self.sample_rate = sample_rate

    def extract(self, video_path: str, output_path: str) -> str:
        """Extract audio from video file.

        Args:
            video_path: Path to input video file
            output_path: Path to save extracted audio

        Returns:
            Path to extracted audio file

        Raises:
            RuntimeError: If extraction fails
        """
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            str(self.sample_rate),  # Sample rate
            "-ac",
            "1",  # Mono
            "-y",  # Overwrite
            output_path,
        ]

        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"Failed to extract audio: {result.stderr}")

        # Get audio duration
        duration = self._get_duration(output_path)
        logger.info(f"Extracted {duration:.2f}s of audio to: {output_path}")

        return output_path

    def _get_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                logger.warning(f"Could not parse duration: {result.stdout}")
                return 0.0
        return 0.0

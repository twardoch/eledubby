# this_file: audio/processor.py
"""Audio processing module for timing preservation."""

import os
import subprocess

from loguru import logger


class AudioProcessor:
    """Handles audio processing for timing preservation."""

    def measure_duration(self, audio_path: str) -> float:
        """Measure precise duration of audio file.

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

    def adjust_duration(self, audio_path: str, target_duration: float, output_path: str) -> str:
        """Adjust audio duration to match target.

        Args:
            audio_path: Path to input audio
            target_duration: Target duration in seconds
            output_path: Path to save adjusted audio

        Returns:
            Path to adjusted audio file
        """
        current_duration = self.measure_duration(audio_path)
        difference = target_duration - current_duration

        logger.debug(
            f"Duration adjustment: current={current_duration:.3f}s, "
            f"target={target_duration:.3f}s, diff={difference:.3f}s"
        )

        if abs(difference) < 0.05:  # Within 50ms tolerance
            # Close enough, just copy
            subprocess.run(["cp", audio_path, output_path], check=True)
            return output_path

        if difference > 0:
            # Need to pad with silence
            return self._pad_audio(audio_path, difference, output_path)
        else:
            # Need to trim
            return self._trim_audio(audio_path, target_duration, output_path)

    def _pad_audio(self, audio_path: str, pad_duration: float, output_path: str) -> str:
        """Pad audio with silence.

        Args:
            audio_path: Path to input audio
            pad_duration: Duration of silence to add
            output_path: Path to save padded audio

        Returns:
            Path to padded audio file
        """
        # Generate silence
        silence_path = output_path + "_silence.wav"
        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=16000:cl=mono:d={pad_duration}",
            "-acodec",
            "pcm_s16le",
            "-y",
            silence_path,
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        # Concatenate with crossfade
        filter_complex = "[0:a][1:a]concat=n=2:v=0:a=1[out]"

        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-i",
            silence_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-acodec",
            "pcm_s16le",
            "-y",
            output_path,
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        # Cleanup
        os.remove(silence_path)

        logger.debug(f"Padded audio with {pad_duration:.3f}s of silence")
        return output_path

    def _trim_audio(self, audio_path: str, target_duration: float, output_path: str) -> str:
        """Trim audio to target duration.

        Args:
            audio_path: Path to input audio
            target_duration: Target duration in seconds
            output_path: Path to save trimmed audio

        Returns:
            Path to trimmed audio file
        """
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-t",
            str(target_duration),
            "-acodec",
            "copy",
            "-y",
            output_path,
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        logger.debug(f"Trimmed audio to {target_duration:.3f}s")
        return output_path

    def normalize_audio(self, audio_path: str, output_path: str, target_db: float = -23.0) -> str:
        """Normalize audio levels.

        Args:
            audio_path: Path to input audio
            output_path: Path to save normalized audio
            target_db: Target loudness in dB

        Returns:
            Path to normalized audio file
        """
        # Use loudnorm filter for EBU R128 normalization
        cmd = [
            "ffmpeg",
            "-i",
            audio_path,
            "-af",
            f"loudnorm=I={target_db}:TP=-1.5:LRA=11",
            "-acodec",
            "pcm_s16le",
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Normalization failed, copying original: {result.stderr}")
            subprocess.run(["cp", audio_path, output_path], check=True)

        return output_path

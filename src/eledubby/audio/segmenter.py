# this_file: audio/segmenter.py
"""Audio segmentation module."""

import os
import subprocess

from loguru import logger


class AudioSegmenter:
    """Handles audio segmentation based on timestamps."""

    def segment(
        self, audio_path: str, segments: list[tuple[float, float]], output_dir: str
    ) -> list[str]:
        """Split audio into segments based on timestamps.

        Args:
            audio_path: Path to input audio file
            segments: List of (start_time, end_time) tuples
            output_dir: Directory to save segments

        Returns:
            List of paths to segment files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        segment_paths = []

        for i, (start, end) in enumerate(segments):
            duration = end - start
            output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")

            cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-y",
                output_path,
            ]

            logger.debug(f"Extracting segment {i}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to extract segment {i}: {result.stderr}")
                raise RuntimeError(f"Segmentation failed: {result.stderr}")

            segment_paths.append(output_path)

        logger.info(f"Created {len(segment_paths)} segments")
        return segment_paths

    def concatenate(self, segment_paths: list[str], output_path: str) -> str:
        """Concatenate audio segments back together.

        Args:
            segment_paths: List of paths to segment files
            output_path: Path to save concatenated audio

        Returns:
            Path to concatenated audio file
        """
        # Create concat file
        concat_file = output_path + ".txt"
        with open(concat_file, "w") as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            "-y",
            output_path,
        ]

        logger.debug(f"Concatenating {len(segment_paths)} segments")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to concatenate: {result.stderr}")
            raise RuntimeError(f"Concatenation failed: {result.stderr}")

        # Cleanup concat file
        os.remove(concat_file)

        logger.info(f"Concatenated audio saved to: {output_path}")
        return output_path

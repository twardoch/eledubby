# this_file: video/remuxer.py
"""Video remuxing module."""

import subprocess

from loguru import logger


class VideoRemuxer:
    """Handles video remuxing with new audio track."""

    def remux(
        self, video_path: str, audio_path: str, output_path: str, copy_video: bool = True
    ) -> str:
        """Replace audio track in video file.

        Args:
            video_path: Path to input video
            audio_path: Path to new audio track
            output_path: Path to save output video
            copy_video: Whether to copy video codec (faster) or re-encode

        Returns:
            Path to output video file

        Raises:
            RuntimeError: If remuxing fails
        """
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            video_path,  # Input video
            "-i",
            audio_path,  # Input audio
            "-map",
            "0:v",  # Take video from first input
            "-map",
            "1:a",  # Take audio from second input
        ]

        if copy_video:
            cmd.extend(["-c:v", "copy"])  # Copy video codec
        else:
            cmd.extend(["-c:v", "libx264", "-crf", "23"])  # Re-encode with x264

        cmd.extend(
            [
                "-c:a",
                "aac",  # Encode audio as AAC
                "-b:a",
                "192k",  # Audio bitrate
                "-movflags",
                "+faststart",  # Optimize for streaming
                "-y",  # Overwrite output
                output_path,
            ]
        )

        logger.debug(f"Remuxing command: {' '.join(cmd)}")

        # Execute remuxing
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Remuxing failed: {result.stderr}")
            raise RuntimeError(f"Failed to remux video: {result.stderr}")

        # Verify output
        if not self._verify_output(output_path, video_path):
            raise RuntimeError("Output video verification failed")

        logger.info(f"Video remuxed successfully: {output_path}")
        return output_path

    def _verify_output(self, output_path: str, original_path: str) -> bool:
        """Verify output video matches original duration.

        Args:
            output_path: Path to output video
            original_path: Path to original video

        Returns:
            True if verification passes
        """
        original_duration = self._get_duration(original_path)
        output_duration = self._get_duration(output_path)

        if original_duration == 0 or output_duration == 0:
            logger.warning("Could not verify video duration")
            return True

        difference = abs(original_duration - output_duration)
        if difference > 0.5:  # More than 0.5 second difference
            logger.error(
                f"Duration mismatch: original={original_duration:.2f}s, "
                f"output={output_duration:.2f}s"
            )
            return False

        logger.debug(f"Duration verified: {output_duration:.2f}s")
        return True

    def _get_duration(self, video_path: str) -> float:
        """Get video duration in seconds.

        Args:
            video_path: Path to video file

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
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                return 0.0
        return 0.0

    def extract_metadata(self, video_path: str) -> dict:
        """Extract video metadata.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary of metadata
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json

            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                logger.warning("Could not parse video metadata")
                return {}
        return {}

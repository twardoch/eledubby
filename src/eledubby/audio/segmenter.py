# this_file: audio/segmenter.py
"""Audio segmentation module."""

import os

from loguru import logger
from pedalboard.io import AudioFile


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

        # Open source file once for all segments
        with AudioFile(audio_path) as src:
            sample_rate = src.samplerate
            num_channels = src.num_channels

            for i, (start, end) in enumerate(segments):
                duration = end - start
                output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")

                # Calculate frame positions
                start_frame = int(start * sample_rate)
                end_frame = int(end * sample_rate)
                num_frames = end_frame - start_frame

                # Seek to start position and read segment
                src.seek(start_frame)
                audio_data = src.read(num_frames)

                logger.debug(f"Extracting segment {i}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")

                # Write segment to file
                with AudioFile(
                    output_path, "w", samplerate=sample_rate, num_channels=num_channels
                ) as dst:
                    dst.write(audio_data)

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
        if not segment_paths:
            raise ValueError("No segments to concatenate")

        # Get format from first segment
        with AudioFile(segment_paths[0]) as first:
            sample_rate = first.samplerate
            num_channels = first.num_channels

        logger.debug(f"Concatenating {len(segment_paths)} segments")

        # Concatenate all segments
        with AudioFile(output_path, "w", samplerate=sample_rate, num_channels=num_channels) as dst:
            for path in segment_paths:
                with AudioFile(path) as src:
                    # Read and write in chunks to handle large files
                    chunk_size = sample_rate  # 1 second chunks
                    while src.tell() < src.frames:
                        chunk = src.read(chunk_size)
                        if chunk.size == 0:
                            break
                        dst.write(chunk)

        logger.info(f"Concatenated audio saved to: {output_path}")
        return output_path

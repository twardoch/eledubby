#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["elevenlabs", "python-dotenv", "fire", "rich", "loguru", "numpy", "scipy"]
# ///
# this_file: eledubby/src/eledubby/eledubby.py

"""
eledubby - Voice dubbing tool using ElevenLabs speech-to-speech API.

Takes an input video and replaces the audio with a new voice using ElevenLabs API.
Performs intelligent audio segmentation and maintains perfect timing synchronization.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

from .api import ElevenLabsClient

# Import our modules
from .audio import AudioExtractor, AudioProcessor, AudioSegmenter, SilenceAnalyzer
from .utils import ProgressTracker, TempFileManager
from .video import VideoRemuxer

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# Constants
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
MIN_SEGMENT_DURATION = 10.0  # seconds
MAX_SEGMENT_DURATION = 20.0  # seconds
SILENCE_THRESHOLD_DB = -40  # dB below max
SAMPLE_RATE = 16000  # Hz for processing


class EleDubby:
    """Main class for video dubbing functionality."""

    def __init__(self, verbose: bool = False):
        """Initialize the dubbing tool.

        Args:
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        self._setup_logging()
        self._check_dependencies()

        # Initialize components
        self.audio_extractor = AudioExtractor(SAMPLE_RATE)
        self.silence_analyzer = SilenceAnalyzer(SILENCE_THRESHOLD_DB)
        self.audio_segmenter = AudioSegmenter()
        self.audio_processor = AudioProcessor()
        self.elevenlabs_client = ElevenLabsClient()
        self.video_remuxer = VideoRemuxer()
        self.progress_tracker = ProgressTracker()
        self.temp_manager = TempFileManager()

    def _setup_logging(self):
        """Configure logging based on verbose flag."""
        logger.remove()  # Remove default handler

        if self.verbose:
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="DEBUG",
            )
        else:
            logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

    def _check_dependencies(self):
        """Check for required system dependencies."""
        import subprocess

        logger.debug("Checking system dependencies...")

        # Check ffmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError("ffmpeg not found")
            logger.debug(f"ffmpeg found: {result.stdout.split('\\n')[0]}")
        except FileNotFoundError:
            console.print("[red]Error: ffmpeg is not installed or not in PATH[/red]")
            console.print("Please install ffmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)

        # Check environment variable
        if not os.getenv("ELEVENLABS_API_KEY"):
            console.print("[red]Error: ELEVENLABS_API_KEY environment variable not set[/red]")
            console.print("Please set your API key: export ELEVENLABS_API_KEY=your_key_here")
            sys.exit(1)

    def process(self, input: str, voice: str = DEFAULT_VOICE_ID, output: str | None = None):
        """Process a video file with voice dubbing.

        Args:
            input: Path to input video file
            voice: ElevenLabs voice ID to use for dubbing
            output: Path to output video file (default: input_dubbed.mp4)
        """
        start_time = time.time()

        # Validate input
        if not os.path.exists(input):
            console.print(f"[red]Error: Input file not found: {input}[/red]")
            sys.exit(1)

        # Set default output if not specified
        if not output:
            input_path = Path(input)
            output = str(input_path.parent / f"{input_path.stem}_dubbed{input_path.suffix}")

        console.print(f"[bold]Processing video:[/bold] {input}")
        console.print(f"[bold]Voice ID:[/bold] {voice}")
        console.print(f"[bold]Output path:[/bold] {output}")

        # Validate voice ID
        console.print("Validating voice ID...")
        if not self.elevenlabs_client.validate_voice_id(voice):
            console.print(f"[yellow]Warning: Voice ID '{voice}' not found, using default[/yellow]")
            voice = DEFAULT_VOICE_ID

        # Check disk space
        estimated_space = self.temp_manager.estimate_space_needed(input)
        if not self.temp_manager.check_disk_space("/tmp", estimated_space):
            console.print(
                f"[red]Error: Insufficient disk space. Need ~{estimated_space:.1f}GB[/red]"
            )
            sys.exit(1)

        with self.temp_manager.temp_directory() as temp_dir:
            try:
                # Step 1: Extract audio from video
                with self.progress_tracker.track_file_operation("Extracting audio from video"):
                    audio_path = self.audio_extractor.extract(
                        input, os.path.join(temp_dir, "extracted_audio.wav")
                    )

                # Step 2: Analyze audio and find segment boundaries
                console.print("Analyzing audio for optimal segmentation...")
                segments = self.silence_analyzer.analyze(
                    audio_path, MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION
                )
                console.print(f"Found {len(segments)} segments")

                # Step 3: Split audio into segments
                with self.progress_tracker.track_file_operation("Splitting audio into segments"):
                    segment_paths = self.audio_segmenter.segment(
                        audio_path, segments, os.path.join(temp_dir, "segments")
                    )

                # Step 4: Process each segment with ElevenLabs
                converted_segments = []
                with self.progress_tracker.track_segments(
                    len(segment_paths), "Converting segments"
                ) as update:
                    for i, (segment_path, (start, end)) in enumerate(
                        zip(segment_paths, segments, strict=False)
                    ):
                        update(description=f"Converting segment {i + 1}/{len(segment_paths)}")

                        # Measure original duration
                        original_duration = end - start

                        # Convert with ElevenLabs
                        converted_path = os.path.join(temp_dir, "converted", f"segment_{i:04d}.mp3")
                        os.makedirs(os.path.dirname(converted_path), exist_ok=True)

                        self.elevenlabs_client.speech_to_speech(segment_path, voice, converted_path)

                        # Adjust timing to match original
                        adjusted_path = os.path.join(temp_dir, "adjusted", f"segment_{i:04d}.wav")
                        os.makedirs(os.path.dirname(adjusted_path), exist_ok=True)

                        self.audio_processor.adjust_duration(
                            converted_path, original_duration, adjusted_path
                        )

                        converted_segments.append(adjusted_path)
                        update(1)

                # Step 5: Concatenate processed segments
                with self.progress_tracker.track_file_operation("Reassembling audio"):
                    final_audio = self.audio_segmenter.concatenate(
                        converted_segments, os.path.join(temp_dir, "final_audio.wav")
                    )

                    # Normalize audio
                    normalized_audio = self.audio_processor.normalize_audio(
                        final_audio, os.path.join(temp_dir, "normalized_audio.wav")
                    )

                # Step 6: Remux video with new audio
                with self.progress_tracker.track_file_operation("Creating final video"):
                    self.video_remuxer.remux(input, normalized_audio, output)

                # Calculate statistics
                elapsed_time = time.time() - start_time
                stats = {
                    "Processing time": f"{elapsed_time:.1f} seconds",
                    "Segments processed": len(segments),
                    "Input video": os.path.basename(input),
                    "Output video": os.path.basename(output),
                    "Voice used": voice,
                }

                self.progress_tracker.print_summary(stats)
                console.print("\n[green]✓ Video processing complete![/green]")

            except Exception as e:
                console.print(f"\n[red]Error during processing: {e}[/red]")
                logger.exception("Processing failed")
                sys.exit(1)


def main(
    input: str | Path,
    voice: str = DEFAULT_VOICE_ID,
    output: str | Path | None = None,
    verbose: bool = False,
):
    """eledubby - Voice dubbing tool using ElevenLabs speech-to-speech API.

    Args:
        input: Path to input video file
        voice: ElevenLabs voice ID to use for dubbing (default: iBR3vm0M6ImfaxXsPgxi)
        output: Path to output video file (default: input_dubbed.mp4)
        verbose: Enable verbose logging output
    """

    dubber = EleDubby(verbose=verbose)
    dubber.process(input, voice, output)

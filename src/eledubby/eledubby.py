#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["elevenlabs", "python-dotenv", "fire", "rich", "loguru", "numpy", "pedalboard", "toml"]
# ///
# this_file: src/eledubby/eledubby.py

"""
eledubby - Voice dubbing tool using ElevenLabs speech-to-speech API.

Takes an input video and replaces the audio with a new voice using ElevenLabs API.
Performs intelligent audio segmentation and maintains perfect timing synchronization.
"""

import glob as glob_module
import mimetypes
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pedalboard
import toml
from dotenv import load_dotenv
from loguru import logger
from pathvalidate import sanitize_filename
from rich.console import Console
from slugify import slugify

from .api import ElevenLabsClient

# Import our modules
from .audio import (
    AudioExtractor,
    AudioProcessor,
    AudioQualityChecker,
    AudioSegmenter,
    SilenceAnalyzer,
)
from .utils import CheckpointManager, ProgressTracker, TempFileManager
from .video import VideoRemuxer

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# Constants
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
DEFAULT_MIN_SEGMENT_DURATION = 10.0  # seconds
DEFAULT_MAX_SEGMENT_DURATION = 20.0  # seconds
SILENCE_THRESHOLD_DB = -40  # dB below max
SAMPLE_RATE = 16000  # Hz for processing
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


class EleDubby:
    """Main class for video dubbing functionality."""

    def __init__(
        self,
        verbose: bool = False,
        seg_min: float = DEFAULT_MIN_SEGMENT_DURATION,
        seg_max: float = DEFAULT_MAX_SEGMENT_DURATION,
        require_elevenlabs: bool = True,
        api_key: str | None = None,
        parallel: int = 1,
        preview: float = 0,
        normalize: bool = True,
        target_db: float = -23.0,
        compress: bool = False,
        resume: bool = False,
        denoise: float = 0.0,
    ):
        """Initialize the dubbing tool.

        Args:
            verbose: Enable verbose logging output
            seg_min: Minimum segment duration in seconds
            seg_max: Maximum segment duration in seconds
            require_elevenlabs: Whether ElevenLabs API is required
            api_key: ElevenLabs API key override (defaults to env var)
            parallel: Number of parallel workers for segment processing (1=sequential)
            preview: Preview mode - process only first N seconds (0=disabled, default: 0)
            normalize: Enable EBU R128 loudness normalization (default: True)
            target_db: Target loudness in dB for normalization (default: -23.0)
            compress: Apply dynamic range compression (default: False)
            resume: Resume from checkpoint if available (default: False)
            denoise: Noise reduction strength 0.0-1.0 (0=disabled, default: 0)
        """
        self.verbose = verbose
        self.parallel = max(1, parallel)
        self.preview = max(0, preview)
        self.seg_min = seg_min
        self.seg_max = seg_max
        self.api_key = api_key
        self.normalize = normalize
        self.target_db = target_db
        self.compress = compress
        self.resume = resume
        self.denoise = max(0.0, min(1.0, denoise))
        self._setup_logging()
        self._check_dependencies(require_elevenlabs)

        # Initialize components
        self.audio_extractor = AudioExtractor(SAMPLE_RATE)
        self.silence_analyzer = SilenceAnalyzer(SILENCE_THRESHOLD_DB)
        self.audio_segmenter = AudioSegmenter()
        self.audio_processor = AudioProcessor()
        if require_elevenlabs:
            self.elevenlabs_client = ElevenLabsClient(api_key=self.api_key)
        self.video_remuxer = VideoRemuxer()
        self.progress_tracker = ProgressTracker()
        self.temp_manager = TempFileManager()
        self.checkpoint_manager = CheckpointManager()

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

    def _check_dependencies(self, require_elevenlabs: bool = True):
        """Check for required system dependencies.

        Args:
            require_elevenlabs: Whether to check for ElevenLabs API key
        """
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

        # Check API key only if required
        if require_elevenlabs and not (self.api_key or os.getenv("ELEVENLABS_API_KEY")):
            console.print("[red]Error: ELEVENLABS_API_KEY environment variable not set[/red]")
            console.print("Please set your API key: export ELEVENLABS_API_KEY=your_key_here")
            sys.exit(1)

    def _resolve_vst3_path(self, plugin_path: str) -> str | None:
        """Resolve VST3 plugin path based on system.

        Args:
            plugin_path: Plugin path or name

        Returns:
            Full path to plugin if found, None otherwise
        """
        # If absolute path and exists, return it
        if os.path.isabs(plugin_path) and os.path.exists(plugin_path):
            return plugin_path

        # Define system-specific VST3 search paths
        search_paths = []
        system = platform.system()

        if system == "Darwin":  # macOS
            home = Path.home()
            search_paths = [
                home / "Library/Audio/Plug-Ins/VST3",
                Path("/Library/Audio/Plug-Ins/VST3"),
            ]
        elif system == "Windows":
            search_paths = [
                Path("C:/Program Files/Common Files/VST3"),
                Path("C:/Program Files (x86)/Common Files/VST3"),
            ]
        elif system == "Linux":
            home = Path.home()
            search_paths = [home / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]

        # Search for plugin in system paths
        plugin_name = Path(plugin_path).name
        for search_path in search_paths:
            if search_path.exists():
                # Direct match
                full_path = search_path / plugin_name
                if full_path.exists():
                    logger.debug(f"Found VST3 plugin at: {full_path}")
                    return str(full_path)

                # Search recursively
                for vst_file in search_path.rglob(plugin_name):
                    logger.debug(f"Found VST3 plugin at: {vst_file}")
                    return str(vst_file)

        logger.warning(f"VST3 plugin not found: {plugin_path}")
        return None

    def _apply_audio_fx(self, audio_path: str, fx_config: str | dict, output_path: str) -> str:
        """Apply audio effects using pedalboard and VST3 plugins.

        Args:
            audio_path: Path to input audio file
            fx_config: Effects configuration (path to TOML or dict)
            output_path: Path to output audio file

        Returns:
            Path to processed audio file
        """
        logger.info("Applying audio post-processing effects...")

        # Load configuration
        if isinstance(fx_config, str):
            config_path = Path(fx_config)
            if not config_path.exists():
                logger.warning(f"FX config file not found: {fx_config}")
                return audio_path
            with open(config_path) as f:
                config = toml.load(f)
        else:
            config = fx_config

        if not config:
            logger.info("No effects configured, skipping post-processing")
            return audio_path

        # Load audio
        with pedalboard.io.AudioFile(audio_path, "r") as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        # Create pedalboard
        board = pedalboard.Pedalboard()

        # Load and configure VST3 plugins
        for plugin_path, params in config.items():
            # Skip comments and metadata
            if plugin_path.startswith("_"):
                continue

            # Resolve plugin path
            resolved_path = self._resolve_vst3_path(plugin_path)
            if not resolved_path:
                logger.warning(f"Skipping plugin: {plugin_path} (not found)")
                continue

            try:
                # Load VST3 plugin
                logger.info(f"Loading VST3 plugin: {resolved_path}")
                plugin = pedalboard.load_plugin(resolved_path)

                # Set parameters
                if isinstance(params, dict):
                    for param_name, param_value in params.items():
                        if hasattr(plugin, param_name):
                            setattr(plugin, param_name, param_value)
                            logger.debug(f"Set {param_name} = {param_value}")
                        else:
                            logger.warning(f"Plugin has no parameter: {param_name}")

                # Add to board
                board.append(plugin)
                logger.info(f"Added plugin to chain: {plugin_path}")

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_path}: {e}")
                continue

        # Apply effects if any plugins were loaded
        if board:
            logger.info(f"Applying {len(board)} effects to audio...")
            processed = board(audio, samplerate)

            # Save processed audio
            with pedalboard.io.AudioFile(output_path, "w", samplerate, processed.shape[0]) as f:
                f.write(processed)

            logger.info(f"Saved processed audio to: {output_path}")
            return output_path
        else:
            logger.warning("No effects were successfully loaded")
            return audio_path

    def _is_video_file(self, file_path: str) -> bool:
        """Check if a file is a video file based on mime type or extension.

        Args:
            file_path: Path to the file

        Returns:
            True if file is video, False if audio
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type.startswith("video/"):
                return True
            elif mime_type.startswith("audio/"):
                return False

        # Fallback to extension checking
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
        audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus"}

        ext = Path(file_path).suffix.lower()
        if ext in video_extensions:
            return True
        elif ext in audio_extensions:
            return False

        # If uncertain, try to probe with ffmpeg
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    file_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() == "video"
        except Exception:
            # Default to assuming it's audio if we can't determine
            return False

    def _determine_output_format(
        self,
        input_path: str,
        output_path: str | None,
        is_input_video: bool,
        force_audio: bool = False,
    ) -> tuple[str, bool]:
        """Determine the output file path and whether it should be video.

        Args:
            input_path: Input file path
            output_path: Specified output path (optional)
            is_input_video: Whether input is video
            force_audio: Force audio output even if input is video

        Returns:
            Tuple of (output_path, is_output_video)
        """
        if output_path:
            # Check if output should be video or audio based on extension
            is_output_video = self._is_video_file(output_path)

            # If input is audio, output must be audio
            if not is_input_video and is_output_video:
                logger.warning(
                    "Input is audio, output must also be audio. Changing extension to .wav"
                )
                output_path = str(Path(output_path).with_suffix(".wav"))
                is_output_video = False
        else:
            # Generate default output path
            input_p = Path(input_path)

            if force_audio or not is_input_video:
                # For audio output
                output_path = str(input_p.parent / f"{input_p.stem}_processed.wav")
                is_output_video = False
            else:
                # For video output
                output_path = str(input_p.parent / f"{input_p.stem}_processed{input_p.suffix}")
                is_output_video = True

        return output_path, is_output_video

    def process(
        self,
        input: str,
        voice: str = DEFAULT_VOICE_ID,
        output: str | None = None,
        fx: str | bool | None = None,
    ):
        """Process a video or audio file with voice dubbing.

        Args:
            input: Path to input video or audio file
            voice: ElevenLabs voice ID to use for dubbing
            output: Path to output file (default: auto-generated based on input)
            fx: Audio effects configuration (False/0/off for none, True/1/on for default, or path to config)
        """
        start_time = time.time()

        # Validate input
        if not os.path.exists(input):
            console.print(f"[red]Error: Input file not found: {input}[/red]")
            sys.exit(1)

        # Determine if input is video or audio
        is_input_video = self._is_video_file(input)

        # Set default output if not specified
        if not output:
            input_path = Path(input)
            if is_input_video:
                output = str(input_path.parent / f"{input_path.stem}_dubbed{input_path.suffix}")
            else:
                output = str(input_path.parent / f"{input_path.stem}_dubbed.wav")

        # Determine output format
        is_output_video = self._is_video_file(output) if output else is_input_video

        # Validate output format
        if not is_input_video and is_output_video:
            console.print("[red]Error: Cannot create video output from audio input[/red]")
            sys.exit(1)

        file_type = "video" if is_input_video else "audio"
        console.print(f"[bold]Processing {file_type}:[/bold] {input}")
        console.print(f"[bold]Voice ID:[/bold] {voice}")
        console.print(f"[bold]Output path:[/bold] {output}")

        # Check for existing checkpoint if resume mode is enabled
        resuming_from_checkpoint = False
        checkpoint_state = None
        if self.resume and self.checkpoint_manager.has_checkpoint(input, voice):
            checkpoint_state = self.checkpoint_manager.load_checkpoint(input, voice)
            remaining = self.checkpoint_manager.get_remaining_segments(input, voice)
            if remaining:
                resuming_from_checkpoint = True
                console.print(
                    f"[cyan]Resuming from checkpoint: {len(checkpoint_state.processed_indices)} of "
                    f"{len(checkpoint_state.segments)} segments already processed[/cyan]"
                )
            else:
                console.print(
                    "[yellow]Checkpoint found but all segments complete. Starting fresh.[/yellow]"
                )
                self.checkpoint_manager.delete_checkpoint(input, voice)

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
                # Preview mode info
                if self.preview > 0:
                    console.print(
                        f"[yellow]Preview mode: processing first {self.preview:.1f} seconds only[/yellow]"
                    )

                # Step 1: Get audio (extract from video or use audio file directly)
                # Build ffmpeg duration limit args for preview mode
                duration_args = ["-t", str(self.preview)] if self.preview > 0 else []

                if is_input_video:
                    with self.progress_tracker.track_file_operation("Extracting audio from video"):
                        if self.preview > 0:
                            # Extract with duration limit for preview
                            preview_audio = os.path.join(temp_dir, "extracted_audio.wav")
                            cmd = [
                                "ffmpeg",
                                "-i",
                                input,
                                *duration_args,
                                "-vn",
                                "-acodec",
                                "pcm_s16le",
                                "-ar",
                                str(SAMPLE_RATE),
                                "-ac",
                                "1",
                                "-y",
                                preview_audio,
                            ]
                            subprocess.run(cmd, capture_output=True, check=True)
                            audio_path = preview_audio
                        else:
                            audio_path = self.audio_extractor.extract(
                                input, os.path.join(temp_dir, "extracted_audio.wav")
                            )
                else:
                    # For audio input, convert to WAV if needed
                    console.print("Processing audio file...")
                    if input.lower().endswith(".wav") and self.preview <= 0:
                        audio_path = input
                    else:
                        audio_path = os.path.join(temp_dir, "input_audio.wav")
                        cmd = [
                            "ffmpeg",
                            "-i",
                            input,
                            *duration_args,
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "44100",
                            "-y",
                            audio_path,
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)

                # Step 1.5: Apply noise reduction preprocessing (optional)
                if self.denoise > 0:
                    console.print(f"Applying noise reduction (strength: {self.denoise:.1%})...")
                    denoised_path = os.path.join(temp_dir, "denoised_audio.wav")
                    audio_path = self.audio_processor.reduce_noise_advanced(
                        audio_path, denoised_path, method="afftdn", strength=self.denoise
                    )
                    logger.debug(f"Noise reduction applied with strength {self.denoise}")

                # Step 2: Get segments (from checkpoint or analyze fresh)
                if resuming_from_checkpoint and checkpoint_state:
                    segments = checkpoint_state.segments
                    console.print(f"Using {len(segments)} segments from checkpoint")
                else:
                    console.print("Analyzing audio for optimal segmentation...")
                    segments = self.silence_analyzer.analyze(audio_path, self.seg_min, self.seg_max)
                    console.print(f"Found {len(segments)} segments")

                # Step 3: Split audio into segments
                with self.progress_tracker.track_file_operation("Splitting audio into segments"):
                    segment_paths = self.audio_segmenter.segment(
                        audio_path, segments, os.path.join(temp_dir, "segments")
                    )

                # Create or verify checkpoint
                if not resuming_from_checkpoint:
                    parameters = {
                        "seg_min": self.seg_min,
                        "seg_max": self.seg_max,
                        "normalize": self.normalize,
                        "target_db": self.target_db,
                        "compress": self.compress,
                    }
                    checkpoint_state = self.checkpoint_manager.create_checkpoint(
                        input, voice, segments, parameters, self.preview
                    )
                    console.print("[dim]Checkpoint created for resume capability[/dim]")

                # Step 4: Process each segment with ElevenLabs
                # Create output directories
                os.makedirs(os.path.join(temp_dir, "converted"), exist_ok=True)
                os.makedirs(os.path.join(temp_dir, "adjusted"), exist_ok=True)

                # Get already-processed segments from checkpoint
                cached_segments = {}
                if resuming_from_checkpoint:
                    cached_segments = self.checkpoint_manager.get_processed_segment_paths(
                        input, voice
                    )
                    logger.debug(f"Loaded {len(cached_segments)} cached segments from checkpoint")

                def process_segment(args: tuple[int, str, float, float]) -> tuple[int, str]:
                    """Process a single segment (for parallel execution)."""
                    i, segment_path, start, end = args
                    original_duration = end - start
                    converted_path = os.path.join(temp_dir, "converted", f"segment_{i:04d}.mp3")
                    adjusted_path = os.path.join(temp_dir, "adjusted", f"segment_{i:04d}.wav")

                    logger.debug(f"Processing segment {i}: {start:.2f}s - {end:.2f}s")
                    self.elevenlabs_client.speech_to_speech(segment_path, voice, converted_path)
                    self.audio_processor.adjust_duration(
                        converted_path, original_duration, adjusted_path
                    )

                    # Update checkpoint after successful processing
                    self.checkpoint_manager.update_checkpoint(input, voice, i, adjusted_path)

                    return i, adjusted_path

                # Build list of segment processing tasks (exclude already-processed)
                segment_tasks = [
                    (i, segment_path, start, end)
                    for i, (segment_path, (start, end)) in enumerate(
                        zip(segment_paths, segments, strict=False)
                    )
                    if i not in cached_segments
                ]

                # Start with cached segments
                converted_segments_map: dict[int, str] = dict(cached_segments)

                # Process remaining segments (parallel or sequential)
                if segment_tasks:
                    with self.progress_tracker.track_segments(
                        len(segment_paths), "Converting segments"
                    ) as update:
                        # Update progress for already-completed segments
                        if cached_segments:
                            update(
                                len(cached_segments),
                                description=f"Restored {len(cached_segments)} segments from checkpoint",
                            )

                        if self.parallel > 1:
                            # Parallel processing
                            mode = f"parallel ({self.parallel} workers)"
                            logger.info(f"Processing {len(segment_tasks)} segments in {mode}")
                            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                                futures = {
                                    executor.submit(process_segment, task): task[0]
                                    for task in segment_tasks
                                }
                                for future in as_completed(futures):
                                    idx, adjusted_path = future.result()
                                    converted_segments_map[idx] = adjusted_path
                                    update(
                                        1,
                                        description=f"Converted segment {len(converted_segments_map)}/{len(segment_paths)}",
                                    )
                        else:
                            # Sequential processing
                            for task in segment_tasks:
                                idx, adjusted_path = process_segment(task)
                                converted_segments_map[idx] = adjusted_path
                                update(
                                    1,
                                    description=f"Converting segment {idx + 1}/{len(segment_paths)}",
                                )
                else:
                    console.print("[green]All segments restored from checkpoint[/green]")

                # Ensure segments are in correct order for concatenation
                converted_segments = [converted_segments_map[i] for i in range(len(segments))]

                # Step 5: Concatenate processed segments
                with self.progress_tracker.track_file_operation("Reassembling audio"):
                    final_audio = self.audio_segmenter.concatenate(
                        converted_segments, os.path.join(temp_dir, "final_audio.wav")
                    )

                    # Apply dynamic range compression (optional)
                    if self.compress:
                        compressed_audio = self.audio_processor.compress_audio(
                            final_audio,
                            os.path.join(temp_dir, "compressed_audio.wav"),
                        )
                        logger.debug("Applied dynamic range compression")
                        processing_audio = compressed_audio
                    else:
                        processing_audio = final_audio

                    # Normalize audio (optional, enabled by default)
                    if self.normalize:
                        normalized_audio = self.audio_processor.normalize_audio(
                            processing_audio,
                            os.path.join(temp_dir, "normalized_audio.wav"),
                            target_db=self.target_db,
                        )
                        logger.debug(f"Normalized audio to {self.target_db} dB")
                    else:
                        normalized_audio = processing_audio
                        logger.debug("Skipping audio normalization")

                    # Apply post-processing effects if configured
                    if fx:
                        fx_config = None
                        if isinstance(fx, bool) or fx in ["1", "on", "true", "True", 1]:
                            # Use default config
                            if DEFAULT_CONFIG_PATH.exists():
                                fx_config = str(DEFAULT_CONFIG_PATH)
                                logger.info(f"Using default FX config: {DEFAULT_CONFIG_PATH}")
                            else:
                                logger.warning("Default FX config not found")
                        elif fx not in ["0", "off", "false", "False", 0, False]:
                            # Use custom config path
                            fx_config = str(fx)

                        if fx_config:
                            fx_output = os.path.join(temp_dir, "fx_audio.wav")
                            normalized_audio = self._apply_audio_fx(
                                normalized_audio, fx_config, fx_output
                            )

                # Step 6: Create output (remux video or save audio)
                if is_output_video:
                    with self.progress_tracker.track_file_operation("Creating final video"):
                        self.video_remuxer.remux(input, normalized_audio, output)
                else:
                    # For audio output
                    with self.progress_tracker.track_file_operation("Saving audio file"):
                        if output.lower().endswith(".wav"):
                            # Already in WAV format, just copy
                            import shutil

                            shutil.copy2(normalized_audio, output)
                        else:
                            # Convert to requested format
                            subprocess.run(
                                ["ffmpeg", "-i", normalized_audio, "-y", output],
                                check=True,
                                capture_output=True,
                            )

                # Delete checkpoint on successful completion
                self.checkpoint_manager.delete_checkpoint(input, voice)
                logger.debug("Checkpoint deleted after successful completion")

                # Calculate statistics
                elapsed_time = time.time() - start_time
                output_type = "video" if is_output_video else "audio"
                stats = {
                    "Processing time": f"{elapsed_time:.1f} seconds",
                    "Segments processed": len(segments),
                    f"Input {file_type}": os.path.basename(input),
                    f"Output {output_type}": os.path.basename(output),
                    "Voice used": voice,
                }
                if resuming_from_checkpoint:
                    stats["Resumed from checkpoint"] = "Yes"

                self.progress_tracker.print_summary(stats)
                console.print(f"\n[green]✓ {file_type.capitalize()} processing complete![/green]")

            except Exception as e:
                console.print(f"\n[red]Error during processing: {e}[/red]")
                if self.resume:
                    console.print(
                        "[yellow]Checkpoint saved. Use --resume to continue later.[/yellow]"
                    )
                logger.exception("Processing failed")
                sys.exit(1)


def _expand_input_paths(input_pattern: str | Path) -> list[Path]:
    """Expand input pattern to list of file paths.

    Supports glob patterns like *.mp4, videos/*.mov, etc.

    Args:
        input_pattern: Single file path or glob pattern

    Returns:
        List of resolved file paths
    """
    pattern = str(input_pattern)

    # Check if it contains glob characters
    if any(c in pattern for c in ["*", "?", "["]):
        # Expand glob pattern
        matches = sorted(glob_module.glob(pattern, recursive=True))
        return [Path(m) for m in matches if Path(m).is_file()]

    # Single file
    path = Path(pattern)
    return [path] if path.is_file() else []


def dub(
    input: str | Path,
    voice: str = DEFAULT_VOICE_ID,
    output: str | Path | None = None,
    api_key: str | None = None,
    verbose: bool = False,
    fx: str | bool | None = None,
    seg_min: float = DEFAULT_MIN_SEGMENT_DURATION,
    seg_max: float = DEFAULT_MAX_SEGMENT_DURATION,
    parallel: int = 1,
    preview: float = 0,
    normalize: bool = True,
    target_db: float = -23.0,
    compress: bool = False,
    resume: bool = False,
    denoise: float = 0.0,
):
    """eledubby - Voice dubbing tool using ElevenLabs speech-to-speech API.

    Args:
        input: Path to input video/audio file or glob pattern (e.g., "*.mp4", "videos/*.mov")
        voice: ElevenLabs voice ID to use for dubbing (default: ELEVENLABS_VOICE_ID environment variable)
        output: Path to output file or directory for batch mode (default: auto-generated)
        api_key: ElevenLabs API key override (defaults to env var)
        verbose: Enable verbose logging output
        fx: Audio effects - 0/off/False for none, 1/on/True for default config, or path to TOML config
        seg_min: Minimum segment duration in seconds (default: 10)
        seg_max: Maximum segment duration in seconds (default: 20)
        parallel: Number of parallel workers for segment processing (default: 1 = sequential)
        preview: Preview mode - process only first N seconds (default: 0 = full file)
        normalize: Enable EBU R128 loudness normalization (default: True)
        target_db: Target loudness in dB for normalization (default: -23.0 dB)
        compress: Apply dynamic range compression before normalization (default: False)
        resume: Resume from checkpoint if processing was interrupted (default: False)
        denoise: Noise reduction strength 0.0-1.0 (0=disabled, default: 0)
    """
    # Expand input pattern to list of files
    input_files = _expand_input_paths(input)

    if not input_files:
        console.print(f"[red]Error: No files found matching: {input}[/red]")
        sys.exit(1)

    # Batch mode: multiple files
    if len(input_files) > 1:
        console.print(f"[bold]Batch mode:[/bold] Found {len(input_files)} files to process")

        # Determine output directory
        if output:
            output_dir = Path(output)
            if output_dir.suffix:  # Has extension, treat as file pattern
                output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None

        dubber = EleDubby(
            verbose=verbose,
            seg_min=seg_min,
            seg_max=seg_max,
            api_key=api_key,
            parallel=parallel,
            preview=preview,
            normalize=normalize,
            target_db=target_db,
            compress=compress,
            resume=resume,
            denoise=denoise,
        )

        success_count = 0
        fail_count = 0
        for i, input_file in enumerate(input_files, 1):
            console.print(
                f"\n[bold cyan]Processing file {i}/{len(input_files)}:[/bold cyan] {input_file.name}"
            )

            # Generate output path for this file
            if output_dir:
                file_output = str(output_dir / f"{input_file.stem}_dubbed{input_file.suffix}")
            else:
                file_output = None

            try:
                dubber.process(str(input_file), voice, file_output, fx)
                success_count += 1
            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")
                fail_count += 1
                if verbose:
                    logger.exception("Processing failed")

        console.print(
            f"\n[bold]Batch complete:[/bold] {success_count} succeeded, {fail_count} failed"
        )
        return

    # Single file mode
    dubber = EleDubby(
        verbose=verbose,
        seg_min=seg_min,
        seg_max=seg_max,
        api_key=api_key,
        parallel=parallel,
        preview=preview,
        normalize=normalize,
        target_db=target_db,
        compress=compress,
        resume=resume,
        denoise=denoise,
    )
    dubber.process(str(input_files[0]), voice, output, fx)


def fx(
    input: str | Path,
    output: str | Path | None = None,
    api_key: str | None = None,
    config: str | None = None,
    verbose: bool = False,
):
    """Apply audio effects to a video or audio file without dubbing.

    Args:
        input: Path to input video or audio file
        output: Path to output file (default: auto-generated with same format as input)
        api_key: ElevenLabs API key override (unused; for interface consistency)
        config: Path to TOML config file for VST3 plugins (default: src/eledubby/config.toml)
        verbose: Enable verbose logging output
    """

    # Create processor without ElevenLabs requirement
    processor = EleDubby(verbose=verbose, require_elevenlabs=False, api_key=api_key)

    # Validate input
    if not os.path.exists(input):
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        sys.exit(1)

    # Determine if input is video or audio
    is_input_video = processor._is_video_file(input)

    # Determine output path and format
    output, is_output_video = processor._determine_output_format(
        input, output, is_input_video, force_audio=False
    )

    # Validate output format
    if not is_input_video and is_output_video:
        console.print("[red]Error: Cannot create video output from audio input[/red]")
        sys.exit(1)

    file_type = "video" if is_input_video else "audio"
    console.print(f"[bold]Processing {file_type}:[/bold] {input}")
    console.print(f"[bold]Output path:[/bold] {output}")

    # Determine config path
    fx_config = config or (str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)

    if not fx_config:
        console.print("[red]Error: No effects configuration found[/red]")
        console.print("Specify a config file with --config or create src/eledubby/config.toml")
        sys.exit(1)

    console.print(f"[bold]Effects config:[/bold] {fx_config}")

    start_time = time.time()

    with processor.temp_manager.temp_directory() as temp_dir:
        try:
            # Step 1: Get audio
            if is_input_video:
                with processor.progress_tracker.track_file_operation("Extracting audio from video"):
                    audio_path = processor.audio_extractor.extract(
                        input, os.path.join(temp_dir, "extracted_audio.wav")
                    )
            else:
                # For audio input, convert to WAV if needed
                console.print("Loading audio file...")
                if input.lower().endswith(".wav"):
                    audio_path = input
                else:
                    audio_path = os.path.join(temp_dir, "input_audio.wav")
                    subprocess.run(
                        ["ffmpeg", "-i", input, "-acodec", "pcm_s16le", "-ar", "44100", audio_path],
                        check=True,
                        capture_output=True,
                    )

            # Step 2: Apply effects
            fx_output = os.path.join(temp_dir, "fx_audio.wav")
            processed_audio = processor._apply_audio_fx(audio_path, fx_config, fx_output)

            # Step 3: Create output
            if is_output_video:
                with processor.progress_tracker.track_file_operation("Creating final video"):
                    processor.video_remuxer.remux(input, processed_audio, output)
            else:
                # For audio output
                with processor.progress_tracker.track_file_operation("Saving audio file"):
                    if output.lower().endswith(".wav"):
                        # Already in WAV format, just copy
                        import shutil

                        shutil.copy2(processed_audio, output)
                    else:
                        # Convert to requested format
                        subprocess.run(
                            ["ffmpeg", "-i", processed_audio, "-y", output],
                            check=True,
                            capture_output=True,
                        )

            # Calculate statistics
            elapsed_time = time.time() - start_time
            output_type = "video" if is_output_video else "audio"
            stats = {
                "Processing time": f"{elapsed_time:.1f} seconds",
                f"Input {file_type}": os.path.basename(input),
                f"Output {output_type}": os.path.basename(output),
                "Effects config": os.path.basename(fx_config),
            }

            processor.progress_tracker.print_summary(stats)
            console.print("\n[green]✓ Effects processing complete![/green]")

        except Exception as e:
            console.print(f"\n[red]Error during processing: {e}[/red]")
            logger.exception("Processing failed")
            sys.exit(1)


def _parse_voice_specs_from_voices_list(voices_list: str) -> list[tuple[str, str | None]]:
    """Parse `--voices_list a,b,c` into voice specs.

    Returns:
        List of (voice_id, api_key_override) tuples.
    """
    voice_ids = [voice_id.strip() for voice_id in voices_list.split(",")]
    return [(voice_id, None) for voice_id in voice_ids if voice_id]


def _parse_voice_specs_from_voices_path(voices_path: str | Path) -> list[tuple[str, str | None]]:
    """Parse `--voices_path` file lines into voice specs.

    Each non-empty line contains `voice_id` or `voice_id;api_key`.
    Lines starting with `#` are ignored.
    """
    path = Path(voices_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    specs: list[tuple[str, str | None]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        voice_part, *rest = line.split(";", maxsplit=1)
        voice_id = voice_part.strip()
        if not voice_id:
            continue

        api_key = rest[0].strip() if rest else None
        specs.append((voice_id, api_key or None))

    return specs


def _default_cast_output_dir(text_path: str | Path, output: str | Path | None) -> Path:
    """Compute output directory for `cast`."""
    if output:
        return Path(output)
    return Path.cwd() / Path(text_path).stem


def _safe_filename_component(value: str) -> str:
    """Make a safe filename component from a string."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(ch if ch in allowed else "_" for ch in value) or "_"


def _slug(text: str) -> str:
    """Create a safe slug from text for use in filenames.

    Args:
        text: Text to slugify

    Returns:
        Sanitized slug suitable for filenames
    """
    return sanitize_filename(slugify(text))


def _unique_output_path(path: Path) -> Path:
    """Return a non-existing path by adding a numeric suffix if needed."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(2, 10_000):
        candidate = path.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique output filename for: {path}")


def cast(
    text_path: str | Path,
    *,
    output: str | Path | None = None,
    voices_path: str | Path | None = None,
    voices_list: str | None = None,
    api_key: str | None = None,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Generate MP3s for a text using one or more voices.

    Voice selection:
    - No voices args: use all non-premade voices for the account
    - `--voices_path`: one voice per line, optional `;api_key` per line
    - `--voices_list`: comma-separated voice IDs

    Args:
        text_path: Path to text file to synthesize
        output: Output folder path
        voices_path: Text file with voice IDs (one per line, optionally `voice_id;api_key`)
        voices_list: Comma-separated voice IDs
        api_key: ElevenLabs API key
        force: Regenerate even if output file exists
        verbose: Enable verbose logging
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="<level>{message}</level>")
    else:
        logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    if voices_path and voices_list:
        console.print("[red]Error: Use only one of --voices_path or --voices_list[/red]")
        sys.exit(1)

    text_file = Path(text_path)
    if not text_file.exists():
        console.print(f"[red]Error: Text file not found: {text_file}[/red]")
        sys.exit(1)

    text = text_file.read_text(encoding="utf-8")
    if not text.strip():
        console.print(f"[red]Error: Text file is empty: {text_file}[/red]")
        sys.exit(1)

    # voice_specs: list of (voice_id, api_key, voice_name)
    # voice_name is used for slug in filename, can be None for explicit lists
    if voices_list:
        # Explicit list: (voice_id, api_key, None)
        parsed = _parse_voice_specs_from_voices_list(voices_list)
        voice_specs = [(vid, key, None) for vid, key in parsed]
    elif voices_path:
        # From file: (voice_id, api_key, None)
        parsed = _parse_voice_specs_from_voices_path(voices_path)
        voice_specs = [(vid, key, None) for vid, key in parsed]
    else:
        # All voices: filter out "premade" category
        try:
            client = ElevenLabsClient(api_key=api_key)
        except ValueError:
            console.print("[red]Error: Missing ElevenLabs API key[/red]")
            console.print("Provide --api_key or set ELEVENLABS_API_KEY")
            sys.exit(1)
        voices = client.list_voices()
        voice_specs = []
        skipped_premade = 0
        for voice in voices:
            if not voice.get("id"):
                continue
            # Skip premade voices
            if voice.get("category", "").lower() == "premade":
                skipped_premade += 1
                continue
            voice_specs.append((voice["id"], api_key, voice.get("name", "")))
        if skipped_premade > 0:
            console.print(f"[dim]Skipped {skipped_premade} premade voices[/dim]")

    if not voice_specs:
        console.print("[red]Error: No voices selected[/red]")
        sys.exit(1)

    output_dir = _default_cast_output_dir(text_file, output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clients_by_key: dict[str | None, ElevenLabsClient] = {}
    generated = 0
    skipped = 0
    failed = 0

    for voice_id, per_voice_key, voice_name in voice_specs:
        effective_key = per_voice_key or api_key
        if effective_key not in clients_by_key:
            try:
                clients_by_key[effective_key] = ElevenLabsClient(api_key=effective_key)
            except ValueError:
                console.print(f"[red]Error: Missing API key for voice {voice_id}[/red]")
                console.print("Provide --api_key or specify `voice_id;api_key` in --voices_path")
                sys.exit(1)

        # Build filename: {voice_id}-{voice_slug}.mp3
        if voice_name:
            voice_slug = _slug(voice_name)
            out_name = f"{voice_id}-{voice_slug}.mp3"
        else:
            # No name available, just use voice_id
            out_name = f"{voice_id}.mp3"

        out_path = output_dir / out_name

        # Skip if file exists and not forcing
        if out_path.exists() and not force:
            console.print(f"[dim]Skipping (exists):[/dim] {voice_id} → {out_path}")
            skipped += 1
            continue

        console.print(f"[bold]Voice:[/bold] {voice_id} → {out_path}")

        try:
            clients_by_key[effective_key].text_to_speech(
                text=text,
                voice_id=voice_id,
                output_path=str(out_path),
            )
            generated += 1
        except Exception as e:
            # Extract meaningful error message
            error_msg = str(e)
            if "detected_captcha_voice" in error_msg:
                console.print("  [yellow]Skipped (requires verification)[/yellow]")
            elif "status_code: 403" in error_msg:
                console.print("  [yellow]Skipped (access denied)[/yellow]")
            elif "status_code: 429" in error_msg:
                console.print("  [yellow]Skipped (rate limited)[/yellow]")
            else:
                console.print(f"  [red]Failed: {error_msg[:100]}[/red]")
            failed += 1
            continue

    console.print(f"\n[green]Generated {generated} files[/green]", end="")
    if skipped > 0 or failed > 0:
        parts = []
        if skipped > 0:
            parts.append(f"skipped {skipped} existing")
        if failed > 0:
            parts.append(f"failed {failed}")
        console.print(f" [dim]({', '.join(parts)})[/dim]")
    else:
        console.print()


def _get_vst3_search_paths() -> list[Path]:
    """Get system-specific VST3 plugin search paths."""
    search_paths = []
    system = platform.system()

    if system == "Darwin":  # macOS
        home = Path.home()
        search_paths = [
            home / "Library/Audio/Plug-Ins/VST3",
            Path("/Library/Audio/Plug-Ins/VST3"),
        ]
    elif system == "Windows":
        search_paths = [
            Path("C:/Program Files/Common Files/VST3"),
            Path("C:/Program Files (x86)/Common Files/VST3"),
        ]
    elif system == "Linux":
        home = Path.home()
        search_paths = [home / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]

    return [p for p in search_paths if p.exists()]


def _discover_vst3_plugins() -> list[dict]:
    """Discover all installed VST3 plugins.

    Returns:
        List of plugin info dicts with name and path
    """
    plugins = []
    search_paths = _get_vst3_search_paths()

    for search_path in search_paths:
        for vst_file in search_path.rglob("*.vst3"):
            plugins.append(
                {
                    "name": vst_file.stem,
                    "path": str(vst_file),
                    "location": str(search_path),
                }
            )

    # Sort by name
    return sorted(plugins, key=lambda p: p["name"].lower())


def plugins(
    json: bool = False,
) -> None:
    """List installed VST3 plugins.

    Args:
        json: Output as JSON instead of table
    """
    import json as json_module

    plugin_list = _discover_vst3_plugins()
    search_paths = _get_vst3_search_paths()

    if not plugin_list:
        console.print("[yellow]No VST3 plugins found[/yellow]")
        console.print(f"Searched: {', '.join(str(p) for p in search_paths)}")
        return

    if json:
        print(json_module.dumps(plugin_list, indent=2))
    else:
        console.print(f"[bold]Found {len(plugin_list)} VST3 plugins:[/bold]\n")
        for plugin in plugin_list:
            console.print(f"  [cyan]{plugin['name']}[/cyan]")
            console.print(f"    {plugin['path']}")


def _resolve_plugin_by_name(plugin_name: str) -> str | None:
    """Find a plugin path by name (partial match supported).

    Args:
        plugin_name: Plugin name or path

    Returns:
        Full path to plugin or None if not found
    """
    # If it's already an absolute path that exists, return it
    if os.path.isabs(plugin_name) and os.path.exists(plugin_name):
        return plugin_name

    # Search for matching plugins
    plugin_list = _discover_vst3_plugins()
    plugin_name_lower = plugin_name.lower()

    # Exact match first
    for p in plugin_list:
        if (
            p["name"].lower() == plugin_name_lower
            or p["name"].lower() == plugin_name_lower + ".vst3"
        ):
            return p["path"]

    # Partial match
    for p in plugin_list:
        if plugin_name_lower in p["name"].lower():
            return p["path"]

    return None


def plugin_params(
    plugin: str,
    json: bool = False,
    toml: bool = False,
) -> None:
    """Show parameters for a VST3 plugin.

    Use this to discover what parameters a plugin exposes for use in preset TOML files.

    Args:
        plugin: Plugin name or path (partial match supported)
        json: Output as JSON
        toml: Output as TOML preset template
    """
    import json as json_module

    resolved = _resolve_plugin_by_name(plugin)
    if not resolved:
        console.print(f"[red]Plugin not found: {plugin}[/red]")
        console.print("Use 'eledubby plugins' to list available plugins")
        return

    try:
        logger.info(f"Loading plugin: {resolved}")
        vst_plugin = pedalboard.load_plugin(resolved)
    except Exception as e:
        console.print(f"[red]Failed to load plugin: {e}[/red]")
        return

    # Get plugin parameters - pedalboard plugins expose params as attributes
    # Filter out private attributes and methods
    params = {}
    plugin_name = Path(resolved).name

    for attr_name in dir(vst_plugin):
        if attr_name.startswith("_"):
            continue
        try:
            attr_val = getattr(vst_plugin, attr_name)
            # Skip methods and non-numeric/string values
            if callable(attr_val):
                continue
            if isinstance(attr_val, (int, float, str, bool)):
                params[attr_name] = attr_val
        except Exception:
            continue

    if not params:
        console.print(f"[yellow]No accessible parameters found for {plugin_name}[/yellow]")
        return

    if json:
        print(
            json_module.dumps(
                {"plugin": plugin_name, "path": resolved, "parameters": params}, indent=2
            )
        )
    elif toml:
        # Output as TOML preset template
        console.print(f"# Preset template for {plugin_name}")
        console.print(f'["{plugin_name}"]')
        for name, value in sorted(params.items()):
            if isinstance(value, str):
                console.print(f'{name} = "{value}"')
            elif isinstance(value, bool):
                console.print(f"{name} = {'true' if value else 'false'}")
            else:
                console.print(f"{name} = {value}")
    else:
        console.print(f"[bold]Parameters for {plugin_name}:[/bold]")
        console.print(f"Path: {resolved}\n")
        for name, value in sorted(params.items()):
            val_type = type(value).__name__
            console.print(f"  [cyan]{name}[/cyan] = {value}  [dim]({val_type})[/dim]")


def presets(
    path: str | Path | None = None,
) -> None:
    """List available FX preset TOML files.

    Searches for .toml preset files in the examples directory and optionally a custom path.

    Args:
        path: Additional directory to search for presets
    """
    # Search locations
    search_dirs = [
        Path(__file__).parent.parent.parent / "examples",  # Package examples
        Path.cwd() / "presets",  # Local presets folder
        Path.cwd(),  # Current directory
    ]

    if path:
        search_dirs.insert(0, Path(path))

    found_presets = []
    seen_paths = set()

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for toml_file in search_dir.glob("*.toml"):
            if toml_file.resolve() in seen_paths:
                continue
            seen_paths.add(toml_file.resolve())

            # Try to extract description from first comment line
            description = ""
            try:
                with open(toml_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") and not line.startswith("# this_file"):
                            description = line.lstrip("#").strip()
                            break
                        if line and not line.startswith("#"):
                            break
            except Exception:
                pass

            found_presets.append(
                {
                    "name": toml_file.stem,
                    "path": str(toml_file),
                    "description": description,
                }
            )

    if not found_presets:
        console.print("[yellow]No preset files found[/yellow]")
        console.print("Create .toml files in ./presets/ or use --path to specify a directory")
        return

    console.print(f"[bold]Found {len(found_presets)} presets:[/bold]\n")
    for preset in sorted(found_presets, key=lambda p: p["name"]):
        console.print(f"  [cyan]{preset['name']}[/cyan]")
        if preset["description"]:
            console.print(f"    {preset['description']}")
        console.print(f"    [dim]{preset['path']}[/dim]")


def quality(
    input: str | Path,
    compare: str | Path | None = None,
    json: bool = False,
    verbose: bool = False,
) -> None:
    """Analyze audio quality and detect issues.

    Checks for clipping, DC offset, low SNR, excessive silence, and other quality issues.

    Args:
        input: Path to audio file to analyze
        compare: Optional second file to compare against (e.g., original vs processed)
        json: Output as JSON instead of formatted text
        verbose: Show all metrics even if passing
    """
    import json as json_module

    if verbose:
        logger.enable("eledubby")

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]File not found: {input}[/red]")
        sys.exit(1)

    checker = AudioQualityChecker()

    if compare:
        compare_path = Path(compare)
        if not compare_path.exists():
            console.print(f"[red]Comparison file not found: {compare}[/red]")
            sys.exit(1)

        result = checker.compare(str(input_path), str(compare_path))

        if json:
            print(json_module.dumps(result, indent=2))
        else:
            console.print("[bold]Quality Comparison Report[/bold]\n")

            console.print("[cyan]Original:[/cyan]")
            _print_quality_report(result["original"], verbose)

            console.print("\n[cyan]Processed:[/cyan]")
            _print_quality_report(result["processed"], verbose)

            console.print("\n[cyan]Changes:[/cyan]")
            delta = result["delta"]
            console.print(f"  Peak: {delta['peak_db']:+.1f} dB")
            console.print(f"  RMS: {delta['rms_db']:+.1f} dB")
            console.print(f"  Dynamic Range: {delta['dynamic_range_db']:+.1f} dB")
    else:
        report = checker.analyze(str(input_path))

        if json:
            print(json_module.dumps(report.to_dict(), indent=2))
        else:
            console.print(f"[bold]Quality Report: {input_path.name}[/bold]\n")
            _print_quality_report(report.to_dict(), verbose)

            if report.passed:
                console.print("\n[green]✓ All quality checks passed[/green]")
            else:
                console.print(f"\n[red]✗ {len(report.issues)} issue(s) found[/red]")


def _print_quality_report(report: dict, verbose: bool = False) -> None:
    """Print formatted quality report."""
    console.print(f"  Duration: {report['duration']:.2f}s")
    console.print(f"  Sample Rate: {report['sample_rate']} Hz")
    console.print(f"  Channels: {report['channels']}")
    console.print(f"  Bit Depth: {report['bit_depth']}")

    if verbose or not report.get("passed", True):
        console.print(f"  Peak: {report['peak_db']:.1f} dB")
        console.print(f"  RMS: {report['rms_db']:.1f} dB")
        if report.get("snr_db") is not None:
            console.print(f"  SNR: {report['snr_db']:.1f} dB")
        console.print(f"  Dynamic Range: {report['dynamic_range_db']:.1f} dB")
        console.print(f"  Crest Factor: {report['crest_factor']:.2f}")
        console.print(f"  DC Offset: {report['dc_offset']:.6f}")
        console.print(f"  Silence: {report['silence_ratio'] * 100:.1f}%")
        console.print(f"  Clipping: {report['clipping_ratio'] * 100:.3f}%")
        if report.get("loudness_lufs") is not None:
            console.print(f"  Loudness: {report['loudness_lufs']:.1f} LUFS")

    if report.get("issues"):
        console.print("\n  [yellow]Issues:[/yellow]")
        for issue in report["issues"]:
            console.print(f"    • {issue}")


def voices(
    api_key: str | None = None,
    json: bool = False,
    detailed: bool = False,
) -> None:
    """List available ElevenLabs voices.

    Args:
        api_key: ElevenLabs API key override (defaults to env var)
        json: Output as JSON instead of CSV
        detailed: Include additional voice metadata (description, preview_url, labels)
    """
    import csv
    import io
    import json as json_module

    try:
        client = ElevenLabsClient(api_key=api_key)
    except ValueError:
        console.print("[red]Error: Missing ElevenLabs API key[/red]")
        console.print("Provide --api_key or set ELEVENLABS_API_KEY")
        sys.exit(1)

    voice_list = client.list_voices(detailed=detailed)

    if not voice_list:
        console.print("[yellow]No voices found[/yellow]")
        sys.exit(0)

    if json:
        print(json_module.dumps(voice_list, indent=2))
    else:
        # CSV output
        if detailed:
            fieldnames = ["id", "name", "category", "description", "preview_url", "labels"]
        else:
            fieldnames = ["id", "name", "category"]

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for voice in voice_list:
            # Convert labels dict to string for CSV
            if "labels" in voice and isinstance(voice["labels"], dict):
                voice = voice.copy()
                voice["labels"] = ";".join(f"{k}={v}" for k, v in voice["labels"].items())
            writer.writerow(voice)
        print(output.getvalue(), end="")


def checkpoints(
    clean: bool = False,
    max_age_days: int = 7,
    json: bool = False,
) -> None:
    """List and manage processing checkpoints.

    Checkpoints allow resuming interrupted dubbing jobs. Use --clean to remove
    old checkpoints.

    Args:
        clean: Remove checkpoints older than max_age_days
        max_age_days: Maximum age in days for cleanup (default: 7)
        json: Output as JSON instead of table
    """
    import json as json_module
    from datetime import datetime

    manager = CheckpointManager()

    if clean:
        removed = manager.cleanup_old_checkpoints(max_age_days)
        console.print(
            f"[green]Removed {removed} checkpoints older than {max_age_days} days[/green]"
        )
        return

    checkpoint_list = manager.list_checkpoints()

    if not checkpoint_list:
        console.print("[yellow]No checkpoints found[/yellow]")
        return

    if json:
        print(json_module.dumps(checkpoint_list, indent=2, default=str))
    else:
        console.print(f"[bold]Found {len(checkpoint_list)} checkpoint(s):[/bold]\n")
        for cp in checkpoint_list:
            modified = datetime.fromtimestamp(cp["modified"]).strftime("%Y-%m-%d %H:%M")
            console.print(f"  [cyan]{cp['job_id']}[/cyan]")
            console.print(f"    Voice: {cp['voice_id']}")
            console.print(f"    Progress: {cp['progress']} segments")
            console.print(f"    Modified: {modified}")


def recover(
    input: str | Path,
    voice: str,
    output: str | Path | None = None,
) -> None:
    """Recover partial results from an interrupted dubbing job.

    Creates an audio file with all processed segments and silence placeholders
    for segments that weren't completed. Useful for previewing partial progress
    or salvaging work from an interrupted job.

    Args:
        input: Path to the original input file (used to identify checkpoint)
        voice: ElevenLabs voice ID (used to identify checkpoint)
        output: Output path for recovered audio (default: input_partial.wav)
    """
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input}[/red]")
        sys.exit(1)

    manager = CheckpointManager()

    # Check if checkpoint exists
    if not manager.has_checkpoint(input_path, voice):
        console.print("[red]Error: No checkpoint found for this input/voice combination[/red]")
        console.print("Use 'eledubby checkpoints' to see available checkpoints")
        sys.exit(1)

    # Get progress info
    progress = manager.get_checkpoint_progress(input_path, voice)

    if not progress["recoverable"]:
        console.print("[red]Error: No segments have been processed yet[/red]")
        sys.exit(1)

    console.print("[bold]Checkpoint Progress:[/bold]")
    console.print(
        f"  Processed: {progress['processed_segments']}/{progress['total_segments']} segments"
    )
    console.print(f"  Complete: {progress['percent_complete']:.1f}%")
    console.print(
        f"  Duration: {progress['processed_duration']:.1f}s / {progress['total_duration']:.1f}s"
    )

    # Set default output path
    if not output:
        output = input_path.parent / f"{input_path.stem}_partial.wav"
    output = Path(output)

    console.print(f"\n[bold]Recovering partial result to:[/bold] {output}")

    try:
        result_path, processed, total = manager.recover_partial_result(input_path, voice, output)
        console.print(f"[green]✓ Recovered {processed}/{total} segments to: {result_path}[/green]")
        console.print("[dim]Note: Missing segments are replaced with silence[/dim]")
    except Exception as e:
        console.print(f"[red]Error recovering partial result: {e}[/red]")
        sys.exit(1)


DEFAULT_PREVIEW_TEXT = "Hello! This is a preview of my voice. How does it sound to you?"


def preview(
    voice: str | None = None,
    text: str = DEFAULT_PREVIEW_TEXT,
    output: str | Path | None = None,
    api_key: str | None = None,
    play: bool = False,
) -> None:
    """Preview a voice by generating a short audio sample.

    Generate a short TTS sample to test how a voice sounds before committing
    to a full dubbing job. If no voice is specified, lists available voices.

    Args:
        voice: Voice ID to preview (if not provided, lists available voices)
        text: Text to speak (default: "Hello! This is a preview...")
        output: Output file path (default: preview_{voice_id}.mp3)
        api_key: ElevenLabs API key override (defaults to env var)
        play: Play the audio after generation (requires ffplay)
    """
    try:
        client = ElevenLabsClient(api_key=api_key)
    except ValueError:
        console.print("[red]Error: Missing ElevenLabs API key[/red]")
        console.print("Provide --api_key or set ELEVENLABS_API_KEY")
        sys.exit(1)

    # If no voice specified, list available voices
    if not voice:
        console.print("[bold]Available voices:[/bold]\n")
        voice_list = client.list_voices(detailed=True)
        for v in voice_list:
            console.print(f"  [cyan]{v['id']}[/cyan] - {v['name']}")
            if v.get("category"):
                console.print(f"    Category: {v['category']}")
        console.print("\n[dim]Use: eledubby preview --voice <voice_id>[/dim]")
        return

    # Validate voice
    if not client.validate_voice_id(voice):
        console.print(f"[red]Error: Voice ID not found: {voice}[/red]")
        console.print("Use 'eledubby voices' to see available voices")
        sys.exit(1)

    # Generate output path
    output = Path.cwd() / f"preview_{voice}.mp3" if not output else Path(output)

    console.print(f"[bold]Voice ID:[/bold] {voice}")
    console.print(f"[bold]Text:[/bold] {text[:50]}{'...' if len(text) > 50 else ''}")
    console.print(f"[bold]Output:[/bold] {output}")

    # Generate preview
    try:
        client.text_to_speech(
            text=text,
            voice_id=voice,
            output_path=str(output),
        )
        console.print(f"[green]✓ Preview saved to: {output}[/green]")

        # Play audio if requested
        if play:
            console.print("[dim]Playing audio...[/dim]")
            try:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", str(output)],
                    capture_output=True,
                    check=True,
                )
            except FileNotFoundError:
                console.print(
                    "[yellow]ffplay not found - install ffmpeg to enable playback[/yellow]"
                )
            except subprocess.CalledProcessError:
                console.print("[yellow]Playback failed[/yellow]")

    except Exception as e:
        console.print(f"[red]Error generating preview: {e}[/red]")
        sys.exit(1)

#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["elevenlabs", "python-dotenv", "fire", "rich", "loguru", "numpy", "scipy", "pedalboard", "toml"]
# ///
# this_file: src/eledubby/eledubby.py

"""
eledubby - Voice dubbing tool using ElevenLabs speech-to-speech API.

Takes an input video and replaces the audio with a new voice using ElevenLabs API.
Performs intelligent audio segmentation and maintains perfect timing synchronization.
"""

import mimetypes
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
import pedalboard
import toml
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
DEFAULT_MIN_SEGMENT_DURATION = 10.0  # seconds
DEFAULT_MAX_SEGMENT_DURATION = 20.0  # seconds
SILENCE_THRESHOLD_DB = -40  # dB below max
SAMPLE_RATE = 16000  # Hz for processing
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


class EleDubby:
    """Main class for video dubbing functionality."""

    def __init__(self, verbose: bool = False, seg_min: float = DEFAULT_MIN_SEGMENT_DURATION, 
                 seg_max: float = DEFAULT_MAX_SEGMENT_DURATION, require_elevenlabs: bool = True):
        """Initialize the dubbing tool.

        Args:
            verbose: Enable verbose logging output
            seg_min: Minimum segment duration in seconds
            seg_max: Maximum segment duration in seconds
            require_elevenlabs: Whether ElevenLabs API is required
        """
        self.verbose = verbose
        self.seg_min = seg_min
        self.seg_max = seg_max
        self._setup_logging()
        self._check_dependencies(require_elevenlabs)

        # Initialize components
        self.audio_extractor = AudioExtractor(SAMPLE_RATE)
        self.silence_analyzer = SilenceAnalyzer(SILENCE_THRESHOLD_DB)
        self.audio_segmenter = AudioSegmenter()
        self.audio_processor = AudioProcessor()
        if require_elevenlabs:
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

        # Check environment variable only if required
        if require_elevenlabs and not os.getenv("ELEVENLABS_API_KEY"):
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
                Path("/Library/Audio/Plug-Ins/VST3")
            ]
        elif system == "Windows":
            search_paths = [
                Path("C:/Program Files/Common Files/VST3"),
                Path("C:/Program Files (x86)/Common Files/VST3")
            ]
        elif system == "Linux":
            home = Path.home()
            search_paths = [
                home / ".vst3",
                Path("/usr/lib/vst3"),
                Path("/usr/local/lib/vst3")
            ]
            
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

    def _apply_audio_fx(self, audio_path: str, fx_config: Union[str, dict], output_path: str) -> str:
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
            with open(config_path, 'r') as f:
                config = toml.load(f)
        else:
            config = fx_config
            
        if not config:
            logger.info("No effects configured, skipping post-processing")
            return audio_path
            
        # Load audio
        with pedalboard.io.AudioFile(audio_path, 'r') as f:
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
            with pedalboard.io.AudioFile(output_path, 'w', samplerate, processed.shape[0]) as f:
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
            if mime_type.startswith('video/'):
                return True
            elif mime_type.startswith('audio/'):
                return False
        
        # Fallback to extension checking
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'}
        
        ext = Path(file_path).suffix.lower()
        if ext in video_extensions:
            return True
        elif ext in audio_extensions:
            return False
        
        # If uncertain, try to probe with ffmpeg
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                 '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', file_path],
                capture_output=True, text=True, check=False
            )
            return result.stdout.strip() == 'video'
        except:
            # Default to assuming it's audio if we can't determine
            return False
    
    def _determine_output_format(self, input_path: str, output_path: str | None, 
                                 is_input_video: bool, force_audio: bool = False) -> tuple[str, bool]:
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
                logger.warning("Input is audio, output must also be audio. Changing extension to .wav")
                output_path = str(Path(output_path).with_suffix('.wav'))
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

    def process(self, input: str, voice: str = DEFAULT_VOICE_ID, output: str | None = None, 
                fx: Union[str, bool, None] = None):
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
                # Step 1: Get audio (extract from video or use audio file directly)
                if is_input_video:
                    with self.progress_tracker.track_file_operation("Extracting audio from video"):
                        audio_path = self.audio_extractor.extract(
                            input, os.path.join(temp_dir, "extracted_audio.wav")
                        )
                else:
                    # For audio input, convert to WAV if needed
                    console.print("Processing audio file...")
                    if input.lower().endswith('.wav'):
                        audio_path = input
                    else:
                        audio_path = os.path.join(temp_dir, "input_audio.wav")
                        subprocess.run(
                            ['ffmpeg', '-i', input, '-acodec', 'pcm_s16le', '-ar', '44100', audio_path],
                            check=True, capture_output=True
                        )

                # Step 2: Analyze audio and find segment boundaries
                console.print("Analyzing audio for optimal segmentation...")
                segments = self.silence_analyzer.analyze(
                    audio_path, self.seg_min, self.seg_max
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
                            normalized_audio = self._apply_audio_fx(normalized_audio, fx_config, fx_output)

                # Step 6: Create output (remux video or save audio)
                if is_output_video:
                    with self.progress_tracker.track_file_operation("Creating final video"):
                        self.video_remuxer.remux(input, normalized_audio, output)
                else:
                    # For audio output
                    with self.progress_tracker.track_file_operation("Saving audio file"):
                        if output.lower().endswith('.wav'):
                            # Already in WAV format, just copy
                            import shutil
                            shutil.copy2(normalized_audio, output)
                        else:
                            # Convert to requested format
                            subprocess.run(
                                ['ffmpeg', '-i', normalized_audio, '-y', output],
                                check=True, capture_output=True
                            )

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

                self.progress_tracker.print_summary(stats)
                console.print(f"\n[green]✓ {file_type.capitalize()} processing complete![/green]")

            except Exception as e:
                console.print(f"\n[red]Error during processing: {e}[/red]")
                logger.exception("Processing failed")
                sys.exit(1)


def dub(
    input: str | Path,
    voice: str = DEFAULT_VOICE_ID,
    output: str | Path | None = None,
    verbose: bool = False,
    fx: Union[str, bool, None] = None,
    seg_min: float = DEFAULT_MIN_SEGMENT_DURATION,
    seg_max: float = DEFAULT_MAX_SEGMENT_DURATION,
):
    """eledubby - Voice dubbing tool using ElevenLabs speech-to-speech API.

    Args:
        input: Path to input video or audio file
        voice: ElevenLabs voice ID to use for dubbing (default: ELEVENLABS_VOICE_ID environment variable)
        output: Path to output file (default: auto-generated). Can be video or audio.
        verbose: Enable verbose logging output
        fx: Audio effects - 0/off/False for none, 1/on/True for default config, or path to TOML config
        seg_min: Minimum segment duration in seconds (default: 10)
        seg_max: Maximum segment duration in seconds (default: 20)
    """

    dubber = EleDubby(verbose=verbose, seg_min=seg_min, seg_max=seg_max)
    dubber.process(input, voice, output, fx)


def fx(
    input: str | Path,
    output: str | Path | None = None,
    config: str | None = None,
    verbose: bool = False,
):
    """Apply audio effects to a video or audio file without dubbing.

    Args:
        input: Path to input video or audio file
        output: Path to output file (default: auto-generated with same format as input)
        config: Path to TOML config file for VST3 plugins (default: src/eledubby/config.toml)
        verbose: Enable verbose logging output
    """
    
    # Create processor without ElevenLabs requirement
    processor = EleDubby(verbose=verbose, require_elevenlabs=False)
    
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
    if config:
        fx_config = config
    else:
        fx_config = str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None
    
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
                if input.lower().endswith('.wav'):
                    audio_path = input
                else:
                    audio_path = os.path.join(temp_dir, "input_audio.wav")
                    subprocess.run(
                        ['ffmpeg', '-i', input, '-acodec', 'pcm_s16le', '-ar', '44100', audio_path],
                        check=True, capture_output=True
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
                    if output.lower().endswith('.wav'):
                        # Already in WAV format, just copy
                        import shutil
                        shutil.copy2(processed_audio, output)
                    else:
                        # Convert to requested format
                        subprocess.run(
                            ['ffmpeg', '-i', processed_audio, '-y', output],
                            check=True, capture_output=True
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
            console.print(f"\n[green]✓ Effects processing complete![/green]")
            
        except Exception as e:
            console.print(f"\n[red]Error during processing: {e}[/red]")
            logger.exception("Processing failed")
            sys.exit(1)

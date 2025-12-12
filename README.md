# Eledubby

[![PyPI version](https://badge.fury.io/py/eledubby.svg)](https://badge.fury.io/py/eledubby)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Eledubby** is a Python tool for automatic voice dubbing of videos using the ElevenLabs API. It performs speech-to-speech conversion, allowing you to replace the original voice in a video with a different voice while preserving timing and synchronization.

## 1. Features

- 🎥 **Automatic voice dubbing** - Replace voices in videos with high-quality AI voices
- 🎯 **Timing preservation** - Maintains original timing by intelligently padding or cropping audio
- 🔊 **Smart audio segmentation** - Splits audio into optimal segments based on silence detection
- 🚀 **Batch processing** - Processes multiple segments in parallel for faster results
- 📊 **Progress tracking** - Real-time progress updates with detailed status information
- 🛡️ **Error resilience** - Automatic retries and graceful error handling
- 🎛️ **Customizable parameters** - Fine-tune silence detection and segmentation settings
- 🎸 **VST3 Plugin Support** - Apply professional audio effects using VST3 plugins via Pedalboard
- 🔄 **Resume capability** - Resume interrupted jobs from checkpoints (`--resume`)
- 🔇 **Noise reduction** - Preprocess audio with noise reduction (`--denoise`)
- 🎤 **Voice preview** - Test voices before dubbing with the `preview` command

## 2. Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 3. Installation

### 3.1. Prerequisites

- Python 3.8 or higher
- [FFmpeg](https://ffmpeg.org/) installed and available in your system PATH
- An [ElevenLabs API key](https://elevenlabs.io/)

### 3.2. Install from PyPI

```bash
pip install eledubby
```

### 3.3. Install from Source

```bash
git clone https://github.com/twardoch/eledubby.git
cd eledubby
pip install -e .
```

### 3.4. Development Installation

```bash
git clone https://github.com/twardoch/eledubby.git
cd eledubby
uv venv
uv sync
```

## 4. Quick Start

1. **Set up your ElevenLabs API key:**

   Create a `.env` file in your project directory:
   ```bash
   echo "ELEVENLABS_API_KEY=your_api_key_here" > .env
   ```

   Or set it as an environment variable:
   ```bash
   export ELEVENLABS_API_KEY=your_api_key_here
   ```

2. **Run eledubby on a video:**

   ```bash
   eledubby --input video.mp4 --voice voice_id --output dubbed_video.mp4
   ```

   To use the default voice:
   ```bash
   eledubby --input video.mp4 --output dubbed_video.mp4
   ```

   With audio post-processing:
   ```bash
   eledubby --input video.mp4 --output dubbed_video.mp4 --fx on
   ```

## 5. Usage

### 5.1. Command Line Interface

The package provides two main commands:

#### 5.1.1. `dub` - Voice Dubbing with ElevenLabs

```bash
eledubby dub [OPTIONS]
```

**Options:**
- `--input` (required): Path to the input video/audio file or glob pattern (e.g., `*.mp4`, `videos/*.mov`)
- `--output`: Path to output file or directory for batch mode (default: auto-generated)
- `--voice`: ElevenLabs voice ID (default: ELEVENLABS_VOICE_ID environment variable)
- `--api_key`: ElevenLabs API key override (default: ELEVENLABS_API_KEY environment variable)
- `--fx`: Audio post-processing effects:
  - `0`, `off`, or `False`: No post-processing
  - `1`, `on`, or `True`: Use default config from `src/eledubby/config.toml`
  - Path to TOML file: Use custom VST3 plugin configuration
- `--seg_min`: Minimum segment duration in seconds (default: 10)
- `--seg_max`: Maximum segment duration in seconds (default: 20)
- `--parallel`: Number of parallel workers for segment processing (default: 1 = sequential)
- `--preview`: Preview mode - process only first N seconds (default: 0 = full file)
- `--normalize`: Enable EBU R128 loudness normalization (default: True)
- `--target_db`: Target loudness in dB for normalization (default: -23.0)
- `--compress`: Apply dynamic range compression before normalization (default: False)
- `--denoise`: Noise reduction strength (0.0-1.0, default: 0 = disabled)
- `--resume`: Resume from checkpoint if available (default: False)
- `--verbose`: Enable verbose logging

#### 5.1.2. `fx` - Audio Effects Only (No Dubbing)

```bash
eledubby fx [OPTIONS]
```

**Options:**
- `--input` (required): Path to the input video or audio file
- `--output`: Path to the output file (default: auto-generated with same format)
- `--api_key`: ElevenLabs API key override (unused; for interface consistency)
- `--config`: Path to TOML config file for VST3 plugins (default: src/eledubby/config.toml)
- `--verbose`: Enable verbose logging

#### 5.1.3. `plugins` - List Installed VST3 Plugins

```bash
eledubby plugins [OPTIONS]
```

**Options:**
- `--json`: Output as JSON instead of formatted list

#### 5.1.4. `voices` - List Available Voices

```bash
eledubby voices [OPTIONS]
```

**Options:**
- `--api_key`: ElevenLabs API key override (default: ELEVENLABS_API_KEY environment variable)
- `--json`: Output as JSON instead of CSV
- `--detailed`: Include additional voice metadata (description, preview_url, labels)

#### 5.1.5. `checkpoints` - Manage Processing Checkpoints

```bash
eledubby checkpoints [OPTIONS]
```

**Options:**
- `--json`: Output as JSON instead of formatted list
- `--clean`: Remove old checkpoints
- `--max_age_days`: Maximum age for checkpoints when cleaning (default: 7)

#### 5.1.6. `preview` - Test Voice Before Dubbing

```bash
eledubby preview [OPTIONS]
```

**Options:**
- `--voice`: ElevenLabs voice ID (lists voices if not provided)
- `--text`: Custom text to synthesize (default: sample preview text)
- `--output`: Output path for preview audio (default: `preview_<voice_id>.mp3`)
- `--api_key`: ElevenLabs API key override
- `--play`: Play audio after generation using ffplay (default: False)

#### 5.1.7. `recover` - Recover Partial Results from Interrupted Jobs

```bash
eledubby recover --input INPUT --voice VOICE_ID [OPTIONS]
```

**Options:**
- `--input` (required): Path to the original input file (used to identify checkpoint)
- `--voice` (required): ElevenLabs voice ID (used to identify checkpoint)
- `--output`: Output path for recovered audio (default: `input_partial.wav`)

#### 5.1.8. `cast` - Generate MP3s for Many Voices

```bash
eledubby cast --text_path TEXT.txt [OPTIONS]
```

**Options:**
- `--text_path` (required): Path to the text file to synthesize
- `--output`: Output folder path (default: `basename_without_extension(text_path)` in current directory)
- `--api_key`: ElevenLabs API key override (default: ELEVENLABS_API_KEY environment variable)
- `--voices_list`: Comma-separated voice IDs (e.g. `a,b,c`)
- `--voices_path`: Text file with one voice per line; optionally `voice_id;api_key` per line
- `--force`: Regenerate files even if they already exist (default: skip existing)
- `--verbose`: Enable verbose logging

**Notes:**
- Output files use format `{voice_id}-{voice_slug}.mp3` where voice_slug is a sanitized version of the voice name
- When using all voices (no `--voices_list` or `--voices_path`), "premade" voices are automatically skipped
- Existing output files are skipped unless `--force` is specified

#### 5.1.9. `plugin-params` - Show Plugin Parameters

```bash
eledubby plugin-params PLUGIN_NAME [OPTIONS]
```

**Options:**
- `--plugin` (required): Plugin name or path (partial match supported)
- `--json`: Output as JSON
- `--toml`: Output as TOML preset template (copy-paste into preset file)

#### 5.1.10. `presets` - List Available Presets

```bash
eledubby presets [OPTIONS]
```

**Options:**
- `--path`: Additional directory to search for presets

Searches for `.toml` preset files in `examples/`, `./presets/`, and current directory.

#### 5.1.11. `quality` - Audio Quality Analysis

```bash
eledubby quality --input AUDIO_FILE [OPTIONS]
```

Analyze audio files for quality issues like clipping, DC offset, low SNR, excessive silence, and more.

**Options:**
- `--input` (required): Path to audio file to analyze
- `--compare`: Optional second file to compare (e.g., original vs processed)
- `--json`: Output as JSON instead of formatted text
- `--verbose`: Show all metrics even if passing

**Metrics Checked:**
- Peak and RMS levels
- Signal-to-noise ratio (SNR)
- Clipping detection
- DC offset
- Dynamic range
- Silence ratio
- Loudness (LUFS)

#### 5.1.12. Examples

**Plugins Examples:**

List installed VST3 plugins:
```bash
eledubby plugins
```

List plugins as JSON:
```bash
eledubby plugins --json
```

Show parameters for a plugin:
```bash
eledubby plugin-params "Compressor"
```

Generate TOML preset template:
```bash
eledubby plugin-params "Compressor" --toml > my_preset.toml
```

List available FX presets:
```bash
eledubby presets
```

**Quality Examples:**

Check audio file quality:
```bash
eledubby quality --input audio.wav
```

Compare original and processed audio:
```bash
eledubby quality --input original.wav --compare processed.wav
```

Get detailed metrics as JSON:
```bash
eledubby quality --input audio.wav --json --verbose
```

**Voices Examples:**

List all voices (CSV):
```bash
eledubby voices
```

List all voices (JSON):
```bash
eledubby voices --json
```

List voices with extra metadata:
```bash
eledubby voices --detailed --json
```

**Dubbing Examples:**

Basic video dubbing:
```bash
eledubby dub --input interview.mp4 --voice rachel_voice_id --output interview_dubbed.mp4
```

Audio file dubbing (audio-to-audio):
```bash
eledubby dub --input podcast.mp3 --voice alex_voice_id --output podcast_dubbed.wav
```

With custom segment durations:
```bash
eledubby dub --input podcast.mp4 --output podcast_dubbed.mp4 \
  --seg_min 8 --seg_max 15 --verbose
```

With audio post-processing:
```bash
eledubby dub --input video.mp4 --output enhanced_video.mp4 \
  --fx ./my_effects.toml --verbose
```

Parallel processing for faster conversion (4 workers):
```bash
eledubby dub --input long_video.mp4 --output dubbed.mp4 \
  --parallel 4 --verbose
```

Preview mode (process first 30 seconds only):
```bash
eledubby dub --input video.mp4 --output preview.mp4 \
  --preview 30 --voice my_voice_id
```

Batch processing with glob patterns:
```bash
eledubby dub --input "videos/*.mp4" --output ./dubbed/ \
  --voice my_voice_id --parallel 4
```

Disable volume normalization:
```bash
eledubby dub --input video.mp4 --output dubbed.mp4 \
  --normalize=False
```

Custom loudness target:
```bash
eledubby dub --input video.mp4 --output dubbed.mp4 \
  --target_db -16.0
```

Apply dynamic range compression:
```bash
eledubby dub --input video.mp4 --output dubbed.mp4 \
  --compress --normalize
```

Apply noise reduction before dubbing:
```bash
eledubby dub --input video.mp4 --output dubbed.mp4 \
  --denoise 0.5 --voice my_voice_id
```

Resume an interrupted job:
```bash
eledubby dub --input video.mp4 --output dubbed.mp4 \
  --voice my_voice_id --resume
```

**Checkpoint Examples:**

List all checkpoints:
```bash
eledubby checkpoints
```

List checkpoints as JSON:
```bash
eledubby checkpoints --json
```

Clean old checkpoints (older than 7 days):
```bash
eledubby checkpoints --clean
```

**Preview Examples:**

List available voices:
```bash
eledubby preview
```

Preview a voice:
```bash
eledubby preview --voice my_voice_id
```

Preview with custom text and play:
```bash
eledubby preview --voice my_voice_id --text "Hello world" --play
```

**Recovery Examples:**

Recover partial results from an interrupted job:
```bash
eledubby recover --input video.mp4 --voice my_voice_id
```

Recover to a specific output path:
```bash
eledubby recover --input video.mp4 --voice my_voice_id --output partial_result.wav
```

**Effects-Only Examples:**

Apply effects to video (no dubbing):
```bash
eledubby fx --input video.mp4 --output enhanced_video.mp4 \
  --config ./voice_enhancement.toml
```

Apply effects to audio file:
```bash
eledubby fx --input recording.wav --output processed_recording.wav
```

Extract and process audio from video:
```bash
eledubby fx --input video.mp4 --output audio_only.wav \
  --config ./mastering.toml
```

**Cast Examples:**

Use all voices for the account:
```bash
eledubby cast --text_path script.txt --api_key YOUR_KEY
```

Use explicit voice IDs:
```bash
eledubby cast --text_path script.txt --voices_list a,b,c --api_key YOUR_KEY
```

Use a voices file (optional per-voice API key):
```bash
eledubby cast --text_path script.txt --voices_path voices.txt
```

Force regeneration of existing files:
```bash
eledubby cast --text_path script.txt --force
```

### 5.2. Python API

```python
from eledubby.eledubby import dub, fx

# Basic video dubbing
dub(
    input="video.mp4",
    output="dubbed_video.mp4",
    voice="voice_id_here"
)

# Audio file dubbing
dub(
    input="podcast.mp3",
    output="dubbed_podcast.wav",
    voice="voice_id_here"
)

# With audio post-processing
dub(
    input="video.mp4",
    output="enhanced_video.mp4",
    voice="voice_id_here",
    fx=True,  # Use default config
    seg_min=8,
    seg_max=15
)

# Apply effects only (no dubbing)
fx(
    input="video.mp4",
    output="enhanced_video.mp4",
    config="./voice_enhancement.toml",
    verbose=True
)

# Extract audio from video and apply effects
fx(
    input="video.mp4",
    output="extracted_audio.wav",  # Audio output from video input
    config="./mastering.toml"
)
```

### 5.3. VST3 Plugin Configuration

Create a TOML file to configure VST3 plugins for audio post-processing:

```toml
# Example: voice_enhancement.toml

# Compression to even out volume
["Compressor.vst3"]
threshold_db = -20.0
ratio = 4.0
attack_ms = 1.0
release_ms = 100.0

# EQ for voice clarity
["Pro-Q 3.vst3"]
preset = "Vocal Presence"

# Subtle reverb for natural sound
["ValhallaRoom.vst3"]
mix = 0.1
room_size = 0.3

# Limiter to prevent clipping
["Limiter.vst3"]
threshold_db = -0.5
release_ms = 50.0
```

**VST3 Plugin Path Resolution:**
- **macOS**: `~/Library/Audio/Plug-Ins/VST3` and `/Library/Audio/Plug-Ins/VST3`
- **Windows**: `C:\Program Files\Common Files\VST3` and `C:\Program Files (x86)\Common Files\VST3`
- **Linux**: `~/.vst3`, `/usr/lib/vst3`, `/usr/local/lib/vst3`

Plugins are applied in the order specified in the configuration file.

```python
# Example with custom parameters
process_video(
    input_path="video.mp4",
    output_path="dubbed_video.mp4",
    voice_id="voice_id_here",
    silence_threshold=-35,
    min_segment_duration=8,
    max_segment_duration=15,
    model="eleven_multilingual_v2",
    stability=0.6,
    similarity_boost=0.8
)
```

## 6. How It Works

Eledubby uses a sophisticated pipeline to perform voice dubbing while maintaining synchronization:

### 6.1. **Audio Extraction**
   - Extracts audio track from the input video using FFmpeg
   - Preserves original audio format and quality settings

### 6.2. **Silence Detection & Analysis**
   - Analyzes the audio waveform to detect periods of silence
   - Uses configurable threshold (dB) and minimum duration parameters
   - Creates a silence map for intelligent segmentation

### 6.3. **Smart Segmentation**
   - Splits audio into segments between 10-20 seconds (configurable)
   - Finds optimal split points at the longest silence within each window
   - Scores silence periods based on both duration and silence level
   - Ensures segments are within the acceptable duration range

### 6.4. **Speech-to-Speech Conversion**
   - Sends each segment to ElevenLabs API for voice conversion
   - Uses the specified voice ID and model parameters
   - Processes multiple segments in parallel for efficiency

### 6.5. **Timing Preservation**
   - Compares the duration of converted segments with originals
   - Pads shorter segments with silence to match original timing
   - Crops longer segments if necessary (with intelligent trimming)
   - Maintains frame-accurate synchronization

### 6.6. **Audio Reassembly**
   - Concatenates all processed segments in order
   - Ensures seamless transitions between segments
   - Produces a final audio track with exact original duration

### 6.7. **Video Remuxing**
   - Replaces the original audio track with the dubbed version
   - Preserves all video streams and metadata
   - Outputs the final dubbed video file

## 7. Architecture

The project is organized into modular components:

```
eledubby/
├── api/               # ElevenLabs API integration
│   └── elevenlabs_client.py
├── audio/            # Audio processing modules
│   ├── analyzer.py   # Silence detection and analysis
│   ├── extractor.py  # Audio extraction from video
│   ├── processor.py  # Main audio processing pipeline
│   └── segmenter.py  # Audio segmentation logic
├── video/            # Video processing modules
│   └── remuxer.py    # Video remuxing operations
├── utils/            # Utility modules
│   ├── progress.py   # Progress tracking
│   └── temp_manager.py # Temporary file management
└── adamdubpy.py     # Main CLI entry point
```

### 7.1. Key Components

- **ElevenLabsClient**: Manages API communication with retry logic and error handling
- **AudioAnalyzer**: Performs silence detection using scipy signal processing
- **AudioSegmenter**: Implements the intelligent segmentation algorithm
- **AudioProcessor**: Orchestrates the entire audio processing pipeline
- **VideoRemuxer**: Handles video operations using FFmpeg
- **ProgressTracker**: Provides real-time progress updates using Rich

## 8. Configuration

### 8.1. Environment Variables

- `ELEVENLABS_API_KEY`: Your ElevenLabs API key (required)
- `ELEDUBBY_TEMP_DIR`: Custom temporary directory (optional)
- `ELEDUBBY_MAX_RETRIES`: Maximum API retry attempts (default: 3)
- `ELEDUBBY_RETRY_DELAY`: Delay between retries in seconds (default: 1)

### 8.2. Voice IDs

You can find available voice IDs in your ElevenLabs account or use the API to list them:

```python
from elevenlabs import voices

# List all available voices
for voice in voices():
    print(f"{voice.voice_id}: {voice.name}")
```

### 8.3. Models

Supported ElevenLabs models:
- `eleven_multilingual_v2` (default) - Best quality, supports multiple languages
- `eleven_monolingual_v1` - English only, faster processing
- `eleven_turbo_v2` - Fastest processing, good quality

## 9. API Reference

### 9.1. Main Functions

#### 9.1.1. `process_video()`

```python
def process_video(
    input_path: str,
    output_path: str,
    voice_id: str = os.getenv("ELEVENLABS_VOICE_ID"),
    silence_threshold: float = -40,
    min_silence_duration: float = 0.5,
    min_segment_duration: float = 10,
    max_segment_duration: float = 20,
    padding_duration: float = 0.1,
    model: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    max_workers: int = 3,
    api_key: Optional[str] = None
) -> None:
    """
    Process a video file by replacing its audio with a dubbed version.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        voice_id: ElevenLabs voice ID to use
        silence_threshold: Threshold for silence detection in dB
        min_silence_duration: Minimum duration of silence in seconds
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds
        padding_duration: Padding to add to segments in seconds
        model: ElevenLabs model to use
        stability: Voice stability parameter (0.0-1.0)
        similarity_boost: Voice similarity boost parameter (0.0-1.0)
        max_workers: Maximum number of parallel workers
        api_key: ElevenLabs API key (uses env var if not provided)
    """
```

### 9.2. Module Classes

#### 9.2.1. `AudioProcessor`

Main class for audio processing operations:

```python
processor = AudioProcessor(
    api_key="your_api_key",
    voice_id="voice_id",
    model="eleven_multilingual_v2",
    max_workers=3
)

# Process audio file
processor.process_audio(
    input_audio_path="audio.wav",
    output_audio_path="dubbed_audio.wav"
)
```

#### 9.2.2. `AudioAnalyzer`

Analyzes audio for silence detection:

```python
analyzer = AudioAnalyzer(
    silence_threshold=-40,
    min_silence_duration=0.5
)

# Detect silence periods
silence_periods = analyzer.detect_silence(audio_data, sample_rate)
```

## 10. Why Eledubby?

### 10.1. The Problem

Traditional dubbing requires voice actors, recording studios, and extensive post-production work. Even with modern AI voice synthesis, maintaining synchronization between video and dubbed audio remains challenging.

### 10.2. The Solution

Eledubby automates the entire dubbing process while solving key synchronization challenges:

1. **Intelligent Segmentation**: Instead of processing the entire audio at once (which can cause drift), Eledubby splits audio at natural pause points.

2. **Timing Preservation**: Each segment is processed individually and adjusted to match the original duration, preventing accumulative timing errors.

3. **Quality Optimization**: By working with smaller segments, the AI voice synthesis produces more consistent and natural results.

4. **Parallel Processing**: Multiple segments are processed simultaneously, significantly reducing total processing time.

### 10.3. Technical Approach

The core innovation is the silence-based segmentation algorithm:

```python
# Pseudocode for segmentation logic
for window in sliding_windows(audio, size=20s, step=10s):
    silence_periods = detect_silence(window)
    best_split = max(silence_periods, key=lambda s: s.duration * s.silence_level)
    segments.append(split_at(audio, best_split))
```

This ensures:
- Natural breaking points that don't cut off speech
- Consistent segment sizes for reliable API processing
- Flexibility to handle various speech patterns

## 11. Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### 11.1. Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/twardoch/eledubby.git
   cd eledubby
   ```

2. Create a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   uv sync
   ```

4. Run tests:
   ```bash
   pytest
   ```

5. Run linting:
   ```bash
   ruff check .
   mypy .
   ```

### 11.2. Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Add docstrings to all public functions and classes
- Write tests for new functionality

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 13. Acknowledgments

- [ElevenLabs](https://elevenlabs.io/) for providing the amazing voice synthesis API
- [FFmpeg](https://ffmpeg.org/) for reliable video/audio processing
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- The Python community for excellent libraries and tools

## 14. Troubleshooting

### 14.1. Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed: `ffmpeg -version`
   - Add FFmpeg to your system PATH
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

2. **API key errors**
   - Verify your API key is correct: `eledubby voices` should list voices
   - Check your ElevenLabs account has sufficient credits
   - Ensure the key has speech-to-speech permissions

3. **Memory issues with large videos**
   - Use `--preview 60` to test with first 60 seconds
   - Reduce `--parallel` workers (try `--parallel 1`)
   - Close other applications to free RAM
   - Consider processing in segments manually

4. **Audio sync issues**
   - Try different segment durations: `--seg_min 8 --seg_max 15`
   - Ensure input video has constant frame rate (use `ffprobe` to check)
   - Convert variable frame rate videos first: `ffmpeg -i input.mp4 -r 30 output.mp4`

5. **Voice sounds unnatural**
   - Try different voices: `eledubby voices --detailed --json`
   - Adjust segment sizes for more natural pauses
   - Use `--compress` for more even dynamics
   - Apply post-processing with `--fx`

6. **VST3 plugins not found**
   - Run `eledubby plugins` to see detected plugins
   - Install plugins to standard system paths
   - Use absolute paths in config TOML files

7. **Processing takes too long**
   - Use `--parallel 4` (or higher) for concurrent processing
   - Use `--preview` to test settings first
   - Consider shorter segment durations

8. **Job interrupted**
   - Use `--resume` to continue from checkpoint
   - Run `eledubby checkpoints` to see saved progress
   - Checkpoints auto-save after each segment

9. **Background noise in source**
   - Use `--denoise 0.5` for moderate noise reduction
   - Higher values (0.7-1.0) for noisy recordings
   - Combines FFT-based and non-local means filtering

### 14.2. Error Messages

| Error | Solution |
|-------|----------|
| `ELEVENLABS_API_KEY not set` | Export your API key: `export ELEVENLABS_API_KEY=your_key` |
| `No files found matching` | Check your glob pattern or file path |
| `Rate limited` | Wait a moment, tool will auto-retry |
| `Voice ID not found` | Use `eledubby voices` to list valid IDs |
| `Insufficient disk space` | Clear temp files, ensure ~10x video size free |

### 14.3. Getting Help

- Check the [Issues](https://github.com/twardoch/eledubby/issues) page
- Create a new issue with detailed information about your problem
- Include error messages, system information, and sample files if possible

---

Made with ❤️ by [Adam Twardoch](https://github.com/twardoch)

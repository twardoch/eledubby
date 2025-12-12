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
- `--input` (required): Path to the input video or audio file
- `--output`: Path to the output file (default: auto-generated)
- `--voice`: ElevenLabs voice ID (default: ELEVENLABS_VOICE_ID environment variable)
- `--api_key`: ElevenLabs API key override (default: ELEVENLABS_API_KEY environment variable)
- `--fx`: Audio post-processing effects:
  - `0`, `off`, or `False`: No post-processing
  - `1`, `on`, or `True`: Use default config from `src/eledubby/config.toml`
  - Path to TOML file: Use custom VST3 plugin configuration
- `--seg_min`: Minimum segment duration in seconds (default: 10)
- `--seg_max`: Maximum segment duration in seconds (default: 20)
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

#### 5.1.3. `cast` - Generate MP3s for Many Voices

```bash
eledubby cast --text_path TEXT.txt [OPTIONS]
```

**Options:**
- `--text_path` (required): Path to the text file to synthesize
- `--output`: Output folder path (default: `basename_without_extension(text_path)` in current directory)
- `--api_key`: ElevenLabs API key override (default: ELEVENLABS_API_KEY environment variable)
- `--voices_list`: Comma-separated voice IDs (e.g. `a,b,c`)
- `--voices_path`: Text file with one voice per line; optionally `voice_id;api_key` per line
- `--verbose`: Enable verbose logging

#### 5.1.4. Examples

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

2. **API key errors**
   - Verify your API key is correct
   - Check your ElevenLabs account has sufficient credits

3. **Memory issues with large videos**
   - Process videos in smaller chunks
   - Reduce the number of parallel workers
   - Use a machine with more RAM

4. **Audio sync issues**
   - Try adjusting the padding duration
   - Experiment with different segment durations
   - Check that the input video has constant frame rate

### 14.2. Getting Help

- Check the [Issues](https://github.com/twardoch/eledubby/issues) page
- Create a new issue with detailed information about your problem
- Include error messages, system information, and sample files if possible

---

Made with ❤️ by [Adam Twardoch](https://github.com/twardoch)

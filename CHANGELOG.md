# CHANGELOG.md

## [Unreleased]

### Added
- New `cast` command to synthesize a text file across multiple voices, supporting `--voices_list` and `--voices_path` (with optional per-line `voice_id;api_key`).

### Changed
- `dub` and `fx` commands accept optional `--api_key` to override `ELEVENLABS_API_KEY` (unused for `fx`).

## [0.2.0] - 2025-08-05

### Added
- **VST3 Plugin Support via Pedalboard**
  - Integration with `pedalboard` library for professional audio post-processing
  - Support for VST3 plugins on macOS, Windows, and Linux
  - Automatic plugin path resolution for system-specific directories
  - TOML-based configuration for plugin parameters
  - Chain multiple effects in specified order

- **New `fx` Command**
  - Apply audio effects without dubbing
  - Works with both video and audio input files
  - Extract audio from video with effects processing
  - `--config` argument for custom VST3 configurations
  - Default configuration support

- **Enhanced Audio/Video File Support**
  - `dub` command now accepts audio files (MP3, WAV, FLAC, AAC, etc.)
  - Audio-to-audio dubbing without video processing
  - Flexible output format handling:
    - Video → Video (dubbed)
    - Video → Audio (extract and dub)
    - Audio → Audio (dub)
  - Smart file type detection using MIME types and ffprobe

- **New Command-Line Arguments**
  - `--fx`: Control audio post-processing in `dub` command
    - `0/off/False`: No post-processing
    - `1/on/True`: Use default config
    - Path to TOML: Custom configuration
  - `--seg_min`: Minimum segment duration (default: 10s)
  - `--seg_max`: Maximum segment duration (default: 20s)
  - `--config`: VST3 configuration path for `fx` command

### Changed
- CLI now uses subcommands: `eledubby dub` and `eledubby fx`
- Output paths are now auto-generated based on input type
- ElevenLabs API is optional for `fx` command
- Improved error messages and validation

### Fixed
- `__main__.py` was calling undefined `main` instead of proper functions
- File path tracking (`this_file` comments) throughout codebase

## [0.1.0] - 2025-08-05

### Added
- Initial implementation of adamdubpy voice dubbing tool
- Core functionality:
  - Audio extraction from video files using ffmpeg
  - Intelligent silence detection and segmentation (10-20 second segments)
  - Speech-to-speech conversion using ElevenLabs API
  - Precise timing preservation with padding/cropping
  - Audio normalization and reassembly
  - Video remuxing with converted audio track
- Modular architecture:
  - `audio/` module for extraction, analysis, segmentation, and processing
  - `api/` module for ElevenLabs integration with retry logic
  - `video/` module for remuxing operations
  - `utils/` module for progress tracking and temp file management
- CLI interface using fire with arguments:
  - `--input` (required): Input video path
  - `--voice` (optional): ElevenLabs voice ID
  - `--output` (optional): Output video path
  - `--verbose` (optional): Detailed logging
- Features:
  - Real-time progress tracking with rich console output
  - Comprehensive error handling and recovery
  - Automatic voice ID validation with fallback
  - Disk space checking before processing
  - Detailed processing statistics
- Documentation:
  - Comprehensive README with installation and usage instructions
  - Inline code documentation with docstrings
  - CLAUDE.md for AI assistant guidance

### Technical Details
- Python 3.12+ with uv script runner support
- Dependencies: elevenlabs, python-dotenv, fire, rich, loguru, numpy, scipy
- Requires ffmpeg for audio/video operations
- Supports various video formats (MP4, AVI, MOV, etc.)
- Processes at 2-5x real-time speed typically

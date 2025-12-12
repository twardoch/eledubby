# WORK.md - Current Work Progress

## Completed Features

### CLI Commands
- `dub` - Voice dubbing with ElevenLabs speech-to-speech
- `fx` - Audio effects only (no dubbing)
- `cast` - Generate MP3s across multiple voices
- `voices` - List available ElevenLabs voices (CSV/JSON)
- `plugins` - List installed VST3 plugins
- `checkpoints` - List and manage processing checkpoints
- `preview` - Test voice before dubbing with TTS sample
- `recover` - Recover partial results from interrupted jobs

### Core Features
- Parallel segment processing (`--parallel N`)
- Preview mode (`--preview N` seconds)
- Batch processing with glob patterns (`--input "*.mp4"`)
- EBU R128 loudness normalization (`--normalize`, `--target_db`)
- Dynamic range compression (`--compress`)
- Noise reduction preprocessing (`--denoise 0.0-1.0`)
- Resume capability (`--resume` flag)
- VST3 plugin support via pedalboard
- Multi-account API key support

### Infrastructure
- 150 unit tests (audio, api, utils, integration, e2e, checkpoint, preview, benchmarks, memory monitoring, recovery)
- GitHub Actions CI/CD (.github/workflows/ci.yml)
- Docker containerization (Dockerfile)
- Example FX configurations (examples/*.toml)
- Troubleshooting guide in README (Section 14)

## Test Status

All 159 tests passing (lint-free)

## Recent Session Work

### Completed This Session
1. **Audio Quality Checker** (`eledubby quality`)
   - New `quality` CLI command for audio quality analysis
   - New `AudioQualityChecker` class in `src/eledubby/audio/quality.py`
   - Metrics: peak/RMS levels, SNR, clipping, DC offset, dynamic range, silence ratio, LUFS
   - Comparison mode to compare original vs processed audio
   - JSON output for programmatic use
   - 9 new tests for quality assessment
   - Test count now at 159

2. **Memory Usage Optimization** (`SilenceAnalyzer`)
   - Streaming analysis for large files (>50MB)
   - Files ≤50MB: fast full-memory analysis
   - Files >50MB: chunk-based streaming with 10-second chunks
   - Two-pass approach: first pass finds global max, second calculates silence scores
   - Added `soundfile` as optional dependency for efficient streaming I/O
   - Graceful fallback if soundfile not installed

3. **Plugin Preset Management** (`eledubby plugin-params`, `eledubby presets`)
   - New `plugin-params` command to show VST3 plugin parameters
   - Supports `--json` and `--toml` output formats
   - `--toml` generates ready-to-use preset template
   - New `presets` command to list available FX preset files
   - Searches examples/, ./presets/, and current directory
   - Updated README with new command documentation

3. **Cast Command Enhancements** (`eledubby cast`)
   - Skip "premade" voices when iterating over all voices
   - New filename format: `{voice_id}-{voice_slug}.mp3` using slugify + pathvalidate
   - Added `--force` flag to regenerate existing output files (default: skip existing)
   - Graceful error handling: skips voices that fail API calls (captcha, 403, 429)
   - Added dependencies: `python-slugify[unidecode]`, `pathvalidate`
   - Updated README documentation with new options and notes

3. **Memory Usage Monitoring** (`tests/test_eledubby/test_benchmarks.py::TestMemoryMonitoring`)
   - Silence analyzer memory usage test (< 100MB for 2-min audio)
   - Checkpoint state memory usage test (< 5MB for 1000 segments)
   - Audio array memory efficiency test
   - Progress tracker memory leak test
   - 4 new memory monitoring tests

3. **Partial Result Recovery** (`eledubby recover` command)
   - New `recover_partial_result()` method in CheckpointManager
   - New `get_checkpoint_progress()` method for detailed progress info
   - New `recover` CLI command to extract processed segments
   - Creates audio with silence placeholders for missing segments
   - 3 new tests for recovery functionality

### Previously This Session
1. **Voice Preview Command** (`eledubby preview`)
   - Lists available voices when no voice specified
   - Generates TTS audio sample for voice testing
   - Custom text and output path support
   - Optional playback with ffplay (`--play` flag)
   - 7 tests for preview functionality

2. **README Documentation Update**
   - Added `--resume` and `--denoise` options to dub command docs
   - Added `checkpoints` command documentation
   - Added `preview` command documentation
   - Added examples for all new features
   - Updated troubleshooting section

3. **Performance Benchmarks** (`tests/test_eledubby/test_benchmarks.py`)
   - Silence analyzer performance test (1 minute audio < 2s)
   - Checkpoint serialization benchmarks
   - Hash computation performance test (10MB < 0.5s)
   - Progress tracker performance test
   - ProcessingState serialization benchmarks
   - Memory usage tests for large segment lists
   - 5-minute audio processing test
   - 7 benchmark tests total

### Previously Completed
1. **Checkpoint System** (`--resume` flag)
   - Created `CheckpointManager` in `src/eledubby/utils/checkpoint.py`
   - Saves processing state after each segment
   - Allows resuming interrupted jobs
   - Added `checkpoints` CLI command to list/clean checkpoints
   - 16 tests for checkpoint functionality

2. **Noise Reduction** (`--denoise` parameter)
   - Added `reduce_noise()` and `reduce_noise_advanced()` to AudioProcessor
   - Uses FFmpeg's `afftdn` (FFT-based) and `anlmdn` (non-local means) filters
   - Strength parameter 0.0-1.0 for user control
   - 6 tests for noise reduction

## Next Priorities (from TODO.md)

### High Value
- [x] Memory usage optimization for large files (completed - streaming analysis)
- [x] Memory usage monitoring (completed)
- [x] Plugin preset management (completed)
- [x] Automated quality checks (completed - `eledubby quality`)

### Medium Value
- [ ] AU (Audio Unit) plugin support on macOS
- [ ] API documentation
- [ ] Source language auto-detection

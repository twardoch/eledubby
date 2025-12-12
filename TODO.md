# TODO.md - Eledubby Future Implementation Tasks

## Performance and Reliability

- [x] Implement parallel segment processing for faster conversion
- [x] Add threading pool for concurrent API calls
- [x] Add batch processing for multiple files with glob pattern support
- [x] Optimize memory usage for large files (streaming analysis for >50MB files)
- [x] Create checkpoint system for resume capability (`--resume` flag, `checkpoints` command)
- [x] Implement partial result recovery (`eledubby recover` command)

## Advanced Audio Features

- [x] Add auto-discovery of installed VST3 plugins (`eledubby plugins` command)
- [x] Implement plugin preset management (`eledubby plugin-params`, `eledubby presets` commands)
- [ ] Support AU (Audio Unit) plugins on macOS
- [x] Add automatic volume leveling (EBU R128 normalization with `--normalize` and `--target_db`)
- [x] Implement noise reduction preprocessing (`--denoise 0.0-1.0` strength)
- [x] Add dynamic range compression (`--compress` flag)
- [ ] Create ML-based scene detection for better splits
- [ ] Implement language-aware segmentation
- [ ] Add musical phrase detection for music videos

## Multi-Language and Voice

- [ ] Implement source language auto-detection
- [ ] Add language-specific voice selection
- [ ] Support cross-language dubbing
- [x] Create voice preview/testing mode (`eledubby preview` command)
- [ ] Add custom voice profiles
- [ ] Implement voice blending for multiple speakers
- [ ] Add speaker diarization
- [ ] Generate subtitles from original audio
- [ ] Sync subtitles with dubbed audio

## User Experience

- [x] Add preview mode (first N seconds)
- [ ] Create side-by-side comparison view
- [ ] Implement real-time parameter adjustment
- [ ] Add audio quality assessment scores
- [ ] Create timing accuracy reports
- [ ] Implement project-based settings
- [ ] Add configuration templates
- [ ] Create import/export for settings

## GUI and Platform

- [ ] Develop desktop GUI application
- [ ] Add drag-and-drop support
- [ ] Implement waveform visualization
- [ ] Create built-in audio player with A/B comparison
- [x] Add Docker containerization
- [x] Create GitHub Actions integration
- [ ] Implement web-based interface
- [ ] Add API for third-party integration

## Advanced Video Features

- [ ] Analyze video for lip sync optimization
- [ ] Implement scene-aware processing
- [ ] Add ambient sound preservation
- [ ] Support background music handling
- [ ] Add support for WebM, MKV formats
- [ ] Implement streaming format support (HLS, DASH)
- [ ] Preserve HDR video
- [ ] Support multi-track audio

## Testing and Quality

- [x] Write comprehensive unit tests (123 tests covering audio, api, utils, integration, e2e modules)
- [x] Create integration test suite for dub command
- [x] Add end-to-end testing (test_e2e.py with mocked API)
- [x] Implement performance benchmarks (`tests/test_eledubby/test_benchmarks.py`)
- [x] Add memory usage monitoring (`tests/test_eledubby/test_benchmarks.py::TestMemoryMonitoring`)
- [x] Create automated quality checks (`eledubby quality` command, `AudioQualityChecker`)
- [x] Set up CI/CD pipeline (GitHub Actions)

## Documentation

- [ ] Write detailed API documentation
- [ ] Create video tutorials
- [x] Add more example configurations (examples/*.toml)
- [x] Write troubleshooting guide (README section 14)
- [ ] Create plugin development guide
- [ ] Add best practices documentation

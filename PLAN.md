# PLAN.md - Eledubby Voice Dubbing Tool

When you output files e.g. with the 'cast' tool, make default paths like f"{voice_id}-{voice_slug}.mp3" where voice_slug is the voice_name processed through: 

```
# python-slugify[unidecode] pathvalidate
from pathvalidate import sanitize_filename
from slugify import slugify

def slug(text): 
	return sanitize_filename(slugify(text))
```

## Project Overview

**Eledubby** is a Python-based voice dubbing and audio processing tool that performs speech-to-speech conversion on video and audio files using ElevenLabs API, with professional audio post-processing capabilities via VST3 plugins.

## Future Enhancements

### Phase 1: Performance and Reliability Improvements

1. **Parallel Processing**

- Implement concurrent segment processing for faster conversion
- Use threading pool for API calls
- Optimize memory usage for large files

2. **Resume Capability**

- Save processing state to checkpoint files
- Allow resuming interrupted conversions
- Implement partial result recovery

3. **Batch Processing**

- Process multiple videos/audio files in sequence
- Support glob patterns for input files
- Generate batch reports

### Phase 2: Advanced Audio Features

1. **Enhanced VST3 Support**

- Auto-discovery of installed VST3 plugins
- Plugin preset management
- Real-time parameter adjustment during processing
- Support for AU (Audio Unit) plugins on macOS

2. **Audio Analysis and Enhancement**

- Automatic volume leveling
- Noise reduction preprocessing
- Dynamic range compression
- Spectral analysis and EQ matching

3. **Advanced Segmentation**

- ML-based scene detection for better split points
- Language-aware segmentation
- Musical phrase detection for music videos

### Phase 3: Multi-Language and Voice Features

1. **Language Support**

- Auto-detect source language
- Language-specific voice selection
- Cross-language dubbing support

2. **Voice Management**

- Voice preview/testing mode
- Custom voice profiles
- Voice blending for multiple speakers
- Speaker diarization for multi-speaker content

3. **Subtitle Integration**

- Generate subtitles from original audio
- Sync subtitles with dubbed audio
- Multi-language subtitle support

### Phase 4: User Experience Enhancements

1. **Preview Mode**

- Process first 30-60 seconds for quick testing
- Side-by-side comparison of original vs dubbed
- Real-time parameter adjustment

2. **Quality Metrics**

- Audio quality assessment scores
- Timing accuracy reports
- Voice similarity metrics
- Processing statistics dashboard

3. **Configuration Management**

- Project-based settings
- Configuration templates
- Import/export settings
- Cloud sync for settings

### Phase 5: Platform and Integration

1. **GUI Application**

- Desktop application with drag-and-drop
- Real-time progress visualization
- Waveform display and editing
- Built-in audio player with A/B comparison

2. **Cloud Processing**

- Upload to cloud for processing
- Distributed processing for long videos
- Web-based interface
- API for third-party integration

3. **Platform Extensions**

- Docker containerization
- GitHub Actions integration
- CI/CD pipeline support
- Plugin architecture for custom processors

### Phase 6: Advanced Video Features

1. **Lip Sync Optimization**

- Analyze video for lip movements
- Adjust audio timing for better sync
- Support for animated content

2. **Scene-Aware Processing**

- Different processing for music vs dialogue
- Ambient sound preservation
- Background music handling

3. **Format Support**

- Additional video formats (WebM, MKV, etc.)
- Streaming format support (HLS, DASH)
- HDR video preservation
- Multi-track audio support

## Technical Debt and Maintenance

### Code Quality

- Add comprehensive unit tests
- Implement integration tests
- Add type hints throughout
- Improve error messages and logging

### Documentation

- API documentation
- Video tutorials
- Example configurations
- Troubleshooting guide

### Performance Optimization

- Profile and optimize bottlenecks
- Reduce memory footprint
- Implement streaming processing
- Cache optimization

## Success Metrics for Future Releases

### Performance Goals

- Process 4K videos efficiently
- Support videos over 2 hours
- Reduce processing time by 50%
- Memory usage under 4GB for 4K videos

### Quality Goals

- 99% audio-video sync accuracy
- Support for 95% of common formats
- Zero quality loss in video
- Natural-sounding voice conversion

### User Experience Goals

- One-click processing for common tasks
- < 5 minute setup time
- Intuitive configuration
- Comprehensive error recovery

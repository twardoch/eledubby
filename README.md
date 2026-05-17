# eledubby

Dub video and audio files with a new voice using the ElevenLabs speech-to-speech API. Timing stays locked to the original.

## The problem

ElevenLabs produces excellent synthetic voices. But when you send it a 3-minute audio clip, two things happen: it rejects anything over its input size limit, and the converted audio almost never comes back at exactly the same duration. Paste the result into your video and it immediately drifts out of sync.

`eledubby` solves both problems:

1. **Splits** the audio at silent moments, keeping each segment within the API's size limits.
2. **Converts** segments in parallel via ElevenLabs speech-to-speech.
3. **Pads or trims** each converted segment to match the exact duration of the original.
4. **Reassembles** the segments and muxes the result back into the video container.

## Install

```bash
pip install eledubby
# or
uv pip install eledubby
```

Requires `ffmpeg` on your system PATH.

Set your API key:

```bash
export ELEVENLABS_API_KEY="sk_..."
export ELEVENLABS_VOICE_ID="<voice-id>"
```

## Quick start

```bash
# Dub a video with the default voice
eledubby dub input.mp4 output.mp4

# Use a specific voice
eledubby dub input.mp4 output.mp4 --voice-id YOUR_VOICE_ID

# Preview: process only the first 30 seconds
eledubby dub input.mp4 preview.mp4 --preview 30

# Resume a job interrupted by rate limiting
eledubby dub input.mp4 output.mp4 --resume

# Process in parallel (3 API calls at once)
eledubby dub input.mp4 output.mp4 --parallel 3
```

## How it works

### Step 1: Audio extraction

`ffmpeg` strips the audio track from the input file and converts it to 16 kHz mono WAV for processing.

### Step 2: Silence detection and segmentation

The audio is scanned for quiet moments. Each candidate split point is scored based on how silent it is and how close it is to an ideal segment boundary. Segments are kept between `seg_min` (default 10 s) and `seg_max` (default 20 s) so each one fits within ElevenLabs' input limits.

### Step 3: Speech-to-speech conversion

Each segment is sent to the ElevenLabs speech-to-speech API with your chosen voice ID. Parallel workers (configurable with `--parallel`) send multiple segments simultaneously to reduce wall-clock time.

### Step 4: Timing synchronisation

The converted segment almost never matches the original duration. `eledubby` measures the difference and either pads the end with silence or trims the tail to match exactly. This ensures no frame drift accumulates across the video.

### Step 5: Reassembly and remuxing

Adjusted segments are concatenated back into a single audio track. `ffmpeg` replaces the original audio in the video container without re-encoding the video stream.

## Options

```
eledubby dub <input> <output> [options]

  --voice-id TEXT       ElevenLabs voice ID (default: ELEVENLABS_VOICE_ID env)
  --seg-min FLOAT       Minimum segment duration in seconds (default: 10.0)
  --seg-max FLOAT       Maximum segment duration in seconds (default: 20.0)
  --parallel INT        Number of concurrent API calls (default: 1)
  --preview FLOAT       Process only the first N seconds (0 = full file)
  --normalize           Apply EBU R128 loudness normalisation (default: on)
  --target-db FLOAT     Target loudness in LUFS (default: -23.0)
  --compress            Apply dynamic range compression
  --denoise FLOAT       Noise reduction strength 0.0–1.0 (default: 0)
  --resume              Resume from a saved checkpoint
  --verbose             Show detailed progress
```

## Checkpointing

Each processed segment is saved to a temporary directory as it completes. If the job is interrupted (API rate limit, network error, keyboard interrupt), run the same command again with `--resume` to pick up where it left off.

## Supported formats

Any container format `ffmpeg` can demux: `.mp4`, `.mov`, `.mkv`, `.avi`, `.mp3`, `.wav`, `.m4a`, and more.

## License

MIT

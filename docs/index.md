---
layout: home
title: eledubby
---

# eledubby

Dub a video or audio file with a new voice using the ElevenLabs speech-to-speech API. The timing stays locked to the original, so nothing drifts out of sync.

![eledubby](assets/icon.png){: style="max-width:320px"}

## Install

```bash
pip install eledubby
```

Requires `ffmpeg` on your system PATH. Set your credentials:

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

# Preview only the first 30 seconds
eledubby dub input.mp4 preview.mp4 --preview 30

# Resume a job interrupted by rate limiting
eledubby dub input.mp4 output.mp4 --resume

# Send three API calls at once
eledubby dub input.mp4 output.mp4 --parallel 3
```

## How it works

1. **Extract** — `ffmpeg` strips the audio to 16 kHz mono WAV.
2. **Segment** — the audio is scanned for quiet moments. Each candidate split point is scored on how silent it is and how close it sits to an ideal boundary, so every segment lands between `seg_min` (10 s) and `seg_max` (20 s) and fits the API's input limit.
3. **Convert** — each segment goes to ElevenLabs speech-to-speech with your voice ID; parallel workers cut wall-clock time.
4. **Sync** — the converted segment rarely matches the original length, so eledubby pads the tail with silence or trims it to match exactly. No frame drift accumulates.
5. **Remux** — segments are concatenated and `ffmpeg` swaps the audio back into the container without re-encoding the video.

## Cost

ElevenLabs bills speech-to-speech per second of audio processed. Dubbing a whole video converts the entire soundtrack, so budget for roughly the clip's full runtime in credits. Use `--preview N` to convert only the first N seconds while you dial in the voice.

## License

MIT. See the [repository](https://github.com/twardoch/eledubby) for source and full documentation.

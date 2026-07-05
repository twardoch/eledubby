# this_file: tests/test_eledubby/wav_helpers.py
"""Stdlib WAV writer for tests (avoids a scipy dependency)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, sample_rate: int, data: np.ndarray) -> None:
    arr = np.asarray(data)
    if arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if arr.ndim == 1 else arr.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(arr.tobytes())

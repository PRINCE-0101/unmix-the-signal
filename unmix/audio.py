"""WAV input/output and deterministic demonstration-signal helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import wavfile


def normalize(signal, peak=0.92):
    signal = np.asarray(signal, dtype=float)
    maximum = np.max(np.abs(signal))
    return signal if maximum < 1e-12 else signal * peak / maximum


def save_wav(path, sample_rate, signal):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.int16(normalize(signal) * 32767))


def load_mono_wav(path):
    sample_rate, data = wavfile.read(path)
    data = data.astype(float)
    if data.ndim == 2:
        data = data.mean(axis=1)
    maximum = np.iinfo(np.int16).max if np.issubdtype(data.dtype, np.integer) else np.max(np.abs(data))
    return sample_rate, normalize(data / max(float(maximum), 1.0))


def make_demo_sources(sample_rate=8000, duration=3.0):
    """Make speech-like, musical, and percussive signals without downloads."""
    time = np.arange(int(sample_rate * duration)) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.1 * time) ** 2
    speech = envelope * (
        np.sin(2 * np.pi * (185 + 18 * np.sin(2 * np.pi * 1.7 * time)) * time)
        + 0.35 * np.sin(2 * np.pi * 370 * time)
    )
    notes = np.array([261.63, 329.63, 392.0, 523.25])
    sections = np.minimum((time / (duration / len(notes))).astype(int), len(notes) - 1)
    music = np.sin(2 * np.pi * notes[sections] * time) + 0.25 * np.sin(4 * np.pi * notes[sections] * time)
    clicks = np.zeros_like(time)
    for location in np.arange(0.1, duration, 0.32):
        start = int(location * sample_rate)
        length = min(int(0.05 * sample_rate), len(clicks) - start)
        clicks[start : start + length] += np.exp(-np.arange(length) / (0.009 * sample_rate))
    return np.vstack([normalize(speech), normalize(music), normalize(clicks)])


def mix_sources(sources, matrix=None):
    count = sources.shape[0]
    if matrix is None:
        matrix = np.array([[1.0, 0.55, 0.25], [0.35, 1.0, 0.45], [0.6, 0.2, 1.0]])
        matrix = matrix[:count, :count]
    return np.asarray(matrix) @ sources, np.asarray(matrix)


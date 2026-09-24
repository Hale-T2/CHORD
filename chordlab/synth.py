"""A toy "band": render chord progressions as audio so every notebook runs before any download.

Two timbres mirror Fig. 1 of the paper: ``'flute'`` (almost a pure sine, one spectral peak) and
``'piano'`` (a stack of decaying harmonics that crowds the spectrum). ``add_drums`` sprinkles
broadband noise bursts on the beats to imitate the percussive saturation of RWC-P.
"""

from __future__ import annotations

import numpy as np
import librosa

from .templates import TRIAD_INTERVALS, label_to_index, PITCH_CLASSES

SR = 22050

_TIMBRES = {
    # name: (harmonic amplitudes, decay time constant in seconds or None for sustained)
    "flute": ((1.0, 0.05), None),
    "organ": ((1.0, 0.5, 0.33, 0.25, 0.2, 0.15), None),
    "piano": ((1.0, 0.6, 0.4, 0.25, 0.2, 0.15, 0.1, 0.08), 0.9),
}


def midi_of_chord(label: str, octave: int = 3) -> list[int]:
    """MIDI note numbers of a reduced label such as ``'G:min'`` in a given octave (root at ``octave``)."""
    state = label_to_index(label)
    quality = "maj" if state < 12 else "min"
    root = state % 12
    base = 12 * (octave + 1) + root
    return [base + iv for iv in TRIAD_INTERVALS[quality]]


def render_note(midi_note: int, duration: float, sr: int = SR, timbre: str = "piano", amp: float = 0.3) -> np.ndarray:
    harmonics, decay = _TIMBRES[timbre]
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    f0 = librosa.midi_to_hz(midi_note)
    y = np.zeros(n)
    for k, a in enumerate(harmonics, start=1):
        if k * f0 < sr / 2:
            y += a * np.sin(2 * np.pi * k * f0 * t)
    env = np.exp(-t / decay) if decay else np.ones(n)
    attack = np.minimum(1.0, t / 0.01)                       # 10 ms attack to avoid clicks
    return amp * y * env * attack / len(harmonics)


def render_progression(labels: list[str], durations: list[float] | float = 2.0, sr: int = SR,
                       timbre: str = "piano", octave: int = 3, strum: float = 0.0,
                       ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Render a chord progression. Returns ``(audio, intervals[N, 2], labels)`` — audio plus its own ground truth."""
    if np.isscalar(durations):
        durations = [float(durations)] * len(labels)
    y_parts, intervals, t0 = [], [], 0.0
    for label, dur in zip(labels, durations):
        chunk = np.zeros(int(round(dur * sr)))
        for i, note in enumerate(midi_of_chord(label, octave)):
            note_audio = render_note(note, dur, sr, timbre)
            off = int(round(i * strum * sr))
            chunk[off:] += note_audio[: len(chunk) - off]
        y_parts.append(chunk)
        intervals.append([t0, t0 + dur])
        t0 += dur
    return np.concatenate(y_parts), np.array(intervals), list(labels)


def add_drums(y: np.ndarray, sr: int = SR, bpm: float = 120.0, level: float = 0.5,
              burst_ms: float = 40.0, seed: int = 0) -> np.ndarray:
    """Add short white-noise bursts on every beat ('four-on-the-floor' as in RWC-P failures)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    period = int(round(60.0 / bpm * sr))
    n_burst = int(burst_ms / 1000 * sr)
    env = np.exp(-np.arange(n_burst) / (n_burst / 4))
    for start in range(0, len(y) - n_burst, period):
        out[start:start + n_burst] += level * env * rng.standard_normal(n_burst)
    return out


def add_melody(y: np.ndarray, intervals: np.ndarray, labels: list[str], sr: int = SR,
               timbre: str = "flute", octave: int = 5, notes_per_chord: int = 4, seed: int = 1) -> np.ndarray:
    """Overlay a wandering melody (chord tones plus random non-chord tones) to imitate a singer."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for (s, e), label in zip(intervals, labels):
        chord_notes = midi_of_chord(label, octave)
        dur = (e - s) / notes_per_chord
        for i in range(notes_per_chord):
            note = rng.choice(chord_notes) if rng.random() < 0.6 else rng.integers(chord_notes[0] - 2, chord_notes[-1] + 3)
            seg = render_note(int(note), dur, sr, timbre, amp=0.25)
            a = int(round((s + i * dur) * sr))
            out[a:a + len(seg)] += seg[: max(0, len(out) - a)]
    return out


DEMO_PROGRESSION = ["C:maj", "G:maj", "A:min", "F:maj", "D:min", "G:maj", "C:maj", "E:min"]

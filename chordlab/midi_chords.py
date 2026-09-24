"""Chord ground truth from audio-aligned MIDI (needed for RWC-R, which has no chord annotations).

Idea: the aligned MIDI tells us which pitches sound when. Aggregate the *pitch-class content*
of every beat (or bar) into a 12-d vector — a *symbolic chroma* — and label it with the same
24 major/minor templates the audio decoder uses. Because the MIDI is aligned to the audio
(re-aligned with DTW for RWC 2.0, see Balke et al. 2026), the resulting labels live on the
audio time axis and can serve as ground truth for the recordings.

Caveats students should know:
* The label of a beat containing passing notes or a melody may be pulled towards a different
  triad; using ``grid='bar'`` averages more notes and is more stable but coarser.
* Beats with fewer than two distinct pitch classes (drum intros, single bass notes) get ``N``.
* This is a *derived* ground truth. The notebook validates the procedure on RWC-P, where
  human chord annotations exist, before trusting it on RWC-R.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pretty_midi

from .templates import TEMPLATES, NO_CHORD, cosine_similarity, index_to_label

EPS = float(np.finfo(np.float64).eps)


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(path))


# ---------------------------------------------------------------------------
# Time grids
# ---------------------------------------------------------------------------

def beat_grid(beat_times: np.ndarray, end_time: float | None = None) -> np.ndarray:
    """Intervals between consecutive annotated beats; the last beat is closed with the median beat period."""
    b = np.asarray(beat_times, dtype=float)
    period = float(np.median(np.diff(b))) if len(b) > 1 else 0.5
    last = end_time if end_time is not None else b[-1] + period
    ends = np.append(b[1:], max(last, b[-1] + 1e-3))
    return np.stack([b, ends], axis=1)


def bar_grid(beat_times: np.ndarray, beat_numbers: np.ndarray, end_time: float | None = None) -> np.ndarray:
    """Intervals from each downbeat (``beat == 1``) to the next one."""
    b = np.asarray(beat_times, dtype=float)
    down = b[np.asarray(beat_numbers) == 1]
    if len(down) == 0:
        return beat_grid(b, end_time)
    period = float(np.median(np.diff(down))) if len(down) > 1 else float(np.median(np.diff(b))) * 4
    last = end_time if end_time is not None else down[-1] + period
    ends = np.append(down[1:], max(last, down[-1] + 1e-3))
    return np.stack([down, ends], axis=1)


# ---------------------------------------------------------------------------
# Symbolic chroma
# ---------------------------------------------------------------------------

def _is_excluded(inst: pretty_midi.Instrument, exclude_drums: bool, exclude_names: Sequence[str]) -> bool:
    if exclude_drums and inst.is_drum:
        return True
    name = inst.name.strip().lower()
    return any(tok.lower() in name for tok in exclude_names)


def pitch_class_activity(pm: pretty_midi.PrettyMIDI, intervals: np.ndarray, exclude_drums: bool = True,
                         exclude_names: Sequence[str] = ("melo",), use_velocity: bool = True) -> np.ndarray:
    """(12, n_intervals) matrix: how long (× velocity) each pitch class sounds inside each interval.

    ``exclude_names`` drops instrument tracks whose name contains one of the tokens — by
    default the ``Melo`` (vocal melody) track of the RWC MIDIs, so passing tones of the tune
    do not blur the accompaniment harmony.
    """
    intervals = np.asarray(intervals, dtype=float)
    iv_start, iv_end = intervals[:, 0][None, :], intervals[:, 1][None, :]
    act = np.zeros((12, len(intervals)))
    for inst in pm.instruments:
        if _is_excluded(inst, exclude_drums, exclude_names) or not inst.notes:
            continue
        starts = np.array([n.start for n in inst.notes])[:, None]
        ends = np.array([n.end for n in inst.notes])[:, None]
        pcs = np.array([n.pitch % 12 for n in inst.notes])
        w = np.array([n.velocity / 127.0 for n in inst.notes]) if use_velocity else np.ones(len(inst.notes))
        overlap = np.clip(np.minimum(ends, iv_end) - np.maximum(starts, iv_start), 0.0, None)   # (notes, intervals)
        overlap *= w[:, None]
        for pc in range(12):
            sel = pcs == pc
            if sel.any():
                act[pc] += overlap[sel].sum(axis=0)
    return act


def bass_pitch_class(pm: pretty_midi.PrettyMIDI, intervals: np.ndarray, exclude_drums: bool = True,
                     exclude_names: Sequence[str] = ("melo",), min_overlap: float = 0.05) -> np.ndarray:
    """Pitch class of the *lowest* note sounding (≥ ``min_overlap`` s) in each interval, ``-1`` if none.

    The bass usually plays the root, so it is a good tie-breaker between chords that share
    notes (A:min7 = A C E G contains both the A:min and the C:maj triad).
    """
    intervals = np.asarray(intervals, dtype=float)
    iv_start, iv_end = intervals[:, 0][None, :], intervals[:, 1][None, :]
    lowest = np.full(len(intervals), 999)
    for inst in pm.instruments:
        if _is_excluded(inst, exclude_drums, exclude_names) or not inst.notes:
            continue
        starts = np.array([n.start for n in inst.notes])[:, None]
        ends = np.array([n.end for n in inst.notes])[:, None]
        pitches = np.array([n.pitch for n in inst.notes])
        sounding = (np.minimum(ends, iv_end) - np.maximum(starts, iv_start)) > min_overlap
        lowest = np.minimum(lowest, np.where(sounding, pitches[:, None], 999).min(axis=0))
    return np.where(lowest < 999, lowest % 12, -1)


def add_bass_bonus(activity: np.ndarray, bass_pc: np.ndarray, bonus: float = 0.5) -> np.ndarray:
    """Boost the bass pitch class by ``bonus`` × (total activity / 3), i.e. about half a triad note."""
    act = activity.copy()
    total = act.sum(axis=0)
    for j in np.flatnonzero(bass_pc >= 0):
        act[bass_pc[j], j] += bonus * total[j] / 3.0
    return act


def label_activity(activity: np.ndarray, min_pitch_classes: int = 2, min_fraction: float = 0.10,
                   templates: np.ndarray = TEMPLATES) -> tuple[np.ndarray, np.ndarray]:
    """Template-match symbolic chroma → states (``-1`` where too little harmonic content).

    A pitch class counts as *active* if it carries at least ``min_fraction`` of the interval's
    total activity; intervals with fewer than ``min_pitch_classes`` active classes are ``N``.
    Returns ``(states, confidence)`` where confidence is the winning cosine similarity.
    """
    total = activity.sum(axis=0)
    sim = cosine_similarity(activity, templates)
    states = sim.argmax(axis=0)
    conf = sim.max(axis=0)
    active = (activity >= min_fraction * (total[None, :] + EPS)).sum(axis=0)
    silent = (total <= EPS) | (active < min_pitch_classes)
    states = np.where(silent, NO_CHORD, states)
    conf = np.where(silent, 0.0, conf)
    return states, conf


def deflicker(states: np.ndarray) -> np.ndarray:
    """Remove isolated one-interval blips: X Y X → X X X (human annotators rarely mark single-beat chords)."""
    s = np.asarray(states).copy()
    for i in range(1, len(s) - 1):
        if s[i] != s[i - 1] and s[i - 1] == s[i + 1]:
            s[i] = s[i - 1]
    return s


def merge_segments(intervals: np.ndarray, labels: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Merge consecutive intervals that carry the same label."""
    intervals = np.asarray(intervals, dtype=float)
    if len(labels) == 0:
        return intervals, list(labels)
    out_iv, out_lab = [list(intervals[0])], [labels[0]]
    for (s, e), lab in zip(intervals[1:], labels[1:]):
        if lab == out_lab[-1] and abs(s - out_iv[-1][1]) < 1e-6:
            out_iv[-1][1] = e
        else:
            out_iv.append([s, e])
            out_lab.append(lab)
    return np.array(out_iv), out_lab


def derive_chords(midi_file: str | Path, beat_times: np.ndarray, beat_numbers: np.ndarray | None = None,
                  grid: str = "beat", end_time: float | None = None, bass_bonus: float = 0.5,
                  smooth: bool = True, exclude_drums: bool = True, exclude_names: Sequence[str] = ("melo",),
                  use_velocity: bool = True) -> tuple[np.ndarray, list[str], np.ndarray]:
    """End-to-end: MIDI + beat annotation → merged ``(intervals, Harte labels, per-segment confidence)``.

    ``grid='beat'`` labels every beat, ``grid='bar'`` every bar (needs ``beat_numbers``).
    ``bass_bonus`` > 0 lets the lowest note vote for the root; ``smooth`` removes single-interval blips.
    Validated on RWC-P (notebook 07): ≈ 73 % frame agreement with the human major/minor labels,
    which is in the range of agreement between *human* annotators reported for pop music.
    """
    pm = load_midi(midi_file)
    end_time = end_time if end_time is not None else pm.get_end_time()
    if grid == "bar":
        if beat_numbers is None:
            raise ValueError("grid='bar' needs beat_numbers (downbeat = 1)")
        intervals = bar_grid(beat_times, beat_numbers, end_time)
    elif grid == "beat":
        intervals = beat_grid(beat_times, end_time)
    else:
        raise ValueError("grid must be 'beat' or 'bar'")
    act = pitch_class_activity(pm, intervals, exclude_drums, exclude_names, use_velocity)
    if bass_bonus:
        act = add_bass_bonus(act, bass_pitch_class(pm, intervals, exclude_drums, exclude_names), bass_bonus)
    states, conf = label_activity(act)
    if smooth:
        states = deflicker(states)
    labels = [index_to_label(int(s)) for s in states]
    merged_iv, merged_lab = merge_segments(intervals, labels)
    seg_conf = []
    for (s, e) in merged_iv:
        members = (intervals[:, 0] >= s - 1e-9) & (intervals[:, 1] <= e + 1e-9)
        seg_conf.append(float(conf[members].mean()) if members.any() else 0.0)
    return merged_iv, merged_lab, np.array(seg_conf)

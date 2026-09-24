"""Evaluation (paper §IV): reduce annotations to 24 triads, then frame-wise precision / recall / F1.

Paper ↔ code
------------
"complex chord annotations were reduced to their fundamental major or minor triads"  :func:`reduce_label`
TP / FP / FN definitions (§IV-A bullet list)                                          :func:`frame_prf`
Eq. (19)-(20) P, R, F1                                                                 :func:`frame_prf`
Table I / Fig. 4 style summaries                                                       :func:`results_table`

For comparability with the wider ACE literature we also expose the standard
``mir_eval.chord`` scores (:func:`mir_eval_scores`), which handle labels outside the
major/minor vocabulary differently from the paper (they are *excluded* rather than counted
as false positives).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import mir_eval.chord as mchord

from .templates import CHORD_LABELS, NO_CHORD, PITCH_CLASSES, index_to_label, label_to_index

#: Quality strings (mir_eval's ``split`` vocabulary) whose triad core is major / minor.
MAJOR_CORE_QUALITIES = {"maj", "maj7", "maj6", "maj9", "maj13", "7", "9", "11", "13"}
MINOR_CORE_QUALITIES = {"min", "min7", "min6", "min9", "min11", "min13", "minmaj7"}


def canonical_root(root: str) -> str:
    """Spell a Harte root with the sharps used by :data:`PITCH_CLASSES` (``Ab`` → ``G#``)."""
    return PITCH_CLASSES[mchord.pitch_class_to_semitone(root)]


def reduce_label(label: str, method: str = "quality") -> str:
    """Map an arbitrary Harte chord label to ``'<root>:maj'``, ``'<root>:min'`` or ``'N'``.

    ``method='quality'`` (paper): sevenths, sixths, ninths … keep the root triad of their
    quality name (``Cmaj7 → C:maj``, ``A:min7 → A:min``); diminished, augmented, suspended and
    ``N``/``X`` become ``'N'``. Omitted notes such as ``E:maj(*3,*5)`` follow the quality name.

    ``method='bitmap'``: decide from the actual pitch-class content — a major third **and** a
    perfect fifth present → ``maj``; minor third and fifth → ``min``; anything else → ``'N'``.
    This is stricter (power chords ``maj(*3)`` → ``'N'``) and closer to what the audio can show.
    """
    label = label.strip()
    root, quality, degrees, _bass = mchord.split(label)
    if root in ("N", "X"):
        return "N"
    if method == "quality":
        if quality in MAJOR_CORE_QUALITIES or (quality not in MINOR_CORE_QUALITIES and quality.startswith("maj")):
            return f"{canonical_root(root)}:maj"
        if quality in MINOR_CORE_QUALITIES or quality.startswith("min"):
            return f"{canonical_root(root)}:min"
        return "N"
    if method == "bitmap":
        _, bitmap, _ = mchord.encode(label)
        if bitmap[4] and bitmap[7]:
            return f"{canonical_root(root)}:maj"
        if bitmap[3] and bitmap[7]:
            return f"{canonical_root(root)}:min"
        return "N"
    raise ValueError("method must be 'quality' or 'bitmap'")


def reduce_labels(labels: Iterable[str], method: str = "quality") -> list[str]:
    return [reduce_label(l, method) for l in labels]


def quality_mapping_table(labels: Iterable[str]) -> pd.DataFrame:
    """For a collection of raw labels: how often each *quality* occurs and how it is reduced."""
    rows = {}
    for lab in labels:
        root, quality, degrees, bass = mchord.split(lab)
        key = quality if root not in ("N", "X") else root
        q_ex = f"{'C' if root not in ('N', 'X') else ''}{':' if key not in ('N', 'X') else ''}{key}"
        rows.setdefault(key, {"count": 0, "example": q_ex})
        rows[key]["count"] += 1
    df = pd.DataFrame(rows).T
    df["reduced (quality)"] = [reduce_label(e, "quality") for e in df["example"]]
    df["reduced (bitmap)"] = [reduce_label(e, "bitmap") for e in df["example"]]
    return df.sort_values("count", ascending=False)


# ---------------------------------------------------------------------------
# Intervals <-> frames
# ---------------------------------------------------------------------------

def labels_to_frames(intervals: np.ndarray, labels: Sequence[str], times: np.ndarray,
                     reduce: str | None = "quality") -> np.ndarray:
    """Sample interval annotations at frame times → integer states (``-1`` outside / no chord)."""
    intervals = np.asarray(intervals, dtype=float)
    labs = reduce_labels(labels, reduce) if reduce else list(labels)
    idx = np.array([label_to_index(l) for l in labs], dtype=np.int64)
    starts, ends = intervals[:, 0], intervals[:, 1]
    order = np.argsort(starts)
    starts, ends, idx = starts[order], ends[order], idx[order]
    pos = np.searchsorted(starts, times, side="right") - 1
    states = np.full(len(times), NO_CHORD, dtype=np.int64)
    ok = pos >= 0
    ok[ok] &= times[ok] < ends[pos[ok]]
    states[ok] = idx[pos[ok]]
    return states


def frames_to_intervals(states: np.ndarray, times: np.ndarray, hop_seconds: float) -> tuple[np.ndarray, list[str]]:
    """Merge runs of identical frame states into ``(intervals, labels)`` (for mir_eval / plotting)."""
    states = np.asarray(states)
    if len(states) == 0:
        return np.zeros((0, 2)), []
    change = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate([[0], change])
    # One boundary per change, shared by the segment that ends and the one that starts there.
    # (Computing end = times[last] + hop separately can exceed the next start by ~1e-15 s,
    # which mir_eval rejects as "Chord Intervals must not overlap".)
    boundaries = np.concatenate([times[starts], [times[-1] + hop_seconds]])
    intervals = np.stack([boundaries[:-1], boundaries[1:]], axis=1)
    labels = [index_to_label(int(states[s])) for s in starts]
    return intervals, labels


# ---------------------------------------------------------------------------
# Paper metric
# ---------------------------------------------------------------------------

def frame_prf(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Frame-wise precision, recall and F1 exactly as defined in §IV-A of the paper.

    * TP: prediction equals the annotated chord (annotation is a chord).
    * FP: a chord is predicted but the annotation is a *different* chord **or silence/N**.
    * FN: the annotation is a chord but the prediction is different (or N).
    A wrong chord therefore counts as both an FP and an FN; a chord predicted on an ``N`` frame
    is an FP only. The 24-state decoders never output ``N``, so every ``N`` frame costs precision.
    """
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have the same number of frames")
    truth_is_chord = truth != NO_CHORD
    pred_is_chord = pred != NO_CHORD
    tp = int(np.sum(truth_is_chord & (pred == truth)))
    fp = int(np.sum(pred_is_chord & (pred != truth)))
    fn = int(np.sum(truth_is_chord & (pred != truth)))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": P, "R": R, "F1": F1, "n_frames": int(len(truth))}


# ---------------------------------------------------------------------------
# Community metric (mir_eval)
# ---------------------------------------------------------------------------

def mir_eval_scores(ref_intervals: np.ndarray, ref_labels: Sequence[str],
                    est_intervals: np.ndarray, est_labels: Sequence[str]) -> dict:
    """Standard ``mir_eval.chord.evaluate`` scores (weighted chord symbol recall).

    ``majmin`` compares major/minor triads and *ignores* reference chords outside that
    vocabulary, ``root`` compares roots only, ``triads`` and ``mirex`` are the other usual columns.
    """
    scores = mchord.evaluate(np.asarray(ref_intervals, float), list(ref_labels),
                             np.asarray(est_intervals, float), list(est_labels))
    return {k: float(v) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def evaluate_track(states: np.ndarray, times: np.ndarray, ref_intervals: np.ndarray,
                   ref_labels: Sequence[str], hop_seconds: float, reduce: str = "quality",
                   with_mir_eval: bool = True) -> dict:
    """Paper metric (+ optional mir_eval scores) for one decoded track."""
    truth = labels_to_frames(ref_intervals, ref_labels, times, reduce)
    out = frame_prf(states, truth)
    if with_mir_eval:
        est_int, est_lab = frames_to_intervals(states, times, hop_seconds)
        me = mir_eval_scores(ref_intervals, ref_labels, est_int, est_lab)
        out.update({f"mir_eval_{k}": v for k, v in me.items() if k in ("root", "majmin", "triads", "mirex")})
    return out


def results_table(per_track: dict[str, dict[str, dict]], metrics: Sequence[str] = ("P", "R", "F1")) -> pd.DataFrame:
    """Build a Table-I-like DataFrame from ``{track: {model: metric_dict}}`` and append the average."""
    frames = {}
    for model, tracks in _swap_levels(per_track).items():
        frames[model] = pd.DataFrame(tracks).T[list(metrics)]
    df = pd.concat(frames, axis=1)
    df.loc["Average"] = df.mean(numeric_only=True)
    return df


def _swap_levels(d: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for track, models in d.items():
        for model, metrics in models.items():
            out.setdefault(model, {})[track] = metrics
    return out


__all__ = ["reduce_label", "reduce_labels", "quality_mapping_table", "labels_to_frames",
           "frames_to_intervals", "frame_prf", "mir_eval_scores", "evaluate_track",
           "results_table", "CHORD_LABELS"]

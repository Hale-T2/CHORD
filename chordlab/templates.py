"""Stateless template matching (paper §III-A).

24 binary templates — 12 major triads (0-4-7) and 12 minor triads (0-3-7) — are compared
to every chroma frame with cosine similarity, Eq. (11); the most similar template wins.

State indexing used everywhere in ``chordlab``: ``0…11`` = C:maj … B:maj, ``12…23`` = C:min … B:min,
and ``-1`` = no chord (``N``).
"""

from __future__ import annotations

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min"]
CHORD_LABELS = [f"{pc}:maj" for pc in PITCH_CLASSES] + [f"{pc}:min" for pc in PITCH_CLASSES]
N_STATES = len(CHORD_LABELS)          # 24
NO_CHORD = -1

TRIAD_INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7)}

#: Enharmonic spellings used by Harte labels → pitch-class index.
_PC_INDEX = {**{pc: i for i, pc in enumerate(PITCH_CLASSES)},
             "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}

EPS = float(np.finfo(np.float64).eps)


def build_templates() -> np.ndarray:
    """(24, 12) binary matrix; row ``c`` is the template of :data:`CHORD_LABELS`[c]."""
    T = np.zeros((N_STATES, 12))
    for q_idx, quality in enumerate(QUALITIES):
        for root in range(12):
            for iv in TRIAD_INTERVALS[quality]:
                T[q_idx * 12 + root, (root + iv) % 12] = 1.0
    return T


TEMPLATES = build_templates()


def cosine_similarity(chroma: np.ndarray, templates: np.ndarray = TEMPLATES, eps: float = EPS) -> np.ndarray:
    """Eq. (11): r_c(x[n]) = ⟨x[n], t_c⟩ / (‖x[n]‖₂ ‖t_c‖₂) for all 24 templates and all frames.

    Returns an array of shape (24, T). With L2-normalised chroma (Eq. 10) the denominator is
    essentially ‖t_c‖₂ = √3, but we keep the general form so raw chroma works as well.
    """
    num = templates @ chroma                                        # (24, T)
    den = np.linalg.norm(templates, axis=1)[:, None] * np.linalg.norm(chroma, axis=0)[None, :]
    return num / (den + eps)


def template_decode(chroma: np.ndarray, templates: np.ndarray = TEMPLATES) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise argmax over the 24 similarities. Returns ``(states[T], similarity[24, T])``."""
    sim = cosine_similarity(chroma, templates)
    return sim.argmax(axis=0), sim


def index_to_label(state: int) -> str:
    """State index → Harte label (``-1`` → ``'N'``)."""
    return "N" if state < 0 else CHORD_LABELS[state]


def label_to_index(label: str) -> int:
    """``'Ab:min'`` → 20, ``'N'`` → -1. Only the 24 triad labels (any spelling) are accepted."""
    if label in ("N", "X"):
        return NO_CHORD
    root, _, quality = label.partition(":")
    quality = quality or "maj"
    if quality not in QUALITIES or root not in _PC_INDEX:
        raise ValueError(f"{label!r} is not a reduced major/minor label; call evaluate.reduce_label first")
    return QUALITIES.index(quality) * 12 + _PC_INDEX[root]


def states_to_labels(states: np.ndarray) -> list[str]:
    return [index_to_label(int(s)) for s in states]

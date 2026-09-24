"""Matplotlib helpers shared by the notebooks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .templates import CHORD_LABELS, PITCH_CLASSES, NO_CHORD


def plot_chroma(chroma: np.ndarray, times: np.ndarray, ax=None, title: str | None = None, cmap: str = "magma"):
    """Pitch-class × time image with note names on the y-axis."""
    ax = ax or plt.gca()
    hop = times[1] - times[0] if len(times) > 1 else 1.0
    extent = [times[0], times[-1] + hop, -0.5, 11.5]
    im = ax.imshow(chroma, aspect="auto", origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels(PITCH_CLASSES)
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title)
    return im


def overlay_segments(ax, intervals: np.ndarray, labels, y: float | None = None, color: str = "w", fontsize: int = 8,
                     min_width: float | None = None):
    """Vertical boundary lines + labels for annotated chord segments.

    Labels of segments narrower than ``min_width`` seconds are skipped (default: 3 % of the x-range)
    so that dense passages stay readable.
    """
    y = ax.get_ylim()[1] - 0.6 if y is None else y
    if min_width is None:
        x0, x1 = ax.get_xlim()
        min_width = 0.03 * (x1 - x0)
    for (s, e), lab in zip(intervals, labels):
        ax.axvline(s, color=color, lw=0.8, alpha=0.7)
        if e - s >= min_width:
            ax.text((s + e) / 2, y, lab, color=color, ha="center", va="top", fontsize=fontsize)


def plot_states(times: np.ndarray, *series, labels=None, ax=None, title: str | None = None):
    """Step plot of chord-state sequences (e.g. truth vs. template vs. HMM)."""
    ax = ax or plt.gca()
    labels = labels or [f"series {i}" for i in range(len(series))]
    styles = [dict(lw=3, alpha=0.4), dict(lw=1.2), dict(lw=1.2)]
    for s, lab, st in zip(series, labels, styles + [dict(lw=1)] * 10):
        s = np.asarray(s, dtype=float)
        s[s == NO_CHORD] = np.nan
        ax.step(times, s, where="post", label=lab, **st)
    ax.set_yticks(range(len(CHORD_LABELS)))
    ax.set_yticklabels(CHORD_LABELS, fontsize=7)
    ax.set_ylim(-1, len(CHORD_LABELS))
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_tf_tiling(sr: int = 22050, fmin: float = 32.7, n_octaves: int = 4, n_fft: int = 2048, ax=None):
    """Fig. 2 of the paper: uniform STFT tiles vs. CQT tiles that widen with frequency."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
    a0, a1 = ax
    # STFT: constant Δf = sr / n_fft and constant window length n_fft / sr
    df, dt = sr / n_fft, n_fft / sr
    fmax = fmin * 2 ** n_octaves
    for k in range(int(fmax // df) + 1):
        for m in range(int(1.0 / dt) + 1):
            a0.add_patch(plt.Rectangle((m * dt, k * df), dt, df, fill=False, lw=0.4, ec="steelblue"))
    a0.set_xlim(0, 1)
    a0.set_ylim(fmin, fmax)
    a0.set_title("STFT: Δt × Δf constant")
    a0.set_xlabel("time (s)")
    a0.set_ylabel("frequency (Hz)")
    # CQT: Δf_k ∝ f_k, window length N_k ∝ 1 / f_k
    q = 1 / (2 ** (1 / 12) - 1)
    for k in range(12 * n_octaves):
        fk = fmin * 2 ** (k / 12)
        bw = fk / q
        win = q / fk
        m = 0.0
        while m < 1.0:
            a1.add_patch(plt.Rectangle((m, fk - bw / 2), win, bw, fill=False, lw=0.4, ec="darkorange"))
            m += win
    a1.set_yscale("log")
    a1.set_xlim(0, 1)
    a1.set_ylim(fmin, fmax)
    a1.set_title("CQT: Q = f_k / Δf_k constant")
    a1.set_xlabel("time (s)")
    a1.set_ylabel("frequency (Hz, log)")
    return ax


def plot_f1_curve(df, hmm_col: str = "HMM_F1", tmpl_col: str = "Template_F1", threshold: float = 0.22, ax=None):
    """Fig. 4 of the paper: tracks sorted by HMM F1, template F1 dashed, template > HMM marked."""
    ax = ax or plt.gca()
    d = df.sort_values(hmm_col, ascending=False).reset_index()
    x = np.arange(len(d))
    ax.plot(x, d[hmm_col], "-", color="navy", lw=2, label="CQT-HMM (Viterbi)")
    ax.plot(x, d[tmpl_col], "--", color="darkorange", lw=1, label="CQT-Template matching")
    better = d[tmpl_col] > d[hmm_col]
    ax.plot(x[better], d[tmpl_col][better], "r^", ms=6, label=f"Template outperforms HMM ({better.sum()} tracks)")
    fail = (d[hmm_col] < threshold) & (d[tmpl_col] < threshold)
    if fail.any():
        ax.axvspan(x[fail].min() - 0.5, x[fail].max() + 0.5, color="red", alpha=0.08, label=f"both F1 < {threshold}")
    ax.axhline(d[hmm_col].mean(), color="navy", ls=":", lw=1, label=f"HMM mean {d[hmm_col].mean():.2f}")
    ax.axhline(d[tmpl_col].mean(), color="darkorange", ls=":", lw=1, label=f"Template mean {d[tmpl_col].mean():.2f}")
    ax.set_xlabel("tracks, sorted by HMM F1")
    ax.set_ylabel("frame-wise F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return ax


def plot_matrix(M: np.ndarray, ax=None, title: str | None = None, labels=CHORD_LABELS, cmap: str = "viridis", log: bool = False):
    """Heat-map of a 24×24 transition matrix."""
    ax = ax or plt.gca()
    im = ax.imshow(np.log10(M + 1e-6) if log else M, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("to state j")
    ax.set_ylabel("from state i")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im

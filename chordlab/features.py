"""Front end: from a stereo WAV to L2-normalised chroma vectors (paper §II).

Paper ↔ code
------------
Eq. (4)  stereo → mono            :func:`stereo_to_mono` (librosa.load(mono=True) does the same)
Eq. (5)  STFT                      :func:`stft_chroma` (via librosa)
Eq. (6)-(9) CQT, N_k, f_k, Q       :func:`cqt`, :func:`cqt_window_lengths`, :data:`Q_FACTOR`
§II-C   chromagram + Eq. (10)      :func:`chroma_from_cqt`, :func:`l2_normalize`, :func:`cqt_chroma`
"""

from __future__ import annotations

import numpy as np
import librosa

#: Paper defaults: CD audio is resampled to 22 050 Hz; CQT starts at C1 ≈ 32.70 Hz.
SR = 22050
HOP_LENGTH = 2048            # ≈ 93 ms per chroma frame
FMIN = librosa.note_to_hz("C1")
BINS_PER_OCTAVE = 12
N_OCTAVES = 7                # C1 … B7 (≈ 3951 Hz, well below Nyquist at 22 050 Hz)
N_BINS = N_OCTAVES * BINS_PER_OCTAVE

#: Eq. (9): Q = f_k / δf_k = 1 / (2^(1/12) − 1).  Note: this evaluates to ≈ 16.82
#: (the paper prints 17.03; the commonly quoted round value in the CQT literature is 17).
Q_FACTOR = 1.0 / (2 ** (1 / BINS_PER_OCTAVE) - 1)

#: Machine epsilon of float64 (≈ 2.22e-16), used as the ε of Eq. (10) and the log-eps of Eq. (14).
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# §II-A  mixing and resampling
# ---------------------------------------------------------------------------

def stereo_to_mono(x: np.ndarray) -> np.ndarray:
    """Eq. (4): arithmetic mean of the channels. Accepts (2, n) or (n, 2) arrays."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.shape[0] != 2 and x.shape[1] == 2:
        x = x.T
    return 0.5 * (x[0] + x[1])


def pcm16_to_float(x_int16: np.ndarray) -> np.ndarray:
    """Scale 16-bit signed PCM (−32768…32767) to floats in [−1, 1] (paper §II-A)."""
    return np.asarray(x_int16, dtype=np.float64) / 32768.0


def load_audio(path, sr: int = SR, offset: float = 0.0, duration: float | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float64 at ``sr`` Hz.

    ``librosa.load`` reads the PCM samples as floats, averages the channels (Eq. 4)
    and resamples (paper: 44.1 kHz → 22.05 kHz).
    """
    y, sr_out = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration, dtype=np.float64)
    return y, int(sr_out)


# ---------------------------------------------------------------------------
# §II-B  STFT vs. CQT
# ---------------------------------------------------------------------------

def cqt_center_frequencies(fmin: float = FMIN, n_bins: int = N_BINS, bins_per_octave: int = BINS_PER_OCTAVE) -> np.ndarray:
    """Eq. (8): f_k = f_min · 2^(k / 12) for k = 0 … K−1."""
    k = np.arange(n_bins)
    return fmin * 2.0 ** (k / bins_per_octave)


def cqt_window_lengths(sr: int = SR, fmin: float = FMIN, n_bins: int = N_BINS,
                       bins_per_octave: int = BINS_PER_OCTAVE) -> np.ndarray:
    """Eq. (7): N_k = Q · F_s / f_k — long windows for low bins, short windows for high bins."""
    q = 1.0 / (2 ** (1 / bins_per_octave) - 1)
    return q * sr / cqt_center_frequencies(fmin, n_bins, bins_per_octave)


def cqt(y: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH, fmin: float = FMIN,
        n_bins: int = N_BINS, bins_per_octave: int = BINS_PER_OCTAVE) -> np.ndarray:
    """Eq. (6): complex CQT matrix of shape (n_bins, n_frames); rows are semitone bins from C1."""
    return librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins,
                       bins_per_octave=bins_per_octave)


def frame_times(n_frames: int, sr: int = SR, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Time stamp (s) of each frame centre."""
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


# ---------------------------------------------------------------------------
# §II-C  chromagram
# ---------------------------------------------------------------------------

def chroma_from_cqt(C: np.ndarray, bins_per_octave: int = BINS_PER_OCTAVE) -> np.ndarray:
    """Fold a CQT magnitude spectrogram over octaves into 12 pitch classes (C, C#, …, B).

    Row 0 of the CQT is C1, so bin ``k`` belongs to pitch class ``k mod 12``. Summing the
    magnitudes of all bins with the same pitch class gives the (un-normalised) chroma.
    """
    mag = np.abs(C)
    n_bins, n_frames = mag.shape
    if n_bins % bins_per_octave:
        raise ValueError("n_bins must be a multiple of bins_per_octave to fold cleanly")
    return mag.reshape(n_bins // bins_per_octave, bins_per_octave, n_frames).sum(axis=0)


def l2_normalize(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Eq. (10): divide every column (frame) by its Euclidean norm + ε.

    ε is float64 machine epsilon; it keeps silent frames from producing 0/0.
    """
    return X / (np.linalg.norm(X, axis=0, keepdims=True) + eps)


def cqt_chroma(y: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH, fmin: float = FMIN,
               n_octaves: int = N_OCTAVES, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Full front end of the paper: mono audio → CQT → 12-d chroma → L2 normalisation.

    Returns ``(chroma[12, T], times[T])``.
    """
    C = cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_octaves * BINS_PER_OCTAVE)
    chroma = chroma_from_cqt(C)
    if normalize:
        chroma = l2_normalize(chroma)
    return chroma, frame_times(chroma.shape[1], sr=sr, hop_length=hop_length)


def stft_chroma(y: np.ndarray, sr: int = SR, hop_length: int = HOP_LENGTH, n_fft: int = 4096,
                normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """STFT-based chroma (fixed window, Eq. 5) for the comparison the paper motivates in §II-B."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, norm=None)
    if normalize:
        chroma = l2_normalize(chroma)
    return chroma, frame_times(chroma.shape[1], sr=sr, hop_length=hop_length)


def harmonic_part(y: np.ndarray, margin: float = 3.0) -> np.ndarray:
    """Harmonic component from harmonic/percussive source separation (a first fix for RWC-P, §IV-B)."""
    return librosa.effects.harmonic(y, margin=margin)

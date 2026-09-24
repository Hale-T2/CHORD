"""First-order hidden Markov model over 24 chord states, decoded with Viterbi (paper §III-B).

Paper ↔ code
------------
Eq. (12) transition matrix with self-transition bias p     :func:`uniform_transition_matrix`
Eq. (13) circular (key-invariant) diagonal averaging         :func:`key_invariant`
Eq. (14) log-likelihood recurrence D_log(c, t)               :func:`viterbi_log`
Eq. (15) softmax emissions from cosine similarities          :func:`softmax_emissions`
Eq. (16)-(18) back-pointers, terminal state, back-tracking   :func:`viterbi_log`
"""

from __future__ import annotations

import numpy as np

from .templates import N_STATES, TEMPLATES, cosine_similarity

EPS = float(np.finfo(np.float64).eps)          # ln(EPS) ≈ −36.04, the paper's "log ε"


# ---------------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------------

def uniform_transition_matrix(n_states: int = N_STATES, p_self: float = 0.90) -> np.ndarray:
    """Eq. (12): a_ij = p on the diagonal and (1 − p)/(n − 1) elsewhere."""
    if not 0.0 < p_self < 1.0:
        raise ValueError("p_self must lie strictly between 0 and 1")
    A = np.full((n_states, n_states), (1.0 - p_self) / (n_states - 1))
    np.fill_diagonal(A, p_self)
    return A


def normalize_rows(A: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Make every row a probability distribution."""
    return A / (A.sum(axis=1, keepdims=True) + eps)


def _circular_average(M: np.ndarray) -> np.ndarray:
    """Eq. (13) for one 12×12 block: average along cyclic diagonals.

    (M_ti)_{i,j} = 1/12 · Σ_k M_{(i+k) mod 12, (j+k) mod 12}. Shifting both indices by the same
    k transposes the whole block to another key, so the result depends only on the
    interval j − i, never on the absolute root.
    """
    out = np.zeros_like(M, dtype=np.float64)
    for k in range(12):
        idx = (np.arange(12) + k) % 12
        out += M[np.ix_(idx, idx)]
    return out / 12.0


def key_invariant(A: np.ndarray) -> np.ndarray:
    """Eq. (13): apply circular diagonal averaging to the four 12×12 quality blocks of a 24×24 matrix.

    Blocks: major→major, major→minor, minor→major, minor→minor. Row sums are preserved, so a
    stochastic matrix stays stochastic. Applying this to the *uniform* matrix of Eq. (12)
    changes nothing (every diagonal is already constant) — it matters when ``A`` was
    **estimated from data**, see :func:`estimate_transition_matrix`.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (24, 24):
        raise ValueError("key_invariant expects a 24×24 major/minor transition matrix")
    out = np.empty_like(A)
    for bi in range(2):
        for bj in range(2):
            sl_i, sl_j = slice(bi * 12, (bi + 1) * 12), slice(bj * 12, (bj + 1) * 12)
            out[sl_i, sl_j] = _circular_average(A[sl_i, sl_j])
    return normalize_rows(out)


def estimate_transition_matrix(state_sequences, n_states: int = N_STATES, alpha: float = 1.0,
                               make_key_invariant: bool = True) -> np.ndarray:
    """Count frame-to-frame transitions in labelled sequences (``-1`` = no chord is skipped).

    ``alpha`` is an additive (Laplace) smoothing pseudo-count so unseen transitions are not
    impossible. This is the "A_est" the paper mentions before Eq. (13).
    """
    counts = np.full((n_states, n_states), float(alpha))
    for seq in state_sequences:
        seq = np.asarray(seq)
        prev, cur = seq[:-1], seq[1:]
        ok = (prev >= 0) & (cur >= 0)
        np.add.at(counts, (prev[ok], cur[ok]), 1.0)
    A = normalize_rows(counts)
    return key_invariant(A) if (make_key_invariant and n_states == 24) else A


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

def softmax_emissions(similarity: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Eq. (15): b_c(x[t]) = exp(r_c / τ) / Σ_c' exp(r_c' / τ), column by column.

    The paper uses τ = 1. Because cosine similarities of non-negative chroma lie in [0, 1],
    the resulting distributions are almost flat (exp(0) … exp(1)); the transition prior then
    dominates the decoding. ``temperature`` < 1 sharpens the emissions — a knob worth exploring.
    """
    z = similarity / temperature
    z = z - z.max(axis=0, keepdims=True)                    # numerical safety, softmax is shift-invariant
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Viterbi in the log domain
# ---------------------------------------------------------------------------

def viterbi_log(B: np.ndarray, A: np.ndarray, pi: np.ndarray | None = None,
                eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Most probable state path for emission probabilities ``B[S, T]`` and transitions ``A[S, S]``.

    Implements Eqs. (14)–(18) with logarithms instead of products to avoid underflow:

        D_log(c, t) = ln(b_c(x[t]) + ε) + max_j [ D_log(j, t−1) + ln(a_jc + ε) ]      (14)
        E(c, t)     = argmax_j [ … ]                                                    (16)
        s*_T        = argmax_c D_log(c, T)                                              (17)
        s*_t        = E(s*_{t+1}, t+1)                                                  (18)

    Returns ``(path[T], D_log[S, T], E[S, T])``. ``E[c, t]`` stores the best predecessor of
    state ``c`` at frame ``t`` (the paper writes this as E(i, t−1)).
    """
    B = np.asarray(B, dtype=np.float64)
    S, T = B.shape
    if A.shape != (S, S):
        raise ValueError("transition matrix must be S×S")
    log_pi = np.log((np.full(S, 1.0 / S) if pi is None else np.asarray(pi)) + eps)
    log_A = np.log(A + eps)
    log_B = np.log(B + eps)

    D = np.empty((S, T))
    E = np.zeros((S, T), dtype=np.int64)
    D[:, 0] = log_pi + log_B[:, 0]
    for t in range(1, T):
        cand = D[:, t - 1][:, None] + log_A            # cand[j, c] = D(j, t−1) + ln a_jc
        E[:, t] = cand.argmax(axis=0)
        D[:, t] = cand[E[:, t], np.arange(S)] + log_B[:, t]

    path = np.empty(T, dtype=np.int64)
    path[-1] = D[:, -1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = E[path[t + 1], t + 1]
    return path, D, E


def hmm_decode(chroma: np.ndarray, A: np.ndarray | None = None, p_self: float = 0.90,
               temperature: float = 1.0, templates: np.ndarray = TEMPLATES) -> tuple[np.ndarray, dict]:
    """Full CQT-HMM decoder of the paper: cosine similarities → softmax emissions → Viterbi.

    Returns ``(states[T], info)`` where ``info`` holds the similarity matrix, emissions,
    transition matrix and the accumulated log-likelihood matrix.
    """
    if A is None:
        A = uniform_transition_matrix(templates.shape[0], p_self)
    sim = cosine_similarity(chroma, templates)
    B = softmax_emissions(sim, temperature)
    path, D, E = viterbi_log(B, A)
    return path, {"similarity": sim, "emissions": B, "transition": A, "loglik": D, "backpointers": E}

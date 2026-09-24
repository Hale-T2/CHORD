import numpy as np
import librosa

from chordlab import templates, hmm, synth, features, evaluate


def test_templates_shape_and_content():
    T = templates.build_templates()
    assert T.shape == (24, 12)
    assert T.sum(axis=1).tolist() == [3.0] * 24
    # C:maj = C E G, A:min = A C E
    assert np.flatnonzero(T[templates.label_to_index("C:maj")]).tolist() == [0, 4, 7]
    assert np.flatnonzero(T[templates.label_to_index("A:min")]).tolist() == [0, 4, 9]


def test_label_roundtrip():
    for i, lab in enumerate(templates.CHORD_LABELS):
        assert templates.label_to_index(lab) == i
        assert templates.index_to_label(i) == lab
    assert templates.label_to_index("N") == -1
    assert templates.label_to_index("Ab:min") == templates.label_to_index("G#:min")


def test_template_decode_on_clean_chord():
    y, iv, labs = synth.render_progression(["D:min"], 1.0, timbre="organ")
    chroma, _ = features.cqt_chroma(y)
    states, sim = templates.template_decode(chroma)
    assert sim.shape[0] == 24
    assert np.bincount(states, minlength=24).argmax() == templates.label_to_index("D:min")


def test_uniform_transition_matrix_is_stochastic_and_key_invariant():
    A = hmm.uniform_transition_matrix(p_self=0.9)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(np.diag(A), 0.9)
    assert np.allclose(hmm.key_invariant(A), A)          # Eq. 13 is a no-op on Eq. 12


def test_key_invariant_depends_only_on_interval():
    rng = np.random.default_rng(0)
    A = hmm.normalize_rows(rng.random((24, 24)))
    K = hmm.key_invariant(A)
    assert np.allclose(K.sum(axis=1), 1.0)
    for block in (slice(0, 12), slice(12, 24)):
        M = K[block, block]
        for shift in range(12):
            assert np.allclose(M[0], np.roll(M[shift], -shift))


def test_estimate_transition_matrix_counts():
    seq = np.array([0, 0, 0, 7, 7, -1, 7, 12])
    A = hmm.estimate_transition_matrix([seq], alpha=0.0, make_key_invariant=False)
    assert A[0, 0] == 2 / 3 and A[0, 7] == 1 / 3           # 0→0 twice, 0→7 once
    assert A[7, 7] == 0.5 and A[7, 12] == 0.5              # the -1 frame is skipped


def test_softmax_emissions_sum_to_one_and_temperature_sharpens():
    sim = np.array([[0.9, 0.2], [0.3, 0.8], [0.1, 0.1]])
    B1 = hmm.softmax_emissions(sim, 1.0)
    B2 = hmm.softmax_emissions(sim, 0.1)
    assert np.allclose(B1.sum(axis=0), 1.0)
    assert B2[0, 0] > B1[0, 0]


def test_viterbi_matches_librosa():
    rng = np.random.default_rng(1)
    B = rng.random((24, 200)); B /= B.sum(axis=0, keepdims=True)
    A = hmm.uniform_transition_matrix(p_self=0.8)
    path, D, E = hmm.viterbi_log(B, A)
    ref = librosa.sequence.viterbi(B, A, p_init=np.full(24, 1 / 24))
    assert np.array_equal(path, ref)
    assert D.shape == (24, 200) and E.shape == (24, 200)


def test_hmm_smooths_isolated_errors():
    # 100 frames of state 3 with a few single-frame glitches: HMM should remove them
    sim = np.full((24, 100), 0.2); sim[3] = 0.95
    for t in (10, 40, 70):
        sim[3, t], sim[9, t] = 0.2, 0.95
    # a glitch costs 2 switches = 2·(ln 0.9 − ln(0.1/23)) ≈ 10.7 nats; the emission gain of a
    # glitch is (0.95 − 0.2)/τ, so τ = 0.2 (3.75 nats) is smoothed away while τ = 0.05 (15 nats) is not
    A = hmm.uniform_transition_matrix(p_self=0.9)
    smooth_path, _, _ = hmm.viterbi_log(hmm.softmax_emissions(sim, temperature=0.2), A)
    sharp_path, _, _ = hmm.viterbi_log(hmm.softmax_emissions(sim, temperature=0.05), A)
    assert (smooth_path == 3).all()
    assert (sharp_path[[10, 40, 70]] == 9).all()

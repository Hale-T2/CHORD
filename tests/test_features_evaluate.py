import numpy as np
import pytest

from chordlab import features, evaluate, synth, templates


def test_q_factor_and_eps():
    assert abs(features.Q_FACTOR - 16.817) < 1e-3          # Eq. (9) evaluates to 16.82, not 17.03
    assert abs(np.log(features.EPS) - (-36.04)) < 0.01     # paper's log-eps


def test_cqt_window_lengths_follow_eq7():
    N = features.cqt_window_lengths(sr=22050, fmin=features.FMIN, n_bins=24)
    f = features.cqt_center_frequencies(features.FMIN, 24)
    assert np.allclose(N * f, features.Q_FACTOR * 22050)
    assert N[0] > N[-1]                                     # long windows at low frequencies


def test_pure_tone_lands_in_the_right_pitch_class():
    sr = 22050
    y = synth.render_note(librosa_note("A4"), 1.0, sr, timbre="flute", amp=0.5)
    chroma, times = features.cqt_chroma(y, sr)
    assert chroma.shape[0] == 12 and len(times) == chroma.shape[1]
    assert chroma.mean(axis=1).argmax() == templates.PITCH_CLASSES.index("A")
    assert np.allclose(np.linalg.norm(chroma, axis=0), 1.0, atol=1e-6)     # Eq. (10)


def librosa_note(name):
    import librosa
    return int(librosa.note_to_midi(name))


def test_stereo_to_mono_eq4():
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert np.allclose(features.stereo_to_mono(x), [2.0, 2.0, 2.0])
    assert np.allclose(features.stereo_to_mono(x.T), [2.0, 2.0, 2.0])


@pytest.mark.parametrize("label,quality,bitmap", [
    ("C:maj7", "C:maj", "C:maj"), ("E:7/3", "E:maj", "E:maj"), ("A:min7", "A:min", "A:min"),
    ("Gb:maj6", "F#:maj", "F#:maj"), ("C:dim", "N", "N"), ("C:aug", "N", "N"), ("C:hdim7", "N", "N"),
    ("B:sus4(b7)", "N", "N"), ("F#:maj(*3,*5)", "F#:maj", "N"), ("A:min(*b3,*5)", "A:min", "N"),
    ("N", "N", "N"), ("X", "N", "N"), ("Db", "C#:maj", "C#:maj"), ("C:minmaj7", "C:min", "C:min"),
])
def test_reduce_label(label, quality, bitmap):
    assert evaluate.reduce_label(label, "quality") == quality
    assert evaluate.reduce_label(label, "bitmap") == bitmap


def test_labels_to_frames_and_back():
    iv = np.array([[0.0, 1.0], [1.0, 2.5], [3.0, 4.0]])
    labs = ["C:maj7", "N", "A:min"]
    times = np.array([0.1, 0.9, 1.2, 2.6, 3.5, 4.5])
    states = evaluate.labels_to_frames(iv, labs, times)
    assert states.tolist() == [0, 0, -1, -1, 21, -1]
    est_iv, est_lab = evaluate.frames_to_intervals(np.array([0, 0, 5, 5, 5]), np.arange(5) * 0.5, 0.5)
    assert est_lab == ["C:maj", "F:maj"]
    assert np.allclose(est_iv, [[0.0, 1.0], [1.0, 2.5]])


def test_frame_prf_definitions():
    truth = np.array([0, 0, 0, -1, 5, 5])
    pred = np.array([0, 0, 7, 7, 5, -1])
    m = evaluate.frame_prf(pred, truth)
    # TP: frames 0,1,4 ; FP: frame 2 (wrong chord) + frame 3 (chord on N) ; FN: frame 2 + frame 5
    assert (m["TP"], m["FP"], m["FN"]) == (3, 2, 2)
    assert abs(m["F1"] - 0.6) < 1e-12


def test_full_pipeline_beats_chance_on_toy_progression():
    from chordlab import hmm
    y, iv, labs = synth.render_progression(synth.DEMO_PROGRESSION, 4.0, timbre="piano")
    chroma, times = features.cqt_chroma(y)
    truth = evaluate.labels_to_frames(iv, labs, times)
    st_t, _ = templates.template_decode(chroma)
    st_h, _ = hmm.hmm_decode(chroma, temperature=0.1)
    assert evaluate.frame_prf(st_t, truth)["F1"] > 0.9
    assert evaluate.frame_prf(st_h, truth)["F1"] > 0.9
    est_iv, est_lab = evaluate.frames_to_intervals(st_h, times, times[1] - times[0])
    scores = evaluate.mir_eval_scores(iv, labs, est_iv, est_lab)
    assert scores["majmin"] > 0.9


def test_frames_to_intervals_never_overlap_and_are_accepted_by_mir_eval():
    # regression: shared boundaries, no floating-point overlaps on a long track
    import mir_eval.chord as mchord
    hop_s = features.HOP_LENGTH / features.SR
    times = features.frame_times(1400, features.SR)
    rng = np.random.default_rng(0)
    states = np.repeat(rng.integers(0, 24, 120), 12)[:1400]
    iv, lab = evaluate.frames_to_intervals(states, times, hop_s)
    assert (iv[:-1, 1] == iv[1:, 0]).all()
    assert np.allclose(iv[:, 1] - iv[:, 0], np.diff(np.append(times[np.r_[0, np.flatnonzero(np.diff(states)) + 1]], times[-1] + hop_s)))
    ref_iv = np.array([[0.0, times[-1] + hop_s]])
    mchord.evaluate(ref_iv, ["C:maj"], iv, lab)          # raises if intervals overlap

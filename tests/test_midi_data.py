import numpy as np
import pretty_midi

from chordlab import midi_chords, data, evaluate


def _toy_midi():
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="PIANO")
    bass = pretty_midi.Instrument(program=33, name="Bass")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    melo = pretty_midi.Instrument(program=73, name="Melo")
    # bar 1 (0-2 s): C major, bar 2 (2-4 s): A minor 7 (A C E G): bass decides the root
    for t0, notes in ((0.0, [60, 64, 67]), (2.0, [60, 64, 67])):
        for n in notes:
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=n, start=t0, end=t0 + 2.0))
    bass.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=2.0))    # C2
    bass.notes.append(pretty_midi.Note(velocity=100, pitch=45, start=2.0, end=4.0))    # A2
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=4.0))
    for t in np.arange(0, 4, 0.25):
        melo.notes.append(pretty_midi.Note(velocity=80, pitch=int(70 + (t * 7) % 5), start=t, end=t + 0.25))
    pm.instruments.extend([piano, bass, drums, melo])
    return pm


def test_beat_and_bar_grids():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    nums = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    bg = midi_chords.beat_grid(beats, 4.0)
    assert bg.shape == (8, 2) and bg[-1, 1] == 4.0
    br = midi_chords.bar_grid(beats, nums, 4.0)
    assert np.allclose(br, [[0.0, 2.0], [2.0, 4.0]])


def test_symbolic_labels_with_bass_tiebreak(tmp_path):
    pm = _toy_midi()
    path = tmp_path / "toy.mid"
    pm.write(str(path))
    beats = np.arange(0, 4, 0.5)
    iv, labels, conf = midi_chords.derive_chords(path, beats, np.tile([1, 2, 3, 4], 2), grid="bar", end_time=4.0)
    assert labels == ["C:maj", "A:min"]
    assert np.allclose(iv, [[0.0, 2.0], [2.0, 4.0]])
    assert (conf > 0.7).all()
    # without the bass bonus the A C E G set is a tie between A:min and C:maj
    iv2, labels2, _ = midi_chords.derive_chords(path, beats, None, grid="beat", end_time=4.0, bass_bonus=0.0)
    assert labels2[0] == "C:maj"


def test_deflicker():
    assert midi_chords.deflicker(np.array([1, 1, 5, 1, 1, 2, 2])).tolist() == [1, 1, 1, 1, 1, 2, 2]


def test_save_and_load_chords_roundtrip(tmp_path):
    iv = np.array([[0.0, 1.234], [1.234, 5.0]])
    labs = ["N", "F#:min"]
    p = data.save_chords(tmp_path / "RWC-R" / "RWC_R999.csv", iv, labs)
    assert p.read_text().splitlines()[0] == "t_start;t_end;chord"
    iv2, labs2 = data.load_chords("RWC_R999", derived_dir=tmp_path)
    assert labs2 == labs and np.allclose(iv2, iv)


def test_zenodo_urls():
    assert data.zenodo_url("R").endswith("RWC-R.zip?download=1")
    assert data.ZENODO_FILES["R"][1] == "63e3b6263656a42c592ce1e90a88caa3"

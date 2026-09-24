"""chordlab — teaching code for *following a chord-recognition paper down to the data*.

The package mirrors the pipeline of

    H. Türeli, "CHORD: Comparing HMM and Constant-Q Representation for
    Decoding Harmony", 2026.

Modules
-------
data          locate / download / parse the RWC 2.0 audio (Zenodo) and annotations (GitHub)
features      mono mixing, resampling, STFT vs. CQT, chromagram, L2 normalisation   (paper §II)
templates     24 binary major/minor templates and cosine-similarity decoding          (paper §III-A)
hmm           transition matrix, key-invariant averaging, softmax emissions, Viterbi (paper §III-B)
evaluate      Harte-label reduction, frame-wise P/R/F1 and mir_eval comparison        (paper §IV)
midi_chords   derive chord labels from aligned MIDI + beat grids (needed for RWC-R)
synth         tiny additive synthesiser to test everything without downloading data
plots         helper plots used throughout the notebooks
paper_results the paper's Tables I/II for side-by-side comparison
"""

__version__ = "0.1.0"

from . import data, features, templates, hmm, evaluate, midi_chords, synth, plots, paper_results  # noqa: F401,E402

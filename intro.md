# From Paper to Data

## Chord recognition with the constant-Q transform, template matching and hidden Markov models — following one paper all the way to its dataset

Most tutorials on audio chord recognition explain an algorithm. This one explains a *paper*: how to read its equations, turn them into code, find and download the dataset it used, reconstruct a ground truth it does not fully describe, reproduce its tables, and then ask your own questions of the data.

The paper is

> H. Türeli, **"CHORD: Comparing HMM and Constant-Q Representation for Decoding Harmony"**, 2026 {cite:p}`Tureli2026`.

It builds a chord recogniser in two variants — stateless **template matching** on constant-Q chroma, and a 24-state **hidden Markov model** decoded with **Viterbi** — and evaluates both on the RWC Music Database {cite:p}`Goto2002,Goto2003,Goto2006`, which was re-released in 2026 as an open, CC BY-NC corpus {cite:p}`Balke2026,Mueller2025`. Every chapter of this book is anchored in a section and a set of equations of that paper, and every chapter is a runnable Jupyter notebook.

```{admonition} Who this is for
:class: tip
Students and early researchers who know some Python and some linear algebra, have heard of the Fourier transform, and want to learn how music information retrieval research is actually *done* — including the unglamorous parts: licenses, checksums, annotation formats and under-specified ground truth.
```

## The map

| # | Chapter | Paper | What you will do |
|---|---------|-------|------------------|
| 0 | [How to run](notebooks/00_setup) | — | Colab or local install, where the data goes |
| 1 | [Finding and fetching the data](notebooks/01_finding_and_fetching_data) | §I-A | Read a Zenodo record, verify an MD5, clone the annotations, discover that RWC-R has no chord labels |
| 2 | [From waveform to pitch](notebooks/02_waveform_to_pitch) | §I, §II-A, Eqs. 1–4 | 12-TET, MIDI numbers, stereo → mono → 22 050 Hz |
| 3 | [STFT versus CQT](notebooks/03_stft_vs_cqt) | §II-B, Eqs. 5–9, Figs. 1–2 | Overtones, spectral crowding, the constant-Q idea, time–frequency tiling |
| 4 | [The chromagram](notebooks/04_chromagram) | §II-C, Eq. 10 | Fold 84 bins into 12 pitch classes, L2 normalisation with ε, what drums and singers do |
| 5 | [Template matching](notebooks/05_template_matching) | §III-A, Eq. 11 | 24 binary templates, cosine similarity, frame-wise decoding and its flicker |
| 6 | [HMM and Viterbi](notebooks/06_hmm_viterbi) | §III-B, Eqs. 12–18 | Transition prior, key-invariant averaging, softmax emissions, log-domain Viterbi, the τ–p trade-off |
| 7 | [Ground truth from MIDI](notebooks/07_ground_truth_from_midi) | §IV-A | Derive chord labels from aligned MIDI and validate the recipe on RWC-P |
| 8 | [Reproducing Table I](notebooks/08_evaluation_rwc_r) | §IV-A, Eqs. 19–20, Table I | Both decoders on RWC-R, the paper's metric vs. `mir_eval`, where the HMM gain comes from |
| 9 | [Scaling to RWC-P](notebooks/09_scaling_rwc_p) | §IV-B, Fig. 4, Table II | 100 tracks, failure analysis with evidence, a learned prior, significance |

Chapters 2–6 run on **synthetic audio** and need no download. Chapter 7 needs only the annotation repository. Chapters 8 and 9 need the RWC audio (320 MB and 4.1 GB respectively).

## What is different about this book

* **Paper ↔ code callouts.** Every notebook starts by naming the equations it implements and the `chordlab` functions that implement them. The package is small (eight modules) and written to be read.
* **Data first.** Getting to the dataset is a chapter, not a footnote. Licenses, checksums, annotation formats and dataset versions are treated as part of the science.
* **Honest reproduction.** The curated RWC annotations contain chord labels for RWC-P only. Chapter 7 reconstructs RWC-R labels from the aligned MIDI and measures how much such a reconstruction can be trusted. Chapter 8 then compares our Table I with the paper's, decimal by decimal, and explains the differences instead of hiding them.
* **Questions the paper leaves open.** Why do softmax emissions of cosine similarities make the HMM lag half a second? Is the HMM's advantage statistically significant over 100 tracks? Does harmonic–percussive separation fix the drum failures? You will find out.

## How this relates to other resources

This book stands on the shoulders of Meinard Müller's *Fundamentals of Music Processing* and its FMP notebooks {cite:p}`Mueller2015`, which cover chroma features, template-based and HMM-based chord recognition in depth and generality. Read them for the theory in full; read this book to see one specific paper carried through, end to end, on the newly opened RWC corpus. The tooling is the community's: `librosa` {cite:p}`McFee2015`, `mir_eval` {cite:p}`Raffel2014`, `pretty_midi`, and the `rwc-annotations` repository maintained by the RWC 2.0 team.

## Licenses and citation

* Code (the `chordlab` package, tests, notebook code) — MIT. Text — CC BY 4.0.
* RWC audio — © AIST, **CC BY-NC 4.0**, downloaded at runtime from Zenodo and never stored in this repository. Cite the five publications the record asks for; they are in {doc}`references`.
* `derived_annotations/` — derived from CC BY-NC 4.0 material and therefore CC BY-NC 4.0.
* The paper is © IEEE. This book restates its equations and reports its numbers for comparison but does not reproduce its text or figures; please read the original.

```{admonition} Contributing
:class: note
Found a mistake, or a better way to derive the RWC-R ground truth? Open an issue or a pull request — the repository button is in the top bar.
```

# From Paper to Data: chord recognition with CQT, templates and HMMs

A Jupyter Book (+ a small Python package) that follows one paper all the way to its dataset:

> H. Türeli, *"CHORD: Comparing HMM and Constant-Q Representation for Decoding Harmony"*, 2026.

I developed ten notebooks take a student from the paper's equations to a working chord recogniser, to the
RWC Music Database on Zenodo (re-released open in 2026), to a reconstructed ground truth, to a
reproduction of the paper's Table I and Fig. 4 — and then beyond the paper.

**Live site:** `https://github.com/Hale-T2/CHORD` 
## Contents

```
intro.md                      landing page and chapter map
notebooks/
  00_setup.md                 how to run (Colab / local), where the data goes
  01_finding_and_fetching_data.ipynb   Zenodo, MD5, license, annotations repo, metadata
  02_waveform_to_pitch.ipynb           12-TET, stereo→mono→22.05 kHz          (paper Eqs. 1–4)
  03_stft_vs_cqt.ipynb                 overtones, STFT vs CQT, Q, tiling      (Eqs. 5–9, Figs. 1–2)
  04_chromagram.ipynb                  fold to 12 pitch classes, L2 norm      (Eq. 10)
  05_template_matching.ipynb           24 templates, cosine similarity        (Eq. 11)
  06_hmm_viterbi.ipynb                 transitions, key invariance, Viterbi   (Eqs. 12–18)
  07_ground_truth_from_midi.ipynb      RWC-R chord labels from aligned MIDI, validated on RWC-P
  08_evaluation_rwc_r.ipynb            Table I reproduction, paper metric vs mir_eval
  09_scaling_rwc_p.ipynb               RWC-P, Fig. 4, Table II, failure analysis, significance
chordlab/                     the code the notebooks import
  data.py        Zenodo download + MD5, annotation clone, parsers   templates.py  Eq. 11
  features.py    Eqs. 4–10 (mono, CQT, chroma, L2)                  hmm.py        Eqs. 12–18
  evaluate.py    label reduction, P/R/F1 (Eqs. 19–20), mir_eval     midi_chords.py  MIDI → chords
  synth.py       toy band for data-free notebooks                   plots.py, paper_results.py
derived_annotations/chords_from_midi/RWC-R/   the reconstructed RWC-R chord labels (CC BY-NC 4.0)
tests/                        35 unit tests (no data needed)
_config.yml, _toc.yml, references.bib          Jupyter Book configuration
.github/workflows/            deploy.yml (GitHub Pages), tests.yml (pytest)
```

## Quick start

```bash
git clone https://github.com/Hale-T2/CHORD.git
cd chord-recognition-book
pip install -e .            # chordlab + dependencies (Python ≥ 3.10; needs libsndfile for soundfile)
pytest -q                   # 35 tests, a few seconds
jupyter lab notebooks/      # start with 02 (no data) or 01 (downloads RWC-R, 320 MB)
```

Or open any notebook in Colab via its badge — the first cell clones the repo and installs the package.

Data goes to `./data/` (override with the `CHORDLAB_DATA` environment variable). Audio is never committed.

## Deploying the site

1. Fork / push this repository to GitHub.
2. Replace the placeholder `Hale-T2/CHORD` everywhere:
   ```bash
   grep -rl "YOUR-GITHUB-USER" --include="*.ipynb" --include="*.yml" --include="*.md" . | xargs sed -i 's#Hale-T2/CHORD#<user>/<repo>#g'
   ```
   and fill in `author` in `_config.yml` and the copyright line in `LICENSE`.
3. In the repository settings, set **Pages → Source** to *GitHub Actions*.
4. Push to `main`. The `deploy-book` workflow builds the book and publishes it.

Notebooks are **not executed** during the build (`execute_notebooks: "off"`), so what the site shows is
what is saved in the `.ipynb` files. Notebooks 02–07 ship with outputs (they run on synthetic audio and
the annotation repository). Notebooks 01, 08 and 09 need the RWC audio: run them once locally, save,
and commit the outputs so readers see the tables and figures.

To build locally: `pip install -r requirements-docs.txt && jupyter-book build .` (Jupyter Book 1.x).

## Things to know before you publish

* **Ground truth.** The curated `rwc-annotations` repository has chord labels for RWC-P only. The RWC-R
  labels in `derived_annotations/` were derived from the aligned MIDI (notebook 07 explains and validates
  the recipe: ≈ 73 % frame agreement with human labels on RWC-P). The paper does not state how 
  RWC-R labels were produced — confirm with the me if you want to match Table I exactly.
* **I had two small numeric slips in the paper**, which are handled gently in notebooks 02 and 03: Eq. (9) gives
  Q = 1/(2^(1/12) − 1) ≈ 16.82 (the paper prints 17.034), and the equal-tempered fifth deviates from
  3:2 by ≈ 0.11 % / 2 cents (the paper says 0.17 %).
* **Licenses.** Code MIT; text CC BY 4.0; derived annotations CC BY-NC 4.0; RWC audio CC BY-NC 4.0
  (downloaded at runtime, cite the publications in `references.bib`); the paper is © IEEE — the book
  restates equations and compares against reported numbers but does not reproduce text or figures.
  Ask me via email haley.tureli@gmail.com before reusing any figure.

## Acknowledgements

RWC Music Database © AIST (Goto et al. 2002, 2003, 2006), re-released as RWC 2.0 by Balke et al. (2026).
Built with librosa, mir_eval, pretty_midi and Jupyter Book. Inspired by Meinard Müller's FMP notebooks and
by Barış Bozkurt's Turkish *Python ile Ses İşlemeye Giriş*.

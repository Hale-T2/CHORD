# 0. How to run this book

Every chapter is a Jupyter notebook you can read here, run in the browser on Google Colab, or run on your own machine.

## Option A — Google Colab (nothing to install)

Click the ![Colab](https://colab.research.google.com/assets/colab-badge.svg) badge at the top of any notebook (or the rocket icon in the top bar). The first cell clones this repository into the Colab runtime and installs `chordlab`. A free Colab CPU runtime is enough for everything except running notebook 9 on all 100 RWC-P tracks.

Downloads land in the Colab session and disappear when it ends. To keep the RWC audio across sessions, uncomment the *Google Drive* cell that appears in the data notebooks; it mounts your Drive and points `chordlab` at a folder there.

## Option B — locally

```bash
git clone https://github.com/YOUR-GITHUB-USER/chord-recognition-book.git
cd chord-recognition-book
python -m venv .venv && source .venv/bin/activate     # or: conda create -n chords python=3.11
pip install -e .                                       # installs chordlab and its dependencies
jupyter lab notebooks/
```

Python 3.10 or newer. `librosa` needs a working `soundfile`; on Linux install `libsndfile1` if you get an import error.

## Where the data goes

`chordlab.data` stores everything under `./data/` (audio in `data/audio/RWC-<subset>/`, annotations in `data/rwc-annotations/`). Set the environment variable `CHORDLAB_DATA` to move it elsewhere, e.g. `export CHORDLAB_DATA=/media/big-disk/rwc`. The folder is in `.gitignore` — audio never enters the repository.

| Notebook | Needs | Size / time |
|---|---|---|
| 1 | Zenodo (RWC-R) + annotation repo | 320 MB + 85 MB, ~2 min on Colab |
| 2–6 | nothing (synthetic audio); notebook 6 also reads the annotation repo | seconds |
| 7 | annotation repo (MIDI + beats + RWC-P chords) | ~1 min |
| 8 | RWC-R audio + derived labels (shipped) | ~3 min |
| 9 | RWC-P: single tracks via `remotezip` (default) or the full 4.1 GB | minutes to hours |

## Running the tests

```bash
pip install pytest
pytest tests/
```

The tests check the maths against the paper (e.g. that Viterbi agrees with `librosa.sequence.viterbi`, that Eq. 13 is a no-op on Eq. 12) and run in a few seconds.

## Building the website yourself

```bash
pip install -r requirements-docs.txt
jupyter-book build .
open _build/html/index.html
```

Notebooks are **not** executed during the build (`execute_notebooks: "off"` in `_config.yml`): the ones that need audio are too heavy for CI. Run a notebook, save it with its outputs, and commit — the site shows what you saved. The GitHub Actions workflow in `.github/workflows/deploy.yml` rebuilds and publishes the site on every push to `main`.

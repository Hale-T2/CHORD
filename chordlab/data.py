"""Getting to the data: RWC 2.0 audio (Zenodo) and annotations (GitHub).

Everything a student needs to go from "the paper says RWC" to files on disk:

* :func:`download_rwc`      – stream a subset zip from Zenodo and verify its MD5
* :func:`ensure_rwc`        – return the audio folder, downloading on first use
* :func:`ensure_annotations`– shallow-clone ``rwc-music/rwc-annotations``
* :func:`load_metadata`, :func:`track_ids`, :func:`audio_path`
* :func:`load_chords`, :func:`save_chords`, :func:`load_beats`, :func:`midi_path`
* :func:`fetch_single_track`– pull one WAV out of the 4.1 GB RWC-P zip (HTTP range requests)

The RWC Music Database is distributed under **CC BY-NC 4.0**. Cite Goto et al.
(2002, 2003), Goto (2006), Müller/Balke/Goto (2025) and Balke et al. (2026)
when you use it — see ``references.bib``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Where things live online
# ---------------------------------------------------------------------------

#: Zenodo record of the CC BY-NC 4.0 re-release ("RWC 2.0", v2, 16 Feb 2026).
ZENODO_RECORD = "18656623"
ZENODO_DOI = "10.5281/zenodo.18656623"
ZENODO_COMMUNITY = "https://zenodo.org/communities/rwc-music/"

#: subset -> (file name on Zenodo, MD5 published on the record page, size in MB)
ZENODO_FILES = {
    "R": ("RWC-R.zip", "63e3b6263656a42c592ce1e90a88caa3", 320),
    "P": ("RWC-P.zip", "960a11a2d7fb603ad0dae8428f53d4f0", 4100),
    "C": ("RWC-C.zip", "2ac9139c4f03a65885ae0d0d299f67f8", 3000),
    "J": ("RWC-J.zip", "c5d7d989e1afb8257ec50a3696d90c37", 2100),
    "G": ("RWC-G.zip", "e78cddfb6fa639bcb6a61ad873f3cceb", 3900),
}

ANNOTATIONS_REPO = "https://github.com/rwc-music/rwc-annotations.git"
ANNOTATIONS_ZIP = "https://github.com/rwc-music/rwc-annotations/archive/refs/heads/main.zip"

SUBSET_NAMES = {"R": "Royalty-Free", "P": "Popular", "C": "Classical", "J": "Jazz", "G": "Genre"}


def zenodo_url(subset: str) -> str:
    """Direct download URL of one subset zip."""
    fname = ZENODO_FILES[subset.upper()][0]
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{fname}?download=1"


# ---------------------------------------------------------------------------
# Local layout
# ---------------------------------------------------------------------------

def data_root() -> Path:
    """Root folder for downloaded data.

    Set the environment variable ``CHORDLAB_DATA`` to relocate it (e.g. to a
    mounted Google Drive folder on Colab so downloads survive the session).
    Defaults to ``./data`` relative to the current working directory.
    """
    return Path(os.environ.get("CHORDLAB_DATA", "data")).expanduser().resolve()


def set_data_root(path: str | os.PathLike) -> Path:
    """Convenience setter for :func:`data_root`."""
    os.environ["CHORDLAB_DATA"] = str(path)
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def audio_dir(subset: str, root: Path | None = None) -> Path:
    return (root or data_root()) / "audio" / f"RWC-{subset.upper()}"


def annotations_dir(root: Path | None = None) -> Path:
    return (root or data_root()) / "rwc-annotations"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def md5sum(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """MD5 of a file, streamed so multi-GB zips do not need to fit in RAM."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: str | os.PathLike, expected_md5: str | None = None,
                  chunk: int = 1 << 20, quiet: bool = False) -> Path:
    """Stream ``url`` to ``dest`` with a plain-text progress report and optional MD5 check."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "chordlab/0.1 (educational)"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and not quiet and done / total >= next_report:
                print(f"  {dest.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB", flush=True)
                next_report += 0.10
    if expected_md5 is not None:
        got = md5sum(dest)
        if got != expected_md5:
            raise IOError(f"MD5 mismatch for {dest.name}: got {got}, expected {expected_md5}. "
                          "The download is probably incomplete — delete the file and retry.")
        if not quiet:
            print(f"  MD5 verified: {got}")
    return dest


def download_rwc(subset: str, root: Path | None = None, verify: bool = True,
                 extract: bool = True, keep_zip: bool = False) -> Path:
    """Download one RWC subset from Zenodo, verify it and unzip it into ``audio/RWC-<subset>``.

    RWC-R is ~320 MB (fine on Colab); RWC-P is ~4.1 GB — consider caching it on
    Google Drive or use :func:`fetch_single_track` for individual songs.
    """
    subset = subset.upper()
    fname, md5, size_mb = ZENODO_FILES[subset]
    root = root or data_root()
    zip_path = root / "zips" / fname
    target = audio_dir(subset, root)
    if not zip_path.exists():
        print(f"Downloading {fname} (~{size_mb} MB) from Zenodo record {ZENODO_RECORD} ...")
        download_file(zenodo_url(subset), zip_path, expected_md5=md5 if verify else None)
    elif verify:
        print(f"{fname} already present, verifying MD5 ...")
        got = md5sum(zip_path)
        if got != md5:
            raise IOError(f"{zip_path} is corrupt (md5 {got}); delete it and rerun.")
    if extract:
        target.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {target} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        if not keep_zip:
            zip_path.unlink()
    return target


def rwc_available(subset: str, root: Path | None = None) -> bool:
    """True if at least one WAV of the subset is on disk."""
    d = audio_dir(subset, root)
    return d.exists() and any(d.rglob("*.wav"))


def ensure_rwc(subset: str, root: Path | None = None, download: bool = True) -> Path:
    """Return the folder holding the WAVs of ``subset``; download it first if allowed."""
    root = root or data_root()
    if rwc_available(subset, root):
        return audio_dir(subset, root)
    if not download:
        raise FileNotFoundError(
            f"RWC-{subset.upper()} audio not found under {audio_dir(subset, root)}.\n"
            f"Run chordlab.data.download_rwc('{subset.upper()}') or download "
            f"{ZENODO_FILES[subset.upper()][0]} from https://doi.org/{ZENODO_DOI}")
    return download_rwc(subset, root)


def ensure_annotations(root: Path | None = None, update: bool = False) -> Path:
    """Shallow-clone (or reuse) the ``rwc-music/rwc-annotations`` repository.

    Falls back to downloading the GitHub zip archive when ``git`` is unavailable.
    """
    dest = annotations_dir(root)
    if dest.exists() and (dest / "metadata.csv").exists():
        if update and (dest / ".git").exists():
            subprocess.run(["git", "-C", str(dest), "pull", "-q"], check=False)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        print(f"Cloning {ANNOTATIONS_REPO} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", "-q", ANNOTATIONS_REPO, str(dest)], check=True)
    else:  # pragma: no cover
        zip_path = dest.parent / "rwc-annotations-main.zip"
        download_file(ANNOTATIONS_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest.parent)
        (dest.parent / "rwc-annotations-main").rename(dest)
        zip_path.unlink()
    return dest


# ---------------------------------------------------------------------------
# Metadata and annotation parsers (formats documented in the repo READMEs)
# ---------------------------------------------------------------------------

def load_metadata(ann_dir: Path | None = None) -> pd.DataFrame:
    """``metadata.csv`` as a DataFrame (semicolon separated, one row per track)."""
    ann_dir = ann_dir or ensure_annotations()
    df = pd.read_csv(ann_dir / "metadata.csv", sep=";")
    return df.set_index("RWCID")


def track_ids(subset: str, ann_dir: Path | None = None) -> list[str]:
    """All track IDs of a subset, e.g. ``['RWC_R001', ..., 'RWC_R015']``."""
    df = load_metadata(ann_dir)
    return sorted(df.index[df["CollID"] == subset.upper()].tolist())


def audio_path(rwc_id: str, root: Path | None = None) -> Path:
    """Locate ``<rwc_id>.wav`` anywhere below the subset's audio folder (zip layouts vary)."""
    subset = rwc_id.split("_")[1][0]
    d = audio_dir(subset, root)
    hits = list(d.rglob(f"{rwc_id}.wav"))
    if not hits:
        raise FileNotFoundError(f"{rwc_id}.wav not found below {d}")
    return hits[0]


def _read_semicolon(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def load_chords(rwc_id: str, ann_dir: Path | None = None,
                derived_dir: str | os.PathLike | None = None) -> tuple[np.ndarray, list[str]]:
    """Chord annotation of a track as ``(intervals[N, 2] in seconds, labels[N])``.

    RWC-P: curated human annotations from ``01_annotations_preprocessed/chords/RWC-P``.
    RWC-R: the annotation repo has **no** chord labels, so pass ``derived_dir`` pointing at
    labels derived from the aligned MIDI (see notebook 07 / ``derived_annotations/``).
    """
    subset = rwc_id.split("_")[1][0]
    if derived_dir is not None:
        path = Path(derived_dir) / f"RWC-{subset}" / f"{rwc_id}.csv"
    else:
        ann_dir = ann_dir or ensure_annotations()
        path = ann_dir / "01_annotations_preprocessed" / "chords" / f"RWC-{subset}" / f"{rwc_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No chord annotation at {path}. (Only RWC-P has curated chords; for RWC-R "
            "derive them from MIDI with chordlab.midi_chords or use derived_annotations/.)")
    df = _read_semicolon(path)
    intervals = df[["t_start", "t_end"]].to_numpy(dtype=float)
    return intervals, df["chord"].astype(str).tolist()


def save_chords(path: str | os.PathLike, intervals: np.ndarray, labels: Sequence[str]) -> Path:
    """Write chords in the same ``t_start;t_end;chord`` format as the annotation repo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_start": np.round(intervals[:, 0], 3),
                  "t_end": np.round(intervals[:, 1], 3),
                  "chord": list(labels)}).to_csv(path, sep=";", index=False)
    return path


def load_beats(rwc_id: str, ann_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Beat annotation as ``(times in seconds, beat position within the bar)``."""
    subset = rwc_id.split("_")[1][0]
    ann_dir = ann_dir or ensure_annotations()
    df = _read_semicolon(ann_dir / "01_annotations_preprocessed" / "beats" / f"RWC-{subset}" / f"{rwc_id}.csv")
    return df["t"].to_numpy(dtype=float), df["beat"].to_numpy(dtype=int)


def midi_path(rwc_id: str, ann_dir: Path | None = None) -> Path:
    """Path of the audio-aligned MIDI file of a track."""
    subset = rwc_id.split("_")[1][0]
    ann_dir = ann_dir or ensure_annotations()
    return ann_dir / "01_annotations_preprocessed" / "MIDI_aligned" / f"RWC-{subset}" / f"{rwc_id}.mid"


def annotation_inventory(ann_dir: Path | None = None) -> pd.DataFrame:
    """Which annotation types exist for which subset (counts of files)."""
    ann_dir = ann_dir or ensure_annotations()
    base = ann_dir / "01_annotations_preprocessed"
    rows = {}
    for kind in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for sub in sorted(p.name for p in (base / kind).iterdir() if p.is_dir()):
            n = sum(1 for f in (base / kind / sub).iterdir() if f.suffix in {".csv", ".mid"})
            rows.setdefault(sub, {})[kind] = n
    return pd.DataFrame(rows).T.fillna(0).astype(int).sort_index()


# ---------------------------------------------------------------------------
# Optional: pull a single track out of the big RWC-P zip without downloading it all
# ---------------------------------------------------------------------------

def fetch_single_track(rwc_id: str, root: Path | None = None) -> Path:
    """Extract one WAV directly from the Zenodo zip using HTTP range requests.

    Needs the optional ``remotezip`` package (``pip install remotezip``). Zenodo serves
    byte ranges, so only the requested member (~30-50 MB) is transferred instead of 4.1 GB.
    """
    try:
        from remotezip import RemoteZip  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install remotezip  (or download the whole subset with download_rwc)") from exc
    subset = rwc_id.split("_")[1][0]
    target = audio_dir(subset, root)
    target.mkdir(parents=True, exist_ok=True)
    with RemoteZip(zenodo_url(subset)) as zf:
        members = [m for m in zf.namelist() if m.endswith(f"{rwc_id}.wav")]
        if not members:
            raise FileNotFoundError(f"{rwc_id}.wav not in {ZENODO_FILES[subset][0]}")
        zf.extract(members[0], path=target)
    return audio_path(rwc_id, root)


def iter_tracks(subset: str, ids: Iterable[str] | None = None, root: Path | None = None):
    """Yield ``(rwc_id, wav_path)`` for a subset (or a chosen list of IDs)."""
    for rwc_id in (ids or track_ids(subset)):
        yield rwc_id, audio_path(rwc_id, root)

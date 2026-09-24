# Derived annotations

## `chords_from_midi/RWC-R/`

Chord labels for the 15 RWC-R tracks, **derived automatically** from the audio-aligned MIDI files and
beat annotations of [`rwc-music/rwc-annotations`](https://github.com/rwc-music/rwc-annotations) with
`chordlab.midi_chords.derive_chords` (beat grid, bass bonus 0.5, single-beat blips removed). Notebook 07
documents and validates the procedure: on RWC-P, where human chord annotations exist, the same procedure
agrees with the human major/minor labels on about 73 % of chord frames.

* Format: semicolon-separated `t_start;t_end;chord`, Harte syntax, 24 major/minor triads plus `N` — the
  same as the curated chord files for RWC-P.
* Time axis: audio time of the RWC 2.0 WAV files (the MIDI was re-aligned to the audio for RWC 2.0).
* These are **not** the labels used in the paper (which does not describe how its RWC-R labels were made).
  Treat them as one reasonable reconstruction; regenerate them with different settings via notebook 07.
* License: derived from CC BY-NC 4.0 material, therefore **CC BY-NC 4.0**. Cite the RWC publications listed
  in `references.bib` when you use them.

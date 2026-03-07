# Audio stitcher

Tool for comping recordings: start from a base take, detect problems, and patch in segments from other takes.

## Setup

Requires [uv](https://docs.astral.sh/uv/). All dependencies are managed automatically.

```
cd automation
uv sync
```

## Usage

```
uv run python app.py <folder> <base_name> [take1,take2,...]
```

- `<folder>` — path to a project folder containing WAV recordings
- `<base_name>` — substring matching the base take filename (the main recording to build on)
- `[take1,take2,...]` — optional comma-separated substrings to filter which takes to load

Then open http://localhost:5000.

### Examples

Use all takes, with 0961 as the base:
```
uv run python app.py "../Projets/20250517 - Etude en fa mineur - Mendelssohn" 0961
```

Use only two specific takes:
```
uv run python app.py "../Projets/20250517 - Etude en fa mineur - Mendelssohn" 0961 0961,0964
```

## Web UI

- **Timeline** — segments color-coded by issue severity (green = clean, red = problems). Blue dot = patched.
- **Playback** — bottom bar plays any full take or the composite. Yellow highlight tracks current segment.
- **Detail panel** — click a segment to see detected issues and all available takes.
  - **Listen** — preview any take's version of that segment
  - **Use this** — replace the base with that take for this segment
  - **Revert to base** — undo a patch
- **Auto-patch** — apply patches automatically above a severity threshold
- **Export** — download the composite WAV

### Keyboard shortcuts

- Arrow keys — navigate segments
- Space — play selected segment (base take)
- Enter — toggle full playback

## How it works

1. Loads WAV files, computes onset/beat/chroma features (librosa)
2. Aligns takes via Dynamic Time Warping on chroma features
3. Segments by detected beats (grouped into bars)
4. Diagnoses the base take for timing issues, hesitations, noise spikes, wrong notes
5. Patches are time-stretched to match the base take's tempo and volume-matched at boundaries
6. Export applies equal-power crossfading at patch boundaries

# Accuracy benchmarks

Reports produced by `scripts/benchmark.py` — one JSON per pipeline run against
a YouTube reference, organized by video ID (for a local file with `--reference`,
by the audio file's name without its extension).

## Layout

```
benchmarks/
  <video_id>/
    <UTC-timestamp>.json   # one per run
  _tmp/                    # gitignored: scratch dir for downloaded audio/subs
```

## Re-running

```bash
# Install dev deps (jiwer, yt-dlp)
pip install -r requirements-dev.txt

# Run against a YouTube URL (engine: TRANSCRIPTION_ENGINE, else the platform
# default: parakeet on Apple Silicon, whisper elsewhere)
python -m scripts.benchmark "https://www.youtube.com/watch?v=<id>"

# Use Parakeet on Apple Silicon for ~3-4x real-time
python -m scripts.benchmark "<url>" --engine parakeet

# Benchmark a specific Whisper model (--model selects Whisper unless --engine is given)
python -m scripts.benchmark "<url>" --model large-v3

# Compare against a hand-corrected reference instead of YouTube captions
python -m scripts.benchmark path/to/audio.wav --reference path/to/truth.vtt
```

The harness reads the project `.env` like every other entry point and never
diarizes. The report's `engine` and `model` fields record what actually ran;
for a Parakeet run, `model` is the `PARAKEET_MODEL` checkpoint.

The harness prefers manual (uploader-provided) captions over auto-generated;
falls back to auto if no manual track exists. The `ref_source` field in the
report records which was used.

## Interpreting WER

| Metric | Meaning                                                            |
|--------|--------------------------------------------------------------------|
| `wer`  | Word Error Rate: (S + I + D) / reference_words                     |
| `mer`  | Match Error Rate: (S + I + D) / (S + I + D + hits)                 |
| `wil`  | Word Information Lost: how much reference info is missing in output |
| `wip`  | Word Information Preserved: 1 - WIL                                |

WER against YouTube auto-captions is a *parity check*, not absolute accuracy —
auto-captions are themselves ASR and noisy. Manual captions are usually
cleaned of filler words ("uh", "um"), so a verbatim ASR output will show
inflated WER even when phonetically correct. Look at `substitutions` for the
count of actual misheard words.

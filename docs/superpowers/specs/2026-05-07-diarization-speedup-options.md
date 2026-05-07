# Diarization speedup — options

**Date:** 2026-05-07
**Status:** Exploration outline (not a committed plan)
**Context:** After landing parakeet-mlx, ASR on a 2hr 13min recording takes 1m 28s. The same recording's diarization (`pyannote/speaker-diarization-community-1` on MPS) takes **16m 44s** — ~12% of audio duration, ~11x the wall time of ASR. Diarization is now the bottleneck for hybrid (parakeet + speaker labels) workflows.

## What's slow and why

`DiarizationEngine.diarize()` runs the full pyannote pipeline on the entire audio in one shot:
1. Voice activity detection (VAD)
2. Speaker embedding extraction (segment-level neural network)
3. Agglomerative clustering of embeddings
4. Resegmentation

On a 2hr file this produced **3,304 raw segments** before downstream merging. The cost dominates regardless of segment count — it's the embedding extraction sweep that scales with audio length.

Caching is already in place (`CacheManager.cache_diarization`), so repeat runs on the same file are free. The cost only matters for first-time and unique-file workflows.

## Ranked options (most-leverage first)

### 1. Try a different pyannote pipeline (lowest effort, unknown gain)

The current default is `pyannote/speaker-diarization-community-1`. Other community pipelines exist with different speed/accuracy tradeoffs. `DIARIZATION_MODEL` is already an env var — drop-in test.

Candidates worth a single timing run on the Schindler file:
- `pyannote/speaker-diarization-3.1` (the prior default before community-1)
- `pyannote/speaker-diarization-3.0`

**Effort:** 0 code. One env var change, one transcribe run.
**Risk:** Quality regression on this specific recording. Speaker count guesses may differ.
**Decision criterion:** If any of these comes in under ~10 minutes for the Schindler file with comparable speaker assignment, switch.

### 2. sherpa-onnx-diarization (medium effort, likely 5-10x speedup)

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ships ONNX-quantized models for VAD + speaker embedding + clustering. CPU-only (no GPU/MPS) but the per-segment work is dramatically lighter than pyannote — published benchmarks show 3-5 minutes for 2hr audio on a recent laptop CPU.

Integration shape mirrors what we just did for parakeet:
- Add `DIARIZATION_ENGINE` env var (`pyannote` | `sherpa`, default `pyannote`)
- Add `SherpaDiarizationEngine` alongside the existing pyannote engine
- Output shape stays the same `[{start, end, speaker}]` so `_combine_segments_with_speakers` is unchanged

**Effort:** ~half a day. Mirrors the parakeet integration: protocol + factory + new implementation + tests with mocks.
**Risk:** Lower accuracy on overlapping speech and short turns. Worth quality-checking on the Schindler hybrid output before committing.
**Decision criterion:** If sherpa gets diarization under 5 minutes AND the speaker turns line up reasonably with pyannote's output (eyeball test on 2-3 minutes of the Schindler interview), commit it.

### 3. NeMo MSDD + TitaNet (highest effort, highest ceiling)

NVIDIA's NeMo offers Multi-Scale Diarization Decoder with TitaNet embeddings. State-of-the-art accuracy and well-optimized for both CPU and CUDA. On Apple Silicon there's no MPS path for NeMo, so this is CPU-bound — but the architecture is faster than pyannote on the same hardware.

**Effort:** ~1-2 days. NeMo has a heavy install footprint (drags in Hydra, OmegaConf, lightning, etc.) and the Python API is more complex than pyannote's pipeline. Probably worth gating with a `requirements-nemo.txt` extras file rather than adding to the base requirements.
**Risk:** Install bloat. Possibly transitive-dep conflicts (we already hit one with pyannote 3.x → 4.x in this session).
**Decision criterion:** Only worth pursuing if (1) and (2) don't get us to a sub-5-minute target, AND if archival genealogy quality justifies the install weight.

### 4. Pre-VAD trim, then diarize only speech regions (low-medium effort)

Run faster-whisper's bundled Silero VAD (`faster_whisper.vad.get_speech_timestamps`) first, concatenate only speech regions into a shortened audio file, diarize that, then map speaker labels back to the original timeline.

For interview recordings where 30-50% of the audio is silence, pauses, or non-speech, this gives a proportional speedup — the Schindler file at 2hr with ~30% silence would diarize as if it were ~1.5hr, knocking ~5 minutes off.

**Effort:** A morning. New helper module, plus careful handling of timestamp mapping.
**Risk:** VAD false-negatives drop content. Mitigated by erring toward keeping borderline regions.
**Decision criterion:** Worth doing as a multiplier on top of (1) or (2) — they compose. Not worth doing alone for a 2-3 minute speedup.

### 5. Diart (streaming pyannote) — probably not worth it here

[diart](https://github.com/juanmc2005/diart) does streaming diarization on top of pyannote. It's optimized for live audio, not batch. For batch files it's typically not faster than vanilla pyannote. Mentioned for completeness; skip unless live transcription becomes a goal.

## Recommended order

1. Spend 5 minutes on **option 1**. If it wins, done.
2. If not, take **option 2 + option 4** as a unit — sherpa-onnx with VAD pre-trim. That's a half-day of work for an expected 3-4x speedup.
3. **Option 3** only if archival quality demands it after (2).

## What this doesn't change

- ASR engine selection — Whisper and Parakeet stay as-is.
- Cache layer — already engine-aware after the parakeet work.
- The output contract (`{start, end, speaker}` list) — every option above produces this shape.

## Open questions worth resolving before committing to any of these

- What's the genealogy-archival quality bar? Is "near pyannote" acceptable, or do we need "match pyannote"? This determines how willing we are to trade accuracy for speed.
- Are there other long recordings in the pipeline beyond Schindler? If this is a one-off file, do nothing — caching means the existing run never repeats. If there's a backlog of family-history recordings, pursue (1) and (2).
- Apple Silicon vs Linux deployment — sherpa-onnx is portable, NeMo is mostly CUDA-flavored. Mirrors the same platform tension we resolved for parakeet (parakeet-mlx Apple-only by design).

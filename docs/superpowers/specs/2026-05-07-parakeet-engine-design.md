# Parakeet ASR Engine — Design

**Date:** 2026-05-07
**Status:** Approved, pending implementation plan
**Scope:** Add NVIDIA Parakeet-TDT-0.6B-v3 as a selectable transcription engine, with Whisper retained as the default.

## Motivation

For English and ~25 European languages, Parakeet-TDT-0.6B-v3 beats Whisper-large-v3-turbo on both accuracy (~6.3% vs ~7.4% WER, Open ASR Leaderboard mid-2026) and speed (roughly an order of magnitude faster). Whisper still wins for the 99+-language long tail and stays the default. Users opt in to Parakeet when their content matches its language coverage and they want the speed/accuracy gain.

## Non-goals

- Replacing Whisper.
- Parakeet streaming / live transcription. Whisper retains the streaming path; Parakeet is batch-only in v1.
- Supporting Parakeet on non-Apple-Silicon platforms. The runtime chosen here (`parakeet-mlx`) is macOS arm64 only by design — see "Runtime choice" below.
- Reusing Handy's bundled INT8 ONNX weights. They are the wrong format for `parakeet-mlx` (which needs MLX-format weights from HuggingFace). Documented as a caveat; not implemented.

## Runtime choice

`parakeet-mlx >= 0.5.1` (Apple Silicon only).

Considered and rejected: `onnx-asr` (cross-platform, would have let Linux/Docker users run Parakeet, but ~3-5x slower on Apple Silicon than `parakeet-mlx`). The user prioritized Mac speed over portability. On non-Apple-Silicon platforms, `engine=parakeet` is rejected at config validation — Whisper remains fully functional.

## Architecture

`src/transcription/engine.py` is refactored to expose a small protocol with two implementations and a factory. The output shape is unchanged from today, so `transcriber.py` and the diarization alignment code work without modification.

```python
class ASREngine(Protocol):
    def ensure_model_loaded(self) -> None: ...
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]: ...

class WhisperEngine:        # current TranscriptionEngine, renamed; retains streaming
class ParakeetEngine:       # new

def make_asr_engine(config, test_mode=False) -> ASREngine:
    if config.transcription_engine == "parakeet":
        return ParakeetEngine(config, test_mode=test_mode)
    return WhisperEngine(config, test_mode=test_mode)

# Backward-compat alias — preserves existing imports and the 91-test suite.
TranscriptionEngine = WhisperEngine
```

`src/transcriber.py` constructs the engine via `make_asr_engine(self.config, test_mode=test_mode)` instead of instantiating `TranscriptionEngine` directly.

### Output shape (unchanged contract)

Both engines return `List[Dict[str, Any]]`:

```python
{
  "start": float,        # seconds
  "end": float,          # seconds
  "text": str,
  "words": [
    {"start": float, "end": float, "word": str},
    ...
  ],
}
```

The diarization alignment in `transcriber.py:_combine_segments_with_speakers` overlaps segment-level ranges (not words), so any segment-shaped output works. Word timestamps remain populated for SRT/VTT/JSON formatters.

### Streaming

`WhisperEngine.transcribe_stream` and `start_async_transcription` are preserved unchanged. `ParakeetEngine` does not implement streaming — calling those paths with `engine=parakeet` raises a clear error pointing the user at the batch path. `parakeet-mlx` does not natively support incremental decoding; adding it is a meaningful follow-up if needed.

## ParakeetEngine internals

### Loading

- `from parakeet_mlx import from_pretrained` is imported lazily inside `_load_model`, not at module top. Importing the module on Linux/Docker stays cheap and fails only when actually invoked.
- `from_pretrained(config.parakeet_model)` accepts either an HF model id or a local path. Default: `mlx-community/parakeet-tdt-0.6b-v3`. First run auto-downloads ~600MB to `~/.cache/huggingface/`.

### Long-form chunking

`parakeet-mlx` has built-in long-form decoding via `transcribe(path, chunk_duration=120, overlap_duration=15)`. The library splits long audio into 120s windows with 15s overlap and reconciles boundaries internally. We pass these defaults statically — no user-facing knob, no custom VAD chunking. If a real-world failure mode emerges later, that becomes a follow-up tuning question.

### Output mapping

`parakeet-mlx` returns an `AlignedResult` with `.sentences[]`. Each sentence has `.text`, `.start`, `.end`, and `.tokens[]` where each token has `.text`, `.start`, `.end`. We map:

- sentence → segment (one transcribed segment per sentence)
- sentence.tokens → segment.words (token.text → word.word)

### Caching

`CacheManager` cache keys today use `prefix="transcription"` regardless of engine, which means switching engines would return stale cross-engine cache hits. Both engines now pass an engine identifier into the prefix:

- `WhisperEngine`: `transcription-whisper-{whisper_model_size}`
- `ParakeetEngine`: `transcription-parakeet-{parakeet_model_id_slug}`

This is the smallest change that prevents the bug. The cache-manager prefix wiring already supports it (the `prefix` argument is just a string).

### Test mode

Mirrors the existing `MockWhisperModel` pattern. `ParakeetEngine` in `test_mode=True` uses a `MockParakeetModel` whose `transcribe` returns an `AlignedResult`-shaped object with two sentences each containing a couple of tokens. This avoids importing `parakeet-mlx` in tests.

## Configuration

`src/config.py` adds:

```python
self.transcription_engine = os.getenv("TRANSCRIPTION_ENGINE", "whisper").strip().lower()
self.parakeet_model = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
```

`Config.validate()` adds:

- `transcription_engine in {"whisper", "parakeet"}`, else fail.
- If `transcription_engine == "parakeet"`: require `sys.platform == "darwin"` and `platform.machine() == "arm64"`. On any other platform, fail with: *"Parakeet requires Apple Silicon (macOS arm64). Set TRANSCRIPTION_ENGINE=whisper or run on macOS arm64."*

`PARAKEET_MODEL` accepts either an HF id or a local absolute path — same input shape `parakeet-mlx.from_pretrained` accepts.

## Dependencies

`requirements.txt` gains one line with a platform marker:

```
parakeet-mlx>=0.5.1; sys_platform == "darwin" and platform_machine == "arm64"
```

The marker means:

- On macOS arm64: `pip install -r requirements.txt` pulls `parakeet-mlx` and its transitive deps (mlx, mlx-lm, etc.).
- On Linux/Docker/Intel macOS: `pip install` is a no-op for that line. The Whisper path remains fully functional. Attempting to set `TRANSCRIPTION_ENGINE=parakeet` is rejected at config validation, before any import is attempted.

## Tests

- **New** `tests/unit/test_parakeet_engine.py` — mirrors structure of existing engine tests. Patches `parakeet_mlx.from_pretrained` to return `MockParakeetModel`. Covers:
  - Engine loads in test_mode without importing `parakeet-mlx`.
  - `transcribe` returns the standard segment shape with populated `words`.
  - Cache key includes the engine identifier (whisper cache and parakeet cache do not collide).
  - Config validation rejects `engine=parakeet` on non-darwin / non-arm64.
- **New fixtures** in `tests/conftest.py`: `mock_parakeet_model`, `test_parakeet_engine`.
- **One end-to-end test** in `tests/integration/test_transcriber.py` parametrized over `engine in ["whisper", "parakeet"]`, asserting the same output shape and that diarization combination produces speaker-labeled output for both engines.
- **Existing 91 tests untouched.** The `TranscriptionEngine = WhisperEngine` alias keeps existing imports valid; the default engine remains `whisper`.

## README updates

A new "Transcription engines" section documents:

- Default: `whisper` (works everywhere, 99+ languages).
- Opt-in: `TRANSCRIPTION_ENGINE=parakeet` for ~6.3% WER and an order-of-magnitude speedup on Apple Silicon. ~25 European languages only. macOS arm64 only.
- First run auto-downloads ~600MB MLX-format weights to `~/.cache/huggingface/`. Override with `PARAKEET_MODEL=/path/to/local/checkpoint` or a different HF id.
- Caveat: Handy's bundled INT8 ONNX weights are *not* compatible with `parakeet-mlx` (different format). Users who want to reuse Handy weights would need a different runtime (e.g. `onnx-asr`); out of scope here.

## Acceptance

1. `TRANSCRIPTION_ENGINE=parakeet python -m src.transcribe <file>` produces a transcript in the same output formats (txt/srt/vtt/json/pretty) as the Whisper path.
2. Diarization (`pyannote`) still produces speaker-labeled output with Parakeet.
3. All 91 existing tests pass; new tests cover the Parakeet path with mocks.
4. README documents the new env vars, the auto-download, the platform constraint, and the Handy-weights caveat.

## Risks & flagged tradeoffs

- **First-run network dependency.** First Parakeet run requires internet to fetch weights from HuggingFace. CI uses mocks so this does not affect tests; offline first-run by an end user surfaces a HuggingFace download error. Documented in the README.
- **Platform lock-in.** Choosing `parakeet-mlx` over `onnx-asr` ties Parakeet to Apple Silicon. If a Docker/Linux user later asks for Parakeet, the answer is "use `onnx-asr` instead" — a separate engine implementation, not a small switch. The Whisper path stays fully cross-platform.
- **Streaming gap.** `engine=parakeet` plus a streaming entry point will error out cleanly. If streaming with Parakeet becomes a real requirement, that's a meaningful follow-up — `parakeet-mlx` does not natively support incremental decoding the way `faster-whisper` does.

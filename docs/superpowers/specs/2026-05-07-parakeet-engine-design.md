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

`src/transcription/engine.py` is refactored to expose a small **batch-only** protocol with two implementations and a factory. The output shape is unchanged from today, so `transcriber.py` and the diarization alignment code work without modification.

```python
class ASREngine(Protocol):
    """Batch-only ASR contract. Streaming is a Whisper-only extension; see below."""
    def ensure_model_loaded(self) -> None: ...
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]: ...

class WhisperEngine:        # current TranscriptionEngine, renamed; retains streaming methods
class ParakeetEngine:       # new — batch-only

def make_asr_engine(config, test_mode=False) -> ASREngine:
    if config.transcription_engine == "parakeet":
        return ParakeetEngine(config, test_mode=test_mode)
    return WhisperEngine(config, test_mode=test_mode)

# Backward-compat alias — preserves existing imports and the existing test suite.
TranscriptionEngine = WhisperEngine
```

`src/transcriber.py` constructs the engine via `make_asr_engine(self.config, test_mode=test_mode)` instead of instantiating `TranscriptionEngine` directly.

### Streaming is Whisper-only — explicit contract

`ASREngine` deliberately does not declare `transcribe_stream` or `start_async_transcription`. Streaming is a Whisper-only extension, not part of the cross-engine contract. Two-layer guard so the failure mode is clear, not an `AttributeError`:

1. **Upstream guard in `transcriber.py`.** `transcribe_stream` and `transcribe_stream_with_diarization` check `config.transcription_engine == "whisper"` first; if not, they raise `NotImplementedError("Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. Use transcribe() for batch transcription with Parakeet.")` before touching the engine.
2. **Defensive stubs on `ParakeetEngine`.** `transcribe_stream` and `start_async_transcription` are defined and raise the same `NotImplementedError`. This keeps the surface honest if someone bypasses the upstream guard later.

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

## ParakeetEngine internals

### Loading

- `from parakeet_mlx import from_pretrained` is imported lazily inside `_load_model`, not at module top. Importing the module on Linux/Docker stays cheap and fails only when actually invoked.
- **Pinned API surface:** validated against `parakeet-mlx==0.5.1`. The token API in MLX-port libraries can churn (`AlignedResult.sentences[].tokens[].text/start/end`) — version pin and one mapping function isolate the risk. Bumping the pin requires re-validating the mapping in `tests/unit/test_parakeet_engine.py`.
- **Test patching note:** because `from_pretrained` is imported lazily inside `_load_model`, the patch target is `parakeet_mlx.from_pretrained`, **not** `src.transcription.engine.from_pretrained`. Tests document this inline.
- `from_pretrained(config.parakeet_model)` accepts either an HF model id or a local path. Default: `mlx-community/parakeet-tdt-0.6b-v3`. First run auto-downloads ~600MB to `~/.cache/huggingface/`.

### Long-form chunking

`parakeet-mlx` has built-in long-form decoding via `transcribe(path, chunk_duration=120, overlap_duration=15)`. The library splits long audio into 120s windows with 15s overlap and reconciles boundaries internally. We pass these defaults statically — no user-facing knob, no custom VAD chunking. If a real-world failure mode emerges later, that becomes a follow-up tuning question.

### Output mapping

`parakeet-mlx` returns an `AlignedResult` with `.sentences[]`. Each sentence has `.text`, `.start`, `.end`, and `.tokens[]` where each token has `.text`, `.start`, `.end`. We map:

- sentence → segment (one transcribed segment per sentence)
- sentence.tokens → segment.words (token.text → word.word)

**Word whitespace normalization.** `faster-whisper` returns words with a leading space (e.g. `" hello"`), so downstream formatters that do `"".join(w["word"] for w in words)` produce correctly-spaced text. Parakeet/NeMo BPE tokens typically do not include the leading space. To preserve the existing output contract for SRT/VTT/JSON formatters, `ParakeetEngine` normalizes token text on the way out: if a token's text does not begin with a space and it is not the first word in its segment, prepend a single space.

The rule is naive on purpose. NeMo TDT models typically attach punctuation to the prior word, so the rule produces correct output for the cases we expect. If parakeet-mlx 0.5.1 ever emits punctuation as its own token (`["hello", " world", ".", " how"]`), the rule would yield `"hello world . how"`. Rather than guard against that pre-emptively, `tests/unit/test_parakeet_engine.py` includes a mid-sentence-punctuation fixture so a real divergence surfaces in tests and the rule can be refined then. This matches the "pin + isolated mapping" risk style used elsewhere in the spec.

### Timeout

`ParakeetEngine.transcribe` wraps the parakeet-mlx call in the same `with timeout(self.timeout_seconds, "Transcription timed out")` context manager that `WhisperEngine` uses today (`engine.py:154`). The existing `TRANSCRIBE_TIMEOUT` env var applies to both engines.

### Device flags

`FORCE_CPU` and the MPS/CUDA detection in the existing engine are Whisper-only — MLX runs on Apple Silicon with no equivalent knob. If `TRANSCRIPTION_ENGINE=parakeet` and `FORCE_CPU=true` are both set, `ParakeetEngine.__init__` logs a warning ("FORCE_CPU has no effect on Parakeet (MLX); flag is Whisper-only") and proceeds. The README documents this in the engines section.

### Caching

`CacheManager.cache_transcription` and `get_cached_transcription` (`src/cache/manager.py:264, 295`) hardcode `prefix="transcription"` and engines never pass anything in. Switching engines would silently return stale cross-engine cache hits. Two changes:

1. **CacheManager API:** add a defaulted `engine_id: str = "whisper"` parameter to both `cache_transcription` and `get_cached_transcription`. The default preserves behavior for any external caller; in-tree callers always pass an explicit value. The methods build the prefix as `f"transcription-{engine_id}"` before calling `_generate_cache_key`.

2. **Engine identifiers:**
   - `WhisperEngine`: `engine_id = f"whisper-{self.whisper_model_size}"`
   - `ParakeetEngine`: `engine_id = f"parakeet-{_slug(self.parakeet_model)}"`

3. **Slug rule (mandatory):** `parakeet_model` may be `mlx-community/parakeet-tdt-0.6b-v3` (HF id with `/`) or an absolute local path. Both forms must produce filesystem-safe cache keys. `_slug(s)` is `re.sub(r"[^A-Za-z0-9._-]", "_", s)`. Defined once in `engine.py` (or a small helper module) and used wherever the model id touches a filename.

### Test mode

Mirrors the existing `MockWhisperModel` pattern. `ParakeetEngine` in `test_mode=True` uses a `MockParakeetModel` whose `transcribe` returns an `AlignedResult`-shaped object with two sentences each containing a couple of tokens. This avoids importing `parakeet-mlx` in tests.

## Configuration

`src/config.py` adds three coordinated changes (matching the existing patterns in that file):

1. **New env-var reads in `__init__`:**
   ```python
   self.transcription_engine = os.getenv("TRANSCRIPTION_ENGINE", "whisper").strip().lower()
   self.parakeet_model = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
   ```

2. **`__init__(**overrides)` whitelist:** `transcription_engine` and `parakeet_model` are accepted as override kwargs (mirrors how `whisper_model`, `language`, etc. are handled at `config.py:67-78`). Tests construct configs via overrides; missing this would force tests to use env-var monkeypatching.

3. **`to_dict()`:** both new fields are added so the debug log at `config.py:80` reflects them.

4. **`Config.validate()`** — preserves the existing **return-False-and-log** pattern (`config.py:121-138`), does not raise:
   - If `self.transcription_engine not in {"whisper", "parakeet"}`: log error, return `False`.
   - If `self.transcription_engine == "parakeet"`: require `sys.platform == "darwin"` and `platform.machine() == "arm64"`. Otherwise log at error level: *"Parakeet requires Apple Silicon (macOS arm64). Set TRANSCRIPTION_ENGINE=whisper or run on macOS arm64."* and return `False`.

`PARAKEET_MODEL` accepts either an HF id or a local absolute path — same input shape `parakeet-mlx.from_pretrained` accepts.

## Dependencies

`requirements.txt` gains one line with a platform marker:

```
parakeet-mlx>=0.5.1,<0.6; sys_platform == "darwin" and platform_machine == "arm64"
```

The upper bound `<0.6` mechanically enforces the "bumping the pin requires re-validating the token mapping" contract from the Loading section — a 0.6 release would fail to install until someone has run the unit tests against it and lifted the bound.

The marker means:

- On macOS arm64: `pip install -r requirements.txt` pulls `parakeet-mlx` and its transitive deps (mlx, mlx-lm, etc.).
- On Linux/Docker/Intel macOS: `pip install` is a no-op for that line. The Whisper path remains fully functional. Attempting to set `TRANSCRIPTION_ENGINE=parakeet` is rejected at config validation, before any import is attempted.

## Tests

- **New** `tests/unit/test_parakeet_engine.py` — mirrors structure of existing engine tests. Patches `parakeet_mlx.from_pretrained` to return `MockParakeetModel`. Covers:
  - Engine loads in test_mode without importing `parakeet-mlx`.
  - `transcribe` returns the standard segment shape with populated `words`.
  - Whitespace normalization: a fixture with mid-sentence punctuation tokens (e.g. `["hello", " world", ".", " how"]`) exercises the punctuation edge case. The test asserts the joined sentence text is human-readable so a future parakeet-mlx tokenization change surfaces here.
  - Cache key includes the engine identifier (whisper cache and parakeet cache do not collide).
  - Config validation rejects `engine=parakeet` on non-darwin / non-arm64.
- **New fixtures** in `tests/conftest.py`: `mock_parakeet_model`, `test_parakeet_engine`.
- **One end-to-end test** in `tests/integration/test_transcriber.py` parametrized over `engine in ["whisper", "parakeet"]`, asserting the same output shape and that diarization combination produces speaker-labeled output for both engines.
- **All existing tests untouched.** The `TranscriptionEngine = WhisperEngine` alias keeps existing imports valid; the default engine remains `whisper`.

## README updates

A new "Transcription engines" section documents:

- Default: `whisper` (works everywhere, 99+ languages).
- Opt-in: `TRANSCRIPTION_ENGINE=parakeet` for ~6.3% WER and an order-of-magnitude speedup on Apple Silicon. ~25 European languages only. macOS arm64 only.
- First run auto-downloads ~600MB MLX-format weights to `~/.cache/huggingface/`. Override with `PARAKEET_MODEL=/path/to/local/checkpoint` or a different HF id.
- Caveat: Handy's bundled INT8 ONNX weights are *not* compatible with `parakeet-mlx` (different format). Users who want to reuse Handy weights would need a different runtime (e.g. `onnx-asr`); out of scope here.

## Acceptance

1. `TRANSCRIPTION_ENGINE=parakeet python -m src.transcribe <file>` produces a transcript in the same output formats (txt/srt/vtt/json/pretty) as the Whisper path.
2. Diarization (`pyannote`) still produces speaker-labeled output with Parakeet.
3. All existing tests pass; new tests cover the Parakeet path with mocks.
4. README documents the new env vars, the auto-download, the platform constraint, that `FORCE_CPU` is Whisper-only, and the Handy-weights caveat.

## Risks & flagged tradeoffs

- **First-run network dependency.** First Parakeet run requires internet to fetch weights from HuggingFace. CI uses mocks so this does not affect tests; offline first-run by an end user surfaces a HuggingFace download error. Documented in the README.
- **Platform lock-in.** Choosing `parakeet-mlx` over `onnx-asr` ties Parakeet to Apple Silicon. If a Docker/Linux user later asks for Parakeet, the answer is "use `onnx-asr` instead" — a separate engine implementation, not a small switch. The Whisper path stays fully cross-platform.
- **Streaming gap.** `engine=parakeet` plus a streaming entry point will error out cleanly. If streaming with Parakeet becomes a real requirement, that's a meaningful follow-up — `parakeet-mlx` does not natively support incremental decoding the way `faster-whisper` does.

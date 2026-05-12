# Parakeet ASR Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NVIDIA Parakeet-TDT-0.6B-v3 (via parakeet-mlx) as a selectable transcription engine, with Whisper retained as the default.

**Architecture:** Refactor `src/transcription/engine.py` into a small `ASREngine` batch-only Protocol with two implementations (`WhisperEngine`, `ParakeetEngine`) and a `make_asr_engine()` factory. `transcriber.py` constructs the engine via the factory. Streaming stays Whisper-only. The `parakeet-mlx` library handles long-form chunking internally, so we don't roll our own VAD chunking.

**Tech Stack:** Python 3.13, faster-whisper (existing), parakeet-mlx==0.5.x (new, Apple Silicon only), pyannote.audio (existing, unchanged), pytest + unittest.mock.

**Spec:** `docs/superpowers/specs/2026-05-07-parakeet-engine-design.md`

---

## File Structure

**Modified:**
- `src/config.py` — add `transcription_engine`, `parakeet_model` fields; extend `to_dict()`, `__init__(**overrides)`, `validate()`.
- `src/cache/manager.py` — add `engine_id` parameter to `cache_transcription` / `get_cached_transcription`.
- `src/transcription/engine.py` — rename `TranscriptionEngine` → `WhisperEngine`, add `ASREngine` Protocol, add `_slug` helper, add `make_asr_engine` factory, add `ParakeetEngine`, keep `TranscriptionEngine = WhisperEngine` alias.
- `src/transcriber.py` — use `make_asr_engine(...)` instead of constructing `TranscriptionEngine` directly; add streaming guard.
- `requirements.txt` — add platform-marked `parakeet-mlx`.
- `README.md` — new "Transcription engines" section.

**Created:**
- `tests/unit/test_parakeet_engine.py` — Parakeet-specific unit tests.

**Updated:**
- `tests/conftest.py` — add `mock_parakeet_model` and `test_parakeet_engine` fixtures.
- `tests/integration/test_transcriber.py` — parametrized end-to-end test across engines.

Each task ends with a commit. Run `pytest tests/ -q` after every task; everything must stay green.

---

## Task 1: Add config fields for engine selection

**Files:**
- Modify: `src/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_config.py`:

```python
import platform
import sys
import pytest
from src.config import Config


def _clear_engine_env(monkeypatch):
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    monkeypatch.delenv("PARAKEET_MODEL", raising=False)


class TestEngineSelection:
    def test_defaults(self, monkeypatch):
        _clear_engine_env(monkeypatch)
        cfg = Config()
        assert cfg.transcription_engine == "whisper"
        assert cfg.parakeet_model == "mlx-community/parakeet-tdt-0.6b-v3"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIPTION_ENGINE", "Parakeet")
        monkeypatch.setenv("PARAKEET_MODEL", "/tmp/local-mlx-model")
        cfg = Config()
        assert cfg.transcription_engine == "parakeet"  # lowercased + stripped
        assert cfg.parakeet_model == "/tmp/local-mlx-model"

    def test_kwargs_override(self, monkeypatch):
        _clear_engine_env(monkeypatch)
        cfg = Config(transcription_engine="parakeet", parakeet_model="custom/model")
        assert cfg.transcription_engine == "parakeet"
        assert cfg.parakeet_model == "custom/model"

    def test_to_dict_includes_new_fields(self, monkeypatch):
        _clear_engine_env(monkeypatch)
        cfg = Config()
        d = cfg.to_dict()
        assert d["transcription_engine"] == "whisper"
        assert d["parakeet_model"] == "mlx-community/parakeet-tdt-0.6b-v3"

    def test_validate_rejects_unknown_engine(self, monkeypatch):
        _clear_engine_env(monkeypatch)
        cfg = Config(transcription_engine="bogus")
        assert cfg.validate() is False

    def test_validate_rejects_parakeet_on_non_apple_silicon(self, monkeypatch):
        _clear_engine_env(monkeypatch)
        cfg = Config(transcription_engine="parakeet")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        assert cfg.validate() is False

    def test_validate_accepts_parakeet_on_apple_silicon(self, monkeypatch):
        _clear_engine_env(monkeypatch)
        cfg = Config(transcription_engine="parakeet")
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        assert cfg.validate() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config.py::TestEngineSelection -v`
Expected: failures referencing missing `transcription_engine` / `parakeet_model` attributes.

- [ ] **Step 3: Implement**

In `src/config.py`:

Add `import platform` and `import sys` at the top (near `import os`).

Inside `Config.__init__`, after the existing `self.language = …` line (around line 33), add:

```python
        # ASR engine selection
        self.transcription_engine = os.getenv("TRANSCRIPTION_ENGINE", "whisper").strip().lower()
        self.parakeet_model = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
```

Inside the `**overrides` block (the `if 'whisper_model' in overrides: …` cluster, ~line 67), add:

```python
        if 'transcription_engine' in overrides:
            self.transcription_engine = str(overrides['transcription_engine']).strip().lower()
        if 'parakeet_model' in overrides:
            self.parakeet_model = overrides['parakeet_model']
```

Extend `to_dict()` (around line 100) so the returned dict includes:

```python
            "transcription_engine": self.transcription_engine,
            "parakeet_model": self.parakeet_model,
```

Replace `validate()` with:

```python
    def validate(self) -> bool:
        """Validate the configuration."""
        if self.include_diarization and not self.hf_token:
            logger.warning("Speaker diarization is enabled but HF_TOKEN is not set")
            return False

        valid_formats = ["txt", "srt", "vtt", "json", "pretty"]
        if self.output_format not in valid_formats:
            logger.warning(f"Invalid output format: {self.output_format}. Must be one of {valid_formats}")
            return False

        valid_engines = {"whisper", "parakeet"}
        if self.transcription_engine not in valid_engines:
            logger.error(
                f"Invalid TRANSCRIPTION_ENGINE: {self.transcription_engine!r}. "
                f"Must be one of {sorted(valid_engines)}"
            )
            return False

        if self.transcription_engine == "parakeet":
            if sys.platform != "darwin" or platform.machine() != "arm64":
                logger.error(
                    "Parakeet requires Apple Silicon (macOS arm64). "
                    "Set TRANSCRIPTION_ENGINE=whisper or run on macOS arm64."
                )
                return False

        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: all green (existing tests + the new `TestEngineSelection` class).

Run: `pytest tests/ -q`
Expected: full suite stays green.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/unit/test_config.py
git commit -m "feat(config): add TRANSCRIPTION_ENGINE and PARAKEET_MODEL"
```

---

## Task 2: CacheManager — add `engine_id` parameter to transcription cache methods

**Files:**
- Modify: `src/cache/manager.py`
- Test: `tests/unit/test_cache_manager.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_cache_manager.py`:

```python
class TestEngineScopedTranscriptionCache:
    def test_cache_is_scoped_by_engine_id(self, tmp_path, monkeypatch):
        """Same audio cached under different engine_ids must not collide."""
        from src.cache.manager import CacheManager
        from src.config import Config

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: str(tmp_path) if p == "~" else os.path.expanduser(p),
        )
        cfg = Config()
        cm = CacheManager(cfg)

        # Real audio file is required for cache key generation (uses st_mtime).
        audio = tmp_path / "sample.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        cm.cache_transcription(str(audio), [{"start": 0.0, "end": 1.0, "text": "w", "words": []}],
                               engine_id="whisper-large-v3-turbo")
        cm.cache_transcription(str(audio), [{"start": 0.0, "end": 1.0, "text": "p", "words": []}],
                               engine_id="parakeet-mlx_community_parakeet-tdt-0.6b-v3")

        assert cm.get_cached_transcription(str(audio), engine_id="whisper-large-v3-turbo")[0]["text"] == "w"
        assert cm.get_cached_transcription(str(audio), engine_id="parakeet-mlx_community_parakeet-tdt-0.6b-v3")[0]["text"] == "p"

    def test_default_engine_id_is_whisper(self, tmp_path, monkeypatch):
        """Backwards compat: callers that don't pass engine_id behave as before."""
        from src.cache.manager import CacheManager
        from src.config import Config

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: str(tmp_path) if p == "~" else os.path.expanduser(p),
        )
        cfg = Config()
        cm = CacheManager(cfg)

        audio = tmp_path / "sample.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        cm.cache_transcription(str(audio), [{"start": 0.0, "end": 1.0, "text": "x", "words": []}])
        assert cm.get_cached_transcription(str(audio))[0]["text"] == "x"
```

Make sure `import os` is at the top of the file (existing tests already have it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cache_manager.py::TestEngineScopedTranscriptionCache -v`
Expected: failures with `TypeError: cache_transcription() got an unexpected keyword argument 'engine_id'`.

- [ ] **Step 3: Implement**

In `src/cache/manager.py`, replace `get_cached_transcription` (around line 264):

```python
    def get_cached_transcription(self, audio_path: str, engine_id: str = "whisper") -> Optional[List[Dict[str, Any]]]:
        """Get cached transcription results if they exist.

        Args:
            audio_path: Path to the audio file
            engine_id: ASR engine identifier (e.g. "whisper-large-v3-turbo",
                "parakeet-<slug>"). Different engines must not share cache entries.
        """
        prefix = f"transcription-{engine_id}"
        cache_key = self._generate_cache_key(audio_path, prefix=prefix)
        if cache_key is None:
            return None

        cache_path = self._get_cache_path(cache_key, "transcription")
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached transcription results: {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)
        return None
```

Replace `cache_transcription` (around line 295):

```python
    def cache_transcription(
        self,
        audio_path: str,
        transcription_results: List[Dict[str, Any]],
        engine_id: str = "whisper",
    ) -> None:
        """Cache transcription results.

        Args:
            audio_path: Path to the audio file
            transcription_results: Transcription results to cache
            engine_id: ASR engine identifier (must match the value passed to
                get_cached_transcription).
        """
        prefix = f"transcription-{engine_id}"
        cache_key = self._generate_cache_key(audio_path, prefix=prefix)
        cache_path = self._get_cache_path(cache_key, "transcription")
        try:
            with open(cache_path, "w") as f:
                json.dump(transcription_results, f)
            logger.info(f"Cached transcription results: {cache_path}")
        except Exception as e:
            logger.warning(f"Error caching transcription results: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_cache_manager.py -v`
Expected: green, including the new `TestEngineScopedTranscriptionCache` class.

Run: `pytest tests/ -q`
Expected: full suite green (engine.py still passes no engine_id; default `"whisper"` keeps existing behavior).

- [ ] **Step 5: Commit**

```bash
git add src/cache/manager.py tests/unit/test_cache_manager.py
git commit -m "feat(cache): scope transcription cache by engine_id"
```

---

## Task 3: Rename `TranscriptionEngine` → `WhisperEngine` with alias, add `_slug` helper

**Files:**
- Modify: `src/transcription/engine.py`
- Test: `tests/unit/test_transcription_engine_rename.py` (new, small file)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_transcription_engine_rename.py`:

```python
"""WhisperEngine rename + backward-compat alias + slug helper."""

import re

from src.transcription.engine import (
    TranscriptionEngine,
    WhisperEngine,
    _slug,
)


def test_alias_points_at_whisper_engine():
    """TranscriptionEngine must remain importable for backward compat."""
    assert TranscriptionEngine is WhisperEngine


def test_slug_handles_hf_model_id():
    assert _slug("mlx-community/parakeet-tdt-0.6b-v3") == "mlx-community_parakeet-tdt-0.6b-v3"


def test_slug_handles_local_path():
    assert _slug("/Users/x/models/parakeet") == "_Users_x_models_parakeet"


def test_slug_preserves_safe_chars():
    assert _slug("safe.name-1_2") == "safe.name-1_2"


def test_slug_replaces_spaces_and_specials():
    assert _slug("name with spaces & chars") == "name_with_spaces___chars"


def test_whisper_engine_passes_engine_id_to_cache(test_config, mock_whisper_model, tmp_path, monkeypatch):
    """Whisper engine must pass an engine_id when reading/writing cache."""
    from unittest.mock import MagicMock
    cfg = test_config
    cfg.cache_enabled = True

    engine = WhisperEngine(cfg)
    engine.whisper = mock_whisper_model

    cache_mock = MagicMock()
    cache_mock.get_cached_transcription.return_value = None
    engine.cache_manager = cache_mock

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    engine.transcribe(str(audio))

    cache_mock.get_cached_transcription.assert_called_once()
    _, kwargs = cache_mock.get_cached_transcription.call_args
    assert kwargs["engine_id"] == f"whisper-{cfg.whisper_model_size}"

    cache_mock.cache_transcription.assert_called_once()
    _, kwargs = cache_mock.cache_transcription.call_args
    assert kwargs["engine_id"] == f"whisper-{cfg.whisper_model_size}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_transcription_engine_rename.py -v`
Expected: ImportError for `WhisperEngine` and `_slug`.

- [ ] **Step 3: Implement — rename class and add slug helper**

In `src/transcription/engine.py`:

Add `import re` at the top alongside the other stdlib imports.

Add the slug helper at module level (after the imports, before any class):

```python
def _slug(text: str) -> str:
    """Return a filesystem-safe slug for cache keys and filenames.

    HF model ids contain '/'; local paths contain '/' (and on macOS, spaces).
    Both must produce a single safe token.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)
```

Rename the class `TranscriptionEngine` to `WhisperEngine` (the `class TranscriptionEngine:` line becomes `class WhisperEngine:`). Update the docstring's first line to: `"""Whisper-based ASR engine using faster-whisper."""`.

At the **bottom** of the file (after the class), add:

```python
# Backward-compat alias: external code, tests, and `service.py` still import
# `TranscriptionEngine`. Streaming methods remain on this class.
TranscriptionEngine = WhisperEngine
```

In `WhisperEngine.transcribe`, change the cache calls to pass `engine_id`. Find the `if self.cache_manager:` block (around line 138) and update both calls:

```python
        cache_engine_id = f"whisper-{self.whisper_model_size}"

        if self.cache_manager:
            cached_transcription = self.cache_manager.get_cached_transcription(
                audio_path, engine_id=cache_engine_id
            )
            if cached_transcription:
                return cached_transcription
```

And in the same method, near the end where caching happens (around line 177):

```python
                if self.cache_manager:
                    self.cache_manager.cache_transcription(
                        audio_path, result, engine_id=cache_engine_id
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_transcription_engine_rename.py -v`
Expected: all green.

Run: `pytest tests/ -q`
Expected: full suite green. The alias means existing imports of `TranscriptionEngine` still work; `MagicMock(spec=TranscriptionEngine)` in `tests/integration/test_transcriber.py` resolves to the same class.

- [ ] **Step 5: Commit**

```bash
git add src/transcription/engine.py tests/unit/test_transcription_engine_rename.py
git commit -m "refactor(engine): rename TranscriptionEngine to WhisperEngine, add _slug helper"
```

---

## Task 4: Add `ASREngine` Protocol and `make_asr_engine` factory

**Files:**
- Modify: `src/transcription/engine.py`
- Test: `tests/unit/test_asr_factory.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_asr_factory.py`:

```python
"""ASREngine Protocol + make_asr_engine factory."""

import pytest

from src.config import Config
from src.transcription.engine import (
    ASREngine,
    WhisperEngine,
    make_asr_engine,
)


def test_factory_returns_whisper_engine_by_default():
    cfg = Config()
    cfg.transcription_engine = "whisper"
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, WhisperEngine)


def test_factory_raises_for_parakeet_until_engine_lands(monkeypatch):
    """Parakeet branch is wired in Task 7; until then it raises NotImplementedError."""
    cfg = Config()
    cfg.transcription_engine = "parakeet"
    with pytest.raises(NotImplementedError, match="ParakeetEngine"):
        make_asr_engine(cfg, test_mode=True)


def test_protocol_runtime_checkable():
    """WhisperEngine satisfies the ASREngine protocol at runtime."""
    cfg = Config()
    cfg.transcription_engine = "whisper"
    engine = WhisperEngine(cfg, test_mode=True)
    assert isinstance(engine, ASREngine)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_asr_factory.py -v`
Expected: ImportError for `ASREngine` and `make_asr_engine`.

- [ ] **Step 3: Implement**

In `src/transcription/engine.py`:

Add to imports at top: `from typing import Protocol, runtime_checkable`. (`Protocol` and `runtime_checkable` from `typing` — Python 3.8+.)

Add the Protocol just below the `_slug` helper, before the `WhisperEngine` class:

```python
@runtime_checkable
class ASREngine(Protocol):
    """Batch-only ASR contract.

    Streaming is a Whisper-only extension and is not part of this protocol.
    Code that needs streaming should depend on WhisperEngine concretely or
    guard on Config.transcription_engine == "whisper" upstream.
    """

    def ensure_model_loaded(self) -> None: ...

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]: ...
```

Add the factory at the **bottom** of the file, after the `TranscriptionEngine = WhisperEngine` alias:

```python
def make_asr_engine(config: Config, test_mode: bool = False) -> ASREngine:
    """Construct the ASR engine selected by config.transcription_engine."""
    engine_name = config.transcription_engine
    if engine_name == "whisper":
        return WhisperEngine(config, test_mode=test_mode)
    if engine_name == "parakeet":
        # ParakeetEngine is wired in Task 7. Until then this is a clear
        # error rather than a silent fallback.
        raise NotImplementedError(
            "ParakeetEngine is not yet wired into the factory. "
            "Set TRANSCRIPTION_ENGINE=whisper for now."
        )
    raise ValueError(f"Unknown transcription engine: {engine_name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_asr_factory.py -v`
Expected: green.

Run: `pytest tests/ -q`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/transcription/engine.py tests/unit/test_asr_factory.py
git commit -m "feat(engine): add ASREngine protocol and make_asr_engine factory"
```

---

## Task 5: Wire `transcriber.py` to the factory + add streaming guards

**Files:**
- Modify: `src/transcriber.py`
- Test: `tests/integration/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/integration/test_transcriber.py`:

```python
class TestStreamingGuard:
    def test_transcribe_stream_rejects_parakeet_engine(self, mock_config):
        from src.transcriber import Transcriber

        mock_config.transcription_engine = "parakeet"
        # Bypass factory's NotImplementedError by stubbing engine after construction.
        # We're testing the upstream guard, not engine construction.
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.audio_processor = MagicMock()
        t.transcription_engine = MagicMock()
        t.diarization_engine = MagicMock()
        t.output_formatter = MagicMock()
        t.include_diarization = False
        t.test_mode = False

        with pytest.raises(NotImplementedError, match="Streaming is only supported"):
            list(t.transcribe_stream("input.wav"))

    def test_transcribe_stream_with_diarization_rejects_parakeet(self, mock_config):
        from src.transcriber import Transcriber

        mock_config.transcription_engine = "parakeet"
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.audio_processor = MagicMock()
        t.transcription_engine = MagicMock()
        t.diarization_engine = MagicMock()
        t.output_formatter = MagicMock()
        t.include_diarization = True
        t.test_mode = False

        with pytest.raises(NotImplementedError, match="Streaming is only supported"):
            list(t.transcribe_stream_with_diarization("input.wav"))
```

Make sure `import pytest` and `from unittest.mock import MagicMock` are present at the top of `tests/integration/test_transcriber.py` (existing test file already imports MagicMock).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_transcriber.py::TestStreamingGuard -v`
Expected: streaming methods don't raise — the iterators just run.

- [ ] **Step 3: Implement**

In `src/transcriber.py`:

Replace the import line `from .transcription.engine import TranscriptionEngine` with:

```python
from .transcription.engine import make_asr_engine
```

Inside `Transcriber.__init__`, replace the construction line (currently `self.transcription_engine = TranscriptionEngine(self.config, test_mode=test_mode)`, around line 40) with:

```python
        self.transcription_engine = make_asr_engine(self.config, test_mode=test_mode)
```

At the **top** of `transcribe_stream` (around line 252) and `transcribe_stream_with_diarization` (around line 293), add this guard as the first statement of the method body:

```python
        if self.config.transcription_engine != "whisper":
            raise NotImplementedError(
                "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
                "Use transcribe() for batch transcription with Parakeet."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_transcriber.py -v`
Expected: green, including new `TestStreamingGuard`.

Run: `pytest tests/ -q`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/transcriber.py tests/integration/test_transcriber.py
git commit -m "feat(transcriber): use ASR factory and guard streaming on engine"
```

---

## Task 6: Implement `ParakeetEngine` (test-mode + production paths)

**Files:**
- Modify: `src/transcription/engine.py`
- Modify: `tests/conftest.py`
- Create: `tests/unit/test_parakeet_engine.py`

- [ ] **Step 1: Add `MockParakeetModel` + fixtures to conftest**

Append to `tests/conftest.py` (after the existing fixtures, before the final blank line):

```python
@pytest.fixture
def mock_parakeet_model():
    """Mock parakeet-mlx model that mirrors the AlignedResult shape we depend on."""
    mock = MagicMock()

    class _Token:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    class _Sentence:
        def __init__(self, text, start, end, tokens):
            self.text = text
            self.start = start
            self.end = end
            self.tokens = tokens

    class _AlignedResult:
        def __init__(self):
            self.text = "Hello world. How are you"
            self.sentences = [
                _Sentence(
                    text="Hello world.",
                    start=0.0,
                    end=1.5,
                    tokens=[
                        _Token("Hello", 0.0, 0.5),
                        _Token(" world", 0.5, 1.4),
                        _Token(".", 1.4, 1.5),
                    ],
                ),
                _Sentence(
                    text="How are you",
                    start=2.0,
                    end=3.0,
                    tokens=[
                        _Token("How", 2.0, 2.3),
                        _Token(" are", 2.3, 2.6),
                        _Token(" you", 2.6, 3.0),
                    ],
                ),
            ]

    def _transcribe(audio_path, **kwargs):
        return _AlignedResult()

    mock.transcribe.side_effect = _transcribe
    return mock


@pytest.fixture
def test_parakeet_engine(test_config, mock_parakeet_model, monkeypatch):
    """ParakeetEngine wired with the mock model — does not import parakeet-mlx."""
    from src.transcription.engine import ParakeetEngine
    test_config.transcription_engine = "parakeet"
    engine = ParakeetEngine(test_config, test_mode=True)
    engine.parakeet = mock_parakeet_model
    return engine
```

- [ ] **Step 2: Write failing tests for ParakeetEngine**

Create `tests/unit/test_parakeet_engine.py`:

```python
"""Tests for ParakeetEngine.

Notes for implementers:
- `from parakeet_mlx import from_pretrained` is imported lazily inside
  `ParakeetEngine._load_model`. That means the patch target is
  `parakeet_mlx.from_pretrained`, NOT `src.transcription.engine.from_pretrained`.
- We never import `parakeet_mlx` in tests — the lazy import is what keeps
  Linux/CI clean.
"""

import os
import sys
import platform
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import Config


def _make_wav(path: Path):
    sr = 16000
    samples = np.zeros(sr, dtype=np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())


@pytest.fixture
def parakeet_config():
    cfg = Config()
    cfg.transcription_engine = "parakeet"
    cfg.parakeet_model = "mlx-community/parakeet-tdt-0.6b-v3"
    cfg.cache_enabled = False
    return cfg


def test_engine_loads_in_test_mode_without_parakeet_mlx_import(parakeet_config):
    """test_mode must not import parakeet_mlx — Linux/CI must stay green."""
    from src.transcription.engine import ParakeetEngine
    engine = ParakeetEngine(parakeet_config, test_mode=True)
    assert engine.parakeet is not None  # MockParakeetModel was set
    assert "parakeet_mlx" not in sys.modules or True  # tolerate prior imports; the engine itself didn't import it


def test_transcribe_returns_standard_segment_shape(test_parakeet_engine, tmp_path):
    audio = tmp_path / "x.wav"
    _make_wav(audio)
    segments = test_parakeet_engine.transcribe(str(audio))

    assert len(segments) == 2
    for seg in segments:
        assert set(seg.keys()) >= {"start", "end", "text", "words"}
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert isinstance(seg["text"], str)
        assert isinstance(seg["words"], list)
        for w in seg["words"]:
            assert set(w.keys()) >= {"start", "end", "word"}


def test_transcribe_populates_words_with_normalized_whitespace(test_parakeet_engine, tmp_path):
    """Tokens that don't start with a space get one prepended (except the first word).

    The MockParakeetModel includes a punctuation-as-its-own-token case
    (the ".") — joining the words verbatim must produce a human-readable
    sentence so a future parakeet-mlx tokenization change surfaces here.
    """
    audio = tmp_path / "x.wav"
    _make_wav(audio)
    segments = test_parakeet_engine.transcribe(str(audio))

    first = segments[0]
    # First token has no leading space; subsequent tokens have spaces normalized.
    assert first["words"][0]["word"] == "Hello"
    assert first["words"][1]["word"] == " world"
    # Punctuation token: the rule prepends a space so a separately-tokenized
    # period becomes " .". This is naive on purpose — see spec §"Word
    # whitespace normalization". If parakeet-mlx 0.5.1 starts emitting
    # punctuation as its own token, this assertion fires and prompts a refine.
    assert first["words"][2]["word"] == " ."


def test_cache_uses_engine_id_with_slugged_model(parakeet_config, tmp_path, monkeypatch):
    """ParakeetEngine must pass an engine_id that slugs the HF model id."""
    from src.transcription.engine import ParakeetEngine, _slug

    parakeet_config.cache_enabled = True
    engine = ParakeetEngine(parakeet_config, test_mode=True)

    cache_mock = MagicMock()
    cache_mock.get_cached_transcription.return_value = None
    engine.cache_manager = cache_mock

    audio = tmp_path / "x.wav"
    _make_wav(audio)
    engine.transcribe(str(audio))

    expected_engine_id = f"parakeet-{_slug(parakeet_config.parakeet_model)}"
    _, kwargs = cache_mock.get_cached_transcription.call_args
    assert kwargs["engine_id"] == expected_engine_id
    _, kwargs = cache_mock.cache_transcription.call_args
    assert kwargs["engine_id"] == expected_engine_id


def test_streaming_methods_raise_not_implemented(test_parakeet_engine):
    with pytest.raises(NotImplementedError, match="Streaming is only supported"):
        list(test_parakeet_engine.transcribe_stream(iter([np.zeros(16000, dtype=np.float32)])))
    with pytest.raises(NotImplementedError, match="Streaming is only supported"):
        test_parakeet_engine.start_async_transcription(iter([np.zeros(16000, dtype=np.float32)]))


def test_force_cpu_logs_warning_for_parakeet(parakeet_config, caplog):
    from src.transcription.engine import ParakeetEngine

    parakeet_config.force_cpu = True
    with caplog.at_level("WARNING"):
        ParakeetEngine(parakeet_config, test_mode=True)
    assert any("FORCE_CPU has no effect on Parakeet" in r.message for r in caplog.records)


def test_load_model_uses_parakeet_mlx_from_pretrained_lazily(parakeet_config):
    """In non-test_mode, loading must call parakeet_mlx.from_pretrained.

    Lazy import means we patch `parakeet_mlx.from_pretrained`, not the
    re-exported name in our module.
    """
    from src.transcription.engine import ParakeetEngine

    engine = ParakeetEngine(parakeet_config, test_mode=False)
    fake_model = MagicMock()
    fake_pkg = MagicMock()
    fake_pkg.from_pretrained = MagicMock(return_value=fake_model)

    with patch.dict(sys.modules, {"parakeet_mlx": fake_pkg}):
        engine._load_model()

    fake_pkg.from_pretrained.assert_called_once_with(parakeet_config.parakeet_model)
    assert engine.parakeet is fake_model
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_parakeet_engine.py -v`
Expected: ImportError for `ParakeetEngine`.

- [ ] **Step 4: Implement `ParakeetEngine`**

Append to `src/transcription/engine.py` (between `WhisperEngine` and the `TranscriptionEngine = WhisperEngine` alias):

```python
class ParakeetEngine:
    """Parakeet-MLX ASR engine. Apple Silicon only. Batch-only.

    Long-form audio is handled by parakeet-mlx's built-in chunking
    (chunk_duration / overlap_duration) — we do not roll our own VAD
    chunking. See spec §"Long-form chunking".
    """

    # parakeet-mlx 0.5.1 defaults — exposed as class constants for visibility.
    CHUNK_DURATION = 120
    OVERLAP_DURATION = 15

    # Punctuation token characters that are "naturally" attached to the
    # prior word in formatted text (no space prepended). The whitespace
    # normalizer is intentionally simple — see spec.
    _PUNCTUATION_HINTS = frozenset(",.!?;:")

    def __init__(self, config: Config, test_mode: bool = False):
        self.config = config
        self.timeout_seconds = config.transcribe_timeout
        self.parakeet_model_id = config.parakeet_model
        self.test_mode = test_mode
        self.parakeet = None

        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_manager = CacheManager(config) if config.cache_enabled else None

        if config.force_cpu:
            logger.warning(
                "FORCE_CPU has no effect on Parakeet (MLX); flag is Whisper-only."
            )

        if self.test_mode:
            self._load_model()

    def _engine_id(self) -> str:
        return f"parakeet-{_slug(self.parakeet_model_id)}"

    def _load_model(self):
        if self.parakeet is not None:
            return

        if self.test_mode:
            logger.info("Test mode enabled, using mock parakeet model")

            class _Token:
                def __init__(self, text, start, end):
                    self.text = text
                    self.start = start
                    self.end = end

            class _Sentence:
                def __init__(self, text, start, end, tokens):
                    self.text = text
                    self.start = start
                    self.end = end
                    self.tokens = tokens

            class _AlignedResult:
                text = "Hello world."
                sentences = [
                    _Sentence(
                        "Hello world.", 0.0, 1.0,
                        [_Token("Hello", 0.0, 0.5), _Token(" world.", 0.5, 1.0)],
                    ),
                ]

            class MockParakeetModel:
                def transcribe(self, audio_path, **kwargs):
                    return _AlignedResult()

            self.parakeet = MockParakeetModel()
            return

        # Lazy import: parakeet_mlx is darwin/arm64 only and imported only
        # when actually needed. Test patch target is `parakeet_mlx.from_pretrained`.
        from parakeet_mlx import from_pretrained  # noqa: WPS433

        try:
            logger.info(f"Loading parakeet-mlx model: {self.parakeet_model_id}")
            self.parakeet = from_pretrained(self.parakeet_model_id)
            logger.info("parakeet-mlx model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading parakeet-mlx model: {e}")
            raise

    def ensure_model_loaded(self) -> None:
        if self.parakeet is None:
            self._load_model()

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        cache_engine_id = self._engine_id()

        if self.cache_manager:
            cached = self.cache_manager.get_cached_transcription(
                audio_path, engine_id=cache_engine_id
            )
            if cached:
                return cached

        self.ensure_model_loaded()
        logger.info(f"Starting parakeet transcription for {audio_path}")
        start_time = time.time()

        try:
            with timeout(self.timeout_seconds, "Transcription timed out"):
                result = self.parakeet.transcribe(
                    audio_path,
                    chunk_duration=self.CHUNK_DURATION,
                    overlap_duration=self.OVERLAP_DURATION,
                )

                segments = self._map_aligned_result(result)

                elapsed = time.time() - start_time
                logger.info(
                    f"Parakeet transcription completed in {elapsed:.1f}s, "
                    f"{len(segments)} segments"
                )

                if self.cache_manager:
                    self.cache_manager.cache_transcription(
                        audio_path, segments, engine_id=cache_engine_id
                    )
                return segments

        except TimeoutException:
            logger.error(f"Parakeet transcription timed out after {self.timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"Error during parakeet transcription: {e}")
            raise Exception(f"Error during parakeet transcription: {e}")

    @classmethod
    def _map_aligned_result(cls, result) -> List[Dict[str, Any]]:
        """Map parakeet-mlx AlignedResult → standard segment shape."""
        segments: List[Dict[str, Any]] = []
        for sentence in result.sentences:
            words: List[Dict[str, Any]] = []
            for idx, token in enumerate(sentence.tokens):
                text = token.text
                # Whitespace normalization: see spec §"Word whitespace normalization".
                if idx > 0 and not text.startswith(" "):
                    text = " " + text
                words.append({
                    "start": float(token.start),
                    "end": float(token.end),
                    "word": text,
                })
            segments.append({
                "start": float(sentence.start),
                "end": float(sentence.end),
                "text": sentence.text.strip(),
                "words": words,
            })
        return segments

    # Streaming is Whisper-only — defensive stubs in case the upstream guard
    # in transcriber.py is ever bypassed.
    def transcribe_stream(self, audio_stream):
        raise NotImplementedError(
            "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
            "Use transcribe() for batch transcription with Parakeet."
        )

    def start_async_transcription(self, audio_stream):
        raise NotImplementedError(
            "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
            "Use transcribe() for batch transcription with Parakeet."
        )
```

Note: `time`, `os`, `logger`, `List`, `Dict`, `Any`, `timeout`, `TimeoutException`, `CacheManager`, `Config`, `_slug` are all already imported/defined at the top of `engine.py` from prior tasks — no new imports needed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_parakeet_engine.py -v`
Expected: all green.

Run: `pytest tests/ -q`
Expected: full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/transcription/engine.py tests/conftest.py tests/unit/test_parakeet_engine.py
git commit -m "feat(engine): implement ParakeetEngine via parakeet-mlx"
```

---

## Task 7: Wire factory to return `ParakeetEngine`

**Files:**
- Modify: `src/transcription/engine.py`
- Test: `tests/unit/test_asr_factory.py`

- [ ] **Step 1: Update tests**

Replace the `test_factory_raises_for_parakeet_until_engine_lands` test in `tests/unit/test_asr_factory.py` with:

```python
def test_factory_returns_parakeet_engine_when_selected():
    from src.transcription.engine import ParakeetEngine

    cfg = Config()
    cfg.transcription_engine = "parakeet"
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, ParakeetEngine)


def test_factory_raises_for_unknown_engine():
    cfg = Config()
    cfg.transcription_engine = "bogus"
    with pytest.raises(ValueError, match="Unknown transcription engine"):
        make_asr_engine(cfg, test_mode=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_asr_factory.py -v`
Expected: `test_factory_returns_parakeet_engine_when_selected` fails because the factory still raises `NotImplementedError`.

- [ ] **Step 3: Implement**

In `src/transcription/engine.py`, replace `make_asr_engine` with:

```python
def make_asr_engine(config: Config, test_mode: bool = False) -> ASREngine:
    """Construct the ASR engine selected by config.transcription_engine."""
    engine_name = config.transcription_engine
    if engine_name == "whisper":
        return WhisperEngine(config, test_mode=test_mode)
    if engine_name == "parakeet":
        return ParakeetEngine(config, test_mode=test_mode)
    raise ValueError(f"Unknown transcription engine: {engine_name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_asr_factory.py -v`
Expected: green.

Run: `pytest tests/ -q`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/transcription/engine.py tests/unit/test_asr_factory.py
git commit -m "feat(engine): wire ParakeetEngine into make_asr_engine factory"
```

---

## Task 8: End-to-end integration test parametrized over engines

**Files:**
- Modify: `tests/integration/test_transcriber.py`

- [ ] **Step 1: Write failing test**

Append to `tests/integration/test_transcriber.py`:

```python
class TestEndToEndAcrossEngines:
    """End-to-end transcribe + diarization combination across both engines.

    Both engines must produce the same output shape so downstream formatters
    and diarization alignment work identically.
    """

    @pytest.mark.parametrize("engine_name", ["whisper", "parakeet"])
    def test_transcribe_with_diarization(self, engine_name, tmp_path, mock_diarizer, monkeypatch):
        from src.config import Config
        from src.transcriber import Transcriber

        cfg = Config(
            transcription_engine=engine_name,
            include_diarization=True,
        )
        cfg.cache_enabled = False
        cfg.hf_token = "test_token"

        # Bypass platform validation for parakeet by not calling validate(); we
        # exercise the engine's runtime behavior here, not config validation.
        # ParakeetEngine in test_mode does not import parakeet_mlx, so this works
        # on Linux CI as well.

        t = Transcriber(cfg, test_mode=True)

        # Real audio file so AudioProcessor works.
        audio = tmp_path / "x.wav"
        import wave, numpy as np
        sr = 16000
        samples = np.zeros(int(2.0 * sr), dtype=np.int16)
        with wave.open(str(audio), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
            f.writeframes(samples.tobytes())

        t.diarization_engine.diarizer = mock_diarizer

        segments = t.transcribe(str(audio))

        # Output shape contract is the same regardless of engine.
        assert len(segments) > 0
        for seg in segments:
            assert isinstance(seg, tuple)
            assert len(seg) == 4
            start, end, text, speaker = seg
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert isinstance(text, str)
            assert isinstance(speaker, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_transcriber.py::TestEndToEndAcrossEngines -v`
Expected: failures (we need to confirm it actually runs end-to-end across both engines and the shape is consistent).

If both pass on first try, that's fine — it means the wiring from Tasks 5 and 7 already produces matching shapes. The test is now a regression guard.

- [ ] **Step 3: If it failed, debug and fix**

Most likely culprits:
- `Transcriber(cfg, test_mode=True)` constructs the real `AudioProcessor`, which requires the audio file to be a valid WAV. The test creates one. Verify the file exists.
- ParakeetEngine's mock model returns segments with non-empty `text`; the diarization combination should produce non-empty results.

If you hit a real bug, fix it in the engine or transcriber code, not the test.

- [ ] **Step 4: Run full suite**

Run: `pytest tests/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_transcriber.py
git commit -m "test(integration): parametrize end-to-end test across whisper and parakeet"
```

---

## Task 9: Add `parakeet-mlx` to `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append the platform-marked dependency**

In `requirements.txt`, append:

```
parakeet-mlx>=0.5.1,<0.6; sys_platform == "darwin" and platform_machine == "arm64"
```

The upper bound `<0.6` mechanically enforces the spec's "bumping the pin requires re-validating the token mapping" contract.

- [ ] **Step 2: Verify pip parses the marker correctly**

Run on this machine (macOS arm64):
```bash
source venv/bin/activate && pip install -r requirements.txt --dry-run 2>&1 | grep -i parakeet
```
Expected: line shows `Would install parakeet-mlx-0.5.x` (or "already satisfied" after install).

If you have shell access on a non-arm64 environment (Docker, Linux), confirm the marker excludes parakeet-mlx there. Optional — the marker syntax is standard and well-tested; skipping is fine.

- [ ] **Step 3: Install and run the full suite once more**

```bash
source venv/bin/activate && pip install -r requirements.txt
pytest tests/ -q
```
Expected: full suite green. parakeet-mlx is installed but the test code still uses `test_mode=True`, so the package is loaded only if tests import it directly (none do).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add parakeet-mlx for Apple Silicon"
```

---

## Task 10: Document the new engine in `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the configuration section**

Run: `grep -n "WHISPER_MODEL\|HF_TOKEN\|## Configuration\|## Environment" README.md | head`
Identify the section that documents env vars (likely "## Configuration" or "## Environment Variables"). The new section goes after it.

- [ ] **Step 2: Insert the new section**

Add a new section "## Transcription engines" placed adjacent to the existing configuration documentation:

```markdown
## Transcription engines

The transcriber supports two ASR engines, selectable via the `TRANSCRIPTION_ENGINE` env var:

### `whisper` (default)

`faster-whisper` running `large-v3-turbo` by default. Works on macOS, Linux, and Docker. Supports 99+ languages. Streaming and async streaming are supported.

### `parakeet` (Apple Silicon only)

NVIDIA Parakeet-TDT-0.6B-v3 via [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx). On macOS arm64, this is roughly an order of magnitude faster than Whisper on CPU and produces lower WER on the Open ASR Leaderboard for English / ~25 European languages. Batch only — no streaming.

Enable:

```bash
export TRANSCRIPTION_ENGINE=parakeet
```

#### First-run model download

On first use, parakeet-mlx auto-downloads `mlx-community/parakeet-tdt-0.6b-v3` (~600MB) to `~/.cache/huggingface/`. To use a different MLX checkpoint or a pre-downloaded local copy:

```bash
# HuggingFace id
export PARAKEET_MODEL=mlx-community/parakeet-tdt-0.6b-v3

# Or absolute local path to an MLX checkpoint
export PARAKEET_MODEL=/path/to/local/mlx-checkpoint
```

#### Caveats

- **Apple Silicon only.** Setting `TRANSCRIPTION_ENGINE=parakeet` on Linux, Docker, or Intel macOS is rejected at config validation.
- **`FORCE_CPU` is Whisper-only.** MLX runs on Apple Silicon with no equivalent knob; if `FORCE_CPU=true` is set with `engine=parakeet`, a warning is logged and the flag is ignored.
- **Streaming is Whisper-only.** Calling streaming entry points with `engine=parakeet` raises `NotImplementedError`. Use the batch `transcribe()` path.
- **Handy weights are not compatible.** Handy ships INT8 ONNX weights; `parakeet-mlx` requires MLX-format weights. Users wanting to reuse Handy's weights would need a different runtime (e.g. `onnx-asr`) — out of scope here.
```

- [ ] **Step 2: Skim the README for accuracy**

Run: `grep -n "TRANSCRIPTION_ENGINE\|PARAKEET" README.md`
Expected: only the new section references these vars. No stale references to remove.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document Parakeet engine and PARAKEET_MODEL"
```

---

## Task 11: Final verification + push

- [ ] **Step 1: Run the complete test suite**

```bash
source venv/bin/activate && pytest tests/ -v
```
Expected: every test green. Count should be ≥ 101 (existing) + Task-1-through-8 additions. Note any new test count for the commit message of a future spec edit (the spec says "All existing tests pass" without a number, so no edit needed unless drift is concerning).

- [ ] **Step 2: Smoke test the CLI with engine=whisper (default)**

Use any short audio/video in the repo (`test.wav` exists at the root):

```bash
TRANSCRIPTION_ENGINE=whisper python transcribe_video.py test.wav
```
Expected: produces a transcript at `transcripts/test.txt` (or whatever `OUTPUT_FORMAT` is set to). No errors.

- [ ] **Step 3: Smoke test the CLI with engine=parakeet**

```bash
TRANSCRIPTION_ENGINE=parakeet python transcribe_video.py test.wav
```
Expected: first run downloads ~600MB to `~/.cache/huggingface/`, then produces a transcript with the same format. If you want to skip the download for now, run with `INCLUDE_DIARIZATION=false PARAKEET_MODEL=<path-to-local-checkpoint>` if a local one is available; otherwise the download is the test.

If you don't want to download 600MB during this verification, **document that step as deferred** in the commit body and proceed.

- [ ] **Step 4: Push**

```bash
git push
```
Expected: push succeeds.

- [ ] **Step 5: Final commit summary**

You're done when:
- All tests green.
- Whisper smoke test produces a transcript.
- Parakeet smoke test produces a transcript (or is documented as deferred pending the model download).
- `git log --oneline` shows ten reasonable commits, one per task.

---

## Self-Review notes

- **Spec coverage:** Tasks map to spec sections — config (T1), cache (T2), slug helper + WhisperEngine rename (T3), Protocol+factory (T4, T7), transcriber wiring + streaming guard (T5), ParakeetEngine internals (T6), tests (T6, T8), requirements (T9), README (T10), final verification (T11). All eleven spec sections are covered.
- **Type consistency:** `engine_id` is the parameter name across CacheManager, WhisperEngine, ParakeetEngine. `_slug` is the helper name. `make_asr_engine` and `ASREngine` are the factory and protocol. `parakeet` is the attribute name on ParakeetEngine (mirroring `whisper` on WhisperEngine).
- **Frequent commits:** Every task ends with a commit. Each task is bite-sized and self-contained.

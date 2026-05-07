"""Tests for ParakeetEngine.

Notes for implementers:
- `parakeet_mlx` is imported lazily inside `ParakeetEngine._load_model`
  via `import parakeet_mlx` + attribute access. The patch target is
  `parakeet_mlx` in `sys.modules`, NOT `src.transcription.engine.from_pretrained`.
- We never import `parakeet_mlx` in tests — the lazy import is what keeps
  Linux/CI clean.
"""

import sys
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
    before = "parakeet_mlx" in sys.modules
    engine = ParakeetEngine(parakeet_config, test_mode=True)
    assert engine.parakeet is not None  # MockParakeetModel was set
    # If parakeet_mlx wasn't loaded before, test_mode boot must not load it.
    if not before:
        assert "parakeet_mlx" not in sys.modules


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

"""ASREngine Protocol + make_asr_engine factory."""

import platform
import sys

import pytest

from src.config import Config
from src.transcription.engine import (
    ASREngine,
    ParakeetEngine,
    WhisperEngine,
    make_asr_engine,
)


def test_factory_default_config_returns_whisper_engine_off_apple_silicon(monkeypatch):
    """Config default (no env override) routes to WhisperEngine off Apple Silicon."""
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    cfg = Config()
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, WhisperEngine)


def test_factory_default_config_returns_parakeet_engine_on_apple_silicon(monkeypatch):
    """Config default (no env override) routes to ParakeetEngine on Apple Silicon."""
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    cfg = Config()
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, ParakeetEngine)


def test_factory_returns_whisper_engine_for_explicit_whisper_config():
    cfg = Config(transcription_engine="whisper")
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, WhisperEngine)


def test_factory_returns_parakeet_engine_when_selected():
    cfg = Config(transcription_engine="parakeet")
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, ParakeetEngine)


def test_factory_raises_for_unknown_engine():
    cfg = Config()
    cfg.transcription_engine = "bogus"
    with pytest.raises(ValueError, match="Unknown transcription engine"):
        make_asr_engine(cfg, test_mode=True)


def test_protocol_runtime_checkable():
    """WhisperEngine satisfies the ASREngine protocol at runtime."""
    cfg = Config()
    cfg.transcription_engine = "whisper"
    engine = WhisperEngine(cfg, test_mode=True)
    assert isinstance(engine, ASREngine)


def test_defaulted_parakeet_falls_back_to_whisper_when_mlx_missing(monkeypatch):
    """A pre-existing venv without parakeet-mlx must not break when the
    platform default flips to parakeet — only an EXPLICIT parakeet request
    should fail loudly."""
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    # Make `import parakeet_mlx` raise ImportError
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)

    cfg = Config()
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, WhisperEngine)


def test_explicit_parakeet_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)

    cfg = Config(transcription_engine="parakeet")
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, ParakeetEngine)

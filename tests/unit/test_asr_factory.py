"""ASREngine Protocol + make_asr_engine factory."""

import pytest

from src.config import Config
from src.transcription.engine import (
    ASREngine,
    ParakeetEngine,
    WhisperEngine,
    make_asr_engine,
)


def test_factory_default_config_returns_whisper_engine(monkeypatch):
    """Config default (no env override) routes to WhisperEngine."""
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    cfg = Config()
    engine = make_asr_engine(cfg, test_mode=True)
    assert isinstance(engine, WhisperEngine)


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

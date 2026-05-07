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

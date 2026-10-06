"""Cached results must not be reused under settings that change them.

Regression: the transcript cache keyed only on the file and the Whisper model,
so changing LANGUAGE or WHISPER_BEAM_SIZE silently returned the old
transcript; the diarization cache ignored DIARIZATION_MODEL.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.cache.manager import CacheManager  # noqa: E402
from src.config import Config  # noqa: E402
from src.transcription.engine import WhisperEngine  # noqa: E402


class _Segment:
    def __init__(self, text):
        self.start, self.end, self.text, self.words = 0.0, 1.0, text, []


class _FakeWhisper:
    """Answers with the language it was asked for, so reuse is visible."""

    def transcribe(self, audio_path, **kwargs):
        return iter([_Segment(f"{kwargs['language']}-beam{kwargs['beam_size']}")]), None


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
    return str(path)


def _transcribe(audio, monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    engine = WhisperEngine(Config(transcription_engine="whisper"), test_mode=True)
    engine.whisper = _FakeWhisper()
    return engine.transcribe(audio)[0]["text"]


def test_transcript_cache_is_reused_for_the_same_settings(audio, monkeypatch):
    assert _transcribe(audio, monkeypatch, LANGUAGE="en") == "en-beam5"

    calls = []
    engine = WhisperEngine(Config(transcription_engine="whisper"), test_mode=True)
    engine.whisper = type("Never", (), {"transcribe": lambda *a, **k: calls.append(1)})()
    assert engine.transcribe(audio)[0]["text"] == "en-beam5"
    assert calls == []


def test_changing_language_does_not_return_the_cached_transcript(audio, monkeypatch):
    assert _transcribe(audio, monkeypatch, LANGUAGE="en") == "en-beam5"
    assert _transcribe(audio, monkeypatch, LANGUAGE="fr") == "fr-beam5"


def test_changing_beam_size_does_not_return_the_cached_transcript(audio, monkeypatch):
    assert _transcribe(audio, monkeypatch, WHISPER_BEAM_SIZE="5") == "en-beam5"
    assert _transcribe(audio, monkeypatch, WHISPER_BEAM_SIZE="1") == "en-beam1"


def test_diarization_cache_is_scoped_by_model(audio):
    cache = CacheManager(Config())
    cache.cache_diarization(audio, [{"speaker": "A"}], model_id="pyannote/model-a")

    assert cache.get_cached_diarization(audio, model_id="pyannote/model-a") == [{"speaker": "A"}]
    assert cache.get_cached_diarization(audio, model_id="pyannote/model-b") is None

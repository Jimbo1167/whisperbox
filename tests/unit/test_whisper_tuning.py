"""Whisper tuning knobs (beam size, CPU threads, batched pipeline) wiring."""

import pytest

from src.config import Config
from src.transcription.engine import WhisperEngine


@pytest.fixture
def config():
    cfg = Config(transcription_engine="whisper")
    cfg.cache_enabled = False
    return cfg


class RecordingModel:
    def __init__(self):
        self.transcribe_kwargs = None

    def transcribe(self, audio_path, **kwargs):
        self.transcribe_kwargs = kwargs
        return iter([]), {"language": "en"}


def test_transcribe_passes_beam_size(config):
    config.whisper_beam_size = 2
    engine = WhisperEngine(config, test_mode=True)
    recorder = RecordingModel()
    engine.whisper = recorder
    engine.transcribe("dummy.wav")
    assert recorder.transcribe_kwargs["beam_size"] == 2


def test_model_load_passes_cpu_threads(config, monkeypatch):
    config.whisper_cpu_threads = 6
    captured = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("src.transcription.engine.WhisperModel", FakeModel)
    engine = WhisperEngine(config, test_mode=False)
    engine.ensure_model_loaded()
    assert captured["cpu_threads"] == 6


def test_batched_pipeline_used_when_batch_size_set(config, monkeypatch):
    config.whisper_batch_size = 8
    engine = WhisperEngine(config, test_mode=True)
    engine.whisper = RecordingModel()

    batched_calls = {}

    class FakeBatched:
        def __init__(self, model):
            batched_calls["model"] = model

        def transcribe(self, audio_path, **kwargs):
            batched_calls["kwargs"] = kwargs
            return iter([]), {"language": "en"}

    monkeypatch.setattr(
        "faster_whisper.BatchedInferencePipeline", FakeBatched
    )
    engine.transcribe("dummy.wav")
    assert batched_calls["model"] is engine.whisper
    assert batched_calls["kwargs"]["batch_size"] == 8


def test_sequential_decoder_used_by_default(config):
    config.whisper_batch_size = 0
    engine = WhisperEngine(config, test_mode=True)
    recorder = RecordingModel()
    engine.whisper = recorder
    engine.transcribe("dummy.wav")
    assert recorder.transcribe_kwargs is not None
    assert "batch_size" not in recorder.transcribe_kwargs

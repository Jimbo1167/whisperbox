"""WhisperEngine rename + backward-compat alias + slug helper."""

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


def test_whisper_engine_id_slugs_unsafe_model_size(test_config, mock_whisper_model, tmp_path):
    """Non-canonical WHISPER_MODEL values (e.g. HF org/model paths) must be slugged."""
    from unittest.mock import MagicMock
    cfg = test_config
    cfg.cache_enabled = True
    cfg.whisper_model_size = "deepdml/faster-distil-whisper-large-v3.5"

    engine = WhisperEngine(cfg)
    engine.whisper = mock_whisper_model

    cache_mock = MagicMock()
    cache_mock.get_cached_transcription.return_value = None
    engine.cache_manager = cache_mock

    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    engine.transcribe(str(audio))

    _, kwargs = cache_mock.get_cached_transcription.call_args
    assert "/" not in kwargs["engine_id"]
    assert kwargs["engine_id"] == "whisper-deepdml_faster-distil-whisper-large-v3.5"

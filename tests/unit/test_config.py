import os
import sys
import pytest
from unittest.mock import patch

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "HF_TOKEN": "test_token",
        "WHISPER_MODEL": "medium",
        "DIARIZATION_MODEL": "test/model",
        "LANGUAGE": "en",
        "OUTPUT_FORMAT": "srt",
        "INCLUDE_DIARIZATION": "true",
        "AUDIO_TIMEOUT": "300",
        "TRANSCRIBE_TIMEOUT": "3600",
        "DIARIZE_TIMEOUT": "3600",
        "FORCE_CPU": "false"
    }
    return env_vars

def test_config_init_defaults():
    """Test Config initialization with default values."""
    with patch.dict(os.environ, {}, clear=True):
        config = Config()
        
        assert config.hf_token is None
        assert config.whisper_model_size == "large-v3-turbo"
        assert config.diarization_model == "pyannote/speaker-diarization-community-1"
        assert config.language == "en"
        assert config.output_format == "txt"
        assert config.include_diarization is False
        assert config.audio_timeout == 300
        assert config.transcribe_timeout == 3600
        assert config.diarize_timeout == 3600
        assert config.force_cpu is False

def test_config_init_from_env(mock_env_vars):
    """Test Config initialization from environment variables."""
    with patch.dict(os.environ, mock_env_vars, clear=True):
        config = Config()
        
        assert config.hf_token == "test_token"
        assert config.whisper_model_size == "medium"
        assert config.diarization_model == "test/model"
        assert config.language == "en"
        assert config.output_format == "srt"
        assert config.include_diarization is True
        assert config.audio_timeout == 300
        assert config.transcribe_timeout == 3600
        assert config.diarize_timeout == 3600
        assert config.force_cpu is False

def test_config_output_format_setter():
    """Test the output_format setter."""
    config = Config()
    
    # Test with valid format
    config.output_format = "srt"
    assert config.output_format == "srt"
    
    # Test with format containing comment
    config.output_format = "vtt # This is a comment"
    assert config.output_format == "vtt"
    
    # Test with empty format
    config.output_format = ""
    assert config.output_format == "txt"
    
    # Test with None
    config.output_format = None
    assert config.output_format == "txt"

def test_config_to_dict():
    """Test the to_dict method."""
    with patch.dict(os.environ, {}, clear=True):
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["whisper_model_size"] == "large-v3-turbo"
        assert config_dict["diarization_model"] == "pyannote/speaker-diarization-community-1"
        assert config_dict["language"] == "en"
        assert config_dict["output_format"] == "txt"
        assert config_dict["include_diarization"] is False
        assert config_dict["audio_timeout"] == 300
        assert config_dict["transcribe_timeout"] == 3600
        assert config_dict["diarize_timeout"] == 3600
        assert config_dict["force_cpu"] is False

def test_config_validate():
    """Test the validate method."""
    # Test valid configuration
    with patch.dict(os.environ, {"HF_TOKEN": "test_token"}, clear=True):
        config = Config()
        assert config.validate() is True
    
    # Test invalid configuration: missing HF token with diarization enabled
    with patch.dict(os.environ, {"HF_TOKEN": "", "INCLUDE_DIARIZATION": "true"}, clear=True):
        config = Config()
        assert config.validate() is False
    
    # Test invalid configuration: invalid output format
    with patch.dict(os.environ, {"HF_TOKEN": "test_token", "OUTPUT_FORMAT": "invalid"}, clear=True):
        config = Config()
        assert config.validate() is False
    
    # Test valid configuration: diarization disabled, no HF token needed
    with patch.dict(os.environ, {"HF_TOKEN": "", "INCLUDE_DIARIZATION": "false"}, clear=True):
        config = Config()
        assert config.validate() is True


import platform

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

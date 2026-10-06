"""
Pytest configuration and fixtures for testing the Whisperbox.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from tests.fixtures.generate_test_files import create_test_video, create_test_wav


@pytest.fixture(autouse=True)
def isolate_from_developer_env(tmp_path):
    """Keep the developer's real files out of every test.

    - Entry points call load_env_file(), which would otherwise read the real
      project .env and leave its values in os.environ for later tests.
    - Engines build a CacheManager under ~/.cache/whisperbox even in test mode
      and prune it on startup; HOME points at a temp dir instead.
    """
    import src.config as config_module

    home = tmp_path / "home"
    home.mkdir()
    with patch.object(config_module, "DEFAULT_ENV_FILE", tmp_path / "no-such.env"), \
            patch.dict(os.environ, {"HOME": str(home)}):
        yield


@pytest.fixture(scope="session", autouse=True)
def ensure_test_media():
    """Generate media fixtures on demand so a fresh checkout can run tests."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    video_path = fixtures_dir / "test_video.mp4"
    audio_path = fixtures_dir / "test_audio.wav"

    fixtures_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        create_test_video(str(video_path))
    if not audio_path.exists():
        create_test_wav(str(audio_path))


@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = Config()
    config.whisper_model_size = "base"
    config.include_diarization = True
    config.output_format = "txt"
    config.hf_token = "test_token"
    return config


@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel."""
    mock = MagicMock()
    
    # Mock the transcribe method
    def mock_transcribe(audio_path, **kwargs):
        # Create mock segments
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []

        segments = [
            MockSegment(0.0, 2.0, "Test segment one"),
            MockSegment(2.0, 4.0, "Test segment two")
        ]
        return segments, None
    
    mock.transcribe.side_effect = mock_transcribe
    return mock

@pytest.fixture
def mock_diarizer():
    """Create a mock diarizer."""
    mock = MagicMock()

    # Mock the __call__ method
    def mock_call(audio_path):
        # Create mock diarization
        class MockDiarization:
            def itertracks(self, yield_label=False):
                class Segment:
                    def __init__(self, start, end):
                        self.start = start
                        self.end = end

                tracks = [
                    (Segment(0.0, 2.0), None, "SPEAKER_01"),
                    (Segment(2.0, 4.0), None, "SPEAKER_02"),
                ]
                for track in tracks:
                    yield track

        return MockDiarization()

    mock.side_effect = mock_call
    return mock


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
    # test_mode=False: skip the inline MockParakeetModel boot; we override
    # `engine.parakeet` with the conftest mock immediately afterwards.
    engine = ParakeetEngine(test_config, test_mode=False)
    engine.parakeet = mock_parakeet_model
    return engine

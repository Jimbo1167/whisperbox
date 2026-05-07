"""
Pytest configuration and fixtures for testing the video transcriber.
"""

import os
import sys
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.transcriber import Transcriber
from src.audio.processor import AudioProcessor
from src.transcription.engine import TranscriptionEngine
from src.diarization.engine import DiarizationEngine
from src.output.formatter import OutputFormatter
from tests.fixtures.generate_test_files import create_test_video, create_test_wav


@pytest.fixture(scope="session", autouse=True)
def ensure_test_media():
    """Generate media fixtures on demand so a fresh checkout can run tests."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    video_path = fixtures_dir / "test_video.mp4"
    silent_video_path = fixtures_dir / "test_video_no_audio.mp4"
    audio_path = fixtures_dir / "test_audio.wav"

    fixtures_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        create_test_video(str(video_path))
    if not silent_video_path.exists():
        create_test_video(str(silent_video_path), with_audio=False)
    if not audio_path.exists():
        create_test_wav(str(audio_path))

@pytest.fixture
def sample_audio():
    """Generate a simple sine wave audio sample for testing."""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio

@pytest.fixture
def mock_video(tmp_path):
    """Create a mock video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Create an empty file
    video_path.touch()
    return str(video_path)

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
def test_audio_processor(test_config):
    """Create a test audio processor."""
    return AudioProcessor(test_config)

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
                tracks = [
                    ((0.0, 2.0), None, "SPEAKER_01"),
                    ((2.0, 4.0), None, "SPEAKER_02")
                ]
                for track in tracks:
                    yield track
        
        return MockDiarization()
    
    mock.__call__.side_effect = mock_call
    return mock

@pytest.fixture
def test_transcription_engine(test_config, mock_whisper_model):
    """Create a test transcription engine with a mock whisper model."""
    engine = TranscriptionEngine(test_config)
    engine.whisper = mock_whisper_model
    return engine

@pytest.fixture
def test_diarization_engine(test_config, mock_diarizer):
    """Create a test diarization engine with a mock diarizer."""
    engine = DiarizationEngine(test_config)
    engine.diarizer = mock_diarizer
    return engine

@pytest.fixture
def test_output_formatter(test_config):
    """Create a test output formatter."""
    return OutputFormatter(test_config)

@pytest.fixture
def test_transcriber(test_config, test_audio_processor, test_transcription_engine, 
                   test_diarization_engine, test_output_formatter):
    """Create a test transcriber with real components but mock models."""
    transcriber = Transcriber(test_config)
    transcriber.audio_processor = test_audio_processor
    transcriber.transcription_engine = test_transcription_engine
    transcriber.diarization_engine = test_diarization_engine
    transcriber.output_formatter = test_output_formatter
    return transcriber

@pytest.fixture
def create_test_audio_file():
    """Create a temporary WAV file for testing."""
    def _create_file(duration=4.0):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Create a simple WAV file
            sample_rate = 16000
            samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
            
            # Write the WAV file
            import wave
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            
            return temp_file.name
    
    return _create_file

@pytest.fixture
def create_test_video_file():
    """Create a temporary MP4 file for testing."""
    def _create_file(duration=4.0):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            # Just create an empty file for now
            # In a real test, we would create a proper video file
            temp_file.write(b"test")
            return temp_file.name
    
    return _create_file

@pytest.fixture
def output_dir(tmp_path):
    """Create and return a temporary directory for test outputs."""
    output_path = tmp_path / "transcripts"
    output_path.mkdir(exist_ok=True)
    return output_path 

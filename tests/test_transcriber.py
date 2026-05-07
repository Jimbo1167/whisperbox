"""
Tests for the video transcriber functionality.
"""

import os
import pytest
from src.transcriber import Transcriber, Config
import json
from pathlib import Path
import tempfile
import torch
import warnings
import wave
import struct
from transcribe_video import get_default_output_path
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from unittest.mock import patch

def create_mock_wav_file(path, duration=4.0):
    """Create a valid WAV file for testing."""
    sample_rate = 16000
    samples = np.zeros(int(duration * sample_rate))
    wavfile.write(path, sample_rate, samples.astype(np.int16))

def create_mock_mp3_file(path):
    """Create a valid MP3 file for testing."""
    wav_path = path.replace('.mp3', '.wav')
    create_mock_wav_file(wav_path)
    audio = AudioSegment.from_wav(wav_path)
    audio.export(path, format='mp3')
    os.remove(wav_path)

# Mock classes for faster testing
class MockWhisperModel:
    def transcribe(self, audio_path, **kwargs):
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        segments = [
            MockSegment(0.0, 2.0, "Test segment one"),
            MockSegment(2.0, 4.0, "Test segment two")
        ]
        return segments, None

class MockDiarizer:
    def __call__(self, waveform, sample_rate):
        return [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"}
        ]

@pytest.fixture
def test_config():
    """Create a test configuration."""
    class TestConfig(Config):
        def __init__(self):
            self.hf_token = "test_token"
            self.whisper_model_size = "base"
            self.diarization_model = "pyannote/speaker-diarization@2.1"
            self.language = "en"
            self.output_format = "txt"
            self.include_diarization = True
            self.audio_timeout = 300
            self.transcribe_timeout = 3600
            self.diarize_timeout = 3600
            self.force_cpu = True
            # Add cache-related attributes
            self.cache_enabled = False
            self.cache_expiration = 7 * 24 * 60 * 60  # 7 days in seconds
            self.max_cache_size = 10 * 1024 * 1024 * 1024  # 10GB
            self.transcription_engine = "whisper"
    return TestConfig()

@pytest.fixture
def test_transcriber(test_config):
    """Create a test transcriber with mocked models."""
    return Transcriber(config=test_config, test_mode=True)

@pytest.fixture
def test_files(request):
    """Get paths to test files."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    return {
        'video': str(fixtures_dir / "test_video.mp4"),
        'video_no_audio': str(fixtures_dir / "test_video_no_audio.mp4"),
        'audio': str(fixtures_dir / "test_audio.wav")
    }

@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary directory for output files."""
    return tmp_path

def test_transcriber_initialization(test_transcriber):
    """Test that the transcriber initializes correctly."""
    assert test_transcriber.transcription_engine is not None
    assert test_transcriber.diarization_engine is not None
    assert test_transcriber.config.whisper_model_size == "base"
    assert test_transcriber.config.language == "en"
    assert test_transcriber.config.include_diarization is True

def test_output_formats(test_transcriber, output_dir):
    """Test different output formats (txt, srt, vtt)."""
    segments = [
        (0.0, 2.0, "Hello world", "SPEAKER_01"),
        (2.0, 4.0, "This is a test", "SPEAKER_02")
    ]
    
    formats = ["txt", "srt", "vtt", "pretty"]
    for fmt in formats:
        test_transcriber.output_format = fmt
        output_path = output_dir / f"test.{fmt}"
        test_transcriber.save_transcript(segments, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Hello world" in content
            assert "This is a test" in content
            
            if fmt == "txt":
                assert "SPEAKER_01:" in content
                assert "SPEAKER_02:" in content
            elif fmt == "srt":
                assert "00:00:00" in content
                assert "-->" in content
            elif fmt == "vtt":
                assert "WEBVTT" in content
                assert "-->" in content
            elif fmt == "pretty":
                assert "SPEAKER_01" in content
                assert "Hello world" in content
                assert "\n\n" in content or "-->" in content

@patch('faster_whisper.WhisperModel.transcribe', MockWhisperModel.transcribe)
def test_transcribe_with_different_inputs(test_transcriber, tmp_path):
    """Test transcription with different input types."""
    wav_file = tmp_path / "test.wav"
    create_mock_wav_file(str(wav_file))

    segments = test_transcriber.transcribe(str(wav_file))
    assert len(segments) == 2
    for segment in segments:
        assert len(segment) == 4  # (start, end, text, speaker)
        assert isinstance(segment[0], float)  # start time
        assert isinstance(segment[1], float)  # end time
        assert isinstance(segment[2], str)    # text
        assert isinstance(segment[3], str)    # speaker
        assert "Test segment" in segment[2]   # Check text content

    test_transcriber.include_diarization = False
    test_transcriber.config.include_diarization = False
    segments = test_transcriber.transcribe(str(wav_file))
    assert len(segments) == 2
    for segment in segments:
        assert len(segment) == 4
        assert segment[3] == ""  # No speaker information
        assert "Test segment" in segment[2]

def test_error_handling_for_audio_files(test_transcriber, tmp_path):
    """Test error handling for audio file processing."""
    with pytest.raises(Exception):
        test_transcriber.transcribe("nonexistent.wav")
    
    invalid_wav = tmp_path / "invalid.wav"
    invalid_wav.write_text("not a real wav file")
    
    # In test mode, the mock models don't actually read the file,
    # so we need to patch the audio processor to raise an exception
    with patch.object(test_transcriber.audio_processor, 'get_audio_path') as mock_get_audio:
        mock_get_audio.side_effect = Exception("Invalid WAV file")
        with pytest.raises(Exception):
            test_transcriber.transcribe(str(invalid_wav))

@patch('faster_whisper.WhisperModel.transcribe', MockWhisperModel.transcribe)
def test_real_file_transcription(test_transcriber, test_files):
    """Test transcription with real test files."""
    segments = test_transcriber.transcribe(test_files['video'])
    assert len(segments) > 0
    for segment in segments:
        assert len(segment) == 4  # (start, end, text, speaker)
        assert isinstance(segment[0], float)
        assert isinstance(segment[1], float)
        assert isinstance(segment[2], str)
        assert isinstance(segment[3], str)
    
    segments = test_transcriber.transcribe(test_files['audio'])
    assert len(segments) > 0
    for segment in segments:
        assert len(segment) == 4
        assert isinstance(segment[0], float)
        assert isinstance(segment[1], float)
        assert isinstance(segment[2], str)
        assert isinstance(segment[3], str)
    
    # Disable diarization
    test_transcriber.include_diarization = False
    test_transcriber.config.include_diarization = False
    segments = test_transcriber.transcribe(test_files['video'])
    assert len(segments) > 0
    for segment in segments:
        assert len(segment) == 4
        assert segment[3] == ""  # No speaker information 

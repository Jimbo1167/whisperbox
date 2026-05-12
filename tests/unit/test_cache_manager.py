"""
Unit tests for the CacheManager class.
"""

import os
import json
import time
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.config import Config
from src.cache.manager import CacheManager

@pytest.fixture
def test_config():
    """Create a test configuration."""
    config = Config()
    config.cache_enabled = True
    config.cache_expiration = 3600  # 1 hour
    config.max_cache_size = 1024 * 1024 * 10  # 10MB
    return config

@pytest.fixture
def cache_manager(test_config):
    """Create a test cache manager."""
    # Create a temporary directory for the cache
    original_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
    test_cache_dir = tempfile.mkdtemp()
    
    # Patch the cache directory
    with patch.object(CacheManager, "__init__", lambda self, config: None):
        manager = CacheManager(test_config)
        manager.config = test_config
        manager.cache_dir = test_cache_dir
        manager.audio_cache_dir = os.path.join(test_cache_dir, "audio")
        manager.transcription_cache_dir = os.path.join(test_cache_dir, "transcription")
        manager.diarization_cache_dir = os.path.join(test_cache_dir, "diarization")
        manager.cache_expiration = test_config.cache_expiration
        manager.max_cache_size = test_config.max_cache_size
        
        # Create the cache directories
        os.makedirs(manager.audio_cache_dir, exist_ok=True)
        os.makedirs(manager.transcription_cache_dir, exist_ok=True)
        os.makedirs(manager.diarization_cache_dir, exist_ok=True)
        
        yield manager
        
        # Clean up the temporary directory
        shutil.rmtree(test_cache_dir)

@pytest.fixture
def test_audio_file():
    """Create a test audio file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"test audio data")
        return f.name

@pytest.fixture
def test_input_file():
    """Create a test input file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"test video data")
        return f.name

def test_generate_cache_key(cache_manager, test_input_file):
    """Test generating a cache key."""
    # Generate a cache key
    cache_key = cache_manager._generate_cache_key(test_input_file)
    
    # Check that the cache key is a string
    assert isinstance(cache_key, str)
    
    # Check that the cache key is not empty
    assert cache_key
    
    # Generate another cache key with a prefix
    cache_key_with_prefix = cache_manager._generate_cache_key(test_input_file, prefix="test")
    
    # Check that the prefix is included
    assert cache_key_with_prefix.startswith("test_")
    
    # Check that the rest of the key is the same
    assert cache_key_with_prefix[5:] == cache_key

def test_get_cache_path(cache_manager):
    """Test getting a cache path."""
    # Generate a cache key
    cache_key = "test_cache_key"
    
    # Get the cache path for audio
    audio_cache_path = cache_manager._get_cache_path(cache_key, "audio")
    
    # Check that the path is correct
    assert audio_cache_path == os.path.join(cache_manager.audio_cache_dir, f"{cache_key}.wav")
    
    # Get the cache path for transcription
    transcription_cache_path = cache_manager._get_cache_path(cache_key, "transcription")
    
    # Check that the path is correct
    assert transcription_cache_path == os.path.join(cache_manager.transcription_cache_dir, f"{cache_key}.json")
    
    # Get the cache path for diarization
    diarization_cache_path = cache_manager._get_cache_path(cache_key, "diarization")
    
    # Check that the path is correct
    assert diarization_cache_path == os.path.join(cache_manager.diarization_cache_dir, f"{cache_key}.json")
    
    # Check that an invalid cache type raises an error
    with pytest.raises(ValueError):
        cache_manager._get_cache_path(cache_key, "invalid")

def test_is_cache_valid(cache_manager):
    """Test checking if a cache is valid."""
    # Create a test file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        test_file = f.name
    
    # Check that the cache is valid
    assert cache_manager._is_cache_valid(test_file)
    
    # Modify the file's modification time to be older than the cache expiration
    os.utime(test_file, (time.time() - cache_manager.cache_expiration - 10, time.time() - cache_manager.cache_expiration - 10))
    
    # Check that the cache is now invalid
    assert not cache_manager._is_cache_valid(test_file)
    
    # Check that a non-existent file is invalid
    assert not cache_manager._is_cache_valid("non_existent_file")
    
    # Clean up
    os.remove(test_file)

def test_cache_audio(cache_manager, test_input_file, test_audio_file):
    """Test caching an audio file."""
    # Cache the audio file
    cached_path = cache_manager.cache_audio(test_input_file, test_audio_file)
    
    # Check that the cached file exists
    assert os.path.exists(cached_path)
    
    # Check that the cached file is in the audio cache directory
    assert cached_path.startswith(cache_manager.audio_cache_dir)
    
    # Check that the cached file has the correct extension
    assert cached_path.endswith(".wav")

def test_get_cached_audio(cache_manager, test_input_file, test_audio_file):
    """Test getting a cached audio file."""
    # Cache the audio file
    cached_path = cache_manager.cache_audio(test_input_file, test_audio_file)
    
    # Get the cached audio file
    retrieved_path = cache_manager.get_cached_audio(test_input_file)
    
    # Check that the retrieved path is the same as the cached path
    assert retrieved_path == cached_path
    
    # Check that a non-existent cache returns None
    assert cache_manager.get_cached_audio("non_existent_file") is None

def test_cache_transcription(cache_manager, test_audio_file):
    """Test caching transcription results."""
    # Create test transcription results
    transcription_results = [
        {"start": 0.0, "end": 1.0, "text": "Test segment one"},
        {"start": 1.0, "end": 2.0, "text": "Test segment two"}
    ]
    
    # Cache the transcription results
    cache_manager.cache_transcription(test_audio_file, transcription_results)

    # Get the cache path (default engine_id="whisper" → prefix="transcription-whisper")
    cache_key = cache_manager._generate_cache_key(test_audio_file, prefix="transcription-whisper")
    cache_path = cache_manager._get_cache_path(cache_key, "transcription")
    
    # Check that the cache file exists
    assert os.path.exists(cache_path)
    
    # Check that the cache file contains the correct data
    with open(cache_path, "r") as f:
        cached_data = json.load(f)
        assert cached_data == transcription_results

def test_get_cached_transcription(cache_manager, test_audio_file):
    """Test getting cached transcription results."""
    # Create test transcription results
    transcription_results = [
        {"start": 0.0, "end": 1.0, "text": "Test segment one"},
        {"start": 1.0, "end": 2.0, "text": "Test segment two"}
    ]
    
    # Cache the transcription results
    cache_manager.cache_transcription(test_audio_file, transcription_results)
    
    # Get the cached transcription results
    retrieved_results = cache_manager.get_cached_transcription(test_audio_file)
    
    # Check that the retrieved results are the same as the original
    assert retrieved_results == transcription_results
    
    # Check that a non-existent cache returns None
    assert cache_manager.get_cached_transcription("non_existent_file") is None

def test_cache_diarization(cache_manager, test_audio_file):
    """Test caching diarization results."""
    # Create test diarization results
    diarization_results = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"}
    ]
    
    # Cache the diarization results
    cache_manager.cache_diarization(test_audio_file, diarization_results)
    
    # Get the cache path
    cache_key = cache_manager._generate_cache_key(test_audio_file, prefix="diarization")
    cache_path = cache_manager._get_cache_path(cache_key, "diarization")
    
    # Check that the cache file exists
    assert os.path.exists(cache_path)
    
    # Check that the cache file contains the correct data
    with open(cache_path, "r") as f:
        cached_data = json.load(f)
        assert cached_data == diarization_results

def test_get_cached_diarization(cache_manager, test_audio_file):
    """Test getting cached diarization results."""
    # Create test diarization results
    diarization_results = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"}
    ]
    
    # Cache the diarization results
    cache_manager.cache_diarization(test_audio_file, diarization_results)
    
    # Get the cached diarization results
    retrieved_results = cache_manager.get_cached_diarization(test_audio_file)
    
    # Check that the retrieved results are the same as the original
    assert retrieved_results == diarization_results
    
    # Check that a non-existent cache returns None
    assert cache_manager.get_cached_diarization("non_existent_file") is None

def test_clear_cache(cache_manager, test_input_file, test_audio_file):
    """Test clearing the cache."""
    # Cache some data
    cache_manager.cache_audio(test_input_file, test_audio_file)
    
    transcription_results = [
        {"start": 0.0, "end": 1.0, "text": "Test segment one"},
        {"start": 1.0, "end": 2.0, "text": "Test segment two"}
    ]
    cache_manager.cache_transcription(test_audio_file, transcription_results)
    
    diarization_results = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"}
    ]
    cache_manager.cache_diarization(test_audio_file, diarization_results)
    
    # Clear the audio cache
    cache_manager.clear_cache("audio")
    
    # Check that the audio cache is empty
    assert len(os.listdir(cache_manager.audio_cache_dir)) == 0
    
    # Check that the other caches still have files
    assert len(os.listdir(cache_manager.transcription_cache_dir)) > 0
    assert len(os.listdir(cache_manager.diarization_cache_dir)) > 0
    
    # Clear all caches
    cache_manager.clear_cache()
    
    # Check that all caches are empty
    assert len(os.listdir(cache_manager.audio_cache_dir)) == 0
    assert len(os.listdir(cache_manager.transcription_cache_dir)) == 0
    assert len(os.listdir(cache_manager.diarization_cache_dir)) == 0

def test_cleanup_cache(cache_manager):
    """Test cleaning up the cache."""
    # Create some test files
    for i in range(5):
        with open(os.path.join(cache_manager.audio_cache_dir, f"test_{i}.wav"), "w") as f:
            f.write("test data")
    
    # Set some files to be older than the cache expiration
    for i in range(2):
        os.utime(
            os.path.join(cache_manager.audio_cache_dir, f"test_{i}.wav"),
            (time.time() - cache_manager.cache_expiration - 10, time.time() - cache_manager.cache_expiration - 10)
        )
    
    # Clean up the cache
    cache_manager._cleanup_cache()
    
    # Check that the expired files were removed
    assert not os.path.exists(os.path.join(cache_manager.audio_cache_dir, "test_0.wav"))
    assert not os.path.exists(os.path.join(cache_manager.audio_cache_dir, "test_1.wav"))
    
    # Check that the non-expired files still exist
    assert os.path.exists(os.path.join(cache_manager.audio_cache_dir, "test_2.wav"))
    assert os.path.exists(os.path.join(cache_manager.audio_cache_dir, "test_3.wav"))
    assert os.path.exists(os.path.join(cache_manager.audio_cache_dir, "test_4.wav"))

def test_get_cache_size(cache_manager):
    """Test getting the cache size."""
    # Create some test files
    for i in range(5):
        with open(os.path.join(cache_manager.audio_cache_dir, f"test_{i}.wav"), "w") as f:
            f.write("test data" * 100)  # 900 bytes per file

    # Get the cache size
    cache_size = cache_manager._get_cache_size()

    # Check that the cache size is correct (5 files * 900 bytes = 4500 bytes)
    assert cache_size == 5 * 900


class TestEngineScopedTranscriptionCache:
    def test_cache_is_scoped_by_engine_id(self, tmp_path, monkeypatch):
        """Same audio cached under different engine_ids must not collide."""
        from src.cache.manager import CacheManager
        from src.config import Config

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: str(tmp_path) if p == "~" else os.path.expanduser(p),
        )
        cfg = Config()
        cm = CacheManager(cfg)

        # Real audio file is required for cache key generation (uses st_mtime).
        audio = tmp_path / "sample.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        cm.cache_transcription(str(audio), [{"start": 0.0, "end": 1.0, "text": "w", "words": []}],
                               engine_id="whisper-large-v3-turbo")
        cm.cache_transcription(str(audio), [{"start": 0.0, "end": 1.0, "text": "p", "words": []}],
                               engine_id="parakeet-mlx_community_parakeet-tdt-0.6b-v3")

        assert cm.get_cached_transcription(str(audio), engine_id="whisper-large-v3-turbo")[0]["text"] == "w"
        assert cm.get_cached_transcription(str(audio), engine_id="parakeet-mlx_community_parakeet-tdt-0.6b-v3")[0]["text"] == "p"

    def test_default_engine_id_is_whisper(self, tmp_path, monkeypatch):
        """Backwards compat: callers that don't pass engine_id behave as before."""
        from src.cache.manager import CacheManager
        from src.config import Config

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: str(tmp_path) if p == "~" else os.path.expanduser(p),
        )
        cfg = Config()
        cm = CacheManager(cfg)

        audio = tmp_path / "sample.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        cm.cache_transcription(str(audio), [{"start": 0.0, "end": 1.0, "text": "x", "words": []}])
        assert cm.get_cached_transcription(str(audio))[0]["text"] == "x"

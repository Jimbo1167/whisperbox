import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import pytest
import imageio_ffmpeg

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.audio.processor import AudioProcessor, TimeoutException

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture
def config():
    """Create a test configuration."""
    config = Config()
    config.audio_timeout = 30
    return config


@pytest.fixture
def audio_processor(config):
    """Audio processor with caching disabled (no shared-state cache dir)."""
    config.cache_enabled = False
    return AudioProcessor(config)


@pytest.fixture
def cached_audio_processor(config, tmp_path):
    """Audio processor whose cache manager is re-pointed at a temp dir."""
    config.cache_enabled = True
    processor = AudioProcessor(config)
    cache_root = tmp_path / "cache"
    processor.cache_manager.cache_dir = str(cache_root)
    processor.cache_manager.audio_cache_dir = str(cache_root / "audio")
    processor.cache_manager.transcription_cache_dir = str(cache_root / "transcription")
    processor.cache_manager.diarization_cache_dir = str(cache_root / "diarization")
    for d in (
        processor.cache_manager.audio_cache_dir,
        processor.cache_manager.transcription_cache_dir,
        processor.cache_manager.diarization_cache_dir,
    ):
        os.makedirs(d, exist_ok=True)
    return processor


def _make_media(tmp_path, name, extra_args):
    """Generate a tiny media file with a 440Hz sine audio track via ffmpeg."""
    out = str(tmp_path / name)
    cmd = [
        FFMPEG, "-y", "-nostdin",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=0.5",
        *extra_args,
        out,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return out


@pytest.fixture
def media_dir(tmp_path):
    d = tmp_path / "media"
    d.mkdir()
    return d


@pytest.fixture
def sample_mp4(media_dir):
    """A 0.5s mp4 with a black video track and stereo 44.1kHz audio."""
    return _make_media(
        media_dir,
        "sample.mp4",
        [
            "-f", "lavfi", "-i", "color=black:s=64x64:d=0.5:r=10",
            "-ac", "2", "-shortest",
        ],
    )


@pytest.fixture
def sample_mp3(media_dir):
    """A 0.5s stereo 44.1kHz mp3."""
    return _make_media(media_dir, "sample.mp3", ["-ac", "2"])


def _wav_params(path):
    with wave.open(path, "rb") as wav_file:
        return wav_file.getframerate(), wav_file.getnchannels()


def test_is_audio_file(audio_processor):
    assert audio_processor.is_audio_file("test.wav") is True
    assert audio_processor.is_audio_file("test.mp3") is True
    assert audio_processor.is_audio_file("test.m4a") is True
    assert audio_processor.is_audio_file("test.aac") is True
    assert audio_processor.is_audio_file("test.flac") is True
    assert audio_processor.is_audio_file("test.ogg") is True
    assert audio_processor.is_audio_file("test.mp4") is False
    assert audio_processor.is_audio_file("test.mov") is False
    assert audio_processor.is_audio_file("test.txt") is False


def test_wav_passthrough(audio_processor, tmp_path):
    wav_path = str(tmp_path / "input.wav")
    samples = np.zeros(1600, dtype=np.int16)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    audio_path, needs_cleanup = audio_processor.get_audio_path(wav_path)
    assert audio_path == wav_path
    assert needs_cleanup is False


def test_missing_input_raises(audio_processor):
    with pytest.raises(FileNotFoundError):
        audio_processor.get_audio_path("/nonexistent/file.mp4")


def test_video_extraction_produces_16k_mono_wav(audio_processor, sample_mp4):
    audio_path, needs_cleanup = audio_processor.get_audio_path(sample_mp4)
    try:
        assert needs_cleanup is True
        rate, channels = _wav_params(audio_path)
        assert rate == 16000
        assert channels == 1
        # No stray WAV written next to the input video
        sibling = sample_mp4.rsplit(".", 1)[0] + ".wav"
        assert not os.path.exists(sibling)
        # Temp output lives in the system temp dir, not next to the input
        assert os.path.dirname(audio_path) != os.path.dirname(sample_mp4)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def test_audio_conversion_produces_16k_mono_wav(audio_processor, sample_mp3):
    audio_path, needs_cleanup = audio_processor.get_audio_path(sample_mp3)
    try:
        assert needs_cleanup is True
        rate, channels = _wav_params(audio_path)
        assert rate == 16000
        assert channels == 1
        sibling = sample_mp3.rsplit(".", 1)[0] + ".wav"
        assert not os.path.exists(sibling)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def test_extraction_writes_directly_into_cache_with_stable_path(
    cached_audio_processor, sample_mp4
):
    """Extraction lands in the audio cache and the path is identical across runs.

    A stable audio path is what keys the downstream transcription/diarization
    caches, so this guards against the run-2 cache-miss regression.
    """
    first_path, first_cleanup = cached_audio_processor.get_audio_path(sample_mp4)
    assert first_cleanup is False
    assert first_path.startswith(cached_audio_processor.cache_manager.audio_cache_dir)
    rate, channels = _wav_params(first_path)
    assert rate == 16000
    assert channels == 1
    first_stat = os.stat(first_path)

    second_path, second_cleanup = cached_audio_processor.get_audio_path(sample_mp4)
    assert second_cleanup is False
    assert second_path == first_path
    # Cache hit: the file was not re-extracted
    assert os.stat(second_path).st_mtime_ns == first_stat.st_mtime_ns

    # No stray WAV written next to the input video
    sibling = sample_mp4.rsplit(".", 1)[0] + ".wav"
    assert not os.path.exists(sibling)


def test_extraction_failure_raises_and_leaves_no_partial_output(
    cached_audio_processor, tmp_path
):
    bogus = str(tmp_path / "corrupt.mp4")
    with open(bogus, "wb") as f:
        f.write(b"this is not a video")

    with pytest.raises(Exception):
        cached_audio_processor.get_audio_path(bogus)

    assert os.listdir(cached_audio_processor.cache_manager.audio_cache_dir) == []


def test_extraction_timeout_raises_timeout_exception(
    audio_processor, sample_mp4, monkeypatch
):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr("src.audio.processor.subprocess.run", fake_run)
    with pytest.raises(TimeoutException):
        audio_processor.get_audio_path(sample_mp4)


def test_load_audio(audio_processor, tmp_path):
    audio_path = tmp_path / "test.wav"

    samples = np.zeros(16000, dtype=np.int16)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    audio = audio_processor.load_audio(str(audio_path))

    assert isinstance(audio, np.ndarray)
    assert len(audio) == 16000


def test_process_audio_stream(audio_processor):
    audio_data = np.zeros(100000)

    bytes_per_sample = audio_data.itemsize
    chunk_samples = AudioProcessor.CHUNK_SIZE // bytes_per_sample

    chunks = list(audio_processor.process_audio_stream(audio_data))

    expected_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
    assert len(chunks) == expected_chunks


class TestRunWithTimeout:
    def test_returns_result_when_fast_enough(self):
        from src.audio.processor import run_with_timeout
        assert run_with_timeout(lambda: 42, 5, "nope") == 42

    def test_raises_timeout_exception_in_caller(self):
        """The old thread-Timer 'timeout' raised in the wrong thread and never
        actually interrupted the caller — this pins the fixed behavior."""
        import time as _time
        from src.audio.processor import run_with_timeout

        with pytest.raises(TimeoutException):
            run_with_timeout(lambda: _time.sleep(2), 0.05, "too slow")

    def test_propagates_fn_exception(self):
        from src.audio.processor import run_with_timeout

        def boom():
            raise ValueError("inner")

        with pytest.raises(ValueError):
            run_with_timeout(boom, 5, "nope")

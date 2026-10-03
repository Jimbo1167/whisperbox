import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Iterator

from src.config import Config
from src.transcription.streaming import StreamingTranscriber, AsyncStreamingTranscriber
from src.transcription.engine import TranscriptionEngine

@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel for testing."""
    mock = MagicMock()
    
    # Mock the transcribe method
    def mock_transcribe(audio_data, **kwargs):
        # Create mock segments based on the length of audio data
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []
        
        # Generate segments based on audio length
        # For testing, create a segment for each 1 second of audio
        segments = []
        
        # Handle both numpy arrays and lists
        if isinstance(audio_data, np.ndarray):
            audio_length = len(audio_data) / 16000  # Assuming 16kHz audio
        else:
            audio_length = 1  # Default for testing
        
        for i in range(int(audio_length) or 1):  # Ensure at least one segment
            segments.append(MockSegment(
                start=float(i),
                end=float(i + 1),
                text=f"Test segment {i+1}"
            ))
        
        return segments, {"language": "en"}
    
    mock.transcribe.side_effect = mock_transcribe
    return mock

@pytest.fixture
def config():
    """Create a test configuration."""
    config = Config()
    config.language = "en"
    config.whisper_model_size = "tiny"
    return config

def create_audio_chunks():
    """Create audio chunks for testing."""
    # Create 3 chunks of 1 second each at 16kHz
    sample_rate = 16000
    return [
        np.zeros(sample_rate, dtype=np.float32),
        np.zeros(sample_rate, dtype=np.float32),
        np.zeros(sample_rate, dtype=np.float32)
    ]

def test_streaming_transcriber_init(mock_whisper_model, config):
    """Test initialization of StreamingTranscriber."""
    transcriber = StreamingTranscriber(mock_whisper_model, config)
    
    assert transcriber.whisper == mock_whisper_model
    assert transcriber.config == config
    assert transcriber.language == config.language
    assert transcriber.sample_rate == 16000
    assert transcriber.buffer_size == 30 * 16000  # 30 seconds buffer
    assert transcriber.min_chunk_size == 5 * 16000  # 5 seconds minimum

def test_streaming_transcriber_process_stream(mock_whisper_model, config):
    """Test processing an audio stream."""
    transcriber = StreamingTranscriber(mock_whisper_model, config)
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Process the stream
    segments = list(transcriber.process_stream(iter(audio_chunks)))
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
        assert isinstance(segment["start"], float)
        assert isinstance(segment["end"], float)
        assert isinstance(segment["text"], str)
        assert isinstance(segment["words"], list)

def test_async_streaming_transcriber(mock_whisper_model, config):
    """Test the AsyncStreamingTranscriber."""
    async_transcriber = AsyncStreamingTranscriber(mock_whisper_model, config)
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Start processing
    async_transcriber.start_processing(iter(audio_chunks))
    
    # Get results
    segments = list(async_transcriber.get_results())
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
        assert isinstance(segment["start"], float)
        assert isinstance(segment["end"], float)
        assert isinstance(segment["text"], str)
        assert isinstance(segment["words"], list)
    
    # Stop the transcriber
    async_transcriber.stop()
    assert not async_transcriber.is_running

@patch('src.transcription.engine.WhisperModel')
def test_transcription_engine_stream(mock_whisper_class, config):
    """Test the transcribe_stream method of TranscriptionEngine."""
    # Create a mock WhisperModel instance
    mock_whisper = MagicMock()
    mock_whisper_class.return_value = mock_whisper
    
    # Mock the transcribe method
    def mock_transcribe(audio_data, **kwargs):
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []
        
        segments = [MockSegment(0.0, 1.0, "Test segment")]
        return segments, {"language": "en"}
    
    mock_whisper.transcribe.side_effect = mock_transcribe
    
    # Create the transcription engine
    engine = TranscriptionEngine(config)
    engine.whisper = mock_whisper
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Process the stream
    segments = list(engine.transcribe_stream(iter(audio_chunks)))
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment

@patch('src.transcription.engine.WhisperModel')
def test_transcription_engine_async_stream(mock_whisper_class, config):
    """Test the start_async_transcription method of TranscriptionEngine."""
    # Create a mock WhisperModel instance
    mock_whisper = MagicMock()
    mock_whisper_class.return_value = mock_whisper
    
    # Mock the transcribe method
    def mock_transcribe(audio_data, **kwargs):
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []
        
        segments = [MockSegment(0.0, 1.0, "Test segment")]
        return segments, {"language": "en"}
    
    mock_whisper.transcribe.side_effect = mock_transcribe
    
    # Create the transcription engine
    engine = TranscriptionEngine(config)
    engine.whisper = mock_whisper
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Start async processing
    async_transcriber = engine.start_async_transcription(iter(audio_chunks))
    
    # Get results
    segments = list(async_transcriber.get_results())
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
    
    # Stop the transcriber
    async_transcriber.stop() 

# ---------------------------------------------------------------------------
# Absolute timestamps and chunk-boundary handling
# ---------------------------------------------------------------------------

SR = 16000


class _Word:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class _Segment:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


class _TimelineWhisper:
    """Fake WhisperModel over a known utterance timeline.

    The audio is a ramp whose sample values are their own absolute time, so
    each call can tell which slice of the stream it was given. Like
    faster-whisper, it returns buffer-relative timestamps, and an utterance
    running past the buffer edge comes back cut off.
    """

    def __init__(self, utterances):
        self.utterances = utterances  # [(abs_start, abs_end, text)]
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"buffer_seconds": len(audio) / SR, **kwargs})
        t0 = float(audio[0])
        t1 = t0 + len(audio) / SR
        eps = 1e-6
        segments = []
        for start, end, text in self.utterances:
            if end <= t0 + eps or start >= t1 - eps:
                continue
            label = text
            if start < t0 - eps:
                label += " (tail)"  # re-transcribing already-covered audio
            if end > t1 + eps:
                label += " (cut)"
            rel_start = max(start, t0) - t0
            rel_end = min(end, t1) - t0
            words = [_Word(rel_start, rel_end, " " + label)] if kwargs.get("word_timestamps") else None
            segments.append(_Segment(rel_start, rel_end, " " + label, words))
        return segments, {"language": "en"}


def _ramp_chunks(total_seconds, chunk_seconds=5.0):
    """Mimic AudioProcessor.stream_audio_from_file: fixed-size chunks."""
    ramp = np.arange(int(total_seconds * SR), dtype=np.float64) / SR
    step = int(chunk_seconds * SR)
    return [ramp[i:i + step] for i in range(0, len(ramp), step)]


def test_word_timestamps_flag_reaches_faster_whisper(config):
    whisper = _TimelineWhisper([(0.0, 1.0, "hi")])

    list(StreamingTranscriber(whisper, config).process_stream(iter(_ramp_chunks(2))))
    assert whisper.calls[-1]["word_timestamps"] is False

    list(StreamingTranscriber(whisper, config, word_timestamps=True)
         .process_stream(iter(_ramp_chunks(2))))
    assert whisper.calls[-1]["word_timestamps"] is True


def test_stream_timestamps_are_absolute_and_boundary_segments_emitted_once(config):
    """Regression: every 5 s chunk restarted timestamps near 0, and the
    0.5 s overlap re-emitted words at each boundary ("two bugs two bugs")."""
    utterances = [
        (0.0, 2.6, "the quick brown fox"),
        (3.3, 4.9, "whisperbox should"),
        (4.9, 7.4, "transcribe this clip"),  # crosses the 5 s boundary
        (7.6, 10.6, "found two bugs today"),  # crosses the 10 s boundary
    ]
    whisper = _TimelineWhisper(utterances)
    transcriber = StreamingTranscriber(whisper, config, word_timestamps=True)

    segments = list(transcriber.process_stream(iter(_ramp_chunks(10.75))))

    assert [(s["start"], s["end"], s["text"]) for s in segments] == utterances
    for segment in segments:
        (word,) = segment["words"]
        assert (word["start"], word["end"]) == (segment["start"], segment["end"])


def test_stream_buffer_stays_bounded_for_one_long_utterance(config):
    """Holding back a segment that never ends must not grow the buffer
    without limit: past buffer_size everything is emitted."""
    whisper = _TimelineWhisper([(0.0, 42.0, "a very long monologue")])
    transcriber = StreamingTranscriber(whisper, config)

    segments = list(transcriber.process_stream(iter(_ramp_chunks(42.0))))

    max_buffer = max(call["buffer_seconds"] for call in whisper.calls)
    assert max_buffer <= transcriber.buffer_size / SR + 5.0
    # Contiguous, non-overlapping coverage of the whole utterance.
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 42.0
    for prev, nxt in zip(segments, segments[1:]):
        assert nxt["start"] == prev["end"]


def test_stream_ending_right_after_a_pass_does_not_retranscribe(config):
    """When the last chunk triggers a pass, the held-back segment was cut only
    by the end of the audio: emit it rather than transcribing the same buffer
    a second time."""
    utterances = [(0.0, 2.0, "hello"), (4.0, 12.0, "a longer sentence")]
    whisper = _TimelineWhisper(utterances)
    transcriber = StreamingTranscriber(whisper, config)

    # Chunks 0-5, 5-10, 10-12: the 10-12 chunk brings the buffer to 8 s.
    segments = list(transcriber.process_stream(iter(_ramp_chunks(12.0))))

    assert [(s["start"], s["end"], s["text"]) for s in segments] == utterances
    assert [call["buffer_seconds"] for call in whisper.calls] == [5.0, 6.0, 8.0]

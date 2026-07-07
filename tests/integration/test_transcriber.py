import os
import sys
import wave
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.transcriber import Transcriber
from src.audio.processor import AudioProcessor
from src.transcription.engine import TranscriptionEngine
from src.diarization.engine import DiarizationEngine
from src.output.formatter import OutputFormatter

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Config()
    config.whisper_model_size = "base"
    config.include_diarization = True
    config.output_format = "txt"
    return config

@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor."""
    mock = MagicMock(spec=AudioProcessor)
    mock.get_audio_path.return_value = ("test.wav", False)
    return mock

@pytest.fixture
def mock_transcription_engine():
    """Create a mock transcription engine."""
    mock = MagicMock(spec=TranscriptionEngine)
    mock.transcribe.return_value = [
        {"start": 0.0, "end": 2.0, "text": "Test segment one"},
        {"start": 2.0, "end": 4.0, "text": "Test segment two"}
    ]
    return mock

@pytest.fixture
def mock_diarization_engine():
    """Create a mock diarization engine."""
    mock = MagicMock(spec=DiarizationEngine)
    mock.diarize.return_value = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"}
    ]
    return mock

@pytest.fixture
def mock_output_formatter():
    """Create a mock output formatter."""
    mock = MagicMock(spec=OutputFormatter)
    return mock

@pytest.fixture
def transcriber(mock_config, mock_audio_processor, mock_transcription_engine, 
               mock_diarization_engine, mock_output_formatter):
    """Create a transcriber with mock components."""
    transcriber = Transcriber(mock_config)
    transcriber.audio_processor = mock_audio_processor
    transcriber.transcription_engine = mock_transcription_engine
    transcriber.diarization_engine = mock_diarization_engine
    transcriber.output_formatter = mock_output_formatter
    return transcriber

def test_transcribe_with_diarization(transcriber, mock_audio_processor, 
                                   mock_transcription_engine, mock_diarization_engine):
    """Test transcribe method with diarization enabled."""
    # Setup mock return values
    mock_audio_processor.get_audio_path.return_value = ("test.wav", False)
    
    # Run the transcribe method
    result = transcriber.transcribe("test.mp4")
    
    # Check that the audio processor was called
    mock_audio_processor.get_audio_path.assert_called_once_with("test.mp4")
    
    # Check that the transcription engine was called
    mock_transcription_engine.transcribe.assert_called_once_with("test.wav")
    
    # Check that the diarization engine was called
    mock_diarization_engine.diarize.assert_called_once_with("test.wav", enabled=True)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == (0.0, 2.0, "Test segment one", "SPEAKER_01")
    assert result[1] == (2.0, 4.0, "Test segment two", "SPEAKER_02")

def test_transcribe_without_diarization(transcriber, mock_audio_processor, 
                                      mock_transcription_engine, mock_diarization_engine):
    """Test transcribe method with diarization disabled."""
    # Setup mock return values
    mock_audio_processor.get_audio_path.return_value = ("test.wav", False)
    
    # Disable diarization
    transcriber.include_diarization = False
    transcriber.config.include_diarization = False
    
    # Run the transcribe method
    result = transcriber.transcribe("test.mp4")
    
    # Check that the audio processor was called
    mock_audio_processor.get_audio_path.assert_called_once_with("test.mp4")
    
    # Check that the transcription engine was called
    mock_transcription_engine.transcribe.assert_called_once_with("test.wav")
    
    # Check that the diarization engine was not called
    mock_diarization_engine.diarize.assert_not_called()
    
    # Check the result
    assert len(result) == 2
    assert result[0][3] == ""  # No speaker information
    assert result[1][3] == ""  # No speaker information

def test_save_transcript(transcriber, mock_output_formatter):
    """Test save_transcript method."""
    # Create test segments
    segments = [
        (0.0, 2.0, "Test segment one", "SPEAKER_01"),
        (2.0, 4.0, "Test segment two", "SPEAKER_02")
    ]
    
    # Run the save_transcript method
    transcriber.save_transcript(segments, "test.txt")
    
    # Check that the output formatter was called
    mock_output_formatter.save_transcript.assert_called_once_with(segments, "test.txt")

def test_combine_segments_with_speakers(transcriber):
    """Test _combine_segments_with_speakers method."""
    # Create test segments
    transcription_segments = [
        {"start": 0.0, "end": 2.0, "text": "Test segment one"},
        {"start": 2.0, "end": 4.0, "text": "Test segment two"}
    ]
    
    diarization_segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"}
    ]
    
    # Run the _combine_segments_with_speakers method
    result = transcriber._combine_segments_with_speakers(transcription_segments, diarization_segments)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == (0.0, 2.0, "Test segment one", "SPEAKER_01")
    assert result[1] == (2.0, 4.0, "Test segment two", "SPEAKER_02")
    
    # Test with no diarization segments
    result = transcriber._combine_segments_with_speakers(transcription_segments, None)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == (0.0, 2.0, "Test segment one", "")
    assert result[1] == (2.0, 4.0, "Test segment two", "")


def test_transcribe_reports_progress(transcriber, mock_audio_processor, mock_transcription_engine):
    """Test that transcribe emits coarse progress updates."""
    mock_audio_processor.get_audio_path.return_value = ("test.wav", False)
    transcriber.include_diarization = False
    transcriber.config.include_diarization = False

    progress_events = []

    result = transcriber.transcribe(
        "test.mp4",
        progress_callback=lambda message, progress: progress_events.append((message, progress)),
    )

    assert len(result) == 2
    assert progress_events
    assert progress_events[0][0] == "Preparing audio"
    assert progress_events[-1][0] == "Finalizing transcript"


class TestStreamingGuard:
    def test_transcribe_stream_rejects_parakeet_engine(self, mock_config):
        from src.transcriber import Transcriber

        mock_config.transcription_engine = "parakeet"
        # Bypass factory's NotImplementedError by stubbing engine after construction.
        # We're testing the upstream guard, not engine construction.
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.audio_processor = MagicMock()
        t.transcription_engine = MagicMock()
        t.diarization_engine = MagicMock()
        t.output_formatter = MagicMock()
        t.include_diarization = False
        t.test_mode = False

        with pytest.raises(NotImplementedError, match="Streaming is only supported"):
            list(t.transcribe_stream("input.wav"))

    def test_transcribe_stream_with_diarization_rejects_parakeet(self, mock_config):
        from src.transcriber import Transcriber

        mock_config.transcription_engine = "parakeet"
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.audio_processor = MagicMock()
        t.transcription_engine = MagicMock()
        t.diarization_engine = MagicMock()
        t.output_formatter = MagicMock()
        t.include_diarization = True
        t.test_mode = False

        with pytest.raises(NotImplementedError, match="Streaming is only supported"):
            list(t.transcribe_stream_with_diarization("input.wav"))


class TestEndToEndAcrossEngines:
    """End-to-end transcribe + diarization combination across both engines.

    Both engines must produce the same output shape so downstream formatters
    and diarization alignment work identically.
    """

    @pytest.mark.parametrize("engine_name", ["whisper", "parakeet"])
    def test_transcribe_with_diarization(self, engine_name, tmp_path, mock_diarizer):
        cfg = Config(
            transcription_engine=engine_name,
            include_diarization=True,
        )
        cfg.cache_enabled = False
        cfg.hf_token = "test_token"

        # Bypass platform validation for parakeet by not calling validate(); we
        # exercise the engine's runtime behavior here, not config validation.
        # ParakeetEngine in test_mode does not import parakeet_mlx, so this works
        # on Linux CI as well.

        t = Transcriber(cfg, test_mode=True)

        # Real audio file so AudioProcessor works.
        audio = tmp_path / "x.wav"
        sr = 16000
        samples = np.zeros(int(2.0 * sr), dtype=np.int16)
        with wave.open(str(audio), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(samples.tobytes())

        t.diarization_engine.diarizer = mock_diarizer

        segments = t.transcribe(str(audio))

        # Output shape contract is the same regardless of engine.
        assert len(segments) > 0
        for seg in segments:
            assert isinstance(seg, tuple)
            assert len(seg) == 4
            start, end, text, speaker = seg
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert isinstance(text, str)
            assert isinstance(speaker, str)


class TestStreamingDiarizationYield:
    def test_segments_yielded_exactly_once_with_speakers(
        self, mock_config, mock_audio_processor,
        mock_transcription_engine, mock_diarization_engine, mock_output_formatter,
    ):
        """Regression: diarized streaming used to yield every segment twice —
        once raw during streaming, then again speaker-labeled — so consumers
        wrote doubled transcripts."""
        mock_config.transcription_engine = "whisper"
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.audio_processor = mock_audio_processor
        t.transcription_engine = mock_transcription_engine
        t.diarization_engine = mock_diarization_engine
        t.output_formatter = mock_output_formatter
        t.include_diarization = True
        t.test_mode = False

        mock_audio_processor.stream_audio_from_file.return_value = iter([])
        mock_transcription_engine.transcribe_stream.return_value = iter([
            {"start": 0.0, "end": 2.0, "text": "Test segment one"},
            {"start": 2.0, "end": 4.0, "text": "Test segment two"},
        ])

        segments = list(t.transcribe_stream_with_diarization("input.mp4"))

        assert len(segments) == 2
        assert [s["text"] for s in segments] == [
            "Test segment one", "Test segment two",
        ]
        assert [s["speaker"] for s in segments] == ["SPEAKER_01", "SPEAKER_02"]


class TestPerRequestDiarization:
    """transcribe(include_diarization=...) overrides the constructed default,
    so callers (e.g. the model server) don't mutate shared config per request."""

    def test_override_enables_diarization(
        self, transcriber, mock_audio_processor,
        mock_transcription_engine, mock_diarization_engine,
    ):
        transcriber.include_diarization = False
        transcriber.config.include_diarization = False
        mock_audio_processor.get_audio_path.return_value = ("test.wav", False)

        result = transcriber.transcribe("test.mp4", include_diarization=True)

        mock_diarization_engine.diarize.assert_called_once()
        assert [seg[3] for seg in result] == ["SPEAKER_01", "SPEAKER_02"]

    def test_override_disables_diarization(
        self, transcriber, mock_audio_processor, mock_diarization_engine,
    ):
        transcriber.include_diarization = True
        transcriber.config.include_diarization = True
        mock_audio_processor.get_audio_path.return_value = ("test.wav", False)

        result = transcriber.transcribe("test.mp4", include_diarization=False)

        mock_diarization_engine.diarize.assert_not_called()
        assert [seg[3] for seg in result] == ["", ""]

    def test_diarized_streaming_yields_incrementally(
        self, mock_config, mock_audio_processor,
        mock_transcription_engine, mock_diarization_engine, mock_output_formatter,
    ):
        """Diarization completes before streaming starts, so each streamed
        segment can be labeled and yielded immediately — consumers keep their
        live progress and Ctrl+C partial-save behavior."""
        mock_config.transcription_engine = "whisper"
        t = Transcriber.__new__(Transcriber)
        t.config = mock_config
        t.audio_processor = mock_audio_processor
        t.transcription_engine = mock_transcription_engine
        t.diarization_engine = mock_diarization_engine
        t.output_formatter = mock_output_formatter
        t.include_diarization = True
        t.test_mode = False

        produced = []

        def segment_stream():
            for seg in [
                {"start": 0.0, "end": 2.0, "text": "Test segment one"},
                {"start": 2.0, "end": 4.0, "text": "Test segment two"},
            ]:
                produced.append(seg)
                yield seg

        mock_audio_processor.stream_audio_from_file.return_value = iter([])
        mock_transcription_engine.transcribe_stream.return_value = segment_stream()

        gen = t.transcribe_stream_with_diarization("input.mp4")
        first = next(gen)
        # Only one source segment consumed when the first labeled one arrives
        assert len(produced) == 1
        assert first["speaker"] == "SPEAKER_01"
        rest = list(gen)
        assert [s["speaker"] for s in rest] == ["SPEAKER_02"]

"""Click CLI → script call contracts in scripts/transcribe.py.

Regression: `transcribe.py stream ... --words` crashed with
"TypeError: main() takes 0 positional arguments but 1 was given" because the
Click subcommand passed an argv list to a main() that took none (and used
flags the stream script didn't define). `server` had the same bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import scripts.model_server as model_server  # noqa: E402
import scripts.stream_transcribe as stream_transcribe  # noqa: E402
from scripts.transcribe import cli  # noqa: E402


class FakeTranscriber:
    """Stands in for src.transcriber.Transcriber inside stream_transcribe."""

    instances: list["FakeTranscriber"] = []
    error: Exception | None = None

    def __init__(self, config):
        self.config = config
        self.stream_kwargs = None
        type(self).instances.append(self)

    def _segments(self, word_timestamps):
        if type(self).error is not None:
            raise type(self).error
        words = (
            [
                {"start": 0.0, "end": 0.4, "word": " Hello"},
                {"start": 0.4, "end": 0.9, "word": " world."},
            ]
            if word_timestamps
            else []
        )
        yield {"start": 0.0, "end": 0.9, "text": "Hello world.", "words": words}

    def transcribe_stream(self, input_path, word_timestamps=False):
        self.stream_kwargs = {"word_timestamps": word_timestamps}
        yield from self._segments(word_timestamps)

    def transcribe_stream_with_diarization(self, input_path, word_timestamps=False):
        self.stream_kwargs = {"word_timestamps": word_timestamps}
        for segment in self._segments(word_timestamps):
            yield {**segment, "speaker": "SPEAKER_00"}

    def save_transcript(self, segments, output_path):
        from src.output.formatter import OutputFormatter

        formatter = OutputFormatter(self.config)
        formatter.save_transcript(segments, output_path)


@pytest.fixture(autouse=True)
def fake_transcriber(monkeypatch):
    FakeTranscriber.instances = []
    FakeTranscriber.error = None
    monkeypatch.setattr(stream_transcribe, "Transcriber", FakeTranscriber)
    # Keep the test independent of the developer's .env.
    monkeypatch.setenv("INCLUDE_DIARIZATION", "false")
    return FakeTranscriber


def test_stream_words_json_writes_word_timestamps(wav_path, tmp_path):
    out = tmp_path / "out.json"
    result = CliRunner().invoke(
        cli,
        ["stream", str(wav_path), "--words", "-l", "en", "-f", "json", "-o", str(out)],
    )

    assert result.exit_code == 0, result.output
    (transcriber,) = FakeTranscriber.instances
    assert transcriber.stream_kwargs == {"word_timestamps": True}
    # Streaming is Whisper-only; Parakeet is the Apple Silicon default.
    assert transcriber.config.transcription_engine == "whisper"
    assert transcriber.config.language == "en"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["text"] == "Hello world."
    assert data[0]["words"] == [
        {"start": 0.0, "end": 0.4, "word": " Hello"},
        {"start": 0.4, "end": 0.9, "word": " world."},
    ]


def test_stream_without_words_omits_word_list(wav_path, tmp_path):
    out = tmp_path / "out.json"
    result = CliRunner().invoke(cli, ["stream", str(wav_path), "-f", "json", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert FakeTranscriber.instances[0].stream_kwargs == {"word_timestamps": False}
    assert "words" not in json.loads(out.read_text(encoding="utf-8"))[0]


def test_stream_words_ignored_for_non_json_format(wav_path, tmp_path):
    out = tmp_path / "out.srt"
    result = CliRunner().invoke(
        cli, ["stream", str(wav_path), "--words", "-f", "srt", "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert FakeTranscriber.instances[0].stream_kwargs == {"word_timestamps": False}
    assert "Hello world." in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("fmt", ["vtt-voice", "json3", "pretty"])
def test_stream_accepts_every_cli_output_format(wav_path, tmp_path, fmt):
    out = tmp_path / f"out.{fmt}"
    result = CliRunner().invoke(cli, ["stream", str(wav_path), "-f", fmt, "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()


def test_stream_failure_propagates_nonzero_exit(wav_path, tmp_path):
    FakeTranscriber.error = RuntimeError("boom")
    result = CliRunner().invoke(
        cli, ["stream", str(wav_path), "-f", "json", "-o", str(tmp_path / "out.json")]
    )

    # A clean exit with the script's return code, not an uncaught crash.
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)


def test_server_passes_args_to_model_server_main(monkeypatch):
    calls = {}
    monkeypatch.setattr(model_server, "initialize_models", lambda path: calls.update(config=path))
    monkeypatch.setattr(
        model_server, "run_server", lambda host, port: calls.update(host=host, port=port)
    )

    result = CliRunner().invoke(cli, ["server", "--port", "9123", "--config", "custom.env"])

    assert result.exit_code == 0, result.output
    assert calls == {"config": "custom.env", "host": "localhost", "port": 9123}



class _Word:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class _Segment:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


class _RecordingWhisper:
    """Fake faster-whisper model: records its kwargs, honors word_timestamps."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
        type(self).calls.append(kwargs)
        words = (
            [_Word(0.0, 0.4, " Hello"), _Word(0.4, 0.9, " world.")]
            if kwargs.get("word_timestamps") else []
        )
        return iter([_Segment(0.0, 0.9, "Hello world.", words)]), None


def test_stream_words_reach_json_through_the_real_pipeline(wav_path, tmp_path, monkeypatch):
    """--words through the real Transcriber, WhisperEngine and StreamingTranscriber.

    The tests above fake the whole Transcriber, so dropping word_timestamps
    anywhere between the CLI and faster-whisper went unnoticed.
    """
    import src.transcription.engine as engine_module
    from src.transcriber import Transcriber

    _RecordingWhisper.calls = []
    monkeypatch.setattr(stream_transcribe, "Transcriber", Transcriber)
    monkeypatch.setattr(engine_module, "WhisperModel", _RecordingWhisper)
    monkeypatch.setenv("CACHE_ENABLED", "false")
    out = tmp_path / "out.json"

    result = CliRunner().invoke(
        cli, ["stream", str(wav_path), "--words", "-f", "json", "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert _RecordingWhisper.calls
    assert all(call["word_timestamps"] is True for call in _RecordingWhisper.calls)
    (segment,) = json.loads(out.read_text(encoding="utf-8"))
    assert [w["word"] for w in segment["words"]] == [" Hello", " world."]


def test_failure_mid_stream_saves_partial_transcript_but_exits_nonzero(
    wav_path, tmp_path, monkeypatch
):
    def one_segment_then_fail(self, input_path, word_timestamps=False):
        yield {"start": 0.0, "end": 0.9, "text": "Hello world."}
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr(FakeTranscriber, "transcribe_stream", one_segment_then_fail)
    out = tmp_path / "out.txt"

    result = CliRunner().invoke(cli, ["stream", str(wav_path), "-f", "txt", "-o", str(out)])

    assert result.exit_code == 1
    assert "Hello world." in out.read_text(encoding="utf-8")


def test_stream_without_diarization_has_no_speaker_labels(wav_path, tmp_path):
    out = tmp_path / "out.txt"

    result = CliRunner().invoke(cli, ["stream", str(wav_path), "-f", "txt", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "SPEAKER" not in out.read_text(encoding="utf-8")

"""`scripts/transcribe.py stream` and `batch`: the argv they hand the wrapped
scripts, and propagation of those scripts' exit status."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_CLI_PATH = _REPO_ROOT / "scripts" / "transcribe.py"
spec = importlib.util.spec_from_file_location("transcribe_cli_module", _CLI_PATH)
transcribe_cli = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(transcribe_cli)

# The module the `stream` command imports, so patches here reach it
import scripts.stream_transcribe as stream_transcribe


class FakeTranscriber:
    """Stands in for src.transcriber.Transcriber without loading any models.

    Like the real one, it can only stream when the config selects Whisper.
    """

    def __init__(self, config):
        self.config = config

    def transcribe_stream(self, input_path):
        if self.config.transcription_engine != "whisper":
            raise NotImplementedError(
                "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper."
            )
        yield {"start": 0.0, "end": 1.5, "text": "hello world"}

    def save_transcript(self, segments, output_path):
        Path(output_path).write_text(" ".join(text for _, _, text, _ in segments))


class FailingTranscriber(FakeTranscriber):
    def transcribe_stream(self, input_path):
        raise RuntimeError("decoder exploded")


@pytest.fixture
def audio_file(tmp_path):
    f = tmp_path / "clip.wav"
    f.write_bytes(b"\x00")
    return str(f)


@pytest.fixture
def whisper_engine(monkeypatch):
    """Pin the engine so tests that aren't about engine choice don't depend on the platform."""
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "whisper")


def _run(*args):
    # catch_exceptions=False: a crash (e.g. a TypeError) should fail the test
    # with its traceback, not pass as exit_code 1.
    return CliRunner().invoke(transcribe_cli.cli, list(args), catch_exceptions=False)


def test_batch_exits_nonzero_when_pattern_matches_no_files(tmp_path):
    result = _run("batch", str(tmp_path / "*.wav"))

    assert result.exit_code == 1


def test_stream_writes_transcript_to_output_path(
    monkeypatch, whisper_engine, audio_file, tmp_path
):
    monkeypatch.setattr(stream_transcribe, "Transcriber", FakeTranscriber)
    output = tmp_path / "out.txt"

    result = _run("stream", audio_file, "--output", str(output))

    assert result.exit_code == 0, result.output
    assert output.read_text() == "hello world"


def test_stream_exits_nonzero_when_transcription_fails(
    monkeypatch, whisper_engine, audio_file
):
    monkeypatch.setattr(stream_transcribe, "Transcriber", FailingTranscriber)

    result = _run("stream", audio_file)

    assert result.exit_code == 1


def test_stream_uses_whisper_when_parakeet_is_only_the_platform_default(
    monkeypatch, audio_file, tmp_path
):
    """Parakeet is the Apple Silicon default but can't stream, so `stream`
    shouldn't fail out of the box there."""
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(stream_transcribe, "Transcriber", FakeTranscriber)
    output = tmp_path / "out.txt"

    result = _run("stream", audio_file, "--output", str(output))

    assert result.exit_code == 0, result.output
    assert output.read_text() == "hello world"


def test_stream_keeps_an_explicitly_chosen_engine(monkeypatch, audio_file):
    """An explicit TRANSCRIPTION_ENGINE=parakeet isn't silently swapped for
    Whisper; the transcriber refuses to stream and the command fails."""
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "parakeet")
    monkeypatch.setattr(stream_transcribe, "Transcriber", FakeTranscriber)

    result = _run("stream", audio_file)

    assert result.exit_code == 1

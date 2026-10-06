"""`transcribe.py batch` end to end through the Click CLI, with a fake transcriber.

Regressions:
- `batch` always exited 0: the Click wrapper dropped batch_transcribe.main()'s
  return code, and main() returned 0 even when files failed.
- `batch -f vtt-voice|json3|pretty` died with an argparse error: Click offered
  seven formats, the batch script accepted four.
- `batch --streaming` failed every file on Apple Silicon (streaming is
  Whisper-only, Parakeet is the platform default) and still exited 0.
- Streaming batch output labelled every line "SPEAKER" without diarization.
"""

from __future__ import annotations

import platform
import sys
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import scripts.batch_transcribe as batch_transcribe  # noqa: E402
from scripts.transcribe import cli  # noqa: E402
from src.output.formatter import OutputFormatter  # noqa: E402


class FakeTranscriber:
    """Mirrors the parts of src.transcriber.Transcriber that batch uses."""

    configs: list = []

    def __init__(self, config):
        self.config = config
        type(self).configs.append(config)

    def transcribe(self, input_path):
        if "bad" in Path(input_path).name:
            raise RuntimeError("decode failed")
        return [(0.0, 1.0, "hello", "")]

    def transcribe_stream_with_diarization(self, input_path, word_timestamps=False):
        if self.config.transcription_engine != "whisper":
            raise NotImplementedError("Streaming is only supported with whisper")
        yield {"start": 0.0, "end": 1.0, "text": "hello"}

    def save_transcript(self, segments, output_path):
        OutputFormatter(self.config).save_transcript(segments, output_path)


@pytest.fixture(autouse=True)
def fake_transcriber(monkeypatch):
    FakeTranscriber.configs = []
    monkeypatch.setattr(batch_transcribe, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(batch_transcribe, "_worker_state", threading.local())


@pytest.fixture
def apple_silicon(monkeypatch):
    """Make Parakeet the platform default, as on the Macs this runs on."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)


def _media(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"\x00")
    return str(tmp_path / "*.wav")


def _batch(*args):
    return CliRunner().invoke(cli, ["batch", *args, "-w", "1"])


def test_exits_nonzero_when_no_files_match(tmp_path):
    result = _batch(str(tmp_path / "*.wav"), "-o", str(tmp_path / "out"))

    assert result.exit_code == 1


def test_exits_nonzero_when_any_file_fails(tmp_path):
    pattern = _media(tmp_path, "good.wav", "bad.wav")

    result = _batch(pattern, "-o", str(tmp_path / "out"))

    assert result.exit_code == 1
    # The good file is still transcribed.
    assert (tmp_path / "out" / "good.txt").exists()


def test_exits_zero_and_writes_every_output_when_all_succeed(tmp_path):
    pattern = _media(tmp_path, "a.wav", "b.wav")

    result = _batch(pattern, "-o", str(tmp_path / "out"), "-f", "txt")

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.txt", "b.txt"]


@pytest.mark.parametrize("fmt", ["vtt-voice", "json3", "pretty"])
def test_accepts_every_cli_output_format(tmp_path, fmt):
    pattern = _media(tmp_path, "a.wav")

    result = _batch(pattern, "-o", str(tmp_path / "out"), "-f", fmt)

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / f"a.{fmt}").exists()


def test_streaming_uses_whisper_when_parakeet_is_the_platform_default(tmp_path, apple_silicon):
    pattern = _media(tmp_path, "a.wav", "b.wav")

    result = _batch(pattern, "-o", str(tmp_path / "out"), "--streaming")

    assert result.exit_code == 0, result.output
    assert FakeTranscriber.configs[0].transcription_engine == "whisper"
    assert len(list((tmp_path / "out").iterdir())) == 2


def test_streaming_without_diarization_has_no_speaker_labels(tmp_path):
    pattern = _media(tmp_path, "a.wav")

    result = _batch(pattern, "-o", str(tmp_path / "out"), "--streaming", "-f", "txt")

    assert result.exit_code == 0, result.output
    text = (tmp_path / "out" / "a.txt").read_text(encoding="utf-8")
    assert "hello" in text
    assert "SPEAKER" not in text

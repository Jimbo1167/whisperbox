"""--model / --engine on the Click CLI.

Regressions: --model only accepted tiny/base/small/medium/large, so the
documented default (large-v3-turbo) couldn't be chosen explicitly, and there
was no way to pick the engine per run.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import scripts.batch_transcribe as batch_transcribe  # noqa: E402
import scripts.transcribe as transcribe_cli  # noqa: E402


class FakeService:
    configs: list = []
    # What the built engine reports; None mirrors the config.
    built_engine = None

    def __init__(self, config, **kwargs):
        self.config = config
        type(self).configs.append(config)

    @property
    def engine_name(self):
        return type(self).built_engine or self.config.transcription_engine

    @property
    def model_name(self):
        if self.engine_name == "parakeet":
            return self.config.parakeet_model
        return self.config.whisper_model_size

    def transcribe_file(self, input_path, output_path=None, progress_callback=None):
        Path(output_path).write_text("hello", encoding="utf-8")
        return {"segments": [], "output_file": output_path}


class FakeTranscriber:
    configs: list = []

    def __init__(self, config):
        type(self).configs.append(config)

    def transcribe(self, input_path):
        return [(0.0, 1.0, "hi", "")]

    def save_transcript(self, segments, output_path):
        Path(output_path).write_text("hi", encoding="utf-8")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeService.configs = []
    FakeService.built_engine = None
    FakeTranscriber.configs = []
    monkeypatch.setattr(transcribe_cli, "TranscriptionService", FakeService)
    monkeypatch.setattr(batch_transcribe, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(batch_transcribe, "_worker_state", threading.local())
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)


def _transcribe(wav_path, tmp_path, *args):
    return CliRunner().invoke(
        transcribe_cli.cli,
        ["transcribe", str(wav_path), "-o", str(tmp_path / "out.txt"), "--progress", "none", *args],
    )


@pytest.mark.parametrize("model", ["large-v3-turbo", "large-v3", "distil-large-v3"])
def test_transcribe_accepts_any_whisper_model_name(wav_path, tmp_path, model):
    result = _transcribe(wav_path, tmp_path, "--model", model)

    assert result.exit_code == 0, result.output
    assert FakeService.configs[0].whisper_model_size == model


@pytest.mark.parametrize("engine", ["whisper", "parakeet"])
def test_transcribe_engine_flag(wav_path, tmp_path, engine):
    result = _transcribe(wav_path, tmp_path, "--engine", engine)

    assert result.exit_code == 0, result.output
    assert FakeService.configs[0].transcription_engine == engine


def test_transcribe_rejects_unknown_engine(wav_path, tmp_path):
    result = _transcribe(wav_path, tmp_path, "--engine", "vosk")

    assert result.exit_code == 2


def test_batch_engine_and_model_flags(wav_path, tmp_path):
    result = CliRunner().invoke(
        transcribe_cli.cli,
        ["batch", str(wav_path), "-o", str(tmp_path / "out"), "-w", "1",
         "--engine", "whisper", "--model", "large-v3-turbo"],
    )

    assert result.exit_code == 0, result.output
    config = FakeTranscriber.configs[0]
    assert config.transcription_engine == "whisper"
    assert config.whisper_model_size == "large-v3-turbo"


def test_jsonl_started_event_names_the_engine_and_its_model(wav_path, tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
    result = CliRunner().invoke(
        transcribe_cli.cli,
        ["transcribe", str(wav_path), "-o", str(tmp_path / "out.txt"),
         "--progress", "jsonl", "--engine", "parakeet"],
    )

    assert result.exit_code == 0, result.output
    started = next(
        json.loads(line) for line in result.output.splitlines()
        if line.startswith("{") and '"started"' in line
    )
    assert started["engine"] == "parakeet"
    assert started["model"] == "mlx-community/parakeet-tdt-0.6b-v3"


def test_jsonl_started_event_names_the_engine_that_was_built(wav_path, tmp_path):
    """A defaulted Parakeet falls back to Whisper without parakeet-mlx; the
    event must say what runs, not what was configured."""
    import json

    FakeService.built_engine = "whisper"
    result = CliRunner().invoke(
        transcribe_cli.cli,
        ["transcribe", str(wav_path), "-o", str(tmp_path / "out.txt"),
         "--progress", "jsonl", "--engine", "parakeet"],
    )

    assert result.exit_code == 0, result.output
    started = next(
        json.loads(line) for line in result.output.splitlines()
        if line.startswith("{") and '"started"' in line
    )
    assert started["engine"] == "whisper"
    assert started["model"] == FakeService.configs[0].whisper_model_size

"""How the project's .env file reaches the configuration.

Regressions:
- The main CLI (`transcribe`, `stream`, `batch`) and `benchmark.py` never
  loaded .env, so settings documented as living there (HF_TOKEN, WHISPER_MODEL,
  ...) were silently ignored.
- Where .env *was* loaded it overrode the shell, so an inline
  `INCLUDE_DIARIZATION=true` (what `make diarize` does) lost to the file.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import src.config as config_module  # noqa: E402
from src.config import Config, load_env_file  # noqa: E402


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    """Point the project .env at a temp file and return a writer for it."""
    env_path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "DEFAULT_ENV_FILE", env_path)

    def write(text: str) -> Path:
        env_path.write_text(text, encoding="utf-8")
        return env_path

    return write


@pytest.fixture
def wav_path(tmp_path):
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
    return path


class TestLoadEnvFile:
    def test_shell_variables_win_over_the_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("INCLUDE_DIARIZATION=false\n", encoding="utf-8")

        with patch.dict(os.environ, {"INCLUDE_DIARIZATION": "true"}, clear=True):
            config = Config(str(env_file))

        assert config.include_diarization is True

    def test_file_fills_variables_the_shell_leaves_unset(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("WHISPER_MODEL=tiny\n", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            config = Config(str(env_file))

        assert config.whisper_model_size == "tiny"

    def test_defaults_to_the_project_env_file(self, project_env):
        path = project_env("WHISPER_MODEL=small\n")

        with patch.dict(os.environ, {}, clear=True):
            assert load_env_file() == path
            assert os.environ["WHISPER_MODEL"] == "small"

    def test_missing_file_is_not_an_error(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True):
            assert load_env_file(tmp_path / "absent.env") is None

    def test_project_env_file_lives_at_the_repo_root(self):
        # DEFAULT_ENV_FILE itself is redirected for every test (conftest), so
        # check the root it is derived from.
        assert config_module.PROJECT_ROOT == _REPO_ROOT


class _FakeService:
    configs: list = []

    def __init__(self, config, **kwargs):
        type(self).configs.append(config)

    def transcribe_file(self, input_path, output_path=None, progress_callback=None):
        Path(output_path).write_text("hello", encoding="utf-8")
        return {"segments": [], "output_file": output_path}


def test_transcribe_command_reads_project_env(project_env, wav_path, tmp_path, monkeypatch):
    import scripts.transcribe as transcribe_cli

    project_env("WHISPER_MODEL=tiny\nLANGUAGE=fr\n")
    _FakeService.configs = []
    monkeypatch.setattr(transcribe_cli, "TranscriptionService", _FakeService)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("LANGUAGE", raising=False)

    result = CliRunner().invoke(
        transcribe_cli.cli,
        ["transcribe", str(wav_path), "-o", str(tmp_path / "out.txt"), "--progress", "none"],
    )

    assert result.exit_code == 0, result.output
    (config,) = _FakeService.configs
    assert config.whisper_model_size == "tiny"
    assert config.language == "fr"


def test_stream_reads_project_env(project_env, wav_path, tmp_path, monkeypatch):
    import scripts.stream_transcribe as stream_transcribe

    seen = []

    class FakeTranscriber:
        def __init__(self, config):
            seen.append(config)

        def transcribe_stream(self, input_path, word_timestamps=False):
            yield {"start": 0.0, "end": 1.0, "text": "hi"}

        def save_transcript(self, segments, output_path):
            Path(output_path).write_text("hi", encoding="utf-8")

    project_env("LANGUAGE=de\nINCLUDE_DIARIZATION=false\n")
    monkeypatch.setattr(stream_transcribe, "Transcriber", FakeTranscriber)
    monkeypatch.delenv("LANGUAGE", raising=False)
    monkeypatch.delenv("INCLUDE_DIARIZATION", raising=False)

    assert stream_transcribe.main([str(wav_path), "-o", str(tmp_path / "out.txt")]) == 0
    assert seen[0].language == "de"


def test_batch_reads_project_env(project_env, wav_path, tmp_path, monkeypatch):
    import scripts.batch_transcribe as batch_transcribe
    import threading

    seen = []

    class FakeTranscriber:
        def __init__(self, config):
            seen.append(config)

        def transcribe(self, input_path):
            return [(0.0, 1.0, "hi", "")]

        def save_transcript(self, segments, output_path):
            Path(output_path).write_text("hi", encoding="utf-8")

    project_env("LANGUAGE=es\n")
    monkeypatch.setattr(batch_transcribe, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(batch_transcribe, "_worker_state", threading.local())
    monkeypatch.delenv("LANGUAGE", raising=False)

    batch_transcribe.main([str(wav_path), "-o", str(tmp_path / "out"), "-w", "1"])
    assert seen[0].language == "es"


def test_benchmark_reads_project_env(project_env, monkeypatch):
    import scripts.benchmark as benchmark

    seen = []
    monkeypatch.setattr(benchmark, "TranscriptionService", lambda config: seen.append(config))
    project_env("WHISPER_MODEL=base\n")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    benchmark.build_service(engine=None, model=None)
    assert seen[0].whisper_model_size == "base"


def test_model_server_reads_project_env_by_default(project_env, monkeypatch):
    import scripts.model_server as model_server

    seen = []
    monkeypatch.setattr(
        model_server, "TranscriptionService",
        lambda config, preload_models: seen.append(config),
    )
    project_env("WHISPER_MODEL=medium\n")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    model_server.initialize_models(None)
    assert seen[0].whisper_model_size == "medium"


# INCLUDE_DIARIZATION in .env must not switch diarization on for the Click
# commands: like `client`, they diarize only when asked with --diarize. (Before
# .env was loaded at all this held by accident; the transcribe-locally skill
# relies on diarization being opt-in.)


@pytest.mark.parametrize("flag, expected", [([], False), (["--diarize"], True)])
def test_transcribe_diarizes_only_with_flag(project_env, wav_path, tmp_path, monkeypatch,
                                            flag, expected):
    import scripts.transcribe as transcribe_cli

    project_env("INCLUDE_DIARIZATION=true\n")
    _FakeService.configs = []
    monkeypatch.setattr(transcribe_cli, "TranscriptionService", _FakeService)
    monkeypatch.delenv("INCLUDE_DIARIZATION", raising=False)

    result = CliRunner().invoke(
        transcribe_cli.cli,
        ["transcribe", str(wav_path), "-o", str(tmp_path / "out.txt"), "--progress", "none", *flag],
    )

    assert result.exit_code == 0, result.output
    assert _FakeService.configs[0].include_diarization is expected


@pytest.mark.parametrize("flag, expected", [([], False), (["--diarize"], True)])
def test_stream_diarizes_only_with_flag(project_env, wav_path, tmp_path, monkeypatch,
                                        flag, expected):
    import scripts.stream_transcribe as stream_transcribe

    seen = []

    class FakeTranscriber:
        def __init__(self, config):
            seen.append(config)

        def transcribe_stream(self, input_path, word_timestamps=False):
            yield {"start": 0.0, "end": 1.0, "text": "hi"}

        transcribe_stream_with_diarization = transcribe_stream

        def save_transcript(self, segments, output_path):
            Path(output_path).write_text("hi", encoding="utf-8")

    project_env("INCLUDE_DIARIZATION=true\n")
    monkeypatch.setattr(stream_transcribe, "Transcriber", FakeTranscriber)
    monkeypatch.delenv("INCLUDE_DIARIZATION", raising=False)

    assert stream_transcribe.main([str(wav_path), "-o", str(tmp_path / "out.txt"), *flag]) == 0
    assert seen[0].include_diarization is expected


@pytest.mark.parametrize("flag, expected", [([], False), (["--diarize"], True)])
def test_batch_diarizes_only_with_flag(project_env, wav_path, tmp_path, monkeypatch,
                                       flag, expected):
    import scripts.batch_transcribe as batch_transcribe
    import threading

    seen = []

    class FakeTranscriber:
        def __init__(self, config):
            seen.append(config)

        def transcribe(self, input_path):
            return [(0.0, 1.0, "hi", "")]

        def save_transcript(self, segments, output_path):
            Path(output_path).write_text("hi", encoding="utf-8")

    project_env("INCLUDE_DIARIZATION=true\n")
    monkeypatch.setattr(batch_transcribe, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(batch_transcribe, "_worker_state", threading.local())
    monkeypatch.delenv("INCLUDE_DIARIZATION", raising=False)

    batch_transcribe.main([str(wav_path), "-o", str(tmp_path / "out"), "-w", "1", *flag])
    assert seen[0].include_diarization is expected


def test_loading_is_quiet_at_info_level(tmp_path, caplog):
    """`transcribe --progress jsonl` keeps stderr parseable; load_env_file runs
    before it lowers the log level, so it must not log at INFO."""
    import logging

    env_file = tmp_path / ".env"
    env_file.write_text("WHISPER_MODEL=tiny\n", encoding="utf-8")
    caplog.set_level(logging.INFO, logger="src.config")

    with patch.dict(os.environ, {}, clear=True):
        load_env_file(env_file)

    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []

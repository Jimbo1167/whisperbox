"""scripts/transcribe_video.py, the legacy single-file entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import scripts.transcribe_video as transcribe_video  # noqa: E402


@pytest.mark.parametrize("fmt", ["vtt-voice", "json3"])
def test_accepts_every_supported_output_format(monkeypatch, tmp_path, fmt):
    calls = []
    monkeypatch.setattr(
        transcribe_video, "transcribe_with_server_fallback",
        lambda input_path, config, output: calls.append((config, output)),
    )
    monkeypatch.setenv("TRANSCRIPTION_ENGINE", "whisper")
    monkeypatch.setattr(
        sys, "argv", ["transcribe_video.py", "in.wav", "-f", fmt, "-o", str(tmp_path / "out")]
    )

    transcribe_video.main()

    ((config, _),) = calls
    assert config.output_format == fmt


def test_model_flag_selects_whisper_like_the_other_commands(monkeypatch, tmp_path):
    import platform

    calls = []
    monkeypatch.setattr(
        transcribe_video, "transcribe_with_server_fallback",
        lambda input_path, config, output: calls.append(config),
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TRANSCRIPTION_ENGINE", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["transcribe_video.py", "in.wav", "-m", "small", "-o", str(tmp_path / "out")],
    )

    transcribe_video.main()

    (config,) = calls
    assert config.transcription_engine == "whisper"
    assert config.whisper_model_size == "small"

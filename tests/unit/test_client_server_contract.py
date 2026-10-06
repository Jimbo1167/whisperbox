"""The real model server against its real clients, over HTTP.

Each side used to be tested only against a hand-written fake of the other,
and they drifted apart:
- `client status` read `models`/`resources`/`queue`; the server sends
  `model`/`stats`, so it printed nothing but the uptime.
- `client -f srt -o x.srt` wrote the client's console rendering, not SRT.
- `client --model/--language` were sent and silently ignored by the server.
- Warm-server routing compared model and language but not the engine, so a
  Parakeet server served a Whisper request.

Only the transcription service is faked; the HTTP server, request handler,
model client and routing code are the production ones.
"""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import scripts.model_client as model_client  # noqa: E402
import scripts.model_server as model_server  # noqa: E402
from src.config import OUTPUT_FORMATS, Config  # noqa: E402
from src.output.formatter import OutputFormatter  # noqa: E402
from src.server_client import try_server_transcribe  # noqa: E402

SEGMENTS = [(0.0, 1.5, "hello world", ""), (1.5, 3.0, "second line", "")]


def _formatted(fmt: str) -> str:
    return OutputFormatter(Config(output_format=fmt)).format_transcript(SEGMENTS)


class FakeService:
    """Stands in for TranscriptionService behind the real request handler."""

    def __init__(self, tmp_path, engine_name="whisper", model_name="large-v3-turbo"):
        self.tmp_path = tmp_path
        self.engine_name = engine_name
        self.model_name = model_name
        self.diarization_error = None
        self.calls = []

    def transcribe_file(self, input_path, output_format=None,
                        progress_callback=None, include_diarization=None):
        self.calls.append({"include_diarization": include_diarization})
        text = _formatted(output_format)
        out = self.tmp_path / f"result.{output_format}"
        out.write_text(text, encoding="utf-8")
        return {
            "segments": SEGMENTS,
            "preview_text": text,
            "output_format": output_format,
            "output_file": str(out),
            "processing_time": 0.01,
            "diarization_error": self.diarization_error,
        }

    def transcribe_existing_audio(self, audio_path, include_diarization=None):
        self.calls.append({"include_diarization": include_diarization})
        return {
            "segments": SEGMENTS,
            "processing_time": 0.01,
            "diarization_error": self.diarization_error,
        }


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def start_server(tmp_path, monkeypatch):
    """Start the real server; returns (url, fake_service)."""
    servers = []
    monkeypatch.setattr(model_client, "POLL_INTERVAL", 0.01)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(model_server, "TRANSCRIPTS_ROOT", transcripts)

    def start(engine="whisper", model="large-v3-turbo", language="en"):
        service = FakeService(tmp_path, engine_name=engine, model_name=model)
        config = Config(output_format="txt", language=language, transcription_engine=engine)
        monkeypatch.setattr(model_server, "config", config)
        monkeypatch.setattr(model_server, "service", service)
        httpd = model_server.ThreadedHTTPServer(
            ("127.0.0.1", _free_port()), model_server.ModelRequestHandler
        )
        threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}", service

    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


# --- model_client (multipart upload + job polling) --------------------------


def test_status_shows_the_engine_model_and_request_stats(start_server, capsys):
    url, _ = start_server(engine="parakeet", model="mlx-community/parakeet-tdt-0.6b-v3")

    assert model_client.main(["--server", url, "status"]) == 0

    out = capsys.readouterr().out
    assert "Engine: parakeet" in out
    assert "Model: mlx-community/parakeet-tdt-0.6b-v3" in out
    # FORCE_CPU is Whisper-only; Parakeet always runs on MLX.
    assert "Device: MLX" in out
    assert "Requests: 0" in out


@pytest.mark.parametrize("fmt", OUTPUT_FORMATS)
def test_output_file_holds_the_transcript_in_the_requested_format(
    start_server, wav_path, tmp_path, fmt
):
    url, _ = start_server()
    out = tmp_path / f"out.{fmt}"

    code = model_client.main(
        ["--server", url, "transcribe", str(wav_path), "-f", fmt, "-o", str(out)]
    )

    assert code == 0
    assert out.read_text(encoding="utf-8") == _formatted(fmt)


def test_model_the_server_isnt_running_is_refused(start_server, wav_path, caplog):
    url, service = start_server(model="large-v3-turbo")

    code = model_client.main(["--server", url, "transcribe", str(wav_path), "--model", "tiny"])

    assert code == 1
    assert service.calls == []
    assert "large-v3-turbo" in caplog.text  # the server says what it runs


def test_model_the_server_is_running_is_accepted(start_server, wav_path):
    url, service = start_server(model="large-v3-turbo")

    code = model_client.main(
        ["--server", url, "transcribe", str(wav_path), "--model", "large-v3-turbo"]
    )

    assert code == 0
    assert len(service.calls) == 1


def test_language_the_whisper_server_isnt_using_is_refused(start_server, wav_path, caplog):
    url, service = start_server(language="en")

    code = model_client.main(["--server", url, "transcribe", str(wav_path), "-l", "fr"])

    assert code == 1
    assert service.calls == []


def test_unknown_format_is_rejected(start_server, wav_path):
    url, service = start_server()

    response = requests.post(
        f"{url}/transcribe",
        files={"file": ("clip.wav", wav_path.read_bytes())},
        data={"format": "docx"},
    )

    assert response.status_code == 400
    assert service.calls == []


# --- JSON endpoint + warm-server routing (src/server_client.py) ------------


def _client_config(**overrides):
    config = Config(**overrides)
    config.include_diarization = False
    return config


def test_routing_round_trips_segments_through_the_server(start_server, wav_path, tmp_path):
    url, service = start_server()
    out = tmp_path / "out.srt"
    config = _client_config(transcription_engine="whisper", whisper_model="large-v3-turbo",
                            language="en", output_format="srt")

    result = try_server_transcribe(str(wav_path), config, str(out), server_url=url)

    assert result is not None
    assert result["segments"] == SEGMENTS
    assert out.read_text(encoding="utf-8") == _formatted("srt")
    assert service.calls == [{"include_diarization": False}]


def test_routing_falls_back_when_the_server_runs_another_engine(start_server, wav_path, tmp_path):
    url, service = start_server(engine="parakeet", model="mlx-community/parakeet-tdt-0.6b-v3")
    config = _client_config(transcription_engine="whisper", whisper_model="large-v3-turbo")

    assert try_server_transcribe(str(wav_path), config, str(tmp_path / "o.txt"), server_url=url) is None
    assert service.calls == []


def test_routing_falls_back_when_the_server_runs_another_model(start_server, wav_path, tmp_path):
    url, service = start_server(engine="whisper", model="large-v3-turbo")
    config = _client_config(transcription_engine="whisper", whisper_model="small")

    assert try_server_transcribe(str(wav_path), config, str(tmp_path / "o.txt"), server_url=url) is None
    assert service.calls == []


def test_routing_to_parakeet_ignores_language(start_server, wav_path, tmp_path):
    # Parakeet detects the language itself, so a language mismatch is no
    # reason to give up the warm server.
    model = "mlx-community/parakeet-tdt-0.6b-v3"
    url, service = start_server(engine="parakeet", model=model, language="en")
    config = _client_config(transcription_engine="parakeet", parakeet_model=model)
    config.language = "fr"

    result = try_server_transcribe(str(wav_path), config, str(tmp_path / "o.txt"), server_url=url)

    assert result is not None
    assert len(service.calls) == 1


def test_json_endpoint_honors_diarization_and_reports_its_failure(start_server, wav_path):
    url, service = start_server()
    service.diarization_error = "pipeline unavailable"

    response = requests.post(
        f"{url}/transcribe",
        json={"audio_path": str(wav_path), "include_diarization": True},
    )

    assert response.status_code == 200
    assert service.calls == [{"include_diarization": True}]
    assert response.json()["diarization_error"] == "pipeline unavailable"

"""CLI-side warm-model-server detection (src/server_client.py)."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.config import Config
from src.server_client import try_server_transcribe


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeModelServer(BaseHTTPRequestHandler):
    segments = [[0.0, 1.5, "hello world", ""]]

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        assert "audio_path" in request
        body = json.dumps(
            {"segments": self.segments, "processing_time": 0.1}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), FakeModelServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def config():
    cfg = Config()
    cfg.include_diarization = False
    return cfg


@pytest.fixture
def input_file(tmp_path):
    f = tmp_path / "input.wav"
    f.write_bytes(b"\x00")
    return str(f)


def test_returns_none_when_no_server(config, input_file, tmp_path):
    url = f"http://127.0.0.1:{_free_port()}"  # nothing listening
    result = try_server_transcribe(
        input_file, config, str(tmp_path / "out.txt"), server_url=url
    )
    assert result is None


def test_uses_server_and_writes_transcript(config, input_file, tmp_path, fake_server):
    output_path = tmp_path / "out.txt"
    result = try_server_transcribe(
        input_file, config, str(output_path), server_url=fake_server
    )
    assert result is not None
    assert result["segments"] == [(0.0, 1.5, "hello world", "")]
    assert output_path.exists()
    assert "hello world" in output_path.read_text()


def test_skips_server_when_diarization_requested(
    config, input_file, tmp_path, fake_server
):
    config.include_diarization = True
    result = try_server_transcribe(
        input_file, config, str(tmp_path / "out.txt"), server_url=fake_server
    )
    assert result is None


def test_skips_server_when_disabled_by_env(
    config, input_file, tmp_path, fake_server, monkeypatch
):
    monkeypatch.setenv("WHISPERBOX_NO_SERVER", "1")
    result = try_server_transcribe(
        input_file, config, str(tmp_path / "out.txt"), server_url=fake_server
    )
    assert result is None

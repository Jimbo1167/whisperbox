"""Server-side per-request diarization handling (scripts/model_server.py).

Regression tests for docs/bugs/2026-08-30-diarization-audiodecoder.md (Issue
B): a server started with INCLUDE_DIARIZATION=true must honor a request's
explicit diarize field instead of applying its env default, and a job whose
diarization post-step failed must still complete with the transcription.
"""

import importlib.util
import io
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_SERVER_PATH = _REPO_ROOT / "scripts" / "model_server.py"
spec = importlib.util.spec_from_file_location("model_server_module", _SERVER_PATH)
model_server = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model_server)

from src.config import Config


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeService:
    """Stands in for TranscriptionService; records per-request diarize flags."""

    def __init__(self, tmp_path, diarization_error=None, fail=False):
        self.tmp_path = tmp_path
        self.diarization_error = diarization_error
        self.fail = fail
        self.calls = []
        self.input_paths = []

    def transcribe_file(self, input_path, output_format=None,
                        progress_callback=None, include_diarization=None):
        self.calls.append({"include_diarization": include_diarization})
        self.input_paths.append(input_path)
        if self.fail:
            raise Exception("simulated pipeline failure")
        out = self.tmp_path / f"result-{uuid.uuid4().hex}.txt"
        out.write_text("hello world", encoding="utf-8")
        return {
            "segments": [(0.0, 1.0, "hello world", "")],
            "preview_text": "hello world",
            "output_format": output_format,
            "output_file": str(out),
            "processing_time": 0.01,
            "diarization_error": self.diarization_error,
        }

    def transcribe_existing_audio(self, audio_path, include_diarization=None):
        self.calls.append({"include_diarization": include_diarization})
        return {
            "segments": [(0.0, 1.0, "hello world", "")],
            "processing_time": 0.01,
            "diarization_error": self.diarization_error,
        }


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Run the real model server with a fake service and diarization-on config."""
    config = Config(include_diarization=True, output_format="txt")
    fake_service = FakeService(tmp_path)

    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    monkeypatch.setattr(model_server, "config", config)
    monkeypatch.setattr(model_server, "service", fake_service)
    monkeypatch.setattr(model_server, "TRANSCRIPTS_ROOT", transcripts_dir)

    port = _free_port()
    httpd = model_server.ThreadedHTTPServer(
        ("127.0.0.1", port), model_server.ModelRequestHandler
    )
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{port}", fake_service
    httpd.shutdown()
    httpd.server_close()


def _submit_and_wait(url, data, deadline=10.0):
    response = requests.post(
        f"{url}/transcribe",
        files={"file": ("audio.wav", io.BytesIO(b"\x00\x01"))},
        data=data,
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    end = time.time() + deadline
    while time.time() < end:
        job = requests.get(f"{url}/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    pytest.fail("job did not finish in time")


def test_explicit_diarize_false_overrides_env_default(server):
    """diarize=false in the request wins over INCLUDE_DIARIZATION=true."""
    url, fake_service = server
    job = _submit_and_wait(url, {"diarize": "false"})

    assert job["status"] == "completed"
    assert fake_service.calls == [{"include_diarization": False}]


def test_explicit_diarize_true_is_forwarded(server):
    url, fake_service = server
    job = _submit_and_wait(url, {"diarize": "true"})

    assert job["status"] == "completed"
    assert fake_service.calls == [{"include_diarization": True}]


def test_missing_diarize_field_uses_server_default(server):
    """Only a request that doesn't specify falls back to the env default."""
    url, fake_service = server
    job = _submit_and_wait(url, {})

    assert job["status"] == "completed"
    assert fake_service.calls == [{"include_diarization": True}]


def test_job_completes_with_diarization_error(server):
    """A failed diarization post-step surfaces as a warning, not a failed job."""
    url, fake_service = server
    fake_service.diarization_error = "name 'AudioDecoder' is not defined"

    job = _submit_and_wait(url, {"diarize": "true"})

    assert job["status"] == "completed"
    result = job["result"]
    assert result["segments"] == [[0.0, 1.0, "hello world", ""]]
    assert "AudioDecoder" in result["diarization_error"]


def test_service_exception_still_fails_job(server):
    url, fake_service = server
    fake_service.fail = True

    job = _submit_and_wait(url, {"diarize": "false"})

    assert job["status"] == "failed"
    assert "simulated pipeline failure" in job["error"]


# --- hardening ---------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/transcribe", "/api/transcribe-sync"])
def test_oversized_upload_is_refused(server, monkeypatch, endpoint):
    url, fake_service = server
    monkeypatch.setattr(model_server, "MAX_UPLOAD_SIZE", 10)

    response = requests.post(
        # Big enough that the client is still sending when the server answers;
        # closing with the body unread used to reset the connection.
        f"{url}{endpoint}", files={"file": ("audio.wav", io.BytesIO(b"\x00" * (2 << 20)))}
    )

    assert response.status_code == 413
    assert fake_service.calls == []


def test_transcript_download_cannot_escape_the_transcripts_dir(server, tmp_path):
    import http.client

    url, _ = server
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    host, port = url.removeprefix("http://").split(":")

    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    conn.request("GET", "/transcripts/../secret.txt")  # sent verbatim
    response = conn.getresponse()
    body = response.read()
    conn.close()

    assert response.status == 404
    assert b"do not serve" not in body


def test_uploaded_temp_file_is_removed_after_the_job(server):
    url, fake_service = server
    _submit_and_wait(url, {"diarize": "false"})

    (temp_path,) = fake_service.input_paths
    end = time.time() + 5
    while Path(temp_path).exists() and time.time() < end:
        time.sleep(0.02)
    assert not Path(temp_path).exists()


@pytest.mark.parametrize("data, expected", [
    ({"diarize": "false"}, False),
    ({"diarize": "true"}, True),
    ({}, None),  # absent: the service applies the server default
])
def test_sync_endpoint_forwards_the_diarize_field(server, data, expected):
    url, fake_service = server

    response = requests.post(
        f"{url}/api/transcribe-sync",
        files={"file": ("audio.wav", io.BytesIO(b"\x00\x01"))},
        data=data,
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert fake_service.calls == [{"include_diarization": expected}]

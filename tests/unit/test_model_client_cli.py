"""`scripts/transcribe.py client transcribe` against a fake async-job model server."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeJobServer(BaseHTTPRequestHandler):
    """Mimics model_server.py: POST queues a job, GET /api/jobs/<id> reports it."""

    job = None
    last_post_body = None

    def do_POST(self):
        type(self).last_post_body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send_json({"job_id": "job1", "status": "queued"}, status=202)

    def do_GET(self):
        if self.path == "/api/jobs/job1":
            self._send_json(type(self).job)
        else:
            self._send_json({"error": "Unknown endpoint"}, status=404)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def job_server(monkeypatch):
    # The Click command imports scripts.model_client; don't wait a real second
    # between job polls.
    import scripts.model_client

    monkeypatch.setattr(scripts.model_client, "POLL_INTERVAL", 0.01)
    FakeJobServer.job = None
    FakeJobServer.last_post_body = None
    server = HTTPServer(("127.0.0.1", _free_port()), FakeJobServer)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def input_file(tmp_path):
    f = tmp_path / "clip.mp3"
    f.write_bytes(b"\x00")
    return str(f)


def _completed_job():
    # Shape produced by model_server._run_transcription_job: segments are the
    # transcriber's (start, end, text, speaker) tuples, JSON-encoded as lists.
    return {
        "job_id": "job1",
        "status": "completed",
        "result": {
            "segments": [[0.0, 1.5, "hello world", ""], [1.5, 3.0, "second line", "SPEAKER_00"]],
            "preview_text": "hello world second line",
            "output_format": "txt",
            "processing_time": 0.1,
            "output_file": "/tmp/transcripts/clip.txt",
            "download_url": "/transcripts/clip.txt",
        },
    }


def _run_client(server_url, *args):
    return CliRunner().invoke(
        transcribe_cli.cli, ["client", "--server", server_url, "transcribe", *args]
    )


def test_client_transcribe_exits_nonzero_when_job_fails(job_server, input_file):
    FakeJobServer.job = {"job_id": "job1", "status": "failed", "error": "boom"}

    result = _run_client(job_server, input_file)

    assert result.exit_code == 1


def test_client_transcribe_prints_transcript_when_job_completes(job_server, input_file):
    FakeJobServer.job = _completed_job()

    result = _run_client(job_server, input_file)

    assert result.exit_code == 0, result.exception
    assert "[00:00.000 --> 00:01.500] hello world" in result.output
    assert "[00:01.500 --> 00:03.000] (SPEAKER_00) second line" in result.output


def test_client_transcribe_forwards_diarize_flag_to_server(job_server, input_file):
    """`client transcribe FILE --diarize` is how the transcribe-locally skill asks for speakers."""
    FakeJobServer.job = _completed_job()

    result = _run_client(job_server, input_file, "--diarize")

    assert result.exit_code == 0, result.output
    body = FakeJobServer.last_post_body
    assert b'name="diarize"' in body
    diarize_value = body.split(b'name="diarize"', 1)[1].split(b"--", 1)[0]
    assert diarize_value.strip() == b"true"


def test_client_transcribe_writes_output_file_when_job_completes(
    job_server, input_file, tmp_path
):
    FakeJobServer.job = _completed_job()
    output = tmp_path / "out.txt"

    result = _run_client(job_server, input_file, "--output", str(output))

    assert result.exit_code == 0, result.exception
    # The file holds the transcript the server formatted in the requested
    # format, not the client's console rendering.
    assert output.read_text() == _completed_job()["result"]["preview_text"]

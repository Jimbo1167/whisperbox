"""Worker-local model reuse in scripts/batch_transcribe.py."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_BATCH_PATH = _REPO_ROOT / "scripts" / "batch_transcribe.py"
spec = importlib.util.spec_from_file_location("batch_transcribe_module", _BATCH_PATH)
batch_transcribe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(batch_transcribe)

from src.config import Config


class FakeTranscriber:
    instances = 0

    def __init__(self, config):
        type(self).instances += 1
        self.config = config

    def transcribe(self, input_path):
        return [(0.0, 1.0, "hello", "")]

    def save_transcript(self, segments, output_path):
        Path(output_path).write_text("hello")


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(batch_transcribe, "Transcriber", FakeTranscriber)
    FakeTranscriber.instances = 0
    # Reset the worker-local cache between tests
    batch_transcribe._worker_state = threading.local()


def _process(config, tmp_path, name="a.wav"):
    src = tmp_path / name
    src.write_bytes(b"\x00")
    out, success, _ = batch_transcribe.process_file(
        str(src), config, str(tmp_path / "out")
    )
    assert success, "process_file failed"
    return out


def test_transcriber_is_reused_across_files_in_same_worker(tmp_path):
    config = Config()
    _process(config, tmp_path, "a.wav")
    _process(config, tmp_path, "b.wav")
    assert FakeTranscriber.instances == 1


def test_transcriber_is_rebuilt_when_config_changes(tmp_path):
    _process(Config(), tmp_path, "a.wav")
    _process(Config(whisper_model="tiny", transcription_engine="whisper"),
             tmp_path, "b.wav")
    assert FakeTranscriber.instances == 2

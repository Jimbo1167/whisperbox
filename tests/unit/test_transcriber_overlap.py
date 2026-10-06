"""Tests that audio extraction overlaps with model preload in Transcriber.

The optimization claim is: in ``Transcriber.transcribe``, the two engines'
``ensure_model_loaded`` calls run on a background thread pool that starts
before ``audio_processor.get_audio_path`` is called, so total wall time
collapses to ``max(ffmpeg, model_load) + max(transcribe, diarize)`` rather
than ``ffmpeg + model_load + max(transcribe, diarize)``.

These tests stub the engines and the audio processor without booting a real
Whisper/pyannote model. Overlap is proven with a barrier rather than wall-clock
limits: each stage blocks until every stage is running at once, so run
sequentially the barrier times out and breaks, however slow the machine.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import Config
from src.transcriber import Transcriber


def _build_transcriber(
    *, include_diarization: bool, barrier: Optional[threading.Barrier] = None
) -> Transcriber:
    config = Config(include_diarization=include_diarization, output_format="txt")
    # test_mode=True wires mock engines that don't touch real models. We then
    # replace specific methods to observe the stages.
    transcriber = Transcriber(config, test_mode=True)

    def stage():
        if barrier is not None:
            barrier.wait()

    def get_audio_path(_input_path: str):
        stage()
        return ("/tmp/fake_audio.wav", False)

    transcriber.audio_processor.get_audio_path = get_audio_path  # type: ignore[attr-defined]
    transcriber.transcription_engine.ensure_model_loaded = stage  # type: ignore[attr-defined]
    transcriber.diarization_engine.ensure_model_loaded = (  # type: ignore[attr-defined]
        lambda force=False: stage()
    )

    # Replace the heavy transcribe/diarize calls with instant fakes so we
    # observe only the load-vs-extract overlap.
    transcriber.transcription_engine.transcribe = lambda _p: [  # type: ignore[attr-defined]
        {"start": 0.0, "end": 1.0, "text": "hello"}
    ]
    transcriber.diarization_engine.diarize = lambda _p, enabled=None: [  # type: ignore[attr-defined]
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
    ]
    return transcriber


def test_audio_extract_overlaps_model_load_with_diarization():
    # Extraction, transcription-model load and diarization-model load.
    barrier = threading.Barrier(3, timeout=5)
    transcriber = _build_transcriber(include_diarization=True, barrier=barrier)

    transcriber.transcribe("ignored_path.mp4")

    assert not barrier.broken


def test_audio_extract_overlaps_model_load_without_diarization():
    # Extraction and transcription-model load only.
    barrier = threading.Barrier(2, timeout=5)
    transcriber = _build_transcriber(include_diarization=False, barrier=barrier)

    transcriber.transcribe("ignored_path.mp4")

    assert not barrier.broken


def test_model_load_error_propagates():
    transcriber = _build_transcriber(include_diarization=True)

    def boom(force=False):
        raise RuntimeError("simulated cuda OOM during preload")

    transcriber.diarization_engine.ensure_model_loaded = boom  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="simulated cuda OOM"):
        transcriber.transcribe("ignored_path.mp4")


def test_preload_skipped_when_diarization_disabled():
    transcriber = _build_transcriber(include_diarization=False)

    calls: List[str] = []

    def track_diar_load(force=False):
        calls.append("diar_load")

    transcriber.diarization_engine.ensure_model_loaded = track_diar_load  # type: ignore[attr-defined]

    transcriber.transcribe("ignored_path.mp4")

    assert "diar_load" not in calls, (
        "Diarization model preload should be skipped when include_diarization=False"
    )

"""Tests that audio extraction overlaps with model preload in Transcriber.

The optimization claim is: in ``Transcriber.transcribe``, the two engines'
``ensure_model_loaded`` calls run on a background thread pool that starts
before ``audio_processor.get_audio_path`` is called, so total wall time
collapses to ``max(ffmpeg, model_load) + max(transcribe, diarize)`` rather
than ``ffmpeg + model_load + max(transcribe, diarize)``.

These tests stub the engines and the audio processor with sleeps to make the
timing observable without booting a real Whisper/pyannote model.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import Config
from src.transcriber import Transcriber


SLEEP = 0.15  # short enough that the suite stays fast, long enough to measure


def _build_transcriber(*, include_diarization: bool) -> Transcriber:
    config = Config(include_diarization=include_diarization, output_format="txt")
    # test_mode=True wires mock engines that don't touch real models. We then
    # replace specific methods to control timing.
    transcriber = Transcriber(config, test_mode=True)

    def slow_get_audio_path(_input_path: str):
        time.sleep(SLEEP)
        return ("/tmp/fake_audio.wav", False)

    transcriber.audio_processor.get_audio_path = slow_get_audio_path  # type: ignore[attr-defined]

    def slow_ensure_transcription_load():
        time.sleep(SLEEP)

    def slow_ensure_diarization_load(force=False):
        time.sleep(SLEEP)

    transcriber.transcription_engine.ensure_model_loaded = slow_ensure_transcription_load  # type: ignore[attr-defined]
    transcriber.diarization_engine.ensure_model_loaded = slow_ensure_diarization_load  # type: ignore[attr-defined]

    # Replace the heavy transcribe/diarize calls with instant fakes so we
    # measure only the load-vs-extract overlap.
    transcriber.transcription_engine.transcribe = lambda _p: [  # type: ignore[attr-defined]
        {"start": 0.0, "end": 1.0, "text": "hello"}
    ]
    transcriber.diarization_engine.diarize = lambda _p, _enabled=None: [  # type: ignore[attr-defined]
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
    ]
    return transcriber


def test_audio_extract_overlaps_model_load_with_diarization():
    transcriber = _build_transcriber(include_diarization=True)

    start = time.perf_counter()
    transcriber.transcribe("ignored_path.mp4")
    elapsed = time.perf_counter() - start

    # If overlap works, total ~= max(SLEEP, SLEEP) + ~tiny stub work = ~SLEEP.
    # If sequential, total >= SLEEP * 3 (extract + transcribe_load + diarize_load).
    # Use 2 * SLEEP as a generous-but-strict ceiling.
    assert elapsed < 2 * SLEEP, (
        f"Expected audio extraction to overlap with model loads "
        f"(elapsed={elapsed:.3f}s, single-stage SLEEP={SLEEP:.3f}s)"
    )


def test_audio_extract_overlaps_model_load_without_diarization():
    transcriber = _build_transcriber(include_diarization=False)

    start = time.perf_counter()
    transcriber.transcribe("ignored_path.mp4")
    elapsed = time.perf_counter() - start

    # Two stages overlap: ffmpeg and transcription-engine load.
    # Sequential would be ~2*SLEEP; overlapped is ~SLEEP.
    assert elapsed < 1.6 * SLEEP, (
        f"Expected audio extraction to overlap with transcription model load "
        f"(elapsed={elapsed:.3f}s, single-stage SLEEP={SLEEP:.3f}s)"
    )


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

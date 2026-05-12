"""Tests for src/utils/progress_events.py."""

from __future__ import annotations

import io
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.progress_events import JsonlProgressEmitter, _slug


def _read_events(buf: io.StringIO):
    return [json.loads(line) for line in buf.getvalue().splitlines() if line]


class TestSlug:
    def test_basic_lowercase(self):
        assert _slug("Preparing audio") == "preparing_audio"

    def test_collapses_punctuation(self):
        assert _slug("Combining transcript!!!") == "combining_transcript"

    def test_strips_leading_trailing(self):
        assert _slug("  -- Done -- ") == "done"

    def test_empty_falls_back(self):
        assert _slug("") == "stage"
        assert _slug("!!!") == "stage"


class TestProgressEmission:
    def test_one_line_per_call(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("Preparing audio", 0.05)
        emit("Transcribing", 0.2)
        emit("Completed", 1.0)
        events = _read_events(buf)
        assert len(events) == 3
        assert all(e["event"] == "progress" for e in events)

    def test_stage_slug_and_label(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("Aligning speakers", 0.78)
        event = _read_events(buf)[0]
        assert event["stage"] == "aligning_speakers"
        assert event["stage_label"] == "Aligning speakers"
        assert event["message"] == "Aligning speakers"

    def test_required_fields_present(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("Transcribing", 0.5)
        event = _read_events(buf)[0]
        for key in ("ts", "event", "stage", "stage_label", "progress",
                    "percent", "elapsed_s", "eta_s", "message"):
            assert key in event, f"missing key: {key}"

    def test_percent_is_rounded_int(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("X", 0.234)
        assert _read_events(buf)[0]["percent"] == 23

    def test_progress_clamped_to_unit_interval(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("Too low", -0.5)
        emit("Too high", 1.5)
        events = _read_events(buf)
        assert events[0]["progress"] == 0.0
        assert events[1]["progress"] == 1.0
        assert events[1]["percent"] == 100

    def test_progress_is_monotonic(self):
        # Pipeline stages can report progress in any order; the emitter must
        # never let the agent see a backwards step.
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("Forward", 0.5)
        emit("Backward", 0.3)
        events = _read_events(buf)
        assert events[0]["progress"] == 0.5
        assert events[1]["progress"] == 0.5
        assert events[1]["stage"] == "backward"  # label preserved

    def test_eta_null_before_threshold(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("Early", 0.01)
        assert _read_events(buf)[0]["eta_s"] is None

    def test_eta_populated_after_threshold(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        # Backdate start so elapsed is large enough to make the math obvious.
        emit.start_time = time.time() - 10.0
        emit("Past threshold", 0.5)
        eta = _read_events(buf)[0]["eta_s"]
        assert eta is not None
        # elapsed=10s at progress=0.5 → projects another ~10s.
        assert 5.0 < eta < 20.0

    def test_eta_null_at_completion(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit.start_time = time.time() - 5.0
        emit("Done", 1.0)
        assert _read_events(buf)[0]["eta_s"] is None


class TestLifecycleEvents:
    def test_started_event(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit.emit_started(input="foo.mp4", format="txt", diarize=True)
        event = _read_events(buf)[0]
        assert event["event"] == "started"
        assert event["input"] == "foo.mp4"
        assert event["format"] == "txt"
        assert event["diarize"] is True
        assert event["elapsed_s"] == 0.0

    def test_started_resets_baseline(self):
        # A second emit_started should reset start_time and last_progress.
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit("First run", 0.9)
        emit.emit_started(input="second.mp4")
        emit("Second run", 0.1)
        events = _read_events(buf)
        # After reset, 0.1 is allowed (not clamped against the prior 0.9).
        progress_events = [e for e in events if e["event"] == "progress"]
        assert progress_events[-1]["progress"] == pytest.approx(0.1)

    def test_completed_event(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit.emit_started(input="foo.mp4")
        emit.emit_completed(output="transcripts/foo.txt", segments=42)
        event = _read_events(buf)[-1]
        assert event["event"] == "completed"
        assert event["output"] == "transcripts/foo.txt"
        assert event["segments"] == 42
        assert event["elapsed_s"] >= 0.0

    def test_error_event(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit.emit_started(input="bad.mp4")
        emit.emit_error("ffmpeg failed", code=42)
        event = _read_events(buf)[-1]
        assert event["event"] == "error"
        assert event["error"] == "ffmpeg failed"
        assert event["code"] == 42


class TestStreamHandling:
    def test_defaults_to_stderr(self):
        emit = JsonlProgressEmitter()
        assert emit.stream is sys.stderr

    def test_well_formed_json_per_line(self):
        buf = io.StringIO()
        emit = JsonlProgressEmitter(stream=buf)
        emit.emit_started(input="foo.mp4")
        emit("Preparing audio", 0.05)
        emit("Transcribing", 0.2)
        emit.emit_completed(output="foo.txt", segments=3)
        # Every non-empty line should be standalone valid JSON.
        for line in buf.getvalue().splitlines():
            if line:
                json.loads(line)

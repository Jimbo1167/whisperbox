"""Structured progress events for programmatic callers (LLM agents, scripts).

``JsonlProgressEmitter`` is a drop-in for the ``progress_callback`` argument
accepted by :func:`src.service.TranscriptionService.transcribe_file`. It emits
one JSON object per line to a configured stream (``sys.stderr`` by default),
so callers can parse the stream incrementally:

    emit = JsonlProgressEmitter()
    emit.emit_started(input="foo.mp4", format="txt", diarize=True)
    service.transcribe_file("foo.mp4", progress_callback=emit)
    emit.emit_completed(output="transcripts/foo.txt", segments=len(result["segments"]))

Schema per ``"progress"`` line::

    {
      "ts":          <unix epoch seconds, float>,
      "event":       "progress",
      "stage":       "<lowercase_slug>",
      "stage_label": "<original message>",
      "progress":    <0.0..1.0>,
      "percent":     <0..100, int>,
      "elapsed_s":   <seconds since emit.emit_started or instance creation>,
      "eta_s":       <float seconds | null until progress >= 0.05>,
      "message":     "<original message>"
    }

Additional events:

* ``{"event": "started", "elapsed_s": 0.0, ...caller fields...}``
* ``{"event": "completed", "elapsed_s": <float>, ...caller fields...}``
* ``{"event": "error", "error": "<message>", "elapsed_s": <float>, ...caller fields...}``

The emitter clamps progress monotonically forward: a stage that reports a
lower value than the previous one is silently raised to the previous level.
This lets pipeline stages that don't know the global percent (e.g. an
alignment step reporting 0.78) coexist with a downstream stage that's already
emitted 0.9 without rewinding the bar for the agent.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Dict, Optional, TextIO


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", (text or "").lower()).strip("_") or "stage"


class JsonlProgressEmitter:
    """Emit one JSON-Lines record per pipeline transition."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.start_time = time.time()
        self._last_progress: float = 0.0

    def _write(self, payload: Dict[str, Any]) -> None:
        payload["ts"] = time.time()
        self.stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        try:
            self.stream.flush()
        except Exception:
            pass

    def emit_started(self, **fields: Any) -> None:
        self.start_time = time.time()
        self._last_progress = 0.0
        payload: Dict[str, Any] = {"event": "started", "elapsed_s": 0.0}
        payload.update(fields)
        self._write(payload)

    def emit_completed(self, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "event": "completed",
            "elapsed_s": time.time() - self.start_time,
        }
        payload.update(fields)
        self._write(payload)

    def emit_error(self, error: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "event": "error",
            "error": error,
            "elapsed_s": time.time() - self.start_time,
        }
        payload.update(fields)
        self._write(payload)

    def __call__(self, message: str, progress: float) -> None:
        progress = max(self._last_progress, min(1.0, max(0.0, float(progress))))
        self._last_progress = progress

        elapsed = time.time() - self.start_time
        eta: Optional[float] = None
        if 0.05 <= progress < 1.0:
            eta = elapsed / progress * (1.0 - progress)

        self._write(
            {
                "event": "progress",
                "stage": _slug(message),
                "stage_label": message,
                "progress": progress,
                "percent": int(round(progress * 100)),
                "elapsed_s": elapsed,
                "eta_s": eta,
                "message": message,
            }
        )

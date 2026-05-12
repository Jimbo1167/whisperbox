# 2026-05-12 — Agent-Facing Progress Stream & YouTube/yt-dlp Format Alignment

## Goal

Make the whisperbox pipeline a better citizen for programmatic callers (LLM
agents, automation scripts):

1. **Cut wall time** — preload models, overlap ffmpeg with model warmup.
2. **Emit structured progress** — JSON Lines on stderr from the CLI; SSE on the
   model server.
3. **Keep accuracy parity** — regression harness using YouTube reference
   transcripts via `yt-dlp`.
4. **Align outputs with YouTube/yt-dlp formats** — `vtt-voice` (WebVTT with
   `<v Speaker>` voice tags) and `json3` (YouTube auto-caption wire format).

Today's PR ships **Phase A** (JSONL progress on the CLI). Other phases are
written here so a follow-up session can pick them up without re-planning.

## Phase A — Structured progress events on the CLI (this PR)

### What ships

- `src/utils/progress_events.py` → `JsonlProgressEmitter`
- `scripts/transcribe.py` → new `--progress {pretty,jsonl,none}` flag on the
  `transcribe` subcommand (default `pretty` to preserve existing behavior)
- `tests/unit/test_progress_events.py`

### Event schema (one JSON object per line on stderr)

```json
{"ts": 1715500000.123, "event": "started",   "input": "foo.mp4", "format": "txt", "diarize": true}
{"ts": 1715500000.456, "event": "progress",  "stage": "preparing_audio",      "stage_label": "Preparing audio",      "progress": 0.05, "percent": 5,  "elapsed_s": 0.42, "eta_s": null,  "message": "Preparing audio"}
{"ts": 1715500011.987, "event": "progress",  "stage": "transcribing",         "stage_label": "Transcribing audio",   "progress": 0.20, "percent": 20, "elapsed_s": 11.9, "eta_s": 47.6,  "message": "Transcribing audio"}
{"ts": 1715500057.012, "event": "completed", "output": "transcripts/foo.txt", "segments": 142, "elapsed_s": 56.8}
```

Error case:

```json
{"ts": 1715500030.111, "event": "error", "error": "Audio extraction timed out", "elapsed_s": 30.0}
```

### Design notes

- **Re-uses existing `progress_callback`** — `JsonlProgressEmitter` is callable
  with `(message: str, progress: float)`, matching the signature
  `service.transcribe_file` already accepts. No changes to `service.py`,
  `transcriber.py`, or the engines.
- **Stage slug** is derived from the message via `re.sub(r"[^a-z0-9]+", "_", msg.lower()).strip("_")`.
  Pipeline message strings (e.g. `"Preparing audio"`, `"Transcribing and diarizing"`)
  stay intact for human-readable modes; the slug is the stable identifier.
- **Progress is monotonic** — the emitter clamps each value forward of the last
  reported one. Agents can trust that progress never rewinds.
- **ETA** is computed once `progress >= 0.05` to avoid wild estimates from a
  cold start.
- **Logging interference** — when `--progress jsonl`, the CLI raises the root
  logger to WARNING and suppresses `click.echo` info lines so stderr stays
  parseable JSONL.

## Phase B — SSE on the model server (follow-up)

Add `GET /api/jobs/{id}/events` returning `text/event-stream` with the same
payload schema as JSONL. Implementation outline:

- One `asyncio.Queue` per job.
- The progress callback handed to `service.transcribe_file` writes to the queue.
- The SSE handler reads from the queue and yields `data: <json>\n\n` frames.
- Existing `GET /api/jobs/{id}` polling endpoint stays for back-compat.

## Phase C — YouTube / yt-dlp format alignment

Add two new output formats to `OutputFormatter`:

- **`vtt-voice`** — WebVTT with `<v Speaker>Text</v>` voice spans, which
  matches the WebVTT voice-tag convention and renders speakers cleanly in
  browser-native players. (Current `vtt` uses `SPEAKER_00: Text` literal.)
- **`json3`** — YouTube auto-caption wire format:

  ```json
  {
    "wireMagic": "pb3",
    "events": [
      {"tStartMs": 0,    "dDurationMs": 1500, "segs": [{"utf8": "First cue"}]},
      {"tStartMs": 1600, "dDurationMs": 1400, "segs": [{"utf8": "Second cue"}]}
    ]
  }
  ```

  yt-dlp consumes this directly and can convert to vtt/srt with
  `yt-dlp --convert-subs vtt`.

The existing flat `json` keeps its name and shape (back-compat). When
faster-whisper emits word-level timestamps, populate `segs[*].tOffsetMs` for
finer-grained alignment.

## Phase D — Accuracy harness via yt-dlp

`scripts/benchmark.py <youtube-url> [--engine whisper|parakeet] [--model …]`:

1. `yt-dlp --write-auto-sub --sub-langs en --skip-download --convert-subs vtt <url>` →
   reference VTT.
2. `yt-dlp -x --audio-format wav <url>` → audio.
3. Run whisperbox pipeline on the audio.
4. Compute WER (Word Error Rate) via `jiwer` against the reference.
5. Emit a JSON report to `benchmarks/<video-id>/<UTC-timestamp>.json` with
   `{wer, cer, segments, elapsed_s, engine, model, ref_source}`.

Add `jiwer` and `yt-dlp` to `requirements-dev.txt`. Seed a fixed set of public
videos in `benchmarks/seed_videos.txt` (3–5 short clips, mixed accents and
domains). A CI job can fail if WER on any seed exceeds `baseline + 5%`.

## Phase E — Performance

| ID  | Change                                                               | Estimated win |
|-----|----------------------------------------------------------------------|---------------|
| E.1 | Preload models in the CLI single-file path (server already does this) | 2–6 s        |
| E.2 | Kick off diarization model load in the background while ffmpeg runs   | 2–5 s        |
| E.3 | Cache device/dtype detection across calls                             | <1 s          |
| E.4 | Drop the `service._lock` for the single-tenant CLI path               | enables parallel batches |

E.1 lands first as a one-line change in `scripts/transcribe.py` (pass
`preload_models=True` to the service). E.2 needs threading discipline — guarded
behind a flag for the first iteration.

## Risks & mitigations

- **JSONL events colliding with Python logging on stderr** → in `jsonl` mode,
  raise root logger level to WARNING and suppress the existing colored
  `click.echo` info lines.
- **Format additions breaking format selection** → extend `OUTPUT_FORMATS` in
  `scripts/transcribe.py` and the dispatch in `OutputFormatter.save_transcript`
  with a clean ValueError on unknown formats. Tests cover each format.
- **yt-dlp / jiwer as new deps** → both land in `requirements-dev.txt` only;
  runtime install is unaffected.
- **Reference transcript accuracy** — YouTube auto-captions are themselves
  ASR-generated and noisy. The harness should also support a `--reference vtt`
  flag pointing at a hand-corrected file.

## Out of scope

- Real-time microphone streaming (existing FUTURE.md item).
- Word-level speaker diarization (needs realignment heuristic).
- WebSocket protocol (SSE is simpler and sufficient for the agent-progress use case).

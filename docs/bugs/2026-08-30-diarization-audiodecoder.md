# Bug: diarization crashes with `NameError: name 'AudioDecoder' is not defined` — and takes the finished transcription down with it

**Date observed:** 2026-08-30, ~20:51 local (macOS, Darwin 25.5.0)
**Repo state:** `main` @ `9c297f9` ("chore(deps): bump tqdm to >=4.67.1,<5") with uncommitted doc edits
**Severity:** high — any run with diarization enabled fails at the end, and the completed transcription result is discarded
**Status:** fixed 2026-08-30 (see §8). Workaround applied downstream (see §6).

Line numbers in this report refer to commit `9c297f9`; the code has changed since.

This report is written to be picked up cold by another agent. Everything needed to reproduce, diagnose, and fix is below; nothing else in the session context is required.

---

## 1. What happened

A 62-minute mp3 (podcast audio, ~57 MB) was submitted through the warm-server path:

```
python scripts/model_server.py --host 0.0.0.0 --port 8000   # server
python scripts/transcribe.py client transcribe <abs-path-to-mp3>   # client — note: NO --diarize
```

Transcription itself **succeeded** (847.8s, 1,063 segments, whisper-large-v3-turbo) and was cached. Then the server ran a diarization post-step — which the client never asked for — and that step raised, so the job was reported as failed and the client printed `Transcription failed` after 14m15s. From the caller's perspective the entire run produced nothing.

Server log excerpt (`/tmp/whisperbox-server.log`):

```
2026-08-30 20:51:10,004 - src.transcription.engine - INFO - Transcription completed in 847.8 seconds, found 1063 segments
2026-08-30 20:51:10,007 - src.cache.manager - INFO - Cached transcription results: /Users/jim/.cache/whisperbox/transcription/transcription-whisper-large-v3-turbo_957b586b20349d3c33f6e29fed79479e.json
2026-08-30 20:51:10,007 - src.transcriber - ERROR - Error during transcription: Error during diarization: name 'AudioDecoder' is not defined
2026-08-30 20:51:10,011 - __main__ - ERROR - Error processing job 35d2cba541334bf8925ec97c896c280d: Error during diarization: name 'AudioDecoder' is not defined
```

The raise site is [src/diarization/engine.py:201-202](../../src/diarization/engine.py) (catch-all that logs and re-raises).

## 2. Root cause chain (verified, not speculation)

There are really **three stacked issues**:

### Issue A — torchcodec cannot load against FFmpeg 9 → pyannote's `AudioDecoder` is never defined

`pyannote-audio 4.0.4` imports torchcodec behind a try/except in
`venv/lib/python3.13/site-packages/pyannote/audio/core/io.py:42-53`:

```python
try:
    import torchcodec
    from torchcodec.decoders import AudioDecoder, AudioStreamMetadata
except Exception as e:
    warnings.warn("torchcodec is not installed correctly so built-in audio decoding will fail...")
```

When that import fails it only *warns*; later, `io.py` uses `AudioDecoder(...)` at lines 86/319/394 → `NameError: name 'AudioDecoder' is not defined` at pipeline runtime.

The import fails on this machine — verified directly:

```
$ ./venv/bin/python -c "import torchcodec"
OSError: Could not load this library:
  .../site-packages/torchcodec/libtorchcodec_core4.dylib
```

torchcodec ships FFmpeg-major-versioned dylibs (`libtorchcodec_core{4..8}.dylib`, tried newest-first) and links against the system FFmpeg shared libraries. The system has **FFmpeg 9.0.1** (`ffmpeg -version` → `ffmpeg version 9.0.1`), which installed torchcodec **0.11.1** has no core variant for — every variant fails to dlopen, the last error shown being the core4 attempt.

Installed versions (from `./venv/bin/pip list`): `pyannote-audio 4.0.4`, `torch 2.11.0`, `torchaudio 2.11.0`, `torchcodec 0.11.1`, `faster-whisper 1.2.1`, python 3.13.

### Issue B — server-side jobs ignore the per-request diarize flag; `.env` forces diarization on

The client command did **not** pass `--diarize`, yet the server diarized anyway. `.env` has:

```
INCLUDE_DIARIZATION=true
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
```

The decision points are `src/transcriber.py:50` (`self.include_diarization = self.config.include_diarization`, i.e. straight from env-backed config) and the usage sites at `src/transcriber.py:156/168/350`. `src/service.py:43-53` shows there IS logic that can flip diarization off (it sets `transcriber.include_diarization = False` in some path — read that block first; it may be the intended opt-out that the model-server job path bypasses). Whether `scripts/model_server.py` even receives a diarize flag per job, and whether it applies it, needs checking — from this incident's behavior, the server used the env default and the client's omission of `--diarize` changed nothing.

### Issue C — a failed post-step destroys a succeeded transcription

The transcription result was complete and cached before diarization ran, but the job-level error handling reports total failure and the client surfaces nothing. The only reason the work wasn't lost: the transcription cache. Recovery was done manually by reading
`~/.cache/whisperbox/transcription/transcription-whisper-large-v3-turbo_957b586b20349d3c33f6e29fed79479e.json`
(a JSON **list** of `{start, end, text, words}` segments — note: bare list, not a dict).

## 3. Reproduction

1. Machine with FFmpeg 9.x as the resolvable FFmpeg shared libs (macOS/homebrew: `ffmpeg version 9.0.1`).
2. This repo's venv (`torchcodec 0.11.1`, `pyannote-audio 4.0.4`).
3. `.env` with `INCLUDE_DIARIZATION=true`.
4. Any transcription through the server path (client without `--diarize`).
5. → transcription completes, then `Error during diarization: name 'AudioDecoder' is not defined`, job fails.

Fast repro without a long audio file: `./venv/bin/python -c "import torchcodec"` shows Issue A immediately; any short mp3 shows B and C.

## 4. Suggested fixes (in priority order)

1. **Issue B (cheapest, biggest win):** make the request-level diarize flag authoritative for server jobs — env default only when the request doesn't specify. A client that doesn't ask for diarization must never pay for (or fail on) it. Add a regression test: server with `INCLUDE_DIARIZATION=true` + client without `--diarize` → no diarization attempted.
2. **Issue C:** if diarization (or any post-step) fails, return the completed transcription with a warning attached (e.g. `{"segments": [...], "diarization_error": "..."}`), not a job failure. The data already exists at failure time — it was cached one line earlier in the log.
3. **Issue A (environment fix + guard):**
   - Environment: either upgrade torchcodec to a release with FFmpeg-9 support (check torchcodec's FFmpeg compat table for the version pairing with torch 2.11), or install/point at FFmpeg ≤ the max major torchcodec 0.11.1 supports (it probes core8→core4, so FFmpeg 8 or lower) — e.g. `brew install ffmpeg@7` and make its libs resolvable to the venv process.
   - Code guard: at diarization-engine init (or server startup when diarization is enabled), do a cheap `import torchcodec` probe and fail fast with an actionable message ("diarization unavailable: torchcodec cannot load against system FFmpeg <v>; install ffmpeg@7 or upgrade torchcodec"), instead of letting pyannote's swallowed warning turn into a mid-job NameError after 14 minutes of work.
4. Optional hardening: pin/assert the FFmpeg-major ↔ torchcodec compatibility in `make setup` or a doctor script, since this will silently re-break on the next `brew upgrade ffmpeg`.

## 5. Acceptance criteria

- [x] Client `transcribe` without `--diarize` never runs diarization regardless of `.env`.
- [x] With diarization requested and torchcodec broken, the run fails **fast** (before transcription) with an actionable message — or degrades to transcription-only with a warning, but never burns the full transcription pass and then errors.
- [x] Post-step failure returns the transcription segments to the client.
- [x] `./venv/bin/python -c "from torchcodec.decoders import AudioDecoder"` succeeds after the environment fix, and a real `--diarize` run completes.

## 6. Workaround used downstream (context only)

The 2026-08-30 run's output was recovered by reading the cached segment list directly and formatting `[mm:ss] text` lines. No repo changes were made. The triggering mp3 was a temp file (session scratchpad) and may no longer exist; any long mp3 reproduces.

## 7. Pointers

- Raise site: `src/diarization/engine.py:201`
- Diarization decision: `src/transcriber.py:50,156,168,350`; opt-out logic to study: `src/service.py:43-53`; config: `src/config.py:23`
- Server job path: `scripts/model_server.py` (jobs polled at `/api/jobs/<id>`)
- pyannote guard: `venv/.../pyannote/audio/core/io.py:42-53`, uses at `:86,:319,:394`
- Full server log from the incident: `/tmp/whisperbox-server.log` (may be gone after reboot; key lines quoted in §1)
- Transcription cache dir: `~/.cache/whisperbox/transcription/`

## 8. Resolution (2026-08-30)

All three issues fixed; acceptance criteria in §5 verified.

**Issue B — request flag now authoritative.** `scripts/model_client.py` always
sends an explicit `diarize` field (`build_transcribe_options`), so a client
that didn't pass `--diarize` sends `diarize=false` and the server's
`INCLUDE_DIARIZATION` default never applies to it. The sync endpoint
(`/api/transcribe-sync`) also honors an optional `diarize` field now. The env
default still applies only when a request genuinely doesn't specify.
Regression tests: `tests/unit/test_model_client.py`,
`tests/unit/test_model_server.py`.

**Issue C — post-step failure no longer destroys the transcription.**
`Transcriber.transcribe` catches a diarization failure after transcription
succeeded, returns the unlabeled transcript, and records the error in
`Transcriber.last_diarization_error`. `TranscriptionService` surfaces it as
`diarization_error` in its result dicts; the model server includes it in job
results and JSON responses; the CLI and model client print a warning. A failed
*transcription* still fails the job. Tests:
`tests/unit/test_diarization_failure.py`.

**Issue A — fail fast + environment fix.**
- Code guard: `src/diarization/engine.py:_check_torchcodec_available()` runs at
  diarization model load (before the pipeline load, which itself happens
  before transcription starts), imports `AudioDecoder` *and decodes a tiny
  in-memory WAV* — necessary because torchcodec >= 0.12 imports successfully
  without FFmpeg and only fails on first decoder use. On failure it raises an
  actionable RuntimeError instead of a mid-job NameError.
- Environment: upgraded torchcodec 0.11.1 → 0.16.0 (compatible with torch
  >= 2.11 per torchcodec's table; ships `libtorchcodec_core9` for FFmpeg 9).
  **Caveat:** 0.16's macOS wheels carry no `LC_RPATH`, so the FFmpeg dylibs
  aren't findable from a uv/pyenv Python whose rpath doesn't include
  homebrew. Fixed by adding the rpath to the venv's torchcodec libraries
  (fish shell syntax):

  ```fish
  cd venv/lib/python3.13/site-packages/torchcodec
  for f in libtorchcodec_core*.dylib libtorchcodec_custom_ops*.dylib
      install_name_tool -add_rpath /opt/homebrew/opt/ffmpeg/lib $f
      codesign -f -s - $f
  end
  ```

  This must be reapplied after any pip reinstall of torchcodec.
- Hardening: `requirements.txt` now floors `torchcodec>=0.16,<1`, and
  `tests/unit/test_diarization_engine.py::test_torchcodec_decodes_on_this_machine`
  asserts the installed torchcodec decodes against the system FFmpeg, so the
  next `brew upgrade ffmpeg` past a supported major fails the test suite
  instead of silently re-breaking diarization runs.

Verified end to end: `transcribe --diarize` on synthesized speech completes
with a speaker label (`Diarization completed in 2.5 seconds, found 1
segments`), and full suite green (190 passed).

**Discovered in passing (out of scope, not fixed here):** with the default
Parakeet engine, the diarize path fails with `There is no Stream(gpu, 0) in
current thread` — parakeet-mlx cannot run inside the ThreadPoolExecutor worker
thread that `Transcriber.transcribe` uses for the parallel
transcribe+diarize block. Whisper is unaffected. With Issue C's fix the job
still fails (it's the transcription step that dies), so this needs its own
fix.

**Update:** fixed in PR #7 (`478f17c`). `ParakeetEngine._load_model` in
`src/transcription/engine.py` now materializes the model weights on the loading
thread (`mx.eval(model.parameters())`), so inference works from worker threads.

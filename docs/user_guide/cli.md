# Command Line Interface Guide

This guide provides detailed information about the Whisperbox command-line interface (CLI) commands, options, and usage examples.

## Overview

The Whisperbox provides several command-line scripts for different use cases:

1. `scripts/transcribe.py` - Unified CLI with subcommands for transcription (recommended)
2. `scripts/model_server.py` - Server for persistent model instances (also serves the web UI)
3. `scripts/model_client.py` - Client for interacting with the model server
4. `scripts/batch_transcribe.py` - Process multiple files in batch
5. `scripts/stream_transcribe.py` - Process files in streaming mode
6. `scripts/benchmark.py` - Accuracy benchmarking against a reference transcript
7. `transcribe_video.py` (repo root) - Minimal legacy script; all options come from `.env` and the environment

Every script except the model client (which has no settings of its own) loads the
project's `.env` from the repository root, wherever you run it from. Variables
already set in your environment take precedence over the file, and command-line
options override both.

Unless a different set is noted below, subcommands share these option values:

- Output formats: `txt`, `srt`, `vtt`, `vtt-voice`, `json`, `json3`, `pretty`
- `--model` takes any model name faster-whisper accepts: `tiny`, `base`, `small`,
  `medium`, `large-v3`, `large-v3-turbo`, `distil-large-v3`, a Hugging Face repo id,
  or a local path. An explicit `--model` selects the Whisper engine for that run,
  unless the engine was also chosen explicitly (`--engine` or `TRANSCRIPTION_ENGINE`);
  then it logs a warning that the model has no effect. Parakeet's model is set by
  `PARAKEET_MODEL`.
- When `--model` is omitted, the `WHISPER_MODEL` setting from `.env` is used
  (default `large-v3-turbo`)
- `--language` sets Whisper's language. Parakeet detects the language itself; an
  explicit `--language` under Parakeet logs a warning and is ignored.
- `--diarize` is an opt-in flag; there is no `--no-diarize`. `transcribe`, `stream`,
  `batch` and `client transcribe` diarize only when it is passed, whatever
  `INCLUDE_DIARIZATION` says (that setting is the default only for the model server
  and `transcribe_video.py`).

## Unified CLI: `transcribe.py`

The `transcribe.py` script provides a unified interface with subcommands for different transcription modes.

```bash
python -m scripts.transcribe [OPTIONS] COMMAND [ARGS]...
```

### Global Options

- `--version`: Show version information and exit
- `--verbose, -v`: Enable verbose (DEBUG) logging
- `--help, -h`: Show help message and exit

### Transcribe Command

The `transcribe` command processes a single video or audio file. It always runs
in-process; it does not use a running model server (use `client transcribe` for
that).

```bash
python -m scripts.transcribe transcribe [OPTIONS] INPUT_PATH
```

#### Options

- `--output, -o PATH`: Output file path (default: `transcripts/<input name>.<format>`)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--model, -m TEXT`: Whisper model (selects Whisper unless `--engine` is given)
- `--engine, -e [whisper|parakeet]`: ASR engine (default: `TRANSCRIPTION_ENGINE`, else parakeet on Apple Silicon and whisper elsewhere)
- `--language, -l TEXT`: Language code (e.g., en, fr, de)
- `--diarize, -d`: Include speaker diarization
- `--progress [pretty|jsonl|none]`: Progress reporting mode (default: pretty).
  `jsonl` emits one JSON event per line on stderr for programmatic callers.
- `--help`: Show help message and exit

The command exits non-zero if transcription fails. If diarization fails after
transcription succeeded, the transcript is still written without speaker labels,
a warning is printed, and the exit status is 0.

#### JSONL progress events

With `--progress jsonl`, logging is limited to warnings and each line on stderr is one
JSON object with an `event` field and a `ts` timestamp:

- `started`: `input`, `output`, `format`, `diarize`, `engine`, `model` (the selected
  engine's model), `language`
- `progress`: `stage`, `stage_label`, `message`, `progress` (0.0–1.0), `percent`,
  `elapsed_s`, `eta_s`
- `completed`: `output`, `segments` (count), `processing_time`, `elapsed_s`, and
  `diarization_error` when diarization failed
- `error`: `error`, `elapsed_s`

The schema is defined in `src/utils/progress_events.py`; see also
[Structured Progress Events](../examples/progress_reporting.md#structured-progress-events-jsonl).

#### Examples

Basic transcription:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4
```

Specify output format and location:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 -f srt -o path/to/output.srt
```

Enable speaker diarization:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --diarize
```

Use a different Whisper model (runs Whisper even where Parakeet is the default):
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 -m medium
```

Choose the engine:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --engine whisper
```

Machine-readable progress events:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --progress jsonl 2> events.jsonl
```

### Stream Command

The `stream` command processes a file in streaming mode to reduce memory usage.

```bash
python -m scripts.transcribe stream [OPTIONS] INPUT_PATH
```

Streaming always uses the Whisper engine (Parakeet has no streaming mode), whatever `TRANSCRIPTION_ENGINE` is set to.

#### Options

- `--output, -o PATH`: Output file path (default: next to the input file, with the format's extension)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--words, -w`: Add word-level timestamps to each segment (JSON output only; ignored with a warning for other formats)
- `--diarize, -d`: Include speaker diarization
- `--model, -m TEXT`: Whisper model
- `--language, -l TEXT`: Language code (e.g., en, fr, de)
- `--help`: Show help message and exit

The command exits with a non-zero status if the input is missing or no segment
was transcribed. With `--diarize` the whole file is diarized before transcription
starts, so a diarization failure exits 1 without a transcript. An error after some
segments were transcribed (or Ctrl-C) still saves the partial transcript, but the
command exits 1 so scripts can tell it is incomplete.

#### Examples

Basic streaming transcription:
```bash
python -m scripts.transcribe stream path/to/video.mp4
```

Streaming with specific options:
```bash
python -m scripts.transcribe stream path/to/video.mp4 -f vtt -m small --diarize
```

Word-level timestamps:
```bash
python -m scripts.transcribe stream interview.wav --words -f json -o interview.json
```

Each segment in the JSON then carries a `words` list with absolute times in seconds:
```json
{
  "start": 0.0,
  "end": 2.66,
  "text": "the quick brown fox jumps over the lazy dog.",
  "speaker": "SPEAKER",
  "words": [
    {"start": 0.0, "end": 0.12, "word": " the"},
    {"start": 0.12, "end": 0.38, "word": " quick"}
  ]
}
```

Word timestamps are only available here. `transcribe` and `batch` JSON output has
no per-word timing.

### Batch Command

The `batch` command processes multiple files matching a glob pattern. Note that
it takes a single quoted pattern, not a list of paths — quote the glob so your
shell doesn't expand it. `**` matches subdirectories.

```bash
python -m scripts.transcribe batch [OPTIONS] INPUT_PATTERN
```

#### Options

- `--output-dir, -o PATH`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--model, -m TEXT`: Whisper model (selects Whisper unless `--engine` is given)
- `--engine, -e [whisper|parakeet]`: ASR engine (default: `TRANSCRIPTION_ENGINE`, else the platform default)
- `--language, -l TEXT`: Language code
- `--diarize, -d`: Include speaker diarization
- `--workers, -w INTEGER`: Number of parallel workers (default: 0 = auto, based on CPU count and free memory)
- `--adaptive, -a`: Use an adaptive worker pool that adjusts to system load
- `--streaming, -s`: Use streaming transcription (reduces memory usage; always uses Whisper)
- `--help`: Show help message and exit

Each worker loads its own copy of the models, so memory use grows with the number
of workers. Transcripts are written as `<output-dir>/<input name>.<format>`; inputs
with the same name in different directories overwrite each other.

The command exits with status 0 only if every matched file was transcribed. It
exits 1 if any file failed, no file matched the pattern, or the run was
interrupted; the log lists the files that failed. A diarization failure after a
file's transcription succeeded is logged as a warning, and the file is written
without speaker labels and counted as transcribed.

#### Examples

Process all MP4 files in a directory:
```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4"
```

Specify output directory and format:
```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4" -o path/to/output -f srt
```

Limit the number of workers:
```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4" -w 2
```

### Server Command

Starts the model server (equivalent to `python -m scripts.model_server`).

```bash
python -m scripts.transcribe server [OPTIONS]
```

#### Options

- `--host TEXT`: Host to bind the server (default: localhost)
- `--port, -p INTEGER`: Port to bind the server (default: 8000)
- `--config, -c PATH`: Path to a `.env` file to load instead of the project's `.env`

### Client Command

Interacts with a running model server (equivalent to `python -m scripts.model_client`;
see [Model Client](#model-client-model_clientpy) for the options).

```bash
python -m scripts.transcribe client [--server URL] {status|transcribe} [ARGS]...
```

#### Examples

```bash
python -m scripts.transcribe client status
python -m scripts.transcribe client transcribe audio.mp3
python -m scripts.transcribe client transcribe audio.mp3 -f srt -o audio.srt
```

### Completion Command

Prints a shell completion script for bash, zsh or fish. The shell defaults to the
one in `$SHELL`.

```bash
python -m scripts.transcribe completion [bash|zsh|fish]
```

The script completes the program name `transcribe.py`, so completion only works
when you run the CLI as `transcribe.py` from your `PATH`, under the project's
virtualenv Python (for example through a small wrapper script named
`transcribe.py`). To install for fish:

```bash
transcribe.py completion fish > ~/.config/fish/completions/transcribe.py.fish
```

`transcribe.py completion --help` shows the bash and zsh equivalents.

## Model Server: `model_server.py`

The `model_server.py` script runs a persistent model server for faster processing. It
also serves the drag-and-drop web UI at the server root (`http://localhost:8000`).

```bash
python -m scripts.model_server [OPTIONS]
```

> **Warning:** the server has no authentication. It binds `localhost` by default,
> but `make server` and the Docker image bind `0.0.0.0`, which makes it reachable
> from your network. Anyone who can reach it can upload files, and the JSON
> `/transcribe` endpoint transcribes any file path on the server's filesystem that
> the server's user can read. Bind beyond localhost only on a trusted network.

The server loads its engine and model once, at startup, from its configuration
(the project's `.env` plus the environment, or `--config`). It can't switch engine,
model or language per request. Transcriptions run one at a time; later requests
wait their turn.

### Options

- `--host TEXT`: Host to bind the server (default: localhost)
- `--port, -p INTEGER`: Port to bind the server (default: 8000)
- `--config, -c TEXT`: Path to a `.env` file to load instead of the project's `.env` (the environment still takes precedence)
- `--verbose, -v`: Enable verbose logging
- `--help`: Show help message and exit

### Examples

Start the server with default settings:
```bash
python -m scripts.model_server
```

Start the server on a specific port:
```bash
python -m scripts.model_server --port 5001
```

### HTTP Endpoints

- `GET /health` — liveness check: `{"status": "ok"}`
- `GET /status`, `GET /api/status` — server status (below)
- `POST /transcribe`, `POST /api/transcribe` with a multipart form — upload a file; starts an async job and returns `202`
- `GET /api/jobs/{id}` — poll an async job
- `POST /transcribe` with a JSON body — transcribe a file already on the server, synchronously
- `POST /api/transcribe-sync` — multipart upload, synchronous plain-text response
- `GET /transcripts/{filename}` — download a finished transcript
- `GET /` — browser UI

Errors are JSON objects of the form `{"error": "..."}`.

#### Status

```json
{
  "status": "running",
  "uptime": 123.4,
  "model": {"engine": "parakeet", "model": "mlx-community/parakeet-tdt-0.6b-v3",
            "model_size": "large-v3-turbo", "language": "en", "device": "MLX (Apple GPU)"},
  "stats": {"requests": 3, "successful": 3, "failed": 0, "avg_processing_time": 12.5}
}
```

`engine` and `model` name what the server actually loaded (a defaulted Parakeet
that fell back to Whisper reports `whisper`). `model_size` is `WHISPER_MODEL`,
kept for older clients. `device` is `MLX (Apple GPU)` for Parakeet; for Whisper
it is `CPU` with `FORCE_CPU=true`, otherwise `GPU (if available)`.

#### Upload (async job)

Multipart form fields:

- `file` (required): the audio or video file. Uploads over 500 MB get `413`.
- `format` (optional): one of the seven output formats; default: the server's `OUTPUT_FORMAT`. An unknown format gets `400`.
- `diarize` (optional): `true`/`false`; when absent, the server's `INCLUDE_DIARIZATION` applies.
- `model`, `language` (optional): must match what the server runs, or the request
  gets `409` with an explanation. `model` is compared with the loaded model;
  `language` is checked only when the server runs Whisper.

The `202` response carries the `job_id`. Poll `GET /api/jobs/{id}`; the job's
`status` is `queued`, `running`, `completed` or `failed`, with `progress` (0–1) and
`message`. A completed job has a `result` with `segments` (a list of
`[start, end, text, speaker]`), `preview_text` (the transcript in the requested
format), `output_format`, `processing_time`, `output_file`, `download_url` and
`diarization_error` (set if diarization failed and the transcript has no speaker
labels). A failed job has an `error`.

The server saves each transcript as `transcripts/<upload name>.<format>` under its
working directory, so uploads with the same name overwrite each other. Jobs live in
the server's memory: they are lost on restart and not shared between server
processes.

#### JSON request (file on the server)

```bash
curl -X POST http://localhost:8000/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"audio_path": "/abs/path/to/audio.wav", "include_diarization": false}'
```

`audio_path` is a path on the server's filesystem (`404` if it doesn't exist).
`include_diarization` is optional; when absent, the server's default applies. The
response is synchronous: `{"segments", "processing_time", "diarization_error"}`.
This is the endpoint [warm-server routing](#warm-server-routing) uses.

#### Synchronous upload

`POST /api/transcribe-sync` takes a multipart `file` and an optional `diarize`
field and returns `{"text": "..."}`, the segment texts joined by spaces. Uploads
over 500 MB get `413`.

## Model Client: `model_client.py`

The `model_client.py` script interacts with the model server.

```bash
python -m scripts.model_client [--server URL] COMMAND [ARGS]...
```

The client exits with status 1 on any failure: the server is unreachable, the
request is refused, the job fails, or the output file can't be written.

### Global Options

- `--server, -s TEXT`: URL of the model server (default: http://localhost:8000)

### Status Command

```bash
python -m scripts.model_client status
```

Prints the server's uptime, engine, model, language and device, and its request
statistics.

Check the status of a specific server:
```bash
python -m scripts.model_client --server http://example.com:8000 status
```

### Transcribe Command

```bash
python -m scripts.model_client transcribe [OPTIONS] FILE_PATH
```

Uploads the file, waits for the job to finish, and prints the segments. With
`--output`, it instead writes the transcript the server formatted in `--format`.

#### Options

- `--output, -o TEXT`: Write the transcript to this file instead of printing segments
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format (default: the server's `OUTPUT_FORMAT`)
- `--model, -m TEXT`: Model the server must be running. The server can't switch models, so a different model is refused (exit status 1, with the server's message).
- `--language, -l TEXT`: Language the server must be using (checked for Whisper only); refused the same way
- `--diarize, -d`: Include speaker diarization (the client always tells the server, so the server's `INCLUDE_DIARIZATION` never applies)
- `--help`: Show help message and exit

#### Examples

Transcribe a file using the default server:
```bash
python -m scripts.model_client transcribe path/to/video.mp4
```

Specify output format and location:
```bash
python -m scripts.model_client transcribe path/to/video.mp4 -f srt -o path/to/output.srt
```

Use a specific server:
```bash
python -m scripts.model_client --server http://example.com:8000 transcribe path/to/video.mp4
```

## Warm-Server Routing

`transcribe_video.py` (both the root script and `scripts/transcribe_video.py`, and
so `make transcribe`) checks for a running model server before loading models
itself:

1. If `WHISPERBOX_NO_SERVER` is set (`1`, `true`, `yes` or `on`), or diarization
   is on for this run, it transcribes in-process. `make diarize` therefore never
   uses the server.
2. It probes `WHISPERBOX_SERVER_URL` (default `http://localhost:8000`). If nothing
   answers within half a second, it transcribes in-process.
3. It compares the server's `/status` with this run's settings: the engine and that
   engine's model must match, and for Whisper the language too. On a mismatch it
   logs the difference and transcribes in-process. A server started from older code
   doesn't report its engine and is never used; restart it.
4. Otherwise it sends the file's absolute path to the server's JSON `/transcribe`
   endpoint (so the server must see the same filesystem) and formats and writes the
   transcript locally. If that request fails, it falls back to in-process
   transcription.

The `transcribe`, `stream` and `batch` commands never route through a server. Use
`client transcribe` to send a file to a server explicitly.

## Batch Transcription: `batch_transcribe.py`

The `batch_transcribe.py` script processes multiple files matching a glob pattern.
The `transcribe.py batch` subcommand delegates to it; use this script directly if
you need the extra worker-pool options. Exit codes are the same as for
[`batch`](#batch-command).

```bash
python -m scripts.batch_transcribe [OPTIONS] INPUT_PATTERN
```

### Options

- `--output-dir, -o TEXT`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--model, -m TEXT`: Whisper model (selects Whisper unless `--engine` is given)
- `--engine, -e [whisper|parakeet]`: ASR engine
- `--language, -l TEXT`: Language code
- `--diarize, -d`: Include speaker diarization
- `--workers, -w INTEGER`: Number of parallel workers (default: 0 = auto)
- `--min-workers INTEGER`: Minimum workers for the adaptive pool (default: 1)
- `--max-workers INTEGER`: Maximum workers for the adaptive pool (default: CPU count)
- `--adaptive, -a`: Use an adaptive worker pool that adjusts to system load
- `--streaming, -s`: Use streaming transcription (always uses Whisper)
- `--verbose, -v`: Enable verbose logging
- `--help`: Show help message and exit

### Examples

Process all MP4 files in a directory:
```bash
python -m scripts.batch_transcribe "path/to/directory/*.mp4"
```

Specify output directory and format:
```bash
python -m scripts.batch_transcribe "path/to/directory/*.mp4" -o path/to/output -f srt
```

## Streaming Transcription: `stream_transcribe.py`

The `stream_transcribe.py` script processes a file in streaming mode to reduce memory usage.

```bash
python -m scripts.stream_transcribe [OPTIONS] INPUT_PATH
```

It takes the same options as the `stream` command above (which calls it), and always uses the Whisper engine. Defaults for the model, language and format come from `.env` (`WHISPER_MODEL`, `LANGUAGE`, `OUTPUT_FORMAT`). Diarization runs only with `--diarize`.

### Options

- `--output, --output-path, -o TEXT`: Output file path (default: next to the input, with the format's extension)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--words, -w`: Add word-level timestamps to each segment (JSON output only)
- `--diarize, -d`: Include speaker diarization
- `--model, -m TEXT`: Whisper model
- `--language, -l TEXT`: Language code
- `--verbose, -v`: Enable verbose logging
- `--help`: Show help message and exit

### Examples

Basic streaming transcription:
```bash
python -m scripts.stream_transcribe path/to/video.mp4
```

Word-level timestamps as JSON:
```bash
python -m scripts.stream_transcribe interview.wav --words -f json -o interview.json
```

Streaming with diarization:
```bash
python -m scripts.stream_transcribe path/to/video.mp4 --diarize
```

## Accuracy Benchmarking: `benchmark.py`

The `benchmark.py` script runs the transcription pipeline against a YouTube video
(comparing to its captions) or a local file with a reference transcript, and writes
a WER report under `benchmarks/`. See `benchmarks/README.md` for details on
interpreting the results. Requires the dev dependencies (`jiwer`, `yt-dlp`).
Benchmark runs never diarize.

```bash
python -m scripts.benchmark [OPTIONS] URL_OR_PATH
```

### Options

- `--engine [whisper|parakeet]`: ASR engine to benchmark (default: `TRANSCRIPTION_ENGINE`, else the platform default)
- `--model TEXT`: Whisper model; selects Whisper unless `--engine` is given
- `--reference TEXT`: Path to a reference VTT file (instead of YouTube captions); the positional argument is then a local audio file
- `--keep-files`: Keep downloaded audio/caption files

The report's `engine` and `model` fields record what actually ran: for a Parakeet
run, `model` is `PARAKEET_MODEL`.

### Examples

```bash
python -m scripts.benchmark "https://www.youtube.com/watch?v=<id>"
python -m scripts.benchmark "<url>" --engine parakeet
python -m scripts.benchmark path/to/audio.wav --reference path/to/truth.vtt
```

## Legacy Script: `transcribe_video.py`

The root-level `transcribe_video.py` is the original minimal entry point. It only
accepts an input path and optional `--output`; everything else (engine, format,
model, diarization) comes from `.env` and the environment. `make transcribe` runs
it, and `make diarize` runs it with `INCLUDE_DIARIZATION=true` set inline. It uses
a running model server when it can (see [Warm-Server Routing](#warm-server-routing)).

```bash
python transcribe_video.py path/to/video.mp4 [-o output.txt]
```

A richer variant lives at `scripts/transcribe_video.py` (adds `--format` including
`pretty`, `--model`, `--language`, `--no-diarization`, `--verbose`). It also
validates the configuration first, and exits with an error if, for example,
diarization is on without `HF_TOKEN` or Parakeet is selected off Apple Silicon.
Prefer the unified `scripts/transcribe.py` CLI for new usage.

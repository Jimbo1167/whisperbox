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
7. `transcribe_video.py` (repo root) - Minimal legacy script; all options come from `.env`

Unless a different set is noted below, subcommands share these option values:

- Output formats: `txt`, `srt`, `vtt`, `vtt-voice`, `json`, `json3`, `pretty`
- Model sizes: `tiny`, `base`, `small`, `medium`, `large`
- When `--model` is omitted, the `WHISPER_MODEL` setting from `.env` is used
  (default `large-v3-turbo`)
- `--diarize` is an opt-in flag; there is no `--no-diarize`. To keep diarization
  off, leave `INCLUDE_DIARIZATION=false` in `.env` and omit the flag.

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

The `transcribe` command processes a single video or audio file.

```bash
python -m scripts.transcribe transcribe [OPTIONS] INPUT_PATH
```

#### Options

- `--output, -o PATH`: Output file path (default: `transcripts/<input name>.<format>`)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--model, -m [tiny|base|small|medium|large]`: Whisper model size
- `--language, -l TEXT`: Language code (e.g., en, fr, de)
- `--diarize, -d`: Include speaker diarization
- `--progress [pretty|jsonl|none]`: Progress reporting mode (default: pretty).
  `jsonl` emits one JSON event per line on stderr for programmatic callers.
- `--help`: Show help message and exit

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

Use a different model:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 -m medium
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

- `--output, -o PATH`: Output file path
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--words, -w`: Add word-level timestamps to each segment (JSON output only; ignored with a warning for other formats)
- `--diarize, -d`: Include speaker diarization
- `--model, -m [tiny|base|small|medium|large]`: Whisper model size
- `--language, -l TEXT`: Language code (e.g., en, fr, de)
- `--help`: Show help message and exit

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

### Batch Command

The `batch` command processes multiple files matching a glob pattern. Note that
it takes a single quoted pattern, not a list of paths — quote the glob so your
shell doesn't expand it.

```bash
python -m scripts.transcribe batch [OPTIONS] INPUT_PATTERN
```

#### Options

- `--output-dir, -o PATH`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--model, -m [tiny|base|small|medium|large]`: Whisper model size
- `--language, -l TEXT`: Language code
- `--diarize, -d`: Include speaker diarization
- `--workers, -w INTEGER`: Number of worker processes (default: 0 = auto)
- `--adaptive, -a`: Use an adaptive worker pool that adjusts to system load
- `--streaming, -s`: Use streaming transcription (reduces memory usage)
- `--help`: Show help message and exit

#### Examples

Process all MP4 files in a directory:
```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4"
```

Specify output directory and format:
```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4" -o path/to/output -f srt
```

Limit the number of worker processes:
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
- `--config, -c PATH`: Path to configuration file (default: .env)

### Client Command

Interacts with a running model server (equivalent to `python -m scripts.model_client`).

```bash
python -m scripts.transcribe client [--server URL] {status|transcribe} [ARGS]...
```

#### Examples

```bash
python -m scripts.transcribe client status
python -m scripts.transcribe client transcribe audio.mp3
```

### Completion Command

Generates a shell completion script for bash, zsh, or fish:

```bash
python -m scripts.transcribe completion
```

## Model Server: `model_server.py`

The `model_server.py` script runs a persistent model server for faster processing. It
also serves the drag-and-drop web UI at the server root (`http://localhost:8000`).

```bash
python -m scripts.model_server [OPTIONS]
```

### Options

- `--host TEXT`: Host to bind the server (default: localhost)
- `--port, -p INTEGER`: Port to bind the server (default: 8000)
- `--config, -c TEXT`: Path to configuration file (default: .env)
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

- `GET /health` — liveness check
- `GET /status`, `GET /api/status` — uptime, model info, stats
- `POST /transcribe`, `POST /api/transcribe` — multipart upload (async job) or JSON `{"audio_path": ...}` for files already on the server
- `GET /api/jobs/{id}` — poll an async job
- `POST /api/transcribe-sync` — multipart upload, synchronous response
- `GET /transcripts/{filename}` — download a finished transcript
- `GET /` — browser UI

Uploads are limited to 500 MB.

## Model Client: `model_client.py`

The `model_client.py` script interacts with the model server.

```bash
python -m scripts.model_client [--server URL] COMMAND [ARGS]...
```

### Global Options

- `--server, -s TEXT`: URL of the model server (default: http://localhost:8000)

### Status Command

```bash
python -m scripts.model_client status
```

Check the status of a specific server:
```bash
python -m scripts.model_client --server http://example.com:8000 status
```

### Transcribe Command

```bash
python -m scripts.model_client transcribe [OPTIONS] FILE_PATH
```

#### Options

- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format
- `--model, -m TEXT`: Whisper model size
- `--language, -l TEXT`: Language code
- `--diarize, -d`: Include speaker diarization
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

## Batch Transcription: `batch_transcribe.py`

The `batch_transcribe.py` script processes multiple files matching a glob pattern.
The `transcribe.py batch` subcommand delegates to it; use this script directly if
you need the extra worker-pool options.

```bash
python -m scripts.batch_transcribe [OPTIONS] INPUT_PATTERN
```

### Options

- `--output-dir, -o TEXT`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|json]`: Output format
- `--model, -m TEXT`: Whisper model size
- `--language, -l TEXT`: Language code
- `--diarize, -d`: Include speaker diarization
- `--workers, -w INTEGER`: Number of worker processes (default: 0 = auto)
- `--min-workers INTEGER`: Minimum workers for the adaptive pool (default: 1)
- `--max-workers INTEGER`: Maximum workers for the adaptive pool (default: CPU count)
- `--adaptive, -a`: Use an adaptive worker pool that adjusts to system load
- `--streaming, -s`: Use streaming transcription
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

It takes the same options as the `stream` command above (which calls it), and always uses the Whisper engine. Defaults for the model, language, format and diarization come from `.env` (`WHISPER_MODEL`, `LANGUAGE`, `OUTPUT_FORMAT`, `INCLUDE_DIARIZATION`).

### Options

- `--output, --output-path, -o TEXT`: Output file path (default: next to the input, with the format's extension)
- `--format, -f [txt|srt|vtt|vtt-voice|json|json3|pretty]`: Output format
- `--words, -w`: Add word-level timestamps to each segment (JSON output only)
- `--diarize, -d`: Include speaker diarization
- `--model, -m TEXT`: Whisper model size
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

```bash
python -m scripts.benchmark [OPTIONS] URL_OR_PATH
```

### Options

- `--engine [whisper|parakeet]`: ASR engine to benchmark (default: `TRANSCRIPTION_ENGINE`, else the platform default)
- `--model TEXT`: Model override
- `--reference TEXT`: Path to a reference transcript (instead of YouTube captions)
- `--keep-files`: Keep downloaded audio/caption files

### Examples

```bash
python -m scripts.benchmark "https://www.youtube.com/watch?v=<id>"
python -m scripts.benchmark "<url>" --engine parakeet
python -m scripts.benchmark path/to/audio.wav --reference path/to/truth.vtt
```

## Legacy Script: `transcribe_video.py`

The root-level `transcribe_video.py` is the original minimal entry point. It only
accepts an input path and optional `--output`; everything else (format, model,
diarization) comes from `.env`.

```bash
python transcribe_video.py path/to/video.mp4 [-o output.txt]
```

A richer variant lives at `scripts/transcribe_video.py` (adds `--format` including
`pretty`, `--model`, `--language`, `--no-diarization`, `--verbose`). Prefer the
unified `scripts/transcribe.py` CLI for new usage.

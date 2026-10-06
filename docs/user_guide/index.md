# Whisperbox User Guide

Welcome to the Whisperbox user guide. This document provides comprehensive instructions for installing, configuring, and using the Whisperbox tool.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Transcription Engines](#transcription-engines)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Speaker Diarization](#speaker-diarization)
8. [Caching](#caching)
9. [Command Line Interface](#command-line-interface)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

## Introduction

Whisperbox is a powerful Python tool for transcribing videos and audio files with speaker diarization. It processes video or audio files, transcribes the speech to text, and identifies different speakers in the conversation.

### Key Features

- **Transcription**: Convert speech to text with NVIDIA's Parakeet (default on Apple Silicon) or OpenAI's Whisper models (default elsewhere)
- **Speaker Diarization**: Identify different speakers in the audio (opt-in per run)
- **Multiple Input Formats**: Support for various video and audio formats
- **Multiple Output Formats**: txt, pretty, srt, vtt, vtt-voice, json, json3
- **Streaming Transcription**: Process large files with minimal memory usage
- **Batch Processing**: Process multiple files in a single command
- **Caching System**: Improve performance by caching results
- **Progress Reporting**: Progress bars for streaming, batch and client runs; JSONL events for programmatic callers
- **Model Server**: Run a persistent model server for faster processing
- **Enhanced CLI**: User-friendly command-line interface with subcommands

## Installation

### Prerequisites

- Python 3.10 or higher (`make setup` creates the virtual environment with `python3.11`)
- FFmpeg: the `ffmpeg` binary for audio extraction, and its shared libraries (major version 4–9) for speaker diarization
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jimbo1167/whisperbox.git
cd whisperbox
```

### Step 2: Set Up the Environment

Using Make (recommended):

```bash
make setup  # Creates venv and installs all dependencies
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure the Environment

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit the `.env` file with your preferred text editor to configure the settings.

## Configuration

Whisperbox reads its settings from environment variables. Every entry point that uses these settings (the `transcribe`, `stream`, `batch` and `server` commands, `stream_transcribe.py`, `batch_transcribe.py`, `model_server.py`, `transcribe_video.py` and `benchmark.py`) first loads the project's `.env` file from the repository root, wherever you run the command from. Variables already set in the environment take precedence over `.env`, whether you exported them in your shell or set them inline for one command:

```bash
INCLUDE_DIARIZATION=true python transcribe_video.py interview.mp4
```

Command-line options such as `--model`, `--engine`, `--language` and `--format` override both for that run. The model server's `--config PATH` option loads a different file in place of the project `.env`, with the same precedence.

Here are the available configuration options:

### General Settings

- `HF_TOKEN`: Your HuggingFace token for accessing models (required for diarization)
- `LANGUAGE`: Target language for Whisper (default: en). Parakeet detects the language itself and ignores it.
- `OUTPUT_FORMAT`: Transcript format (txt, srt, vtt, vtt-voice, json, json3, pretty)

### Model Settings

- `TRANSCRIPTION_ENGINE`: ASR engine, `whisper` or `parakeet` (default: parakeet on Apple Silicon, whisper elsewhere; parakeet is Apple Silicon only). See [Transcription Engines](#transcription-engines).
- `WHISPER_MODEL`: Whisper model; any name faster-whisper accepts, such as tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3, a Hugging Face repo id or a local path (default: large-v3-turbo)
- `WHISPER_BEAM_SIZE`: Beam size (default: 5; 1 = greedy, ~2x faster)
- `WHISPER_CPU_THREADS`: CPU threads for Whisper (default: 0 = ctranslate2's default of 4)
- `WHISPER_BATCH_SIZE`: >0 enables batched (parallel chunk) decoding (default: 0 = sequential)
- `PARAKEET_MODEL`: HF model id or local path to MLX weights (default: mlx-community/parakeet-tdt-0.6b-v3)
- `DIARIZATION_MODEL`: Diarization model to use (default: pyannote/speaker-diarization-community-1)
- `FORCE_CPU`: Run Whisper and pyannote diarization on the CPU even when a GPU is available (default: false; `.env.example` sets it to true). No effect on Parakeet.

### Feature Toggles

- `INCLUDE_DIARIZATION`: Default diarization setting for the model server (for requests that don't specify) and for `transcribe_video.py` / `make transcribe` (true/false, default: false). The `transcribe`, `stream`, `batch` and `client transcribe` commands ignore it and diarize only when you pass `--diarize`.
- `CACHE_ENABLED`: Enable/disable caching system (true/false, default: true)

### Caching Settings

- `CACHE_EXPIRATION`: Cache expiration time in seconds (default: 604800 = 7 days)
- `MAX_CACHE_SIZE`: Maximum cache size in bytes (default: 10GB)

### Timeout Settings

- `AUDIO_TIMEOUT`: Timeout for audio extraction in seconds (default: 300)
- `TRANSCRIBE_TIMEOUT`: Timeout for transcription in seconds (default: 3600)
- `DIARIZE_TIMEOUT`: Timeout for diarization in seconds (default: 3600)

### Warm-Server Routing

- `WHISPERBOX_SERVER_URL`: Model server that `transcribe_video.py` tries before loading models itself (default: http://localhost:8000)
- `WHISPERBOX_NO_SERVER`: Set to `1` to always transcribe in-process

See [Warm-server routing](cli.md#warm-server-routing) for when the server is used.

The model server's host and port are set with the `--host`/`--port` CLI flags,
not environment variables (default: localhost:8000).

## Transcription Engines

Whisperbox has two ASR engines. `TRANSCRIPTION_ENGINE` sets the default; `--engine whisper|parakeet` overrides it for one run of `transcribe` or `batch` (and `benchmark.py`).

| | Whisper (`faster-whisper`) | Parakeet (`parakeet-mlx`) |
|---|---|---|
| Platforms | macOS, Linux, Docker | Apple Silicon (macOS arm64) only |
| Default on | everything except Apple Silicon | Apple Silicon (falls back to Whisper if `parakeet-mlx` isn't installed) |
| Model setting | `WHISPER_MODEL`, or `--model` per run | `PARAKEET_MODEL` |
| Language | `LANGUAGE`, or `--language` per run | Detected automatically; `LANGUAGE` and `--language` are ignored |
| Streaming (`stream`, `batch --streaming`) | Yes | No; those commands always run Whisper |
| Word timestamps (`stream --words -f json`) | Yes | No (streaming only) |
| Device | CUDA if available, otherwise CPU; never Apple's GPU (MPS) | Apple GPU through MLX |
| `FORCE_CPU` | Forces the CPU | No effect (logs a warning) |
| Speed settings | `WHISPER_BEAM_SIZE`, `WHISPER_CPU_THREADS`, `WHISPER_BATCH_SIZE` | None |

An explicit `--model` selects Whisper for that run, unless you chose the engine explicitly (with `--engine` or `TRANSCRIPTION_ENGINE`). If that explicit choice is Parakeet, `--model` logs a warning that it has no effect, and Parakeet runs. Passing `--language` while Parakeet runs also logs a warning.

Diarization (pyannote) is separate from the engine choice. It runs on CUDA or Apple's GPU (MPS) when available, and on the CPU with `FORCE_CPU=true`.

## Basic Usage

### Transcribing a Video or Audio File

Using the unified CLI:

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4
```

This will transcribe the file using the default settings and save the transcript to the `transcripts` directory.

### Specifying Output Format

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --format srt
```

### Specifying Output Location

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --output path/to/output.txt
```

### Enabling Speaker Diarization

Diarization is off unless you pass `--diarize`. `INCLUDE_DIARIZATION` in `.env`
does not turn it on for this command. Diarization needs `HF_TOKEN`; see
[Speaker Diarization](#speaker-diarization).

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --diarize
```

### Selecting a Different Whisper Model

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --model medium
```

`--model` takes any faster-whisper model name and runs Whisper for that run, even
on Apple Silicon where Parakeet is the default. To choose the engine directly,
pass `--engine whisper` or `--engine parakeet`.

## Advanced Usage

### Streaming Transcription (Low Memory Usage)

For large files or systems with limited memory, use the streaming transcription:

```bash
python -m scripts.transcribe stream path/to/video.mp4
```

This processes the audio in chunks, significantly reducing memory usage. Streaming
always uses the Whisper engine. Unless you pass `--output`, the transcript is
written next to the input file.

### Batch Processing Multiple Files

Process multiple files matching a glob pattern (quote the pattern — the command
takes one pattern, not a list of paths):

```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4"
```

Or specify an output directory:

```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4" --output-dir path/to/output
```

The command exits with status 1 unless every matched file was transcribed, so
scripts can detect a partial batch.

### Using the Model Server

Start the model server:

```bash
python -m scripts.model_server
```

This listens on `localhost:8000`. `make server` binds `0.0.0.0` instead, which
makes the server reachable from your network. The server has no authentication,
and its JSON endpoint transcribes any file path on the server's filesystem, so
only bind it beyond localhost on a trusted network.

Check the server status:

```bash
python -m scripts.model_client status
```

Transcribe a file using the server:

```bash
python -m scripts.model_client transcribe path/to/your/video.mp4
```

The server keeps the engine, model and language it started with. A client
request for a different `--model` or `--language` is refused rather than served
with the wrong model. See the [CLI guide](cli.md#model-server-model_serverpy) for
the HTTP API.

## Speaker Diarization

Diarization labels each segment with a speaker, using the pyannote model set in
`DIARIZATION_MODEL`.

**When it runs.** The `transcribe`, `stream`, `batch` and `client transcribe`
commands diarize only when you pass `--diarize`. `INCLUDE_DIARIZATION` sets the
default only for the model server (for requests that don't say; the CLI client
and the web UI always do) and for `transcribe_video.py` / `make transcribe`. `make diarize` turns it on for a single run.

**Requirements.**

- `HF_TOKEN` (in `.env` or the environment) for a Hugging Face account that has
  accepted the terms of the model in `DIARIZATION_MODEL`.
- pyannote decodes audio through torchcodec, which loads the system FFmpeg shared
  libraries. torchcodec 0.16 (the minimum in `requirements.txt`) supports FFmpeg
  major versions 4–9. On macOS with a uv or pyenv Python, torchcodec may not find
  Homebrew's FFmpeg until you add an rpath; see
  [Troubleshooting](#torchcodec-cannot-load-ffmpeg).
- The pyannote model is downloaded on first use to `~/.cache/whisperbox/diarization/`.

**When it fails.**

- If the diarization model can't load (missing token, broken torchcodec), the run
  fails before transcription starts and reports why.
- If diarization fails after transcription succeeded, the transcript is still
  written, without speaker labels, and the error is reported: a warning from the
  CLI and the model client, a `diarization_error` field in the jsonl `completed`
  event and in the model server's job results and JSON responses, and a logged
  warning for each affected file in `batch`.
- Streaming diarizes the whole file before it starts transcribing, so a
  diarization failure there ends the run with exit status 1 and no transcript.
- A model server started with `INCLUDE_DIARIZATION=true` whose diarization model
  can't load logs a warning and turns its diarization default off.

## Caching

Whisperbox caches its intermediate results in `~/.cache/whisperbox/`:

- `audio/`: the 16 kHz mono WAV extracted from non-WAV inputs
- `transcription/`: transcripts
- `diarization/`: diarization results (`*.json`)

The same directory also holds model downloads: Whisper models in `whisper/` and
the pyannote model in a subdirectory of `diarization/`. Parakeet models go to the
Hugging Face cache (`~/.cache/huggingface/`).

**Keys.** Each entry is keyed on the input file's path, size and modification
time, so editing or moving a file means a fresh run. Transcripts are also keyed on
the engine and its model, and for Whisper on `LANGUAGE` and `WHISPER_BEAM_SIZE`.
Diarization results are also keyed on `DIARIZATION_MODEL`. Streaming runs
(`stream`, `batch --streaming`) reuse cached audio and diarization but don't cache
transcripts.

**Expiry and size.** Each time a run or the model server starts, entries older
than `CACHE_EXPIRATION` seconds (default 7 days) are removed. If the cache is then larger than
`MAX_CACHE_SIZE` bytes (default 10 GB), the oldest entries are removed until it
fits. Model downloads are not counted or removed.

**Disabling and clearing.** Set `CACHE_ENABLED=false` to turn the cache off. There
is no clear command; delete the cached results instead:

```bash
rm -rf ~/.cache/whisperbox/audio ~/.cache/whisperbox/transcription
find ~/.cache/whisperbox/diarization -maxdepth 1 -name '*.json' -delete
```

Deleting all of `~/.cache/whisperbox` also works, but removes the downloaded
Whisper and pyannote models, which are then downloaded again on the next run.

## Command Line Interface

The Whisperbox provides a unified command-line interface with several subcommands:

### Global Options

- `--help`: Show help message and exit
- `--version`: Show version information and exit

### Transcribe Command

```bash
python -m scripts.transcribe transcribe [OPTIONS] INPUT_PATH
```

Options:
- `--output, -o`: Output file path
- `--format, -f`: Output format (txt, srt, vtt, vtt-voice, json, json3, pretty)
- `--model, -m`: Whisper model (any faster-whisper name; selects Whisper unless `--engine` is given)
- `--engine, -e`: ASR engine (whisper, parakeet)
- `--language, -l`: Language code
- `--diarize, -d`: Include speaker diarization
- `--progress`: Progress mode (pretty, jsonl, none)

### Stream Command

```bash
python -m scripts.transcribe stream [OPTIONS] INPUT_PATH
```

Options:
- Same as transcribe command, minus `--progress` and `--engine` (streaming always uses Whisper), plus `--words, -w` for word-level timestamps

### Batch Command

```bash
python -m scripts.transcribe batch [OPTIONS] INPUT_PATTERN
```

Options:
- `--output-dir, -o`: Output directory
- `--format, -f`: Output format
- `--model, -m`: Whisper model (selects Whisper unless `--engine` is given)
- `--engine, -e`: ASR engine (whisper, parakeet)
- `--language, -l`: Language code
- `--diarize, -d`: Include speaker diarization
- `--workers, -w`: Number of workers (0 = auto)
- `--adaptive, -a`: Adaptive worker pool
- `--streaming, -s`: Use streaming transcription (always Whisper)

### Model Server Commands

Start the server:
```bash
python -m scripts.model_server [OPTIONS]
```

Options:
- `--host`: Host to bind the server (default: localhost)
- `--port, -p`: Port to bind the server (default: 8000)
- `--config, -c`: Path to a `.env` file (default: the project's `.env`)
- `--verbose, -v`: Enable verbose logging

Client commands:
```bash
python -m scripts.model_client status
python -m scripts.model_client transcribe [OPTIONS] INPUT_PATH
```

See the [CLI guide](cli.md) for the full option reference.

## Troubleshooting

### Common Issues

#### FFmpeg Not Found

Error: `RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.`

Audio extraction uses the `ffmpeg` binary that `imageio-ffmpeg` provides: its
bundled copy when the installed package includes one, otherwise the `ffmpeg` on
your `PATH`.

Solution: Install FFmpeg and ensure it's in your PATH, or point
`IMAGEIO_FFMPEG_EXE` at an `ffmpeg` binary.

```bash
# On macOS with Homebrew
brew install ffmpeg

# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On Windows with Chocolatey
choco install ffmpeg
```

#### CUDA Not Available

Log: `Using CPU for processing (no GPU acceleration available)`

Solution: Install a CUDA build of PyTorch for your GPU (`make install-torch`
installs CPU-only wheels), plus the NVIDIA libraries that faster-whisper's GPU
support needs (see the faster-whisper documentation). On Apple Silicon, use the
Parakeet engine for GPU speed; Whisper does not use Apple's GPU.

#### Out of Memory Errors

Error: `RuntimeError: CUDA out of memory`

Solutions:
- Use a smaller Whisper model
- Use streaming transcription
- Use fewer `batch` workers (each worker loads its own copy of the models)
- Increase your system's swap space
- Use a machine with more GPU memory

#### Diarization Errors

Error: `Could not download pyannote pipeline` or HTTP 401/403 from HuggingFace, or the warning `HF_TOKEN is not set`

Solutions:
- Set a valid `HF_TOKEN` in your `.env` file or your environment
- Accept the model's terms on its HuggingFace page for the model set in `DIARIZATION_MODEL` (default: pyannote/speaker-diarization-community-1)

#### torchcodec Cannot Load FFmpeg

Error: `Diarization is unavailable: torchcodec's AudioDecoder cannot be imported ...`

pyannote decodes audio through torchcodec, which loads the system FFmpeg shared
libraries and supports only some FFmpeg major versions (4–9 for torchcodec 0.16).

Solutions:
- Install a supported FFmpeg major version, or upgrade torchcodec to a release that supports yours
- On macOS with a uv or pyenv Python, torchcodec's libraries may not find Homebrew's FFmpeg. Add an rpath as described in [the diarization bug report, §8](../bugs/2026-08-30-diarization-audiodecoder.md#8-resolution-2026-08-30), and reapply it after reinstalling torchcodec
- Verify with `python -c 'from torchcodec.decoders import AudioDecoder'`, or run `pytest -m env`

#### Warm Model Server Not Used

`transcribe_video.py` and `make transcribe` use a running model server only when
it can produce the same transcript ([details](cli.md#warm-server-routing)). They
transcribe in-process, and log why, when:

- no server answers at `WHISPERBOX_SERVER_URL` (default http://localhost:8000), or `WHISPERBOX_NO_SERVER` is set
- diarization is on for the run (so `make diarize` never uses the server)
- the server runs a different engine or model, or (for Whisper) a different language
- the server was started from older code that doesn't report its engine in `/status`. Restart the server.

The `transcribe`, `stream` and `batch` commands never use the server. To send a
file to the server explicitly, use `client transcribe`, which fails (exit status 1)
with the server's explanation if you ask for a model or language it isn't running.

#### Parakeet Fails on Linux or Intel Macs

Error: `No module named 'mlx'`

Parakeet is Apple Silicon only. Leave `TRANSCRIPTION_ENGINE` unset, or set it to
`whisper`.

#### Batch Exits With Status 1

`batch` exits with status 1 unless every matched file was transcribed: a file
failed, no file matched the pattern, or the run was interrupted. The log lists the
files that failed. Quote the glob so the command, not your shell, expands it.

### Logging

The Whisperbox logs information to the console by default. For detailed DEBUG
logging, pass the `--verbose`/`-v` flag to any of the CLI scripts:

```bash
python -m scripts.transcribe --verbose transcribe path/to/video.mp4
```

## FAQ

### What file formats are supported?

The Whisperbox supports most video and audio formats:
- Video: mov, mp4, avi, mkv, etc. (anything FFmpeg can decode)
- Audio: wav (direct processing), mp3, m4a, aac, etc.

### How much memory does it need?

Memory requirements depend on the model size and file length:
- Tiny/Base models: 2-4GB RAM
- Medium model: 8GB RAM recommended
- Large model: 16GB RAM recommended

For large files, use streaming transcription to reduce memory usage.

### How accurate is the transcription?

Accuracy depends on the model size, engine, audio quality, and language. Larger
Whisper models are more accurate but slower. You can measure accuracy on your own
content with the benchmarking harness (`python -m scripts.benchmark`) — see
`benchmarks/README.md` for how to run it and interpret the WER numbers.

### How accurate is the speaker diarization?

Diarization quality depends on audio quality and the number of speakers; it
degrades as speaker count grows and with overlapping speech.

### Can it transcribe languages other than English?

Yes. Parakeet detects the language itself (it covers about 25 European
languages, English among them). Whisper supports many more languages but transcribes the one
set in `LANGUAGE` (or `--language`). Set it in your `.env` file:

```bash
LANGUAGE=fr  # French
LANGUAGE=es  # Spanish
LANGUAGE=de  # German
# etc.
```

### How can I improve transcription quality?

- Use a larger Whisper model
- Ensure good audio quality (reduce background noise)
- Use a directional microphone when recording
- Process audio files directly when possible
- For non-English content, specify the language explicitly

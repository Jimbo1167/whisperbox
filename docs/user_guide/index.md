# Whisperbox User Guide

Welcome to the Whisperbox user guide. This document provides comprehensive instructions for installing, configuring, and using the Whisperbox tool.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [Command Line Interface](#command-line-interface)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Introduction

Whisperbox is a powerful Python tool for transcribing videos and audio files with speaker diarization. It processes video or audio files, transcribes the speech to text, and identifies different speakers in the conversation.

### Key Features

- **Transcription**: Convert speech to text using OpenAI's Whisper models
- **Speaker Diarization**: Identify different speakers in the audio
- **Multiple Input Formats**: Support for various video and audio formats
- **Multiple Output Formats**: Support for TXT, SRT, VTT, and JSON formats
- **Streaming Transcription**: Process large files with minimal memory usage
- **Batch Processing**: Process multiple files in a single command
- **Caching System**: Improve performance by caching results
- **Progress Reporting**: Track progress with detailed progress bars
- **Model Server**: Run a persistent model server for faster processing
- **Enhanced CLI**: User-friendly command-line interface with subcommands

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video/audio processing)
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

The Whisperbox can be configured using environment variables or a `.env` file. Here are the available configuration options:

### General Settings

- `HF_TOKEN`: Your HuggingFace token for accessing models
- `LANGUAGE`: Target language for transcription (default: en)
- `OUTPUT_FORMAT`: Transcript format (txt, srt, vtt, vtt-voice, json, json3, pretty)

### Model Settings

- `TRANSCRIPTION_ENGINE`: ASR engine, `whisper` or `parakeet` (default: parakeet on Apple Silicon, whisper elsewhere; parakeet is Apple Silicon only)
- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo; default: large-v3-turbo)
- `WHISPER_BEAM_SIZE`: Beam size (default: 5; 1 = greedy, ~2x faster)
- `WHISPER_CPU_THREADS`: CPU threads for Whisper (default: 0 = ctranslate2's default of 4)
- `WHISPER_BATCH_SIZE`: >0 enables batched (parallel chunk) decoding (default: 0 = sequential)
- `PARAKEET_MODEL`: HF model id or local path to MLX weights (default: mlx-community/parakeet-tdt-0.6b-v3)
- `DIARIZATION_MODEL`: Diarization model to use (default: pyannote/speaker-diarization-community-1)
- `FORCE_CPU`: Force CPU for Whisper even when a GPU is available (default: false)

### Feature Toggles

- `INCLUDE_DIARIZATION`: Enable/disable speaker diarization (true/false, default: false)
- `CACHE_ENABLED`: Enable/disable caching system (true/false, default: true)

### Caching Settings

- `CACHE_EXPIRATION`: Cache expiration time in seconds (default: 604800 = 7 days)
- `MAX_CACHE_SIZE`: Maximum cache size in bytes (default: 10GB)

### Timeout Settings

- `AUDIO_TIMEOUT`: Timeout for audio extraction in seconds (default: 300)
- `TRANSCRIBE_TIMEOUT`: Timeout for transcription in seconds (default: 3600)
- `DIARIZE_TIMEOUT`: Timeout for diarization in seconds (default: 3600)

The model server's host and port are set with the `--host`/`--port` CLI flags,
not environment variables (default: localhost:8000).

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

Diarization is off by default. Enable it per run with the flag, or for every run
with `INCLUDE_DIARIZATION=true` in `.env`:

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --diarize
```

### Selecting a Different Whisper Model

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --model medium
```

## Advanced Usage

### Streaming Transcription (Low Memory Usage)

For large files or systems with limited memory, use the streaming transcription:

```bash
python -m scripts.transcribe stream path/to/video.mp4
```

This processes the audio in chunks, significantly reducing memory usage.

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

### Using the Model Server

Start the model server:

```bash
python -m scripts.model_server
```

Check the server status:

```bash
python -m scripts.model_client status
```

Transcribe a file using the server:

```bash
python -m scripts.model_client transcribe path/to/your/video.mp4
```

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
- `--model, -m`: Whisper model size
- `--language, -l`: Language code
- `--diarize, -d`: Include speaker diarization
- `--progress`: Progress mode (pretty, jsonl, none)

### Stream Command

```bash
python -m scripts.transcribe stream [OPTIONS] INPUT_PATH
```

Options:
- Same as transcribe command, minus `--progress`, plus `--words, -w` for word-level timestamps

### Batch Command

```bash
python -m scripts.transcribe batch [OPTIONS] INPUT_PATTERN
```

Options:
- `--output-dir, -o`: Output directory
- `--format, -f`: Output format
- `--model, -m`: Whisper model size
- `--language, -l`: Language code
- `--diarize, -d`: Include speaker diarization
- `--workers, -w`: Number of worker processes (0 = auto)
- `--adaptive, -a`: Adaptive worker pool
- `--streaming, -s`: Use streaming transcription

### Model Server Commands

Start the server:
```bash
python -m scripts.model_server [OPTIONS]
```

Options:
- `--host`: Host to bind the server (default: localhost)
- `--port, -p`: Port to bind the server (default: 8000)
- `--config, -c`: Path to configuration file (default: .env)
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

Error: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

Solution: Install FFmpeg and ensure it's in your PATH.

```bash
# On macOS with Homebrew
brew install ffmpeg

# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On Windows with Chocolatey
choco install ffmpeg
```

#### CUDA Not Available

Warning: `CUDA is not available, using CPU for transcription`

Solution: Install CUDA and the appropriate PyTorch version for your GPU.

#### Out of Memory Errors

Error: `RuntimeError: CUDA out of memory`

Solutions:
- Use a smaller Whisper model
- Use streaming transcription
- Increase your system's swap space
- Use a machine with more GPU memory

#### Diarization Errors

Error: `Could not download pyannote pipeline` or HTTP 401/403 from HuggingFace

Solutions:
- Set a valid `HF_TOKEN` in your `.env` file
- Accept the model's terms on its HuggingFace page for the model set in `DIARIZATION_MODEL` (default: pyannote/speaker-diarization-community-1)

### Logging

The Whisperbox logs information to the console by default. For detailed DEBUG
logging, pass the `--verbose`/`-v` flag to any of the CLI scripts:

```bash
python -m scripts.transcribe --verbose transcribe path/to/video.mp4
```

## FAQ

### What file formats are supported?

The Whisperbox supports most video and audio formats:
- Video: mov, mp4, avi, mkv, etc. (any format supported by MoviePy)
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

Yes, Whisper supports multiple languages. Set the language in your `.env` file:

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
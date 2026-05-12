# Command Line Interface Guide

This guide provides detailed information about the Whisperbox command-line interface (CLI) commands, options, and usage examples.

## Overview

The Whisperbox provides several command-line scripts for different use cases:

1. `transcribe.py` - Unified CLI with subcommands for transcription
2. `model_server.py` - Server for persistent model instances
3. `model_client.py` - Client for interacting with the model server
4. `batch_transcribe.py` - Process multiple files in batch
5. `stream_transcribe.py` - Process files in streaming mode
6. `transcribe_video.py` - Legacy script for basic transcription

## Unified CLI: `transcribe.py`

The `transcribe.py` script provides a unified interface with subcommands for different transcription modes.

```bash
python -m scripts.transcribe [OPTIONS] COMMAND [ARGS]...
```

### Global Options

- `--help`: Show help message and exit
- `--version`: Show version information and exit

### Transcribe Command

The `transcribe` command processes a single video or audio file.

```bash
python -m scripts.transcribe transcribe [OPTIONS] INPUT_PATH
```

#### Options

- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
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

Disable speaker diarization:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --no-diarize
```

Use a different model:
```bash
python -m scripts.transcribe transcribe path/to/video.mp4 -m medium
```

### Stream Command

The `stream` command processes a file in streaming mode to reduce memory usage.

```bash
python -m scripts.transcribe stream [OPTIONS] INPUT_PATH
```

#### Options

Same as the `transcribe` command.

#### Examples

Basic streaming transcription:
```bash
python -m scripts.transcribe stream path/to/video.mp4
```

Streaming with specific options:
```bash
python -m scripts.transcribe stream path/to/video.mp4 -f vtt -m small --no-diarize
```

### Batch Command

The `batch` command processes multiple files in batch.

```bash
python -m scripts.transcribe batch [OPTIONS] INPUT_PATHS...
```

#### Options

- `--output-dir, -o TEXT`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--workers, -w INTEGER`: Number of worker processes (default: auto)
- `--help`: Show help message and exit

#### Examples

Process multiple files:
```bash
python -m scripts.transcribe batch path/to/video1.mp4 path/to/video2.mp4
```

Process all MP4 files in a directory:
```bash
python -m scripts.transcribe batch path/to/directory/*.mp4
```

Specify output directory and format:
```bash
python -m scripts.transcribe batch path/to/directory/*.mp4 -o path/to/output -f srt
```

Limit the number of worker processes:
```bash
python -m scripts.transcribe batch path/to/directory/*.mp4 -w 2
```

## Model Server: `model_server.py`

The `model_server.py` script runs a persistent model server for faster processing.

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

Start the server with a specific port:
```bash
python -m scripts.model_server --port 5001
```

Start the server with verbose logging:
```bash
python -m scripts.model_server --verbose
```

## Model Client: `model_client.py`

The `model_client.py` script interacts with the model server.

```bash
python -m scripts.model_client [OPTIONS] COMMAND [ARGS]...
```

### Commands

- `status`: Check the server status
- `transcribe`: Transcribe a file using the server

### Status Command

```bash
python -m scripts.model_client status [OPTIONS]
```

#### Options

- `--server-url TEXT`: URL of the model server (default: http://localhost:8000)
- `--help`: Show help message and exit

#### Examples

Check the status of the default server:
```bash
python -m scripts.model_client status
```

Check the status of a specific server:
```bash
python -m scripts.model_client status --server-url http://example.com:8000
```

### Transcribe Command

```bash
python -m scripts.model_client transcribe [OPTIONS] INPUT_PATH
```

#### Options

- `--server-url TEXT`: URL of the model server (default: http://localhost:8000)
- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
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
python -m scripts.model_client transcribe path/to/video.mp4 --server-url http://example.com:8000
```

## Batch Transcription: `batch_transcribe.py`

The `batch_transcribe.py` script processes multiple files in batch.

```bash
python -m scripts.batch_transcribe [OPTIONS] INPUT_PATHS...
```

### Options

- `--output-dir, -o TEXT`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--workers, -w INTEGER`: Number of worker processes (default: auto)
- `--help`: Show help message and exit

### Examples

Process multiple files:
```bash
python -m scripts.batch_transcribe path/to/video1.mp4 path/to/video2.mp4
```

Process all MP4 files in a directory:
```bash
python -m scripts.batch_transcribe path/to/directory/*.mp4
```

Specify output directory and format:
```bash
python -m scripts.batch_transcribe path/to/directory/*.mp4 -o path/to/output -f srt
```

## Streaming Transcription: `stream_transcribe.py`

The `stream_transcribe.py` script processes a file in streaming mode to reduce memory usage.

```bash
python -m scripts.stream_transcribe [OPTIONS] INPUT_PATH
```

### Options

- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--help`: Show help message and exit

### Examples

Basic streaming transcription:
```
# Whisperbox Documentation

Welcome to the Whisperbox documentation. This documentation provides comprehensive information about the Whisperbox tool, including installation, configuration, usage, and API reference.

## Table of Contents

### User Guide
- [Getting Started](user_guide/index.md)
- [Command Line Interface](user_guide/cli.md)
- [Configuration](user_guide/index.md#configuration)
- [Transcription Engines](user_guide/index.md#transcription-engines)
- [Speaker Diarization](user_guide/index.md#speaker-diarization)
- [Caching](user_guide/index.md#caching)
- [Model Server and HTTP API](user_guide/cli.md#model-server-model_serverpy)
- [Troubleshooting](user_guide/index.md#troubleshooting)
- [FAQ](user_guide/index.md#faq)

### API Reference
- [API Overview](api/index.md)
- [Transcriber](api/index.md#transcriber)
- [Transcription Service](api/index.md#transcription-service)
- [Audio Processor](api/index.md#audio-processor)
- [Transcription Engine](api/index.md#transcription-engine)
- [Diarization Engine](api/index.md#diarization-engine)
- [Output Formatter](api/index.md#output-formatter)
- [Cache Manager](api/index.md#cache-manager)
- [Progress Reporter](api/index.md#progress-reporter)
- [Configuration](api/index.md#configuration)

### Examples
- [Progress Reporting](examples/progress_reporting.md)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Jimbo1167/whisperbox.git
cd whisperbox

# Set up the environment
make setup

# Configure the environment
cp .env.example .env
```

### Basic Usage

```bash
# Transcribe a video file
python -m scripts.transcribe transcribe path/to/your/video.mp4

# Transcribe with specific options
python -m scripts.transcribe transcribe path/to/your/video.mp4 --format srt --model medium --diarize

# Process multiple files in batch (quote the glob pattern)
python -m scripts.transcribe batch "path/to/directory/*.mp4"

# Use streaming transcription for large files
python -m scripts.transcribe stream path/to/your/large_video.mp4
```

### Using the Model Server

```bash
# Start the model server
python -m scripts.model_server

# Check the server status
python -m scripts.model_client status

# Transcribe a file using the server
python -m scripts.model_client transcribe path/to/your/video.mp4
```

The server listens on `localhost` by default. `make server` binds `0.0.0.0` instead, which exposes it, with no authentication, to your network; see the [CLI guide](user_guide/cli.md#model-server-model_serverpy) before using it.

## Features

- **Transcription**: Convert speech to text with Parakeet (default on Apple Silicon) or Whisper (default elsewhere)
- **Speaker Diarization**: Identify different speakers in the audio (opt-in with `--diarize`)
- **Multiple Input Formats**: Support for various video and audio formats
- **Multiple Output Formats**: txt, pretty, srt, vtt, vtt-voice, json, json3
- **Streaming Transcription**: Process large files with minimal memory usage
- **Batch Processing**: Process multiple files in a single command
- **Caching System**: Improve performance by caching results
- **Progress Reporting**: Progress bars for `stream`, `batch` and the model client; machine-readable JSONL events from `transcribe --progress jsonl`
- **Model Server + Web UI**: Persistent model server with a drag-and-drop browser UI
- **Enhanced CLI**: User-friendly command-line interface with subcommands

## Contributing

Contributions are welcome! Please see the [Contributing Guide](../README.md#contributing) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](../README.md#license) file for details. 
# Whisperbox Documentation

Welcome to the Whisperbox documentation. This documentation provides comprehensive information about the Whisperbox tool, including installation, configuration, usage, and API reference.

## Table of Contents

### User Guide
- [Getting Started](user_guide/index.md)
- [Command Line Interface](user_guide/cli.md)
- [Configuration](user_guide/index.md#configuration)
- [Troubleshooting](user_guide/index.md#troubleshooting)
- [FAQ](user_guide/index.md#faq)

### API Reference
- [API Overview](api/index.md)
- [Transcriber](api/index.md#transcriber)
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
git clone https://github.com/yourusername/whisperbox.git
cd whisperbox

# Set up the environment
make setup

# Configure the environment
cp .env.example .env
```

### Basic Usage

```bash
# Transcribe a video file
./scripts/transcribe.py transcribe path/to/your/video.mp4

# Transcribe with specific options
./scripts/transcribe.py transcribe path/to/your/video.mp4 --format srt --model medium --diarize

# Process multiple files in batch
./scripts/transcribe.py batch path/to/directory/*.mp4

# Use streaming transcription for large files
./scripts/transcribe.py stream path/to/your/large_video.mp4
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

## Features

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

## Contributing

Contributions are welcome! Please see the [Contributing Guide](../README.md#contributing) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](../README.md#license) file for details. 
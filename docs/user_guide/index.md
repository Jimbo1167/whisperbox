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
git clone https://github.com/yourusername/whisperbox.git
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
- `OUTPUT_FORMAT`: Transcript format (txt, srt, vtt, json)

### Model Settings

- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large-v3)
- `DIARIZATION_MODEL`: Diarization model to use (default: pyannote/speaker-diarization@2.1)

### Feature Toggles

- `INCLUDE_DIARIZATION`: Enable/disable speaker diarization (true/false)
- `CACHE_ENABLED`: Enable/disable caching system (true/false)

### Caching Settings

- `CACHE_EXPIRATION`: Cache expiration time in seconds (default: 7 days)
- `MAX_CACHE_SIZE`: Maximum cache size in bytes (default: 10GB)

### Timeout Settings

- `AUDIO_TIMEOUT`: Timeout for audio extraction in seconds
- `TRANSCRIBE_TIMEOUT`: Timeout for transcription in seconds
- `DIARIZE_TIMEOUT`: Timeout for diarization in seconds

### Model Server Settings

- `MODEL_SERVER_HOST`: Host for the model server (default: localhost)
- `MODEL_SERVER_PORT`: Port for the model server (default: 5000)

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

### Enabling/Disabling Speaker Diarization

```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4 --diarize  # Enable diarization
python -m scripts.transcribe transcribe path/to/your/video.mp4 --no-diarize  # Disable diarization
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

Process multiple files at once:

```bash
python -m scripts.transcribe batch path/to/directory/*.mp4
```

Or specify an output directory:

```bash
python -m scripts.transcribe batch path/to/directory/*.mp4 --output-dir path/to/output
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
./scripts/transcribe.py transcribe [OPTIONS] INPUT_PATH
```

Options:
- `--output, -o`: Output file path
- `--format, -f`: Output format (txt, srt, vtt, json)
- `--model, -m`: Whisper model size
- `--language, -l`: Language code
- `--diarize/--no-diarize`: Enable/disable speaker diarization

### Stream Command

```bash
./scripts/transcribe.py stream [OPTIONS] INPUT_PATH
```

Options:
- Same as transcribe command

### Batch Command

```bash
./scripts/transcribe.py batch [OPTIONS] INPUT_PATHS...
```

Options:
- `--output-dir, -o`: Output directory
- `--format, -f`: Output format
- `--model, -m`: Whisper model size
- `--language, -l`: Language code
- `--diarize/--no-diarize`: Enable/disable speaker diarization
- `--workers, -w`: Number of worker processes

### Model Server Commands

Start the server:
```bash
./scripts/model_server.py start [OPTIONS]
```

Options:
- `--host`: Host to bind the server
- `--port`: Port to bind the server
- `--model`: Whisper model size

Client commands:
```bash
./scripts/model_client.py status
./scripts/model_client.py transcribe [OPTIONS] INPUT_PATH
```

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

Error: `ImportError: cannot import name 'Inference' from 'pyannote.audio'`

Solutions:
- Use a compatible version of pyannote.audio
- Set `DIARIZATION_MODEL=pyannote/speaker-diarization@2.1.1` in your `.env` file
- Downgrade libraries: `pip install pyannote.audio==0.0.1 torch==1.10.0`

### Logging

The Whisperbox logs information to the console by default. You can adjust the logging level in your `.env` file:

```bash
LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

For more detailed logging, set the level to DEBUG.

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

Accuracy depends on the model size, audio quality, and language:
- Tiny model: ~80% accuracy on clear English speech
- Base model: ~85% accuracy
- Small model: ~90% accuracy
- Medium model: ~94% accuracy
- Large model: ~96% accuracy

### How accurate is the speaker diarization?

Speaker diarization accuracy depends on audio quality and number of speakers:
- 2-3 speakers: ~90% accuracy
- 4-6 speakers: ~80% accuracy
- 7+ speakers: accuracy decreases significantly

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
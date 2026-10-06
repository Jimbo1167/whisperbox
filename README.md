# Whisperbox

A Python tool for transcribing videos and audio files with speaker diarization. This tool processes video or audio files, transcribes the speech to text, and identifies different speakers in the conversation.

## Current State

- Core transcription flow is shared across CLI and server through `src/service.py`
- The test suite runs on a fresh checkout without models or network access (`make test`; see [Running the tests](#running-the-tests))
- A simple browser UI is available from the local model server for drag-and-drop uploads
- Output files are written to `transcripts/`

## Features

- Support for both video and audio files (WAV processed directly, other formats converted)
- Two ASR engines: Parakeet via `parakeet-mlx` (default on Apple Silicon) and Whisper via `faster-whisper` (default everywhere else)
- Optional speaker diarization via pyannote.audio (`--diarize`)
- Multiple output formats: txt, pretty, srt, vtt, vtt-voice, json, json3
- Streaming transcription for large files with minimal memory usage
- Batch processing of multiple files
- Caching of audio extraction, transcription, and diarization results
- Progress reporting (human-readable or machine-readable JSONL events)
- Model server with browser UI for drag-and-drop uploads
- Hardware acceleration: Parakeet runs on the Apple GPU (MLX); Whisper uses CUDA when available, otherwise the CPU
- Configurable via a `.env` file or environment variables (the environment wins)
- Docker support and an AWS deployment guide

## Architecture

The project has been restructured into a modular architecture:

```
whisperbox/
├── src/
│   ├── audio/         # Audio extraction, conversion, validation
│   ├── transcription/ # ASR engines (Whisper, Parakeet) and streaming
│   ├── diarization/   # Speaker diarization
│   ├── output/        # Output formatting
│   ├── cache/         # Caching system
│   ├── utils/         # Progress reporting, resource monitoring
│   ├── config.py      # Configuration handling
│   ├── service.py     # High-level service facade (shared by CLI and server)
│   ├── transcriber.py # Main orchestrator
├── scripts/           # CLI entry points and model server
├── web/               # Browser UI served by the model server
├── tests/
│   ├── unit/          # Unit tests (fake models; no downloads)
│   ├── fixtures/      # Generated test media
```

### Key Components

- **TranscriptionService**: High-level facade used by both the CLI and the model server
- **Transcriber**: Orchestrates audio processing, transcription, diarization, and output
- **WhisperEngine / ParakeetEngine**: ASR engines selected via `make_asr_engine(config)`
- **DiarizationEngine**: Handles speaker identification
- **OutputFormatter**: Formats transcripts in various output formats
- **CacheManager**: Manages caching of audio, transcription, and diarization results

## Caching System

The new caching system improves performance by:

- Caching extracted audio files to avoid repeated extraction
- Caching transcription results for previously processed files
- Caching diarization results for previously processed files
- Automatically managing cache expiration and size limits

Configure caching in the `.env` file:
```bash
CACHE_ENABLED=true       # Enable/disable caching
CACHE_EXPIRATION=604800  # Cache expiration in seconds (default: 7 days)
MAX_CACHE_SIZE=10737418240  # Maximum cache size in bytes (default: 10GB)
```

The cache lives in `~/.cache/whisperbox/`. See [Caching](docs/user_guide/index.md#caching) for what the cache keys cover and how to clear it.

## Supported Formats

### Input Formats
- Video: mov, mp4, etc. (anything FFmpeg can decode; audio is extracted with the `ffmpeg` binary that `imageio-ffmpeg` provides)
- Audio: wav (direct processing), mp3, m4a, aac (auto-converted to wav)

### Output Formats
- `txt`: Raw text format with speaker labels and fine-grained timestamps
- `pretty`: Readable text format with merged same-speaker paragraphs
- `srt`: SubRip subtitle format with timestamps
- `vtt`: WebVTT format for web video subtitles
- `vtt-voice`: WebVTT with `<v Speaker>` voice tags
- `json`: Structured segments with timestamps and speakers
- `json3`: YouTube auto-caption wire format (consumable by yt-dlp)

## Whisper Models

The system supports different Whisper model sizes, each with its own trade-offs:

| Model | Size | Memory | Speed | Accuracy | Use Case |
|-------|------|---------|--------|-----------|-----------|
| tiny | ~75MB | Minimal | Fastest | Basic | Quick tests, simple audio |
| base | ~150MB | Low | Fast | Good | General use, clear audio |
| small | ~500MB | Medium | Moderate | Better | Professional use |
| medium | ~1.5GB | High | Slower | Very Good | Complex audio |
| large-v3 | ~3GB | Very High | Slowest | Best | Critical accuracy needs |
| large-v3-turbo | ~1.6GB | High | Fast | Very Good | Default — near large-v3 accuracy at much higher speed |

Choose your model in the `.env` file:
```bash
WHISPER_MODEL=large-v3-turbo  # Common options: tiny, base, small, medium, large-v3, large-v3-turbo
```

Any model name faster-whisper accepts works (for example `distil-large-v3`, a Hugging Face repo id, or a local path). `WHISPER_MODEL` only applies when the Whisper engine runs. On Apple Silicon, where Parakeet is the default, set `TRANSCRIPTION_ENGINE=whisper`, or pass `--model` for a single run (an explicit `--model` selects Whisper).

## Requirements

- Python 3.10+ (developed on 3.13; Docker image uses 3.12)
- FFmpeg: the `ffmpeg` binary for audio extraction (`imageio-ffmpeg` uses its bundled copy when it has one, otherwise the one on your `PATH`), and the FFmpeg shared libraries (major version 4–9) for diarization, which decodes audio through torchcodec
- PyTorch — not in `requirements.txt`; installed separately via `make install-torch` (CPU wheels) or your own CUDA build
- Other dependencies listed in requirements.txt

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/Jimbo1167/whisperbox.git
cd whisperbox
```

2. Set up the environment and install dependencies:
```bash
make setup  # Creates venv and installs all dependencies
```
   Or manually:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

### Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/Jimbo1167/whisperbox.git
cd whisperbox
```

2. Build and run the Docker container:
```bash
make docker-build
make docker-run
```

3. For more details, see the [Docker Deployment Guide](./docs/deployment/docker.md).

## Configuration

Edit the `.env` file to configure the settings below. Every entry point that uses these settings reads the project's `.env`, wherever you run it from. Variables already set in your environment take precedence, whether exported or set inline (`INCLUDE_DIARIZATION=true python transcribe_video.py ...`). The full list is in the [user guide](docs/user_guide/index.md#configuration).

- `HF_TOKEN`: Your HuggingFace token for accessing models
- `TRANSCRIPTION_ENGINE`: ASR engine to use (`whisper` or `parakeet`; default `parakeet` on Apple Silicon, `whisper` elsewhere). See [Transcription engines](#transcription-engines) below.
- `WHISPER_MODEL`: Whisper model (any faster-whisper name, e.g. tiny, base, small, medium, large-v3, large-v3-turbo; default: large-v3-turbo)
- `WHISPER_BEAM_SIZE`, `WHISPER_CPU_THREADS`, `WHISPER_BATCH_SIZE`: Whisper speed/accuracy tuning (see `.env.example`)
- `PARAKEET_MODEL`: HF model id or local path to MLX-format weights (default `mlx-community/parakeet-tdt-0.6b-v3`). Only used when the Parakeet engine runs.
- `LANGUAGE`: Target language for Whisper (default: en). Parakeet detects the language itself.
- `OUTPUT_FORMAT`: Transcript format (txt, pretty, srt, vtt, vtt-voice, json, json3)
- `INCLUDE_DIARIZATION`: Default diarization setting for the model server and `transcribe_video.py` / `make transcribe` (default: false). The `transcribe`, `stream`, `batch` and `client transcribe` commands diarize only when you pass `--diarize`.
- `DIARIZATION_MODEL`: pyannote model id (default: pyannote/speaker-diarization-community-1)
- `FORCE_CPU`: Run Whisper and pyannote diarization on the CPU even when a GPU is available (default: false; `.env.example` sets it to true). No effect on Parakeet.
- `CACHE_ENABLED`: Enable/disable caching system
- `CACHE_EXPIRATION`: Cache expiration time in seconds
- `MAX_CACHE_SIZE`: Maximum cache size in bytes
- Various timeout settings

## Transcription engines

Two ASR engines are selectable via `TRANSCRIPTION_ENGINE`. When it is unset, the
default is `parakeet` on Apple Silicon (falling back to `whisper` if `parakeet-mlx`
isn't installed) and `whisper` everywhere else.

### `whisper` (default off Apple Silicon)

`faster-whisper` running `large-v3-turbo` by default. Works on macOS, Linux, and Docker. Supports 99+ languages. Streaming and async streaming are supported.

### `parakeet` (Apple Silicon only, default there)

NVIDIA Parakeet-TDT-0.6B-v3 via [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx). On macOS arm64, this is roughly an order of magnitude faster than Whisper on CPU and produces lower WER on the Open ASR Leaderboard for English / ~25 European languages. Batch only — no streaming.

To force Whisper on Apple Silicon instead:

```bash
export TRANSCRIPTION_ENGINE=whisper
```

#### First-run model download

On first use, parakeet-mlx auto-downloads `mlx-community/parakeet-tdt-0.6b-v3` (~600MB) to `~/.cache/huggingface/`. To use a different MLX checkpoint or a pre-downloaded local copy:

```bash
# HuggingFace id
export PARAKEET_MODEL=mlx-community/parakeet-tdt-0.6b-v3

# Or absolute local path to an MLX checkpoint
export PARAKEET_MODEL=/path/to/local/mlx-checkpoint
```

#### Caveats

- **Apple Silicon only.** The `parakeet-mlx` dependency in `requirements.txt` carries a platform marker, so Linux, Docker and Intel macOS installs skip it. Leave `TRANSCRIPTION_ENGINE` unset (or `whisper`) there: an explicit `parakeet` fails when the model loads (`No module named 'mlx'`), and `scripts/transcribe_video.py` rejects it up front.
- **`FORCE_CPU` has no effect on Parakeet.** MLX runs on Apple Silicon with no equivalent setting; if `FORCE_CPU=true` is set with `engine=parakeet`, a warning is logged and the flag is ignored for transcription. It still applies to diarization.
- **Streaming is Whisper-only.** `stream` and `batch --streaming` always run Whisper, whatever `TRANSCRIPTION_ENGINE` says.
- **Language and model.** Parakeet detects the language itself, so `LANGUAGE` and `--language` are ignored (an explicit `--language` logs a warning). Its model is set by `PARAKEET_MODEL`. `--model` names a Whisper model and selects Whisper for that run.
- **Handy weights are not compatible.** Handy ships INT8 ONNX weights; `parakeet-mlx` requires MLX-format weights. Users wanting to reuse Handy's weights would need a different runtime (e.g. `onnx-asr`) — out of scope here.

The [engine comparison](docs/user_guide/index.md#transcription-engines) in the user guide lists the differences side by side. To pick an engine for a single run, pass `--engine whisper|parakeet` to `transcribe` or `batch`.

## Usage

### Recommended CLI

Process a video file:
```bash
python -m scripts.transcribe transcribe path/to/your/video.mp4
```

Process an audio file (WAV files are processed directly):
```bash
python -m scripts.transcribe transcribe path/to/your/audio.wav
```

Enable diarization for a run. It is off unless you pass `--diarize`; `INCLUDE_DIARIZATION` in `.env` does not turn it on for this command. Diarization needs `HF_TOKEN` and working FFmpeg libraries (see [Speaker diarization](docs/user_guide/index.md#speaker-diarization)):
```bash
python -m scripts.transcribe transcribe path/to/your/audio.wav --diarize
```

### Web UI

Start the local server:

```bash
python -m scripts.model_server
```

Then open `http://localhost:8000` in your browser and drag a file onto the page.

`make server` starts the same server bound to `0.0.0.0`, so other machines on your network can reach it. The server has no authentication, and its JSON `/transcribe` endpoint transcribes any file path on the server's filesystem. Only use `make server` on a trusted network. The [CLI guide](docs/user_guide/cli.md#model-server-model_serverpy) documents the server and its HTTP API.

### Streaming Transcription (Low Memory Usage)

For large files or systems with limited memory, use the streaming transcription:

```bash
python -m scripts.stream_transcribe path/to/video.mp4
```

This processes the audio in chunks, significantly reducing memory usage. Streaming always uses the Whisper engine.

For word-level timestamps, add `--words` with JSON output:

```bash
python -m scripts.stream_transcribe path/to/audio.wav --words -f json -o transcript.json
```

### With Speaker Diarization

```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --diarize
```

Or with streaming:

```bash
python -m scripts.stream_transcribe path/to/video.mp4 --diarize
```

### Specify Output Format

```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --format srt
```

### Specify Output Location

```bash
python -m scripts.transcribe transcribe path/to/your/file.mp4 -o path/to/output.txt
```

### Batch Processing

```bash
python -m scripts.transcribe batch "path/to/directory/*.mp4" -f srt
```

Quote the glob pattern; the command takes one pattern. It exits with status 1 unless every matched file was transcribed.

## Development

### Using Make Commands

The project includes several make commands to simplify common operations:

```bash
make help         # Show all available commands
make setup        # Create virtual environment and install dependencies
make venv         # Create virtual environment only
make install      # Install dependencies into existing virtual environment
make test         # Run tests
make clean        # Remove Python cache files and temporary files
make docker-build # Build Docker image
make docker-run   # Run Docker container
make docker-stop  # Stop Docker container
make docker-clean # Clean Docker resources
```

### Manual Development Setup

For development work without using Make:
```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

### Running the tests

`make test` (or `pytest`; `pytest.ini` sets the test paths) runs the suite in a few seconds. It uses fake models, so it needs no model downloads, network access or `HF_TOKEN`, and it never reads your `.env` or touches `~/.cache`. One test, marked `env`, checks that this machine's torchcodec can decode with the system FFmpeg. Skip it where the FFmpeg libraries aren't installed:

```bash
pytest -m "not env"
```

## Performance Considerations

- **Streaming Mode**: Use streaming transcription for large files to reduce memory usage
- **Caching**: Enable caching for improved performance when processing the same files multiple times
- **Model Selection**: Choose the appropriate model size based on your accuracy needs and hardware capabilities
- **Hardware Acceleration**: On Apple Silicon, Parakeet runs on the GPU through MLX. Whisper runs on CUDA (NVIDIA) when available and otherwise on the CPU; it does not use Apple's GPU (MPS)
- **Memory Usage**: Large files may require significant memory, especially with larger models
- **Containerization**: Use Docker for consistent deployment across environments
- **Cloud Deployment**: Deploy to AWS for scalable processing of large volumes of media

## Known Issues

- Large video files may require significant memory
- Some hardware acceleration features require specific hardware/drivers
- Non-WAV audio files will be converted to WAV before processing
- Diarization requires a HuggingFace token (`HF_TOKEN`) with access to the pyannote model set in `DIARIZATION_MODEL`, and FFmpeg shared libraries that torchcodec can load (see [Troubleshooting](docs/user_guide/index.md#troubleshooting))

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

Copyright (c) 2025 James Schindler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- OpenAI's Whisper (via faster-whisper) and NVIDIA's Parakeet (via parakeet-mlx) for transcription
- Pyannote.audio for speaker diarization
- FFmpeg (via imageio-ffmpeg) for audio extraction

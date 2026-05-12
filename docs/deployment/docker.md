# Docker Deployment Guide

This guide explains how to deploy the Whisperbox application using Docker for better scalability and isolation.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- Basic understanding of Docker concepts

## Quick Start

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd whisperbox
```

2. Create an `.env` file from the example:

```bash
cp .env.example .env
```

3. Edit the `.env` file to set your configuration values, especially:
   - `HF_TOKEN` (if you plan to use speaker diarization)
   - `WHISPER_MODEL` (default is "base")
   - `FORCE_CPU` (set to "false" if you want to use GPU)

4. Build and start the Docker container:

```bash
make docker-build
make docker-run
```

The transcription server will be available at http://localhost:8000.

## Configuration Options

You can configure the Docker deployment through environment variables in your `.env` file or by overriding them in the `docker-compose.yml` file:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| WHISPER_MODEL | Whisper model size (tiny, base, small, medium, large) | base |
| OUTPUT_FORMAT | Output format (txt, srt, vtt, json) | txt |
| INCLUDE_DIARIZATION | Enable speaker diarization | false |
| FORCE_CPU | Force CPU usage instead of GPU | true |
| CACHE_ENABLED | Enable caching of results | true |
| HF_TOKEN | HuggingFace token for diarization | (required for diarization) |

## Using the Transcription Server

Once the server is running, you can use it to transcribe audio and video files using:

1. The model client script:

```bash
python scripts/model_client.py --server http://localhost:8000 transcribe path/to/your/file.mp4
```

2. Direct HTTP requests:

```bash
curl -F "file=@path/to/your/file.mp4" http://localhost:8000/transcribe
```

3. JSON API (for files on the server):

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/server/file.mp4"}'
```

## Docker Commands

The project includes several Makefile targets for Docker management:

- `make docker-build`: Build the Docker image
- `make docker-run`: Start the Docker container
- `make docker-stop`: Stop the Docker container
- `make docker-clean`: Remove Docker containers and perform cleanup

## Scaling with Docker Compose

For increased capacity, you can scale the service:

```bash
docker-compose up -d --scale transcription-server=3
```

Note: This requires a load balancer setup to distribute requests.

## GPU Support

To use GPU with Docker:

1. Ensure you have nvidia-docker installed
2. Modify the `docker-compose.yml`:

```yaml
services:
  transcription-server:
    # other settings...
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      - FORCE_CPU=false
      # other env variables...
```

## Troubleshooting

- If you encounter issues with model loading, ensure you have enough memory available
- For GPU issues, verify that nvidia-docker is properly installed
- Check logs with `docker-compose logs transcription-server`
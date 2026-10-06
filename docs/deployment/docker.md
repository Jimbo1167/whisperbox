# Docker Deployment Guide

This guide explains how to run the Whisperbox model server (HTTP API and web UI) in Docker.

> **Warning:** The server has no authentication. In the container it listens on all interfaces, and `docker-compose.yml` publishes port 8000 on all of the host's interfaces, so anyone who can reach the port can upload files, download transcripts by name, and use the JSON `/transcribe` endpoint, which transcribes any path inside the container. Run it only on a trusted network, or publish the port on the host's loopback interface only (`"127.0.0.1:8000:8000"`) and put an authenticating reverse proxy in front of it.

## Prerequisites

- Docker installed on your system
- Docker Compose 2.24 or later installed on your system (the Makefile targets run `docker-compose`)
- Basic understanding of Docker concepts

## Quick Start

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Jimbo1167/whisperbox.git
cd whisperbox
```

2. Create an `.env` file from the example. `docker-compose.yml` passes everything in it to the container (`env_file`); without one, the image's defaults apply:

```bash
cp .env.example .env
```

3. Edit the `.env` file to set your configuration values, especially:
   - `HF_TOKEN` (if you plan to use speaker diarization)
   - `WHISPER_MODEL` (default is "large-v3-turbo")
   - `FORCE_CPU` (`.env.example` sets "true"; set "false" to use an NVIDIA GPU, see [GPU Support](#gpu-support))
   - `TRANSCRIPTION_ENGINE`: leave it unset or set it to "whisper". The container runs Linux, so the Parakeet engine (Apple Silicon only) is not available.

4. Build and start the Docker container:

```bash
make docker-build
make docker-run
```

The transcription server will be available at http://localhost:8000, with the web UI at that address and the HTTP API described below. On startup the server downloads the Whisper model if needed and loads it before it accepts connections, which can take several minutes; follow it with `docker-compose logs -f transcription-server`. Transcripts are written to `transcripts/` in the project directory, which is mounted into the container.

The downloaded models are stored inside the container (under `/root/.cache`), so they are downloaded again whenever the container is recreated, for example by `make docker-stop` followed by `make docker-run`. Mount a volume at `/root/.cache` to keep them.

## Configuration Options

Every variable in your `.env` file is passed to the container, as for a local run. Variables it doesn't set fall back to the defaults built into the image (below) or into the code:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| WHISPER_MODEL | Whisper model: any name faster-whisper accepts (e.g. tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3) | large-v3-turbo |
| OUTPUT_FORMAT | Default output format (txt, srt, vtt, vtt-voice, json, json3, pretty) | txt |
| INCLUDE_DIARIZATION | Diarize requests that don't say whether to | false |
| FORCE_CPU | Run Whisper and pyannote on the CPU even when a GPU is available | true |
| CACHE_ENABLED | Enable caching of results | true |
| HF_TOKEN | HuggingFace token for diarization | (required for diarization) |

The model and language are fixed when the server starts: requests can't switch them. See the [Configuration section of the user guide](../user_guide/index.md#configuration) for the other variables.

## Using the Transcription Server

Once the server is running, you can use it to transcribe audio and video files using:

1. The web UI at http://localhost:8000: drop a file, pick a format, and download the transcript.

2. The model client script (run on the host, from a checkout with the project's dependencies installed). It uploads the file, waits for the job to finish, and prints the segments; `-o` saves the transcript in the `-f` format (default: the server's `OUTPUT_FORMAT`):

```bash
python scripts/model_client.py --server http://localhost:8000 transcribe path/to/your/file.mp4
python scripts/model_client.py --server http://localhost:8000 transcribe path/to/your/file.mp4 -f srt -o file.srt
python scripts/model_client.py --server http://localhost:8000 status
```

3. Direct HTTP requests. An upload starts a job and returns `202` with a `job_id`; poll the job until its `status` is `completed` or `failed`:

```bash
curl -F "file=@path/to/your/file.mp4" -F "format=srt" http://localhost:8000/api/transcribe
# {"job_id": "<job-id>", "status": "queued", "progress": 0.1, "message": "Upload complete"}

curl http://localhost:8000/api/jobs/<job-id>
# While it runs:  {"status": "running", "progress": 0.2, "message": "Transcribing audio", ...}
# When finished:  {"status": "completed", "result": {"segments": [...], "preview_text": "...",
#                  "download_url": "/transcripts/file.srt", "diarization_error": null, ...}, ...}
# On failure:     {"status": "failed", "error": "...", ...}

curl -o file.srt http://localhost:8000/transcripts/file.srt
```

Form fields: `file` (required); `format` (one of the seven formats above; default: the server's `OUTPUT_FORMAT`); `diarize` (`true` or `false`; default: `INCLUDE_DIARIZATION`); `model` and `language` (optional; the request is refused with `409` unless they match what the server runs). Uploads are limited to 500 MB (`413`). `diarization_error` is set when diarization failed after transcription succeeded; the transcript is then saved without speaker labels.

For a synchronous request that returns only the text (no timestamps or speakers), use `/api/transcribe-sync` (fields `file` and optional `diarize`):

```bash
curl -F "file=@path/to/your/file.mp4" http://localhost:8000/api/transcribe-sync
# {"text": "..."}
```

4. JSON API (for files inside the container, such as a mounted volume). This request is synchronous: it returns when transcription finishes, with each segment as `[start, end, text, speaker]`, and writes no transcript file:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/in/container/file.mp4"}'
# {"segments": [[0.0, 2.5, "Hello.", ""], ...], "processing_time": 12.3, "diarization_error": null}
```

Add `"include_diarization": true` or `false` to override the server's default.

`GET /health` returns `{"status": "ok"}`, and `GET /api/status` reports the engine, model, language, device, and request statistics.

## Docker Commands

The project includes several Makefile targets for Docker management:

- `make docker-build`: Build the Docker image
- `make docker-run`: Start the Docker container
- `make docker-stop`: Stop and remove the Docker container
- `make docker-clean`: Run `docker-compose down -v` (removes the container and any volumes Compose created; `transcripts/` on the host is kept), then `docker system prune -f`, which also removes stopped containers, unused networks, dangling images and build cache belonging to other projects on this machine

## Running More Than One Server

`docker-compose up --scale transcription-server=3` does not work with the provided `docker-compose.yml`: it publishes the fixed host port 8000, which only one container can bind. Beyond that, each server keeps its jobs in memory. A job ID is known only to the server that accepted the upload, and jobs are lost when that server restarts.

To run several servers, give each its own host port (or none) and put a load balancer in front that sends an upload and all of its `/api/jobs/<id>` polls to the same server (sticky sessions), or use only the synchronous endpoints (`/api/transcribe-sync` and the JSON `/transcribe`). Each server loads its own copy of the model and transcribes one file at a time, so more servers means more files in parallel and more memory.

## GPU Support

To use GPU with Docker:

1. Ensure you have nvidia-docker installed
2. Set `FORCE_CPU=false` in your `.env`
3. Modify the `docker-compose.yml`:

```yaml
services:
  transcription-server:
    # other settings...
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

## Troubleshooting

- If `docker-compose` rejects the `env_file` entry in `docker-compose.yml`, upgrade to Docker Compose 2.24 or later
- If you encounter issues with model loading, ensure you have enough memory available
- For GPU issues, verify that nvidia-docker is properly installed
- Check logs with `docker-compose logs transcription-server`

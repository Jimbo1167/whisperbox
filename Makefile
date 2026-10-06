.PHONY: setup clean install test venv install-torch \
       transcribe diarize server client batch \
       docker-build docker-run docker-stop docker-clean help

# Configuration
VENV := . venv/bin/activate &&
FILE ?=
OUTPUT ?=
FORMAT ?= txt
MODEL ?= large-v3-turbo
LANGUAGE ?= en
PORT ?= 8000
WORKERS ?= 0

# ── Setup ──────────────────────────────────────────────

venv:
	python3.11 -m venv venv
	$(VENV) pip install --upgrade pip

install-torch:
	$(VENV) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

install: venv install-torch
	$(VENV) pip install -r requirements.txt
	$(VENV) pip install -r requirements-dev.txt

setup: clean venv install

# ── Transcription ──────────────────────────────────────

# Transcribe a single file (uses .env config)
#   make transcribe FILE=video.mp4
#   make transcribe FILE=video.mp4 OUTPUT=out.txt
transcribe:
ifndef FILE
	$(error FILE is required. Usage: make transcribe FILE=video.mp4)
endif
	$(VENV) python transcribe_video.py "$(FILE)" $(if $(OUTPUT),-o "$(OUTPUT)")

# Transcribe with speaker diarization
#   make diarize FILE=interview.mp4
#   make diarize FILE=interview.mp4 OUTPUT=out.txt
diarize:
ifndef FILE
	$(error FILE is required. Usage: make diarize FILE=video.mp4)
endif
	$(VENV) INCLUDE_DIARIZATION=true python transcribe_video.py "$(FILE)" $(if $(OUTPUT),-o "$(OUTPUT)")

# Batch transcribe multiple files
#   make batch FILE="videos/*.mp4"
#   make batch FILE="videos/*.mp4" WORKERS=4
batch:
ifndef FILE
	$(error FILE is required. Usage: make batch FILE="videos/*.mp4")
endif
	$(VENV) python scripts/transcribe.py batch "$(FILE)" --workers $(WORKERS) --format $(FORMAT)

# ── Server ─────────────────────────────────────────────

# Start the model server and browser UI
#   make server
#   make server PORT=9000
server:
	$(VENV) python scripts/model_server.py --host 0.0.0.0 --port $(PORT)

# Check server status
client-status:
	$(VENV) python scripts/transcribe.py client status

# Transcribe via the running server
#   make client-transcribe FILE=audio.mp3
client-transcribe:
ifndef FILE
	$(error FILE is required. Usage: make client-transcribe FILE=audio.mp3)
endif
	$(VENV) python scripts/transcribe.py client transcribe "$(FILE)"

# ── Testing & Cleanup ─────────────────────────────────

test:
	$(VENV) python -m pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type f -name ".coverage" -delete

# ── Docker ─────────────────────────────────────────────

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker system prune -f

# ── Help ───────────────────────────────────────────────

help:
	@echo ""
	@echo "Usage:"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup              Create venv and install all dependencies"
	@echo "    make install            Install dependencies into existing venv"
	@echo ""
	@echo "  Transcription:"
	@echo "    make transcribe FILE=video.mp4              Transcribe a file"
	@echo "    make transcribe FILE=video.mp4 OUTPUT=o.txt Transcribe with custom output"
	@echo "    make diarize FILE=video.mp4                 Transcribe with speaker diarization"
	@echo "    make batch FILE=\"videos/*.mp4\"               Batch transcribe multiple files"
	@echo "    make batch FILE=\"*.mp4\" WORKERS=4            Batch with parallel workers"
	@echo ""
	@echo "  Server:"
	@echo "    make server                     Start the model server and web UI on 0.0.0.0:8000"
	@echo "                                    (reachable from your network, no authentication)"
	@echo "    make server PORT=9000           Start on a custom port"
	@echo "    make client-status              Check server status"
	@echo "    make client-transcribe FILE=f   Transcribe via running server"
	@echo ""
	@echo "  Testing & Maintenance:"
	@echo "    make test               Run tests"
	@echo "    make clean              Remove cache/temp files"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build       Build Docker image"
	@echo "    make docker-run         Run Docker container"
	@echo "    make docker-stop        Stop Docker container"
	@echo "    make docker-clean       Clean Docker resources"
	@echo ""

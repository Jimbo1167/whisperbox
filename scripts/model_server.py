#!/usr/bin/env python3
"""
Model Server for persistent Whisper model instances.

This script implements a simple server that keeps Whisper models loaded in memory
and provides an API for transcription requests, avoiding the overhead of loading
the model for each transcription.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import tempfile
import uuid
from email.parser import BytesParser
from email.policy import default as default_policy
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.service import TranscriptionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Maximum upload size: 500MB
MAX_UPLOAD_SIZE = 500 * 1024 * 1024
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
TRANSCRIPTS_ROOT = Path.cwd() / "transcripts"

# Global variables
config = None
service = None
stats_lock = threading.Lock()
stats = {
    "requests": 0,
    "successful": 0,
    "failed": 0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}
jobs_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}


def _update_stats(**kwargs):
    """Thread-safe stats update."""
    with stats_lock:
        for key, value in kwargs.items():
            if key in stats:
                stats[key] += value


def _parse_multipart(content_type, body):
    """Parse multipart form data without the deprecated cgi module.

    Returns:
        Dict mapping field names to (filename, data) tuples for file fields
        or (None, value_string) for regular fields.
    """
    # Build a valid MIME message for the email parser
    header = f"Content-Type: {content_type}\r\n\r\n".encode()
    msg = BytesParser(policy=default_policy).parsebytes(header + body)

    fields = {}
    if msg.is_multipart():
        for part in msg.iter_parts():
            disposition = part.get_content_disposition()
            name = part.get_param('name', header='content-disposition')
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True)
            if filename:
                fields[name] = (filename, payload)
            else:
                fields[name] = (None, payload.decode('utf-8', errors='replace'))
    return fields


def _set_job_state(job_id: str, **updates):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def _create_job(filename: str, output_format: str, include_diarization: bool) -> str:
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "filename": filename,
            "format": output_format,
            "include_diarization": include_diarization,
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
    return job_id


def _run_transcription_job(
    job_id: str,
    temp_path: str,
    original_filename: str,
    output_format: str,
    include_diarization: bool,
):
    start_time = time.time()

    def progress_callback(message: str, progress: float):
        _set_job_state(
            job_id,
            status="running",
            message=message,
            progress=max(0.0, min(progress, 1.0)),
        )

    try:
        logger.info("Processing uploaded file for job %s: %s", job_id, temp_path)
        progress_callback("Starting", 0.05)

        # Diarization is passed per-request; mutating the shared config here
        # raced with concurrently submitted jobs.
        result = service.transcribe_file(
            temp_path,
            output_format=output_format,
            progress_callback=progress_callback,
            include_diarization=include_diarization,
        )

        processing_time = time.time() - start_time
        _update_stats(successful=1, total_processing_time=processing_time)

        output_path = str(TRANSCRIPTS_ROOT / f"{Path(original_filename).stem}.{output_format}")
        os.replace(result["output_file"], output_path)
        logger.info("Saved transcript to %s", output_path)

        _set_job_state(
            job_id,
            status="completed",
            progress=1.0,
            message="Completed",
            result={
                "segments": result["segments"],
                "preview_text": result["preview_text"],
                "output_format": result["output_format"],
                "processing_time": processing_time,
                "output_file": output_path,
                "download_url": f"/transcripts/{Path(output_path).name}",
            },
        )
    except Exception as e:
        logger.error("Error processing job %s: %s", job_id, str(e))
        _update_stats(failed=1)
        _set_job_state(
            job_id,
            status="failed",
            message="Failed",
            error=str(e),
        )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning("Failed to clean up temp file: %s", temp_path)


class ModelRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the model server."""

    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_file_response(self, file_path: Path, content_type: str):
        """Send a static file response."""
        if not file_path.exists() or not file_path.is_file():
            self._send_error("File not found", 404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

        with open(file_path, "rb") as file_handle:
            self.wfile.write(file_handle.read())

    def _send_error(self, message: str, status: int = 400):
        """Send an error response."""
        self._send_json_response({"error": message}, status)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        # Health check endpoint
        if path == '/health':
            self._send_json_response({"status": "ok"})
            return

        # Handle status endpoint
        if path in ('/status', '/api/status'):
            # Calculate uptime
            with stats_lock:
                uptime = time.time() - stats["start_time"]
                avg_time = 0.0
                if stats["successful"] > 0:
                    avg_time = stats["total_processing_time"] / stats["successful"]
                stats_snapshot = {
                    "requests": stats["requests"],
                    "successful": stats["successful"],
                    "failed": stats["failed"],
                    "avg_processing_time": avg_time
                }

            # Get model info
            device = "CPU" if config.force_cpu else "GPU (if available)"
            model_info = {
                "model_size": config.whisper_model_size,
                "language": config.language,
                "device": device
            }

            self._send_json_response({
                "status": "running",
                "uptime": uptime,
                "model": model_info,
                "stats": stats_snapshot
            })
            return

        if path.startswith("/api/jobs/"):
            job_id = Path(path).name
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self._send_error("Job not found", 404)
                return
            self._send_json_response(job)
            return

        if path == "/":
            self._send_file_response(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return

        if path == "/app.js":
            self._send_file_response(WEB_ROOT / "app.js", "application/javascript; charset=utf-8")
            return

        if path == "/app.css":
            self._send_file_response(WEB_ROOT / "app.css", "text/css; charset=utf-8")
            return

        if path.startswith("/transcripts/"):
            filename = Path(path.removeprefix("/transcripts/")).name
            self._send_file_response(TRANSCRIPTS_ROOT / filename, "text/plain; charset=utf-8")
            return

        # Handle unknown endpoints
        self._send_error("Unknown endpoint", 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urllib.parse.urlparse(self.path)

        # Handle transcribe endpoint
        if parsed_path.path in ('/transcribe', '/api/transcribe'):
            # Check content type
            content_type = self.headers.get('Content-Type', '')

            # Enforce upload size limit
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_UPLOAD_SIZE:
                self._send_error(
                    f"Upload too large ({content_length} bytes). Max: {MAX_UPLOAD_SIZE} bytes.",
                    413
                )
                return

            # Handle multipart form data (file upload)
            if content_type.startswith('multipart/form-data'):
                self._handle_multipart_transcribe(content_type, content_length)

            # Handle JSON request (legacy)
            else:
                self._handle_json_transcribe(content_length)

        elif parsed_path.path == '/api/transcribe-sync':
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > MAX_UPLOAD_SIZE:
                self._send_error(
                    f"Upload too large ({content_length} bytes). Max: {MAX_UPLOAD_SIZE} bytes.",
                    413
                )
                return
            if not content_type.startswith('multipart/form-data'):
                self._send_error("Expected multipart/form-data")
                return
            self._handle_sync_transcribe(content_type, content_length)

        else:
            self._send_error(f"Unknown endpoint: {parsed_path.path}", 404)

    def _handle_multipart_transcribe(self, content_type, content_length):
        """Handle multipart file upload transcription."""
        if content_length == 0:
            self._send_error("Empty request body")
            return

        body = self.rfile.read(content_length)
        fields = _parse_multipart(content_type, body)

        # Check if file was uploaded
        if 'file' not in fields:
            self._send_error("No file uploaded")
            return

        filename, file_data = fields['file']
        if not file_data:
            self._send_error("Empty file")
            return

        # Get the original filename
        original_filename = os.path.basename(filename) if filename else "uploaded_file"

        # Get output format from form fields (don't mutate global config)
        output_format = config.output_format
        if 'format' in fields:
            _, fmt_value = fields['format']
            if fmt_value in ('txt', 'srt', 'vtt', 'json', 'pretty'):
                output_format = fmt_value

        include_diarization = config.include_diarization
        if 'diarize' in fields:
            _, diarize_value = fields['diarize']
            include_diarization = diarize_value.lower() in ("true", "1", "yes", "on")

        # Save the uploaded file to a temporary location
        temp_path = None
        try:
            suffix = Path(original_filename).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                temp_file.write(file_data)

            _update_stats(requests=1)

            job_id = _create_job(original_filename, output_format, include_diarization)
            _set_job_state(job_id, status="running", progress=0.1, message="Upload complete")

            threading.Thread(
                target=_run_transcription_job,
                args=(job_id, temp_path, original_filename, output_format, include_diarization),
                daemon=True,
            ).start()
            temp_path = None

            self._send_json_response({
                "job_id": job_id,
                "status": "queued",
                "progress": 0.1,
                "message": "Upload complete",
            }, status=202)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            _update_stats(failed=1)
            self._send_error(f"Error processing request: {str(e)}")
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.warning(f"Failed to clean up temp file: {temp_path}")

    def _handle_sync_transcribe(self, content_type, content_length):
        """Handle synchronous multipart file upload transcription."""
        if content_length == 0:
            self._send_error("Empty request body")
            return

        body = self.rfile.read(content_length)
        fields = _parse_multipart(content_type, body)

        if 'file' not in fields:
            self._send_error("No file uploaded")
            return

        filename, file_data = fields['file']
        if not file_data:
            self._send_error("Empty file")
            return

        original_filename = os.path.basename(filename) if filename else "uploaded_file"

        temp_path = None
        try:
            suffix = Path(original_filename).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                temp_file.write(file_data)

            _update_stats(requests=1)
            logger.info("Sync transcription of uploaded file: %s", temp_path)

            result = service.transcribe_existing_audio(temp_path)
            segments = result.get("segments", [])
            text = " ".join(seg[2] for seg in segments)

            _update_stats(successful=1)
            self._send_json_response({"text": text})

        except Exception as e:
            logger.error("Error in sync transcription: %s", str(e))
            _update_stats(failed=1)
            self._send_json_response({"error": str(e)}, status=500)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.warning("Failed to clean up temp file: %s", temp_path)

    def _handle_json_transcribe(self, content_length):
        """Handle JSON-based transcription request."""
        if content_length == 0:
            self._send_error("Empty request body")
            return

        # Parse request body
        try:
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_error("Invalid JSON")
            return
        except UnicodeDecodeError:
            self._send_error("Invalid encoding, expected UTF-8")
            return

        input_path = request_data.get("audio_path") or request_data.get("input_path")
        if not input_path:
            self._send_error("Missing required field: audio_path")
            return

        if not os.path.exists(input_path):
            self._send_error(f"Audio file not found: {input_path}", 404)
            return

        _update_stats(requests=1)

        try:
            start_time = time.time()
            logger.info(f"Processing audio file: {input_path}")

            result = service.transcribe_existing_audio(input_path)
            processing_time = time.time() - start_time
            _update_stats(successful=1, total_processing_time=processing_time)

            self._send_json_response({
                "segments": result["segments"],
                "processing_time": processing_time
            })

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            _update_stats(failed=1)
            self._send_error(f"Error processing request: {str(e)}")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


def initialize_models(config_path: Optional[str] = None):
    """Initialize the models and processors."""
    global config, service

    logger.info("Initializing models...")

    # Load configuration
    config = Config(config_path or ".env")
    service = TranscriptionService(config=config, preload_models=True)

    logger.info("Models initialized successfully")


def run_server(host: str, port: int):
    """Run the model server."""
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, ModelRequestHandler)

    logger.info(f"Starting model server on {host}:{port}")
    device = "CPU" if config.force_cpu else "GPU (if available)"
    logger.info(f"Model: {config.whisper_model_size}, Device: {device}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        httpd.server_close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run a model server for persistent Whisper model instances"
    )
    parser.add_argument("--host", default="localhost",
                       help="Host to bind the server to (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="Port to bind the server to (default: 8000)")
    parser.add_argument("--config", "-c",
                       help="Path to configuration file (default: .env)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize models
    initialize_models(args.config)

    # Run server
    run_server(args.host, args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())

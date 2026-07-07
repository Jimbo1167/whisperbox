"""Client-side detection of a running warm model server.

The model server (scripts/model_server.py) keeps models loaded across jobs.
CLI entry points call try_server_transcribe() first: if a server is listening,
the transcription runs there — skipping the multi-second torch import + model
load a fresh process would pay — and the transcript is formatted and saved
locally. If no server is up (or the request can't be served remotely), the
caller falls back to in-process transcription.

This module is used by CLI entry points only. It must never be called from
TranscriptionService: the server itself uses the service, and a probe there
would make the server POST to itself.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .config import Config
from .output.formatter import OutputFormatter

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://localhost:8000"
HEALTH_PROBE_TIMEOUT = 0.5


def _server_is_healthy(server_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{server_url}/health", timeout=HEALTH_PROBE_TIMEOUT
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _server_matches_config(server_url: str, config: Config) -> bool:
    """Check the server was loaded with the model/language this run expects.

    A long-running server keeps the config it started with; silently
    transcribing with a different model or language than the caller's
    config would change output quality with no warning.
    """
    try:
        with urllib.request.urlopen(
            f"{server_url}/status", timeout=HEALTH_PROBE_TIMEOUT
        ) as response:
            status = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False

    model = status.get("model") or {}
    if model.get("model_size") != config.whisper_model_size:
        logger.info(
            "Model server runs %s but this request wants %s; transcribing locally",
            model.get("model_size"), config.whisper_model_size,
        )
        return False
    if model.get("language") != config.language:
        logger.info(
            "Model server language %s differs from requested %s; transcribing locally",
            model.get("language"), config.language,
        )
        return False
    return True


def try_server_transcribe(
    input_path: str,
    config: Config,
    output_path: str,
    server_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Transcribe via a running model server, if one can serve this request.

    Returns a result dict ({"segments", "output_file"}) when the server
    handled the job, or None when the caller should transcribe in-process
    (no server running, diarization requested, or opt-out via
    WHISPERBOX_NO_SERVER).
    """
    if os.getenv("WHISPERBOX_NO_SERVER", "").strip().lower() in ("1", "true", "yes", "on"):
        return None

    # The server's JSON endpoint transcribes with the server's own settings;
    # it cannot honor a per-request diarization ask, so fall back locally.
    if config.include_diarization:
        return None

    server_url = (server_url or os.getenv("WHISPERBOX_SERVER_URL", DEFAULT_SERVER_URL)).rstrip("/")

    if not _server_is_healthy(server_url):
        return None

    if not _server_matches_config(server_url, config):
        return None

    logger.info(f"Using warm model server at {server_url}")
    payload = json.dumps({
        "audio_path": os.path.abspath(input_path),
        # Explicit, so a server whose own default has diarization on doesn't
        # apply it to this request.
        "include_diarization": config.include_diarization,
    }).encode()
    request = urllib.request.Request(
        f"{server_url}/transcribe",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Long files legitimately take minutes; cap at the same ceiling the
        # local path enforces so a wedged server can't hang the CLI forever.
        with urllib.request.urlopen(
            request, timeout=config.transcribe_timeout
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning(f"Model server request failed, falling back locally: {exc}")
        return None

    raw_segments = body.get("segments")
    if raw_segments is None:
        logger.warning(f"Model server returned no segments, falling back locally: {body.get('error')}")
        return None

    # JSON turns (start, end, text, speaker) tuples into lists; restore them.
    segments = [tuple(segment) for segment in raw_segments]

    OutputFormatter(config).save_transcript(segments, output_path)

    return {"segments": segments, "output_file": output_path}


def transcribe_with_server_fallback(
    input_path: str,
    config: Config,
    output_path: str,
) -> Dict[str, Any]:
    """Transcribe via a running warm server when possible, else in-process.

    Single entry-point helper so every CLI gets identical routing policy
    (probe, compatibility check, fallback) instead of re-implementing it.
    """
    result = try_server_transcribe(input_path, config, output_path)
    if result is not None:
        return result

    # Imported here, not at module top: TranscriptionService pulls in the
    # model stack, and the server-served path never needs it.
    from .service import TranscriptionService

    logger.info("No compatible model server; transcribing in-process")
    service = TranscriptionService(config)
    return service.transcribe_file(input_path, output_path=output_path)

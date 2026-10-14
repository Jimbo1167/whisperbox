import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import logging

from .config import Config
from .output.formatter import OutputFormatter
from .transcriber import Transcriber
from .transcription.engine import ParakeetEngine

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Reusable application service for file transcription workflows."""

    def __init__(
        self,
        config: Optional[Config] = None,
        test_mode: bool = False,
        preload_models: bool = False,
    ):
        self.config = config or Config()
        self.test_mode = test_mode
        self._transcriber: Optional[Transcriber] = None
        self._lock = threading.Lock()

        if preload_models:
            self.preload_models()

    @property
    def transcriber(self) -> Transcriber:
        """Construct the transcriber lazily so tests and CLI startup stay cheap."""
        if self._transcriber is None:
            self._transcriber = Transcriber(self.config, test_mode=self.test_mode)
        return self._transcriber

    @property
    def engine_name(self) -> str:
        """Name of the ASR engine this service actually runs.

        Read from the constructed engine rather than the config, because a
        defaulted parakeet selection silently falls back to whisper when
        parakeet-mlx isn't installed.
        """
        engine = self.transcriber.transcription_engine
        return "parakeet" if isinstance(engine, ParakeetEngine) else "whisper"

    @property
    def model_name(self) -> str:
        """Model the ASR engine loads (a Parakeet model id or Whisper model)."""
        engine = self.transcriber.transcription_engine
        if isinstance(engine, ParakeetEngine):
            return engine.parakeet_model_id
        return engine.whisper_model_size

    def preload_models(self) -> None:
        """Warm transcription and diarization models ahead of the first request."""
        transcriber = self.transcriber
        transcriber.transcription_engine.ensure_model_loaded()
        if transcriber.include_diarization:
            try:
                transcriber.diarization_engine.ensure_model_loaded()
            except Exception as exc:
                logger.warning(
                    "Disabling diarization because the model could not be loaded: %s",
                    exc,
                )
                self.config.include_diarization = False
                transcriber.include_diarization = False
                transcriber.diarization_engine.include_diarization = False

    def build_output_path(self, input_path: str, output_format: Optional[str] = None) -> str:
        output_dir = Path("transcripts")
        output_dir.mkdir(exist_ok=True)
        extension = output_format or self.config.output_format or "txt"
        return str(output_dir / f"{Path(input_path).stem}.{extension}")

    def transcribe_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        include_diarization: Optional[bool] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        with self._lock:
            segments = self.transcriber.transcribe(
                input_path,
                progress_callback=progress_callback,
                include_diarization=include_diarization,
            )
            diarization_error = self.transcriber.last_diarization_error

        output_format = output_format or self.config.output_format
        output_path = output_path or self.build_output_path(input_path, output_format)

        formatter = OutputFormatter(self.config)
        formatter.format = output_format
        if progress_callback:
            progress_callback("Saving transcript", 0.96)
        # Format once and write the same string (save_transcript would
        # re-format the full segment list a second time).
        preview_text = formatter.format_transcript(segments)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(preview_text, encoding="utf-8")
        if progress_callback:
            progress_callback("Completed", 1.0)

        return {
            "segments": segments,
            "preview_text": preview_text,
            "output_format": output_format,
            "output_file": output_path,
            "processing_time": time.time() - start_time,
            # Set when diarization was requested but failed; the segments
            # above are then the completed transcription without speakers.
            "diarization_error": diarization_error,
        }

    def transcribe_existing_audio(
        self,
        audio_path: str,
        include_diarization: Optional[bool] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        with self._lock:
            segments = self.transcriber.transcribe(
                audio_path, include_diarization=include_diarization
            )
            diarization_error = self.transcriber.last_diarization_error

        return {
            "segments": segments,
            "processing_time": time.time() - start_time,
            "diarization_error": diarization_error,
        }

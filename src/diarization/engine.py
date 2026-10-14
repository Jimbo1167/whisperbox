import os
import time
import logging
from typing import List, Dict, Any, Optional
import concurrent.futures

from pyannote.audio import Pipeline
import torch
import warnings

from ..config import Config
from ..audio.processor import run_with_timeout, TimeoutException
from ..cache.manager import CacheManager

logger = logging.getLogger(__name__)


def _tiny_wav_bytes() -> bytes:
    """A minimal valid mono 16-bit WAV (a few ms of silence) for probing."""
    import io
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<160h", *([0] * 160)))
    return buffer.getvalue()


def _check_torchcodec_available():
    """Fail fast when pyannote's audio decoder cannot work on this machine.

    pyannote-audio 4.x decodes audio through torchcodec's AudioDecoder, but
    guards the import with a warning only — a broken torchcodec then surfaces
    as ``NameError: name 'AudioDecoder' is not defined`` in the middle of a
    job. torchcodec dlopens the system FFmpeg shared libraries and supports a
    bounded range of FFmpeg major versions, so a system FFmpeg newer than the
    installed torchcodec supports breaks the import.

    Raises:
        RuntimeError: with an actionable message when torchcodec's
            AudioDecoder cannot be imported or cannot decode.
    """
    try:
        from torchcodec.decoders import AudioDecoder

        # torchcodec >= 0.12 imports fine even when the FFmpeg libraries
        # can't be loaded and only fails on first decoder use, so probe
        # with a real decode of a tiny in-memory WAV.
        AudioDecoder(_tiny_wav_bytes())
    except Exception as exc:
        raise RuntimeError(
            "Diarization is unavailable: torchcodec's AudioDecoder cannot be "
            f"imported ({exc}). pyannote-audio decodes audio via torchcodec, "
            "which loads the system FFmpeg shared libraries and only supports "
            "a bounded range of FFmpeg major versions. Fix: install an FFmpeg "
            "major version your torchcodec build supports (e.g. `brew install "
            "ffmpeg@7`) or upgrade torchcodec to a build that supports your "
            "FFmpeg, then verify with: python -c 'from torchcodec.decoders "
            "import AudioDecoder'"
        ) from exc


class DiarizationEngine:
    """Handles speaker diarization for audio files."""
    
    def __init__(self, config: Config, test_mode: bool = False):
        """Initialize the diarization engine.
        
        Args:
            config: Configuration object
            test_mode: If True, use mock models for testing
        """
        self.config = config
        self.timeout_seconds = config.diarize_timeout
        self.include_diarization = config.include_diarization
        self.hf_token = config.hf_token
        self.diarization_model = config.diarization_model
        self.test_mode = test_mode
        
        # Cache directory for models
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache manager if caching is enabled
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
        
        # Configure warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.utils.autocast")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.core.notebook")
        
        # Determine the best available device
        if config.force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forcing CPU usage as specified in configuration")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA for acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for processing (no GPU acceleration available)")
        
        # Load the diarization model if needed
        self.diarizer = None
        if self.include_diarization and self.test_mode:
            self._load_model()
    
    def _load_model(self, force: bool = False):
        """Load the diarization model."""
        if self.diarizer is not None:
            return

        if not (self.include_diarization or force):
            logger.info("Diarization is disabled, skipping model loading")
            return
            
        if self.test_mode:
            logger.info("Test mode enabled, using mock diarizer")
            # Create a mock diarizer for testing
            class MockDiarizer:
                def __init__(self, device):
                    self.device = device
                
                def to(self, device):
                    return self
                
                def __call__(self, audio_path):
                    class MockDiarization:
                        def itertracks(self, yield_label=False):
                            class Segment:
                                def __init__(self, start, end):
                                    self.start = start
                                    self.end = end
                            
                            return [
                                (Segment(0.0, 2.0), None, "SPEAKER_01"),
                                (Segment(2.0, 4.0), None, "SPEAKER_02")
                            ]
                    return MockDiarization()
            
            self.diarizer = MockDiarizer(self.device)
            logger.info("Mock diarization model loaded successfully")
            return
            
        if not self.hf_token:
            logger.warning("HF_TOKEN is not set. Speaker diarization may not work properly.")

        # Probe before the (slow) pipeline load so a broken audio decoder
        # fails here, at model-load time, not mid-job after transcription.
        _check_torchcodec_available()

        try:
            logger.info(f"Loading diarization model: {self.diarization_model}")
            self.diarizer = Pipeline.from_pretrained(
                self.diarization_model,
                token=self.hf_token,
                cache_dir=os.path.join(self.cache_dir, "diarization")
            )
            # Use the device property directly
            self.diarizer = self.diarizer.to(self.device)
            logger.info("Diarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading diarization model: {e}")
            raise

    def ensure_model_loaded(self, force: bool = False):
        """Load the diarization model on demand.

        Args:
            force: Load even when the configured default has diarization off
                (used for per-request diarization overrides).
        """
        if (self.include_diarization or force) and self.diarizer is None:
            logger.info(f"Loading diarization model: {self.diarization_model}")
            self._load_model(force=force)

    def _unwrap_diarization_result(self, diarization_result: Any):
        """Normalize pyannote outputs across old and new API shapes."""
        if hasattr(diarization_result, "itertracks"):
            return diarization_result

        for attr_name in ("speaker_diarization", "exclusive_speaker_diarization", "annotation"):
            annotation = getattr(diarization_result, attr_name, None)
            if annotation is not None and hasattr(annotation, "itertracks"):
                return annotation

        raise TypeError(
            f"Unsupported diarization output type: {type(diarization_result).__name__}"
        )
    
    def diarize(
        self, audio_path: str, enabled: Optional[bool] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Perform speaker diarization with timeout.

        Args:
            audio_path: Path to the audio file
            enabled: Per-call override of the configured diarization setting
                (None = use the configured default)

        Returns:
            List of diarization segments or None if diarization is disabled

        Raises:
            TimeoutException: If diarization times out
            Exception: If diarization fails
        """
        effective = self.include_diarization if enabled is None else enabled
        if not effective:
            logger.info("Diarization is disabled, skipping")
            return None

        # Check if we have cached results
        if self.cache_manager:
            cached_diarization = self.cache_manager.get_cached_diarization(
                audio_path, model_id=self.config.diarization_model
            )
            if cached_diarization:
                return cached_diarization

        if not self.diarizer:
            logger.warning("Diarizer not initialized, attempting to load model")
            self.ensure_model_loaded(force=effective)
            if not self.diarizer:
                logger.error("Failed to load diarization model")
                return None
        
        logger.info(f"Starting speaker diarization for {audio_path}")
        start_time = time.time()
        
        def _do_diarize():
            diarization = self._unwrap_diarization_result(self.diarizer(audio_path))

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            segments.sort(key=lambda x: x["start"])
            return segments

        try:
            segments = run_with_timeout(
                _do_diarize, self.timeout_seconds, "Diarization timed out"
            )

            elapsed = time.time() - start_time
            logger.info(f"Diarization completed in {elapsed:.1f} seconds, found {len(segments)} segments")

            # Cache the results if caching is enabled
            if self.cache_manager:
                self.cache_manager.cache_diarization(
                    audio_path, segments, model_id=self.config.diarization_model
                )

            return segments

        except TimeoutException:
            logger.error(f"Diarization timed out after {self.timeout_seconds} seconds")
            raise
        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
            raise Exception(f"Error during diarization: {str(e)}")
    
    def diarize_with_progress(self, audio_path: str) -> Optional[List[Dict[str, Any]]]:
        """Perform speaker diarization with progress reporting.
        
        This method runs diarization in a separate thread and reports progress.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of diarization segments or None if diarization is disabled
        """
        if not self.include_diarization:
            logger.info("Diarization is disabled, skipping")
            return None
            
        logger.info("Starting speaker diarization with progress reporting")
        
        # Create a future to store the result
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.diarize, audio_path)
            
            # Wait for the result with progress reporting
            while not future.done():
                logger.info("Speaker diarization in progress...")
                time.sleep(5)
            
            # Get the result or raise the exception
            return future.result() 

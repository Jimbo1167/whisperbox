import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator, Generator, Protocol, runtime_checkable
import concurrent.futures
from contextlib import contextmanager
import threading

from faster_whisper import WhisperModel
import torch
import numpy as np

from ..config import Config
from ..audio.processor import timeout, TimeoutException
from ..cache.manager import CacheManager
from .streaming import StreamingTranscriber, AsyncStreamingTranscriber

logger = logging.getLogger(__name__)


def _slug(text: str) -> str:
    """Return a filesystem-safe slug for cache keys and filenames.

    HF model ids contain '/'; local paths contain '/' (and on macOS, spaces).
    Both must produce a single safe token.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


@runtime_checkable
class ASREngine(Protocol):
    """Batch-only ASR contract.

    Streaming is a Whisper-only extension and is not part of this protocol.
    Code that needs streaming should depend on WhisperEngine concretely or
    guard on Config.transcription_engine == "whisper" upstream.
    """

    def ensure_model_loaded(self) -> None: ...

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]: ...


class WhisperEngine:
    """Whisper-based ASR engine using faster-whisper."""
    
    def __init__(self, config: Config, test_mode: bool = False):
        """Initialize the transcription engine.
        
        Args:
            config: Configuration object
            test_mode: If True, use mock models for testing
        """
        self.config = config
        self.timeout_seconds = config.transcribe_timeout
        self.language = config.language
        self.whisper_model_size = config.whisper_model_size
        self.test_mode = test_mode
        self.whisper = None
        self._batched_pipeline = None

        # Cache directory for models
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache manager if caching is enabled
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
        
        # Determine the best available device
        if config.force_cpu:
            self.device = "cpu"
            logger.info("Forcing CPU usage as specified in configuration")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA for acceleration")
        else:
            self.device = "cpu"
            logger.info("Using CPU for processing (no GPU acceleration available)")
        
        if self.test_mode:
            logger.info(f"Loading Whisper model ({self.whisper_model_size})...")
            self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        if self.whisper is not None:
            return

        if self.test_mode:
            logger.info("Test mode enabled, using mock whisper model")
            # Create a mock whisper model for testing
            class MockWhisperModel:
                def __init__(self, model_size, device, compute_type, download_root):
                    self.model_size = model_size
                    self.device = device
                    self.compute_type = compute_type
                
                def transcribe(self, audio_path, **kwargs):
                    class MockSegment:
                        def __init__(self, start, end, text):
                            self.start = start
                            self.end = end
                            self.text = text
                            self.words = []
                    
                    segments = [
                        MockSegment(0.0, 2.0, "Test segment one"),
                        MockSegment(2.0, 4.0, "Test segment two")
                    ]
                    info = {"language": "en"}
                    return segments, info
            
            self.whisper = MockWhisperModel(
                self.whisper_model_size,
                self.device,
                "int8",
                os.path.join(self.cache_dir, "whisper")
            )
            logger.info("Mock whisper model loaded successfully")
            return
            
        try:
            # Whisper works better on CPU for Apple Silicon
            compute_type = "int8"
            if self.device == "cuda" and self.whisper_model_size in ["medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]:
                compute_type = "float16"  # Use float16 for larger models on CUDA
                
            self.whisper = WhisperModel(
                self.whisper_model_size,
                device="cpu" if self.device == "mps" else self.device,
                compute_type=compute_type,
                cpu_threads=self.config.whisper_cpu_threads,
                download_root=os.path.join(self.cache_dir, "whisper")
            )
            logger.info(f"Whisper model loaded successfully: {self.whisper_model_size}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def ensure_model_loaded(self):
        """Load the model on demand."""
        if self.whisper is None:
            logger.info(f"Loading Whisper model ({self.whisper_model_size})...")
            self._load_model()
    
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe an audio file with timeout.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of transcription segments
            
        Raises:
            TimeoutException: If transcription times out
            Exception: If transcription fails
        """
        # Check if we have cached results
        cache_engine_id = f"whisper-{_slug(self.whisper_model_size)}"

        if self.cache_manager:
            cached_transcription = self.cache_manager.get_cached_transcription(
                audio_path, engine_id=cache_engine_id
            )
            if cached_transcription:
                return cached_transcription
        
        if not self.whisper:
            logger.warning("Whisper model not initialized, attempting to load model")
            self.ensure_model_loaded()
            if not self.whisper:
                logger.error("Failed to load Whisper model")
                raise Exception("Failed to load Whisper model")
        
        logger.info(f"Starting transcription for {audio_path}")
        start_time = time.time()
        
        try:
            with timeout(self.timeout_seconds, "Transcription timed out"):
                transcribe_kwargs = dict(
                    language=self.language,
                    beam_size=self.config.whisper_beam_size,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                if self.config.whisper_batch_size > 0:
                    # BatchedInferencePipeline decodes chunks in parallel —
                    # the single biggest faster-whisper throughput knob.
                    from faster_whisper import BatchedInferencePipeline
                    if self._batched_pipeline is None:
                        self._batched_pipeline = BatchedInferencePipeline(
                            model=self.whisper
                        )
                    segments, _ = self._batched_pipeline.transcribe(
                        audio_path,
                        batch_size=self.config.whisper_batch_size,
                        **transcribe_kwargs,
                    )
                else:
                    segments, _ = self.whisper.transcribe(
                        audio_path, **transcribe_kwargs
                    )
                
                # Convert segments to a list of dictionaries for easier processing
                result = []
                for segment in segments:
                    result.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "words": [{"start": word.start, "end": word.end, "word": word.word} for word in segment.words] if segment.words else []
                    })
                
                elapsed = time.time() - start_time
                logger.info(f"Transcription completed in {elapsed:.1f} seconds, found {len(result)} segments")
                
                # Cache the results if caching is enabled
                if self.cache_manager:
                    self.cache_manager.cache_transcription(
                        audio_path, result, engine_id=cache_engine_id
                    )
                
                return result
                
        except TimeoutException as e:
            logger.error(f"Transcription timed out after {self.timeout_seconds} seconds")
            raise
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise Exception(f"Error during transcription: {str(e)}")
    
    def transcribe_with_progress(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio file with progress reporting.
        
        This method runs transcription in a separate thread and reports progress.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of transcription segments
        """
        logger.info("Starting transcription with progress reporting")
        
        # Create a future to store the result
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.transcribe, audio_path)
            
            # Wait for the result with progress reporting
            while not future.done():
                logger.info("Transcription in progress...")
                time.sleep(5)
            
            # Get the result or raise the exception
            return future.result()
    
    def transcribe_stream(self, audio_stream: Iterator[np.ndarray]) -> Generator[Dict[str, Any], None, None]:
        """
        Transcribe an audio stream and yield segments as they become available.
        
        Args:
            audio_stream: Iterator yielding chunks of audio data as numpy arrays
            
        Yields:
            Transcription segments as they become available
            
        Raises:
            Exception: If transcription fails
        """
        if not self.whisper:
            logger.warning("Whisper model not initialized, attempting to load model")
            self.ensure_model_loaded()
            if not self.whisper:
                logger.error("Failed to load Whisper model")
                raise Exception("Failed to load Whisper model")
        
        logger.info("Starting streaming transcription")
        start_time = time.time()
        
        try:
            streaming_transcriber = StreamingTranscriber(self.whisper, self.config)
            segment_count = 0
            
            for segment in streaming_transcriber.process_stream(audio_stream):
                segment_count += 1
                yield segment
                
            elapsed = time.time() - start_time
            logger.info(f"Streaming transcription completed in {elapsed:.1f} seconds, found {segment_count} segments")
                
        except Exception as e:
            logger.error(f"Error during streaming transcription: {str(e)}")
            raise Exception(f"Error during streaming transcription: {str(e)}")
    
    def start_async_transcription(self, audio_stream: Iterator[np.ndarray]) -> AsyncStreamingTranscriber:
        """
        Start asynchronous transcription of an audio stream.
        
        Args:
            audio_stream: Iterator yielding chunks of audio data as numpy arrays
            
        Returns:
            AsyncStreamingTranscriber instance that can be used to get results
            
        Raises:
            Exception: If transcription fails to start
        """
        if not self.whisper:
            logger.warning("Whisper model not initialized, attempting to load model")
            self.ensure_model_loaded()
            if not self.whisper:
                logger.error("Failed to load Whisper model")
                raise Exception("Failed to load Whisper model")
        
        logger.info("Starting asynchronous streaming transcription")
        
        try:
            async_transcriber = AsyncStreamingTranscriber(self.whisper, self.config)
            async_transcriber.start_processing(audio_stream)
            return async_transcriber
                
        except Exception as e:
            logger.error(f"Error starting async transcription: {str(e)}")
            raise Exception(f"Error starting async transcription: {str(e)}")


class ParakeetEngine:
    """Parakeet-MLX ASR engine. Apple Silicon only. Batch-only.

    Long-form audio is handled by parakeet-mlx's built-in chunking
    (chunk_duration / overlap_duration) — we do not roll our own VAD
    chunking. See spec §"Long-form chunking".
    """

    # parakeet-mlx 0.5.1 defaults — exposed as class constants for visibility.
    CHUNK_DURATION = 120
    OVERLAP_DURATION = 15

    def __init__(self, config: Config, test_mode: bool = False):
        self.config = config
        self.timeout_seconds = config.transcribe_timeout
        self.parakeet_model_id = config.parakeet_model
        self.test_mode = test_mode
        self.parakeet = None

        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_manager = CacheManager(config) if config.cache_enabled else None

        if config.force_cpu:
            logger.warning(
                "FORCE_CPU has no effect on Parakeet (MLX); flag is Whisper-only."
            )

        if self.test_mode:
            self._load_model()

    def _engine_id(self) -> str:
        return f"parakeet-{_slug(self.parakeet_model_id)}"

    def _load_model(self):
        if self.parakeet is not None:
            return

        if self.test_mode:
            logger.info("Test mode enabled, using mock parakeet model")

            class _Token:
                def __init__(self, text, start, end):
                    self.text = text
                    self.start = start
                    self.end = end

            class _Sentence:
                def __init__(self, text, start, end, tokens):
                    self.text = text
                    self.start = start
                    self.end = end
                    self.tokens = tokens

            class _AlignedResult:
                text = "Hello world."
                sentences = [
                    _Sentence(
                        "Hello world.", 0.0, 1.0,
                        [_Token("Hello", 0.0, 0.5), _Token(" world.", 0.5, 1.0)],
                    ),
                ]

            class MockParakeetModel:
                def transcribe(self, audio_path, **kwargs):
                    return _AlignedResult()

            self.parakeet = MockParakeetModel()
            return

        # Lazy import: parakeet_mlx is darwin/arm64 only and imported only
        # when actually needed. Test patch target is `parakeet_mlx.from_pretrained`.
        import parakeet_mlx  # noqa: WPS433

        try:
            logger.info(f"Loading parakeet-mlx model: {self.parakeet_model_id}")
            self.parakeet = parakeet_mlx.from_pretrained(self.parakeet_model_id)
            logger.info("parakeet-mlx model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading parakeet-mlx model: {e}")
            raise

    def ensure_model_loaded(self) -> None:
        if self.parakeet is None:
            self._load_model()

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        cache_engine_id = self._engine_id()

        if self.cache_manager:
            cached = self.cache_manager.get_cached_transcription(
                audio_path, engine_id=cache_engine_id
            )
            if cached:
                return cached

        self.ensure_model_loaded()
        logger.info(f"Starting parakeet transcription for {audio_path}")
        start_time = time.time()

        try:
            with timeout(self.timeout_seconds, "Transcription timed out"):
                result = self.parakeet.transcribe(
                    audio_path,
                    chunk_duration=self.CHUNK_DURATION,
                    overlap_duration=self.OVERLAP_DURATION,
                )

                segments = self._map_aligned_result(result)

                elapsed = time.time() - start_time
                logger.info(
                    f"Parakeet transcription completed in {elapsed:.1f}s, "
                    f"{len(segments)} segments"
                )

                if self.cache_manager:
                    self.cache_manager.cache_transcription(
                        audio_path, segments, engine_id=cache_engine_id
                    )
                return segments

        except TimeoutException:
            logger.error(f"Parakeet transcription timed out after {self.timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"Error during parakeet transcription: {e}")
            raise Exception(f"Error during parakeet transcription: {e}")

    @classmethod
    def _map_aligned_result(cls, result) -> List[Dict[str, Any]]:
        """Map parakeet-mlx AlignedResult → standard segment shape."""
        segments: List[Dict[str, Any]] = []
        for sentence in result.sentences:
            words: List[Dict[str, Any]] = []
            for idx, token in enumerate(sentence.tokens):
                text = token.text
                # Whitespace normalization: see spec §"Word whitespace normalization".
                if idx > 0 and not text.startswith(" "):
                    text = " " + text
                words.append({
                    "start": float(token.start),
                    "end": float(token.end),
                    "word": text,
                })
            segments.append({
                "start": float(sentence.start),
                "end": float(sentence.end),
                "text": sentence.text.strip(),
                "words": words,
            })
        return segments

    # Streaming is Whisper-only — defensive stubs in case the upstream guard
    # in transcriber.py is ever bypassed.
    def transcribe_stream(self, audio_stream):
        raise NotImplementedError(
            "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
            "Use transcribe() for batch transcription with Parakeet."
        )

    def start_async_transcription(self, audio_stream):
        raise NotImplementedError(
            "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
            "Use transcribe() for batch transcription with Parakeet."
        )


# Backward-compat alias: src/transcriber.py and existing tests import
# `TranscriptionEngine` by name. Streaming methods remain on this class.
TranscriptionEngine = WhisperEngine


def make_asr_engine(config: Config, test_mode: bool = False) -> ASREngine:
    """Construct the ASR engine selected by config.transcription_engine."""
    engine_name = config.transcription_engine
    if engine_name == "whisper":
        return WhisperEngine(config, test_mode=test_mode)
    if engine_name == "parakeet":
        return ParakeetEngine(config, test_mode=test_mode)
    raise ValueError(f"Unknown transcription engine: {engine_name!r}")

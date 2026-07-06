import os
import platform
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration class to handle all settings for the Whisperbox.
    
    This class loads configuration from environment variables and provides
    validation and type conversion for the settings.
    """
    def __init__(self, env_file: Optional[str] = None, **overrides):
        """Initialize configuration from environment variables.

        Args:
            env_file: Optional path to a .env file to load
            **overrides: Optional keyword arguments to override env values.
                Supported keys: whisper_model, language, output_format,
                include_diarization, diarization_model, force_cpu,
                transcription_engine, parakeet_model
        """
        if env_file:
            logger.info(f"Loading configuration from {env_file}")
            load_dotenv(env_file, override=True)
        else:
            logger.info("Using existing environment variables for configuration")

        # API tokens and model settings
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper_model_size = os.getenv("WHISPER_MODEL", "large-v3-turbo")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
        self.language = os.getenv("LANGUAGE", "en")

        # ASR engine selection. Parakeet (MLX) is roughly an order of magnitude
        # faster than Whisper-on-CPU, but only exists on Apple Silicon.
        default_engine = (
            "parakeet"
            if sys.platform == "darwin" and platform.machine() == "arm64"
            else "whisper"
        )
        self.transcription_engine = os.getenv("TRANSCRIPTION_ENGINE", default_engine).strip().lower()
        self.parakeet_model = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")

        # Parse output format
        self._output_format = None
        output_format = os.getenv("OUTPUT_FORMAT")
        if output_format:
            output_format = output_format.strip()
            if "#" in output_format:
                output_format = output_format.split("#")[0].strip()
        self._output_format = output_format if output_format else "txt"

        # Parse diarization setting
        diarization = os.getenv("INCLUDE_DIARIZATION", "false")
        self.include_diarization = diarization.strip().lower() in ["true", "1", "yes", "on"]

        # Timeouts
        self.audio_timeout = int(os.getenv("AUDIO_TIMEOUT", "300"))
        self.transcribe_timeout = int(os.getenv("TRANSCRIBE_TIMEOUT", "3600"))
        self.diarize_timeout = int(os.getenv("DIARIZE_TIMEOUT", "3600"))

        # Whisper (faster-whisper/ctranslate2) tuning.
        # WHISPER_BEAM_SIZE=1 (greedy) is ~2x faster than the default 5-way beam.
        # WHISPER_CPU_THREADS=0 keeps ctranslate2's default (4); higher values
        # use more of the machine on the CPU-bound path.
        # WHISPER_BATCH_SIZE>0 enables BatchedInferencePipeline (parallel
        # chunk decoding); 0 keeps the sequential decoder.
        self.whisper_beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
        self.whisper_cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "0"))
        self.whisper_batch_size = int(os.getenv("WHISPER_BATCH_SIZE", "0"))

        # Device settings
        self.force_cpu = os.getenv("FORCE_CPU", "false").strip().lower() in ["true", "1", "yes", "on"]

        # Cache settings
        cache_enabled = os.getenv("CACHE_ENABLED", "true")
        self.cache_enabled = cache_enabled.strip().lower() in ["true", "1", "yes", "on"]

        # Cache expiration in seconds (default: 7 days)
        self.cache_expiration = int(os.getenv("CACHE_EXPIRATION", str(7 * 24 * 60 * 60)))

        # Maximum cache size in bytes (default: 10GB)
        self.max_cache_size = int(os.getenv("MAX_CACHE_SIZE", str(10 * 1024 * 1024 * 1024)))

        # Apply any keyword overrides
        if 'whisper_model' in overrides:
            self.whisper_model_size = overrides['whisper_model']
        if 'language' in overrides:
            self.language = overrides['language']
        if 'output_format' in overrides:
            self.output_format = overrides['output_format']
        if 'include_diarization' in overrides:
            self.include_diarization = bool(overrides['include_diarization'])
        if 'diarization_model' in overrides:
            self.diarization_model = overrides['diarization_model']
        if 'force_cpu' in overrides:
            self.force_cpu = bool(overrides['force_cpu'])
        if 'transcription_engine' in overrides:
            self.transcription_engine = str(overrides['transcription_engine']).strip().lower()
        if 'parakeet_model' in overrides:
            self.parakeet_model = overrides['parakeet_model']

        logger.debug(f"Configuration loaded: {self.to_dict()}")

    @property
    def output_format(self) -> str:
        """Get the output format for transcripts."""
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        """Set the output format for transcripts.
        
        Args:
            value: The output format (txt, srt, vtt, json, pretty)
        """
        if value:
            value = value.strip()
            if "#" in value:
                value = value.split("#")[0].strip()
        self._output_format = value if value else "txt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dict containing all configuration values
        """
        return {
            "whisper_model_size": self.whisper_model_size,
            "diarization_model": self.diarization_model,
            "language": self.language,
            "output_format": self.output_format,
            "include_diarization": self.include_diarization,
            "audio_timeout": self.audio_timeout,
            "transcribe_timeout": self.transcribe_timeout,
            "diarize_timeout": self.diarize_timeout,
            "force_cpu": self.force_cpu,
            "cache_enabled": self.cache_enabled,
            "cache_expiration": self.cache_expiration,
            "max_cache_size": self.max_cache_size,
            "transcription_engine": self.transcription_engine,
            "parakeet_model": self.parakeet_model,
            "whisper_beam_size": self.whisper_beam_size,
            "whisper_cpu_threads": self.whisper_cpu_threads,
            "whisper_batch_size": self.whisper_batch_size,
        }
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if self.include_diarization and not self.hf_token:
            logger.warning("Speaker diarization is enabled but HF_TOKEN is not set")
            return False

        valid_formats = ["txt", "srt", "vtt", "json", "pretty"]
        if self.output_format not in valid_formats:
            logger.warning(f"Invalid output format: {self.output_format}. Must be one of {valid_formats}")
            return False

        valid_engines = {"whisper", "parakeet"}
        if self.transcription_engine not in valid_engines:
            logger.error(
                f"Invalid TRANSCRIPTION_ENGINE: {self.transcription_engine!r}. "
                f"Must be one of {sorted(valid_engines)}"
            )
            return False

        if self.transcription_engine == "parakeet":
            if sys.platform != "darwin" or platform.machine() != "arm64":
                logger.error(
                    "Parakeet requires Apple Silicon (macOS arm64). "
                    "Set TRANSCRIPTION_ENGINE=whisper or run on macOS arm64."
                )
                return False

        return True

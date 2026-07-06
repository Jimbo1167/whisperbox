import os
import subprocess
import tempfile
import time
import logging
import wave
from typing import Tuple, Optional, Generator, Iterator
import numpy as np
import imageio_ffmpeg
import concurrent.futures

from ..config import Config
from ..cache.manager import CacheManager

logger = logging.getLogger(__name__)

# Whisper and Parakeet both consume 16kHz mono; extracting at the target rate
# keeps temp WAVs ~5x smaller than source-rate stereo and skips a resample.
TARGET_SAMPLE_RATE = 16000


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio with linear interpolation."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / float(orig_sr)
    target_length = max(1, int(round(duration * target_sr)))
    source_positions = np.arange(len(audio), dtype=np.float32)
    target_positions = np.linspace(0, max(len(audio) - 1, 0), num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)

class TimeoutException(Exception):
    """Raised when an operation times out."""
    pass


def run_with_timeout(fn, seconds, message="Operation timed out"):
    """Run fn() and raise TimeoutException in the caller if it exceeds seconds.

    Replaces the old threading.Timer-based context manager, which raised in
    the timer thread and therefore never actually interrupted anything. The
    work runs in a helper thread; on timeout the caller is unblocked and the
    abandoned thread is left to finish in the background (Python cannot kill
    a thread stuck in a C extension).
    """
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="timeout-guard"
    )
    future = executor.submit(fn)
    try:
        return future.result(timeout=seconds)
    except concurrent.futures.TimeoutError:
        raise TimeoutException(message)
    finally:
        executor.shutdown(wait=False)

class AudioProcessor:
    """Handles audio extraction and processing for transcription."""
    
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
    
    def __init__(self, config: Config):
        """Initialize the audio processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.timeout_seconds = config.audio_timeout
        
        # Initialize cache manager if caching is enabled
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if the file is an audio file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is an audio file, False otherwise
        """
        return file_path.lower().endswith(('.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'))
    
    def get_audio_path(self, input_path: str) -> tuple[str, bool]:
        """
        Get the path to an audio file, converting if necessary.
        
        Args:
            input_path: Path to the input file (audio or video)
            
        Returns:
            tuple: (audio_path, needs_cleanup) where:
                - audio_path: Path to the WAV audio file
                - needs_cleanup: Boolean indicating if the file needs to be cleaned up after use
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the input file is not a supported format
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check if we have a cached version
        if self.cache_manager:
            cached_audio = self.cache_manager.get_cached_audio(input_path)
            if cached_audio:
                return cached_audio, False

        # If it's already a WAV file, return it
        if input_path.lower().endswith('.wav'):
            logger.info(f"Input is already a WAV file: {input_path}")
            return input_path, False

        # Extract/convert directly into the audio cache when caching is on:
        # the cached WAV then has a stable path across runs, which keeps the
        # downstream transcription/diarization cache keys stable too.
        if self.cache_manager:
            wav_path = self.cache_manager.audio_cache_path(input_path)
            self._extract_to_wav(input_path, wav_path)
            return wav_path, False

        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="whisperbox_")
        os.close(fd)
        try:
            self._extract_to_wav(input_path, wav_path)
        except Exception:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            raise
        return wav_path, True

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from a video file to a temporary 16kHz mono WAV.

        Args:
            video_path: Path to the video file

        Returns:
            Path to the extracted audio file (caller owns cleanup)

        Raises:
            TimeoutException: If audio extraction times out
            Exception: If audio extraction fails
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="whisperbox_")
        os.close(fd)
        try:
            self._extract_to_wav(video_path, wav_path)
        except Exception:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            raise
        return wav_path

    def _extract_to_wav(self, input_path: str, wav_path: str):
        """Decode any audio/video input straight to a 16kHz mono PCM WAV.

        Runs ffmpeg as a single subprocess (no PCM piping through Python) so
        the extraction is one pass and the timeout is actually enforced.
        """
        logger.info(f"Extracting 16kHz mono audio: {input_path} -> {wav_path}")
        # Write to a temp name and rename into place so a killed run can never
        # leave a truncated WAV at the final path (which the cache would then
        # treat as a valid hit).
        partial_path = f"{wav_path}.part{os.getpid()}.wav"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y", "-nostdin",
            "-i", input_path,
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            "-f", "wav",
            partial_path,
        ]
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=self.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            logger.error(f"Audio extraction timed out after {self.timeout_seconds}s")
            raise TimeoutException("Audio extraction timed out")

        if result.returncode != 0:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
            logger.error(f"ffmpeg failed for {input_path}: {stderr_tail}")
            raise Exception(f"Error extracting audio: {stderr_tail}")

        os.replace(partial_path, wav_path)
        elapsed = time.time() - start_time
        logger.info(f"Audio extraction complete in {elapsed:.1f} seconds")
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load audio file into memory with efficient processing.
        
        Args:
            audio_path: Path to the audio file
            target_sr: Target sample rate
            
        Returns:
            Audio data as numpy array
        """
        logger.info(f"Loading audio file: {audio_path}")
        try:
            with wave.open(audio_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                frames = wav_file.readframes(wav_file.getnframes())

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            if sample_width not in dtype_map:
                raise ValueError(f"Unsupported WAV sample width: {sample_width}")

            audio = np.frombuffer(frames, dtype=dtype_map[sample_width]).astype(np.float32)
            scale = float(2 ** (8 * sample_width - 1))
            audio = audio / scale

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            audio = _resample_audio(audio, sample_rate, target_sr)
            logger.info(f"Loaded audio: {len(audio)/target_sr:.1f} seconds at {target_sr}Hz")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def process_audio_stream(self, audio_data: np.ndarray, 
                           chunk_size: int = CHUNK_SIZE) -> Generator[np.ndarray, None, None]:
        """Process audio data in chunks to avoid memory issues.
        
        Args:
            audio_data: Audio data as numpy array
            chunk_size: Size of each chunk in bytes
            
        Yields:
            Chunks of audio data
        """
        # Calculate chunk size in samples
        bytes_per_sample = audio_data.itemsize
        chunk_samples = chunk_size // bytes_per_sample
        
        # Process in chunks
        for i in range(0, len(audio_data), chunk_samples):
            yield audio_data[i:i+chunk_samples]
    
    def stream_audio_from_file(self, audio_path: str, chunk_duration: float = 5.0, 
                              target_sr: int = 16000) -> Iterator[np.ndarray]:
        """Stream audio from a file in chunks.
        
        Args:
            audio_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            target_sr: Target sample rate
            
        Yields:
            Chunks of audio data as numpy arrays
        """
        logger.info(f"Streaming audio from file: {audio_path}")
        try:
            with wave.open(audio_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                total_frames = wav_file.getnframes()
                total_duration = total_frames / float(sample_rate)
                chunk_size = int(chunk_duration * sample_rate)

                dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                if sample_width not in dtype_map:
                    raise ValueError(f"Unsupported WAV sample width: {sample_width}")

                logger.info(f"Audio file: {total_duration:.1f} seconds at {sample_rate}Hz")
                logger.info(f"Streaming in chunks of {chunk_duration:.1f} seconds")

                while wav_file.tell() < total_frames:
                    frames = wav_file.readframes(chunk_size)
                    if not frames:
                        break

                    chunk = np.frombuffer(frames, dtype=dtype_map[sample_width]).astype(np.float32)
                    scale = float(2 ** (8 * sample_width - 1))
                    chunk = chunk / scale

                    if channels > 1:
                        chunk = chunk.reshape(-1, channels).mean(axis=1)

                    yield _resample_audio(chunk, sample_rate, target_sr)

            logger.info(f"Finished streaming audio from {audio_path}")
            
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            raise

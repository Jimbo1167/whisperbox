import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Generator, Iterator, Callable
import concurrent.futures

from .config import Config
from .audio.processor import AudioProcessor, TimeoutException
from .transcription.engine import make_asr_engine
from .diarization.engine import DiarizationEngine
from .output.formatter import OutputFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Transcriber:
    """Main class that orchestrates the transcription process."""
    
    def __init__(self, config: Optional[Config] = None, test_mode: bool = False):
        """Initialize the transcriber.
        
        Args:
            config: Configuration object or None to use default configuration
            test_mode: If True, use mock models for testing
        """
        # Use provided config or create new one
        self.config = config or Config()
        self.test_mode = test_mode
        
        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.transcription_engine = make_asr_engine(self.config, test_mode=test_mode)
        self.diarization_engine = DiarizationEngine(self.config, test_mode=test_mode)
        self.output_formatter = OutputFormatter(self.config)
        
        # Copy config values to instance variables for backward compatibility
        self.hf_token = self.config.hf_token
        self.whisper_model_size = self.config.whisper_model_size
        self.diarization_model = self.config.diarization_model
        self.language = self.config.language
        self.output_format = self.config.output_format
        self.include_diarization = self.config.include_diarization
        self.audio_timeout = self.config.audio_timeout
        self.transcribe_timeout = self.config.transcribe_timeout
        self.diarize_timeout = self.config.diarize_timeout
        
        # Log configuration
        logger.info("\nConfiguration loaded:")
        logger.info(f"- Whisper model size: {self.whisper_model_size}")
        logger.info(f"- Language: {self.language}")
        logger.info(f"- Include diarization: {self.include_diarization}")
        logger.info(f"- Output format: {self.output_format}")
        if test_mode:
            logger.info(f"- Test mode: Enabled")
    
    def _combine_segments_with_speakers(self,
                                       transcription_segments: Union[List[Dict[str, Any]], List[Tuple[float, float, str]]],
                                        diarization_segments: Optional[List[Dict[str, Any]]]) -> List[Tuple[float, float, str, str]]:
        """Combine transcription segments with speaker information.
        
        Args:
            transcription_segments: List of transcription segments (either as dicts with start, end, text keys or tuples)
            diarization_segments: List of diarization segments
            
        Returns:
            List of tuples containing (start_time, end_time, text, speaker)
        """
        result = []
        
        if not diarization_segments:
            # No diarization, return transcription segments with empty speaker
            if transcription_segments and isinstance(transcription_segments[0], dict):
                # Handle dictionary-style segments
                return [(s["start"], s["end"], s["text"], "") for s in transcription_segments]
            else:
                # Handle tuple-style segments
                return [(s[0], s[1], s[2], "") for s in transcription_segments]
        
        logger.info("Combining transcription with speaker information...")
        
        # Create a mapping of time ranges to speakers
        speaker_map = {}
        for segment in diarization_segments:
            speaker_map[(segment["start"], segment["end"])] = segment["speaker"]
        
        # Process each transcription segment
        for segment in transcription_segments:
            # Extract segment information based on type
            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
            else:
                start = segment[0]
                end = segment[1]
                text = segment[2]
            
            # Find the speaker with the most overlap
            max_overlap = 0
            dominant_speaker = ""
            
            for (spk_start, spk_end), speaker in speaker_map.items():
                # Calculate overlap
                overlap_start = max(start, spk_start)
                overlap_end = min(end, spk_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    dominant_speaker = speaker
            
            result.append((start, end, text, dominant_speaker))
        
        return result
    
    def transcribe(
        self,
        input_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[tuple[float, float, str, str]]:
        """
        Transcribe an audio or video file.
        
        Args:
            input_path: Path to the audio or video file
            
        Returns:
            List of tuples containing (start_time, end_time, text, speaker)
            
        Raises:
            Exception: If transcription fails
        """
        audio_path = None
        needs_cleanup = False

        try:
            if progress_callback:
                progress_callback("Preparing audio", 0.05)

            # Overlap audio extraction (ffmpeg) with model warmup so the
            # downstream parallel block doesn't pay a cold-load penalty. Both
            # `ensure_model_loaded` calls are idempotent, so re-invocation
            # inside the engines' own `transcribe`/`diarize` is a no-op.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="model-preload"
            ) as load_pool:
                load_futures = [load_pool.submit(self.transcription_engine.ensure_model_loaded)]
                if self.include_diarization:
                    load_futures.append(
                        load_pool.submit(self.diarization_engine.ensure_model_loaded)
                    )
                audio_path, needs_cleanup = self.audio_processor.get_audio_path(input_path)
                # Executor __exit__ joins all submitted tasks before returning.

            # Surface any model-load errors here so callers see them before
            # we kick off the heavier transcribe/diarize block.
            for future in load_futures:
                future.result()

            if self.include_diarization:
                logger.info("\nStarting transcription and diarization...")
                if progress_callback:
                    progress_callback("Transcribing and diarizing", 0.2)
                
                # Start transcription and diarization concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_transcription = executor.submit(
                        self.transcription_engine.transcribe, audio_path
                    )
                    future_diarization = executor.submit(
                        self.diarization_engine.diarize, audio_path
                    )
                    
                    # Get transcription results
                    transcription_segments = future_transcription.result()
                    logger.info(f"\nTranscription complete. Found {len(transcription_segments)} segments.")
                    if progress_callback:
                        progress_callback("Aligning speakers", 0.78)
                    
                    # Convert transcription segments to a standard format
                    # Handle both dictionary format and object format
                    standardized_segments = []
                    for segment in transcription_segments:
                        if isinstance(segment, dict):
                            standardized_segments.append((
                                segment["start"],
                                segment["end"],
                                segment["text"]
                            ))
                        else:
                            # Assume it's an object with start, end, and text attributes
                            standardized_segments.append((
                                segment.start,
                                segment.end,
                                segment.text
                            ))
                    
                    # Get diarization results
                    diarization_segments = future_diarization.result()
                    logger.info("Speaker diarization complete.")
                    
                    # Combine transcription with speaker information
                    logger.info("Combining transcription with speaker information...")
                    if progress_callback:
                        progress_callback("Combining transcript", 0.9)
                    return self._combine_segments_with_speakers(standardized_segments, diarization_segments)
            else:
                # Only run transcription if diarization is disabled
                logger.info("\nStarting transcription...")
                if progress_callback:
                    progress_callback("Transcribing audio", 0.2)
                
                # Get transcription results
                transcription_segments = self.transcription_engine.transcribe(audio_path)
                logger.info(f"\nTranscription complete. Found {len(transcription_segments)} segments.")
                if progress_callback:
                    progress_callback("Finalizing transcript", 0.9)
                
                # Convert transcription segments to a standard format
                standardized_segments = []
                for segment in transcription_segments:
                    if isinstance(segment, dict):
                        standardized_segments.append((
                            segment["start"],
                            segment["end"],
                            segment["text"]
                        ))
                    else:
                        # Assume it's an object with start, end, and text attributes
                        standardized_segments.append((
                            segment.start,
                            segment.end,
                            segment.text
                        ))
                
                # Return transcription without speaker information
                return [(s[0], s[1], s[2], "") for s in standardized_segments]
                
        except Exception as e:
            logger.error(f"\nError during transcription: {str(e)}")
            raise
        finally:
            # Clean up temporary files if needed
            if needs_cleanup and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {audio_path}: {str(e)}")
    
    def save_transcript(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript to file.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        # Set the output format in the formatter
        self.output_formatter.format = self.output_format
        self.output_formatter.save_transcript(segments, output_path)
    
    def transcribe_stream(self, input_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Transcribe an audio or video file using streaming to reduce memory usage.

        Args:
            input_path: Path to the audio or video file

        Yields:
            Transcription segments as they become available

        Raises:
            NotImplementedError: If TRANSCRIPTION_ENGINE != "whisper". Because this
                is a generator function, the error fires on first iteration
                (e.g. ``next()`` or ``list()``), not at call time.
            Exception: If transcription fails
        """
        if self.config.transcription_engine != "whisper":
            raise NotImplementedError(
                "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
                "Use transcribe() for batch transcription with Parakeet."
            )

        logger.info(f"Starting streaming transcription for {input_path}")
        start_time = time.time()

        try:
            # Get the audio path
            audio_path, needs_cleanup = self.audio_processor.get_audio_path(input_path)
            
            # Stream audio from the file
            audio_stream = self.audio_processor.stream_audio_from_file(audio_path)
            
            # Transcribe the audio stream
            for segment in self.transcription_engine.transcribe_stream(audio_stream):
                # For now, we don't include diarization in streaming mode
                # as it requires the full audio file
                yield segment
            
            # Clean up temporary files if needed
            if needs_cleanup and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed temporary audio file: {audio_path}")
            
            elapsed = time.time() - start_time
            logger.info(f"Streaming transcription completed in {elapsed:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error during streaming transcription: {str(e)}")
            raise
    
    def transcribe_stream_with_diarization(self, input_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Transcribe an audio or video file using streaming with diarization.
        This method first performs diarization on the entire file, then streams the transcription.

        Args:
            input_path: Path to the audio or video file

        Yields:
            Transcription segments with speaker information as they become available

        Raises:
            NotImplementedError: If TRANSCRIPTION_ENGINE != "whisper". Because this
                is a generator function, the error fires on first iteration
                (e.g. ``next()`` or ``list()``), not at call time.
            Exception: If transcription or diarization fails
        """
        if self.config.transcription_engine != "whisper":
            raise NotImplementedError(
                "Streaming is only supported with TRANSCRIPTION_ENGINE=whisper. "
                "Use transcribe() for batch transcription with Parakeet."
            )

        logger.info(f"Starting streaming transcription with diarization for {input_path}")
        start_time = time.time()

        try:
            # Get the audio path
            audio_path, needs_cleanup = self.audio_processor.get_audio_path(input_path)
            
            # First, perform diarization on the entire file
            if self.include_diarization:
                logger.info("Performing diarization before streaming transcription")
                diarization_segments = self.diarization_engine.diarize(audio_path)
            else:
                diarization_segments = None
            
            # Stream audio from the file
            audio_stream = self.audio_processor.stream_audio_from_file(audio_path)
            
            # Keep track of all segments to combine with speakers later
            all_segments = []
            
            # Transcribe the audio stream
            for segment in self.transcription_engine.transcribe_stream(audio_stream):
                all_segments.append(segment)
                
                # For streaming output, we'll yield the segment without speaker info first
                # and update it later when all segments are available
                yield segment
            
            # If diarization was performed, combine with transcription
            if diarization_segments:
                logger.info("Combining transcription segments with speaker information")
                combined_segments = self._combine_segments_with_speakers(all_segments, diarization_segments)
                
                # Yield the updated segments with speaker information
                for segment in combined_segments:
                    yield {
                        "start": segment[0],
                        "end": segment[1],
                        "text": segment[2],
                        "speaker": segment[3]
                    }
            
            # Clean up temporary files if needed
            if needs_cleanup and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed temporary audio file: {audio_path}")
            
            elapsed = time.time() - start_time
            logger.info(f"Streaming transcription with diarization completed in {elapsed:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error during streaming transcription with diarization: {str(e)}")
            raise 

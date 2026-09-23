#!/usr/bin/env python3
"""
Stream Transcribe Script

This script demonstrates the streaming transcription functionality.
It processes audio in chunks to reduce memory usage.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.transcriber import Transcriber
from src.utils.progress import ProgressReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main(argv=None):
    """Main function for the streaming transcription script."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio or video files using streaming to reduce memory usage"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to the input audio or video file"
    )
    
    parser.add_argument(
        "--output-path", "-o",
        help="Path to save the transcript (default: input_file_name.txt)"
    )
    
    parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Include speaker diarization"
    )
    
    parser.add_argument(
        "--model", "-m",
        help="Whisper model size (tiny, base, small, medium, large)"
    )
    
    parser.add_argument(
        "--language", "-l",
        help="Language code (e.g., en, fr, de)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "srt", "vtt", "json"],
        help="Output format (default: txt)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args(argv)
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.isfile(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return 1
    
    # Create configuration
    config_kwargs = {}
    if args.model:
        config_kwargs['whisper_model'] = args.model
    if args.language:
        config_kwargs['language'] = args.language
    if args.format:
        config_kwargs['output_format'] = args.format
    if args.diarize:
        config_kwargs['include_diarization'] = True
    
    config = Config(**config_kwargs)
    
    # Streaming is Whisper-only. Parakeet may just be the Apple Silicon
    # default, so stream with Whisper rather than failing out of the box; an
    # explicitly chosen engine is kept (and refused by the transcriber).
    if config.transcription_engine_defaulted and config.transcription_engine != "whisper":
        logger.info(
            f"Streaming requires Whisper; using it instead of the default "
            f"{config.transcription_engine} engine"
        )
        config.transcription_engine = "whisper"
    
    # Generate output path if not specified
    if not args.output_path:
        input_file = Path(args.input_path)
        ext = config.output_format or 'txt'
        output_path = str(input_file.with_suffix(f".{ext}"))
    else:
        output_path = args.output_path
    
    # Create transcriber
    transcriber = Transcriber(config)
    
    # Process the file
    logger.info(f"Processing {args.input_path} using streaming transcription...")
    logger.info(f"Model: {config.whisper_model_size}, Language: {config.language or 'auto'}")
    logger.info(f"Diarization: {'Enabled' if config.include_diarization else 'Disabled'}")
    
    start_time = time.time()
    segments = []
    
    # Create progress reporter
    progress = ProgressReporter(
        desc="Transcribing",
        unit="segment",
        color="blue"
    )
    
    try:
        with progress:
            # Use streaming transcription with diarization
            if config.include_diarization:
                for segment in transcriber.transcribe_stream_with_diarization(args.input_path):
                    segments.append((
                        segment['start'],
                        segment['end'],
                        segment['text'],
                        segment.get('speaker', 'SPEAKER')
                    ))
                    
                    # Update progress
                    progress.update(1, f"Segment {len(segments)}")
                    progress.set_description(f"Transcribed {len(segments)} segments")
                    progress.set_postfix(
                        time=f"{segment['end']:.1f}s",
                        speaker=segment.get('speaker', 'SPEAKER')
                    )
            else:
                # Use streaming transcription without diarization
                for segment in transcriber.transcribe_stream(args.input_path):
                    segments.append((
                        segment['start'],
                        segment['end'],
                        segment['text'],
                        "SPEAKER"
                    ))
                    
                    # Update progress
                    progress.update(1, f"Segment {len(segments)}")
                    progress.set_description(f"Transcribed {len(segments)} segments")
                    progress.set_postfix(time=f"{segment['end']:.1f}s")
    
    except KeyboardInterrupt:
        logger.warning("Transcription interrupted by user")
        if len(segments) > 0:
            logger.info(f"Saving partial transcript with {len(segments)} segments...")
        else:
            logger.error("No segments processed. Exiting without saving.")
            return 1
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        if len(segments) > 0:
            logger.info(f"Saving partial transcript with {len(segments)} segments...")
        else:
            logger.error("No segments processed. Exiting without saving.")
            return 1
    
    # Save transcript
    if len(segments) > 0:
        try:
            transcriber.save_transcript(segments, output_path)
            logger.info(f"Transcript saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving transcript: {str(e)}")
            return 1
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Processed {len(segments)} segments in {elapsed_time:.2f} seconds")
    
    # Get resource usage summary
    resource_summary = progress.get_average_resource_usage()
    logger.info("\nResource usage summary:")
    logger.info(f"  CPU: {resource_summary.get('cpu_percent', 0):.1f}%")
    logger.info(f"  Memory: {resource_summary.get('memory_used_gb', 0):.2f} GB")
    
    if 'gpu_memory_used_gb' in resource_summary and resource_summary['gpu_memory_used_gb'] > 0:
        logger.info(f"  GPU Memory: {resource_summary['gpu_memory_used_gb']:.2f} GB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
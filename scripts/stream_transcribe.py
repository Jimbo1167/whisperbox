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

from src.config import Config, load_env_file
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

# Every format OutputFormatter can write; kept in sync with the Click CLI's
# OUTPUT_FORMATS so `transcribe.py stream -f <fmt>` never trips argparse.
OUTPUT_FORMATS = ["txt", "srt", "vtt", "vtt-voice", "json", "json3", "pretty"]


def main(argv=None):
    """Main function for the streaming transcription script.

    Args:
        argv: Argument list (defaults to sys.argv[1:]). The Click CLI in
            scripts/transcribe.py calls this with an explicit list.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio or video files using streaming to reduce memory usage"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to the input audio or video file"
    )
    
    parser.add_argument(
        "--output", "--output-path", "-o",
        dest="output_path",
        help="Path to save the transcript (default: input_file_name.txt)"
    )
    
    parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Include speaker diarization"
    )

    parser.add_argument(
        "--words", "-w",
        action="store_true",
        help="Include word-level timestamps (JSON output only)"
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
        choices=OUTPUT_FORMATS,
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
    
    load_env_file()

    # Create configuration
    config_kwargs = {}
    if args.model:
        config_kwargs['whisper_model'] = args.model
    if args.language:
        config_kwargs['language'] = args.language
    if args.format:
        config_kwargs['output_format'] = args.format
    # Diarization is opt-in per run; INCLUDE_DIARIZATION only sets the model
    # server's default.
    config_kwargs['include_diarization'] = args.diarize
    # Streaming is Whisper-only (Transcriber.transcribe_stream raises for any
    # other engine), and Parakeet is the platform default on Apple Silicon.
    config_kwargs['transcription_engine'] = 'whisper'

    config = Config(**config_kwargs)

    # Only the JSON formatter has a place for per-word timing; skip the extra
    # alignment work when the words would be dropped anyway.
    word_timestamps = args.words and config.output_format == "json"
    if args.words and not word_timestamps:
        logger.warning(
            f"--words only applies to json output; ignoring it for {config.output_format}"
        )
    
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
    logger.info(f"Word timestamps: {'Enabled' if word_timestamps else 'Disabled'}")
    
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
                for segment in transcriber.transcribe_stream_with_diarization(
                    args.input_path, word_timestamps=word_timestamps
                ):
                    segments.append(_to_row(
                        segment, segment.get('speaker', 'SPEAKER'), word_timestamps
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
                for segment in transcriber.transcribe_stream(
                    args.input_path, word_timestamps=word_timestamps
                ):
                    segments.append(_to_row(segment, "SPEAKER", word_timestamps))
                    
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


def _to_row(segment, speaker, word_timestamps):
    """Convert a streamed segment dict to an OutputFormatter row.

    Rows are (start, end, text, speaker), plus the word list as a fifth
    element when word timestamps were requested (JSON output only).
    """
    row = (segment['start'], segment['end'], segment['text'], speaker)
    if word_timestamps:
        row += (segment.get('words', []),)
    return row


if __name__ == "__main__":
    sys.exit(main()) 
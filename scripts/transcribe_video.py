#!/usr/bin/env python3
"""
Whisperbox - Command line tool to transcribe video and audio files.

This script provides a command-line interface to the Whisperbox library,
allowing users to transcribe video and audio files with optional speaker diarization.
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.server_client import try_server_transcribe
from src.service import TranscriptionService

logger = logging.getLogger(__name__)

def get_default_output_path(input_path, output_format):
    """Generate default output path based on input filename.
    
    Args:
        input_path: Path to the input file
        transcriber: Transcriber instance
        
    Returns:
        Path to the output file
    """
    # Get the input filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # Create output path in transcripts directory with appropriate extension
    return os.path.join("transcripts", f"{base_name}.{output_format}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Transcribe a video or audio file with speaker diarization. '
                   'Supports video files (mov, mp4, etc.) and audio files (wav, mp3, m4a, aac).'
    )
    parser.add_argument('input_path', 
                       help='Path to the video or audio file to transcribe')
    parser.add_argument('--output', '-o', 
                       help='Output path for the transcript (default: transcripts/<input_filename>.<format>)')
    parser.add_argument('--format', '-f', choices=['txt', 'srt', 'vtt', 'json', 'pretty'], 
                       help='Output format (default: from config)')
    parser.add_argument('--no-diarization', action='store_true',
                       help='Disable speaker diarization')
    parser.add_argument('--model', '-m', 
                       help='Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)')
    parser.add_argument('--language', '-l',
                       help='Language code (e.g., en, fr, de)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("\n=== Starting Transcription Process ===")
    start_time = time.time()
    
    logger.info("\nInitializing transcriber...")
    config = Config(".env")  # Explicitly load from .env file
    
    # Override config with command line arguments
    if args.format:
        config.output_format = args.format
    if args.no_diarization:
        config.include_diarization = False
    if args.model:
        config.whisper_model_size = args.model
    if args.language:
        config.language = args.language
    
    # Validate the configuration
    if not config.validate():
        logger.error("Invalid configuration. Please check your settings.")
        sys.exit(1)
    
    # Set default output path if not specified
    if args.output is None:
        args.output = get_default_output_path(args.input_path, config.output_format)

    logger.info(f"\nProcessing {args.input_path}...")
    # Prefer a running warm model server unless CLI flags change the model,
    # which the server (loaded with its own config) can't honor.
    result = None
    if not (args.model or args.language):
        result = try_server_transcribe(args.input_path, config, args.output)
    if result is None:
        service = TranscriptionService(config)
        service.transcribe_file(args.input_path, output_path=args.output)
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nDone! Transcript saved.")
    logger.info(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    logger.info("=====================================")

if __name__ == "__main__":
    main() 

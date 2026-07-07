import sys
from src.config import Config
from src.server_client import transcribe_with_server_fallback
import time
import argparse
import os

def get_default_output_path(input_path, output_format):
    """Generate default output path based on input filename."""
    # Get the input filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # Create output path in transcripts directory with appropriate extension
    return os.path.join("transcripts", f"{base_name}.{output_format}")

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe a video or audio file with speaker diarization. '
                   'Supports video files (mov, mp4, etc.) and audio files (wav, mp3, m4a, aac).'
    )
    parser.add_argument('input_path', 
                       help='Path to the video or audio file to transcribe')
    parser.add_argument('--output', '-o', 
                       help='Output path for the transcript (default: transcripts/<input_filename>.<format>)')
    
    args = parser.parse_args()
    
    print("\n=== Starting Transcription Process ===")
    start_time = time.time()
    
    config = Config(".env")  # Explicitly load from .env file

    # Set default output path if not specified
    if args.output is None:
        args.output = get_default_output_path(args.input_path, config.output_format)

    print(f"\nProcessing {args.input_path}...")
    # Prefer a running warm model server (skips per-run model loading)
    transcribe_with_server_fallback(args.input_path, config, args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nDone! Transcript saved.")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=====================================")

if __name__ == "__main__":
    main() 

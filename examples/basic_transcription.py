#!/usr/bin/env python3
"""
Basic example of using the Whisperbox.
This example shows different ways to use the transcriber,
including different output formats and speaker diarization options.

Usage:
    python examples/basic_transcription.py path/to/your/video.mp4
    python examples/basic_transcription.py path/to/your/video.mp4 --example srt

Transcripts are written to transcripts/ in the current directory.
"""

import argparse
import os
import sys

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config, load_env_file
from src.transcriber import Transcriber


def basic_txt_transcription(video_path):
    """Basic transcription with the settings from .env and the environment"""
    transcriber = Transcriber(Config(output_format="txt"))
    output_path = "transcripts/basic.txt"

    print(f"Transcribing {video_path}...")
    segments = transcriber.transcribe(video_path)
    transcriber.save_transcript(segments, output_path)  # creates transcripts/
    print(f"Transcript saved to {output_path}")


def srt_transcription_no_diarization(video_path):
    """Transcription without speaker diarization, output as SRT"""
    # Keyword overrides take precedence over .env and the environment
    config = Config(include_diarization=False, output_format="srt")

    transcriber = Transcriber(config)
    output_path = "transcripts/no_speakers.srt"

    print(f"Transcribing {video_path} without speaker diarization...")
    segments = transcriber.transcribe(video_path)
    transcriber.save_transcript(segments, output_path)
    print(f"SRT file saved to {output_path}")


def vtt_transcription_with_diarization(video_path):
    """Transcription with speaker diarization, output as VTT"""
    # Choosing a Whisper model selects the Whisper engine, unless
    # TRANSCRIPTION_ENGINE=parakeet is set (then the model is ignored with a
    # warning). base is much faster than the default large-v3-turbo, and less
    # accurate.
    config = Config(include_diarization=True, output_format="vtt", whisper_model="base")

    transcriber = Transcriber(config)
    output_path = "transcripts/with_speakers.vtt"

    print(f"Transcribing {video_path} with speaker diarization...")
    segments = transcriber.transcribe(video_path)
    if transcriber.last_diarization_error:
        print(f"Diarization failed; no speaker labels: {transcriber.last_diarization_error}")
    transcriber.save_transcript(segments, output_path)
    print(f"VTT file saved to {output_path}")


def main():
    """Run the example transcriptions"""
    parser = argparse.ArgumentParser(description="Run the Whisperbox example transcriptions.")
    parser.add_argument("video_path", help="Path to the video or audio file to transcribe")
    parser.add_argument("--example", choices=["txt", "srt", "vtt", "all"], default="all",
                        help="Which example to run (default: all)")
    args = parser.parse_args()

    # Read the project .env (HF_TOKEN, WHISPER_MODEL, ...) as the CLI does;
    # variables already set in the environment win.
    load_env_file()

    if args.example in ("txt", "all"):
        print("=== Running Basic TXT Transcription ===")
        basic_txt_transcription(args.video_path)

    if args.example in ("srt", "all"):
        print("\n=== Running SRT Transcription (No Diarization) ===")
        srt_transcription_no_diarization(args.video_path)

    if args.example in ("vtt", "all"):
        print("\n=== Running VTT Transcription (With Diarization) ===")
        if not os.getenv("HF_TOKEN"):
            print("Skipped: speaker diarization needs HF_TOKEN (set it in .env)")
        else:
            vtt_transcription_with_diarization(args.video_path)


if __name__ == "__main__":
    main()

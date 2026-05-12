#!/usr/bin/env python3
"""
Basic example of using the Whisperbox.
This example shows different ways to use the transcriber,
including different output formats and speaker diarization options.
"""

import os
from src.transcriber import Transcriber
from dotenv import load_dotenv

def basic_txt_transcription():
    """Basic transcription with default settings (txt output)"""
    transcriber = Transcriber()
    video_path = "path/to/your/video.mp4"
    output_path = "transcripts/basic.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Transcribing {video_path}...")
    segments = transcriber.transcribe(video_path)
    transcriber.save_transcript(segments, output_path)
    print(f"Transcript saved to {output_path}")

def srt_transcription_no_diarization():
    """Transcription without speaker diarization, output as SRT"""
    # Load environment variables
    load_dotenv()
    
    # Override environment settings
    os.environ["INCLUDE_DIARIZATION"] = "false"
    os.environ["OUTPUT_FORMAT"] = "srt"
    
    transcriber = Transcriber()
    video_path = "path/to/your/video.mp4"
    output_path = "transcripts/no_speakers.srt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Transcribing {video_path} without speaker diarization...")
    segments = transcriber.transcribe(video_path)
    transcriber.save_transcript(segments, output_path)
    print(f"SRT file saved to {output_path}")

def vtt_transcription_with_diarization():
    """Transcription with speaker diarization, output as VTT"""
    # Load environment variables
    load_dotenv()
    
    # Override environment settings
    os.environ["INCLUDE_DIARIZATION"] = "true"
    os.environ["OUTPUT_FORMAT"] = "vtt"
    os.environ["WHISPER_MODEL"] = "base"  # Use base model for faster processing
    
    transcriber = Transcriber()
    video_path = "path/to/your/video.mp4"
    output_path = "transcripts/with_speakers.vtt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Transcribing {video_path} with speaker diarization...")
    segments = transcriber.transcribe(video_path)
    transcriber.save_transcript(segments, output_path)
    print(f"VTT file saved to {output_path}")

def main():
    """Run all example transcriptions"""
    print("=== Running Basic TXT Transcription ===")
    basic_txt_transcription()
    
    print("\n=== Running SRT Transcription (No Diarization) ===")
    srt_transcription_no_diarization()
    
    print("\n=== Running VTT Transcription (With Diarization) ===")
    vtt_transcription_with_diarization()

if __name__ == "__main__":
    main()
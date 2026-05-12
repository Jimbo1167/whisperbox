"""Script to generate test files for the Whisperbox tests."""

import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip
import wave
import os
from pathlib import Path

def create_test_wav(output_path: str, duration: float = 2.0, sample_rate: int = 16000):
    """Create a test WAV file with a simple sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def create_test_video(output_path: str, duration: float = 2.0, fps: int = 30, with_audio: bool = True):
    """Create a test video file with optional audio."""
    # Create a simple color clip (blue screen)
    size = (320, 240)  # Small size for tests
    color_clip = ColorClip(size, color=(0, 0, 255), duration=duration)
    
    if with_audio:
        # First create a WAV file
        temp_wav = str(Path(output_path).with_suffix('.temp.wav'))
        create_test_wav(temp_wav, duration=duration, sample_rate=44100)
        
        # Load the WAV file as an AudioFileClip
        audio = AudioFileClip(temp_wav)
        
        # Set the audio for the video
        color_clip = color_clip.set_audio(audio)
    
    # Write the video file
    color_clip.write_videofile(output_path, fps=fps, audio=with_audio, logger=None)
    
    # Cleanup
    color_clip.close()
    if with_audio:
        audio.close()
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def main():
    """Generate all test files."""
    # Create fixtures directory
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)
    
    # Generate test files
    print("Generating test files...")
    
    # 1. Basic video with audio
    print("Creating test video with audio...")
    create_test_video(str(fixtures_dir / "test_video.mp4"))
    
    # 2. Video without audio
    print("Creating test video without audio...")
    create_test_video(str(fixtures_dir / "test_video_no_audio.mp4"), with_audio=False)
    
    # 3. Basic WAV file
    print("Creating test WAV file...")
    create_test_wav(str(fixtures_dir / "test_audio.wav"))
    
    print("Done generating test files.")

if __name__ == "__main__":
    main() 
# Progress Reporting Examples

This document provides examples of how to use the progress reporting features in your own code.

## Basic Progress Reporting

The `ProgressReporter` class provides a simple way to track progress during long-running operations. Here's a basic example:

```python
from src.utils.progress import ProgressReporter
import time

# Create a progress reporter with a total of 100 steps
with ProgressReporter(total=100, description="Processing") as progress:
    for i in range(100):
        # Do some work
        time.sleep(0.1)
        
        # Update the progress
        progress.update(1)
        
        # Optionally, set a postfix to show additional information
        progress.set_postfix(item=i, status="processing")
```

This will display a progress bar with the current progress, estimated time remaining, and resource usage.

## Changing Description During Processing

You can change the description of the progress bar during processing:

```python
from src.utils.progress import ProgressReporter
import time

with ProgressReporter(total=100, description="Initializing") as progress:
    # Initial phase
    for i in range(30):
        time.sleep(0.1)
        progress.update(1)
    
    # Change description for the next phase
    progress.set_description("Processing data")
    for i in range(40):
        time.sleep(0.1)
        progress.update(1)
    
    # Change description for the final phase
    progress.set_description("Finalizing")
    for i in range(30):
        time.sleep(0.1)
        progress.update(1)
```

## Adding Checkpoints

You can add checkpoints to track important milestones during processing:

```python
from src.utils.progress import ProgressReporter
import time

with ProgressReporter(total=100, description="Processing") as progress:
    # First phase
    for i in range(25):
        time.sleep(0.1)
        progress.update(1)
    
    # Add a checkpoint
    progress.add_checkpoint("Data loaded")
    
    # Second phase
    for i in range(50):
        time.sleep(0.1)
        progress.update(1)
    
    # Add another checkpoint
    progress.add_checkpoint("Processing complete")
    
    # Final phase
    for i in range(25):
        time.sleep(0.1)
        progress.update(1)
    
    # Add a final checkpoint
    progress.add_checkpoint("Finalized")
```

## Multiple Progress Bars

The `MultiProgressReporter` class allows you to manage multiple progress bars simultaneously:

```python
from src.utils.progress import MultiProgressReporter
import time
import threading

def process_task(multi_progress, name, total, delay):
    for i in range(total):
        time.sleep(delay)
        multi_progress.update(name, 1)

# Create a multi-progress reporter
multi_progress = MultiProgressReporter()

# Add progress reporters for different tasks
multi_progress.add_reporter(name="task1", total=50, description="Task 1")
multi_progress.add_reporter(name="task2", total=100, description="Task 2")
multi_progress.add_reporter(name="task3", total=75, description="Task 3")

# Start threads for each task
threads = []
threads.append(threading.Thread(target=process_task, args=(multi_progress, "task1", 50, 0.2)))
threads.append(threading.Thread(target=process_task, args=(multi_progress, "task2", 100, 0.1)))
threads.append(threading.Thread(target=process_task, args=(multi_progress, "task3", 75, 0.15)))

# Start all threads
for thread in threads:
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Get a summary of all tasks
summary = multi_progress.get_summary()
print("\nSummary:")
for task, task_summary in summary.items():
    print(f"{task}: {task_summary['elapsed_time']:.2f}s, CPU: {task_summary['cpu_percent']:.1f}%, Memory: {task_summary['memory_percent']:.1f}%")
```

## Progress Callback for Libraries

If you're working with a library that reports progress as `(completed, status)`, you can adapt it to your own handler with `create_callback_progress`:

```python
from src.utils.progress import create_callback_progress
import time

def on_progress(completed, total, status):
    print(f"{completed}/{total} {status or ''}")

# Wraps on_progress, filling in the fixed total
callback = create_callback_progress(on_progress, total=100, desc="Processing with callback")

for i in range(100):
    time.sleep(0.1)
    callback(i + 1, "processing")
```

## Resource Usage Monitoring

The progress reporter automatically monitors system resources (CPU, memory, and GPU if available). You can access this information in the summary:

```python
from src.utils.progress import ProgressReporter
import time

with ProgressReporter(total=100, description="Resource monitoring") as progress:
    for i in range(100):
        # Do some work
        time.sleep(0.1)
        progress.update(1)
    
    # Get the summary
    summary = progress.get_summary()
    
    print("\nResource Usage:")
    print(f"CPU: {summary['cpu_percent']:.1f}%")
    print(f"Memory: {summary['memory_percent']:.1f}%")
    if 'gpu_percent' in summary:
        print(f"GPU: {summary['gpu_percent']:.1f}%")
    
    print("\nTime Information:")
    print(f"Elapsed time: {summary['elapsed_time']:.2f}s")
    print(f"Items per second: {summary['items_per_second']:.2f}")
```

## Integration with Transcription

Here's an example of how progress reporting is integrated into the transcription process:

```python
from src.transcriber import Transcriber
from src.config import Config
from src.utils.progress import MultiProgressReporter

# Create a configuration
config = Config()

# Create a transcriber
transcriber = Transcriber(config)

# Create a multi-progress reporter
multi_progress = MultiProgressReporter()

# Add progress reporters for different stages
multi_progress.add_reporter(name="audio", total=100, description="Extracting audio")
multi_progress.add_reporter(name="transcription", total=100, description="Transcribing")
multi_progress.add_reporter(name="diarization", total=100, description="Identifying speakers")

# Process a file with progress reporting
def process_file(input_path, output_path, multi_progress):
    # Get the audio path
    audio_reporter = multi_progress.get_reporter("audio")
    audio_reporter.set_description(f"Extracting audio from {os.path.basename(input_path)}")
    audio_path, needs_cleanup = transcriber.audio_processor.get_audio_path(input_path)
    audio_reporter.update(100)  # Mark as complete
    
    # Transcribe the audio
    transcription_reporter = multi_progress.get_reporter("transcription")
    transcription_reporter.set_description(f"Transcribing {os.path.basename(audio_path)}")
    segments = transcriber.transcription_engine.transcribe(audio_path)
    transcription_reporter.update(100)  # Mark as complete
    
    # Diarize the audio if enabled
    diarization_reporter = multi_progress.get_reporter("diarization")
    if transcriber.config.include_diarization:
        diarization_reporter.set_description(f"Identifying speakers in {os.path.basename(audio_path)}")
        diarization_segments = transcriber.diarization_engine.diarize(audio_path)
        diarization_reporter.update(100)  # Mark as complete
    else:
        diarization_reporter.set_description("Speaker identification disabled")
        diarization_segments = None
        diarization_reporter.update(100)  # Mark as complete
    
    # Combine segments
    combined_segments = transcriber._combine_segments_with_speakers(segments, diarization_segments)
    
    # Save the transcript
    transcriber.save_transcript(combined_segments, output_path)
    
    # Clean up if needed
    if needs_cleanup and os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Return the summary
    return multi_progress.get_summary()

# Process a file
input_path = "path/to/video.mp4"
output_path = "path/to/output.txt"
summary = process_file(input_path, output_path, multi_progress)

# Print the summary
print("\nProcessing Summary:")
for task, task_summary in summary.items():
    print(f"{task}: {task_summary['elapsed_time']:.2f}s, CPU: {task_summary['cpu_percent']:.1f}%, Memory: {task_summary['memory_percent']:.1f}%")
```

This example shows how to use the `MultiProgressReporter` to track progress for different stages of the transcription process. 
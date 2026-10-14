# Progress Reporting Examples

This document provides examples of how to use the progress reporting features in your own code.

## Basic Progress Reporting

The `ProgressReporter` class provides a simple way to track progress during long-running operations. Here's a basic example:

```python
from src.utils.progress import ProgressReporter
import time

# Create a progress reporter with a total of 100 steps
with ProgressReporter(total=100, desc="Processing") as progress:
    for i in range(100):
        # Do some work
        time.sleep(0.1)
        
        # Update the progress
        progress.update(1)
        
        # Optionally, set a postfix to show additional information
        progress.set_postfix(item=i, status="processing")
```

This displays a tqdm progress bar on stdout with the count, rate and estimated time remaining. The bar is created when the `with` block starts (or when you call `start()`) and closed when it ends (`close()`). Every `log_interval` seconds (default 10) the reporter also logs progress at INFO level through the `src.utils.progress` logger.

## Changing Description During Processing

You can change the description of the progress bar during processing:

```python
from src.utils.progress import ProgressReporter
import time

with ProgressReporter(total=100, desc="Initializing") as progress:
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

`update()` also takes a `status` string, which is shown after the description (`"Processing data - <status>"`).

## Adding Checkpoints

You can add checkpoints to track important milestones during processing:

```python
from src.utils.progress import ProgressReporter
import time

with ProgressReporter(total=100, desc="Processing") as progress:
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

for checkpoint in progress.get_summary()["checkpoints"]:
    print(f"{checkpoint['name']}: {checkpoint['completed']} items")
```

Each checkpoint records its `name`, `time`, the number of items `completed` so far, optional `data` (the second argument to `add_checkpoint`), and, while the reporter monitors resources, a `metrics` snapshot.

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

# Create a multi-progress reporter; the with block starts its shared resource
# monitor and closes every progress bar at the end
with MultiProgressReporter() as multi_progress:
    # Add progress reporters for different tasks (each bar starts immediately)
    multi_progress.add_reporter(name="task1", total=50, desc="Task 1")
    multi_progress.add_reporter(name="task2", total=100, desc="Task 2")
    multi_progress.add_reporter(name="task3", total=75, desc="Task 3")

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

    # Get a summary of all tasks and the shared resource usage
    summary = multi_progress.get_summary()
    usage = multi_progress.get_average_resource_usage()

print("\nSummary:")
for task, task_summary in summary.items():
    print(f"{task}: {task_summary['completed']}/{task_summary['total']} ({task_summary['percent']:.0f}%)")
print(f"CPU: {usage['cpu_percent']:.1f}%, Memory: {usage['memory_percent']:.1f}%")
```

`get_summary()` returns one entry per reporter with `total`, `completed`, `percent`, `elapsed`, `elapsed_formatted`, `start_time`, `end_time` and `checkpoints`. It has no resource fields. `elapsed` is measured from when the reporter started (for `add_reporter`, when it was added) to the moment `get_summary()` is called, so it is not the time a finished task took. Record a checkpoint when a task finishes if you need that.

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

Unless you pass `monitor_resources=False`, a `ProgressReporter` samples CPU and memory usage (and GPU memory and utilization on CUDA) once a second in a background thread. Read the samples with `get_resource_usage()` (latest) or `get_average_resource_usage(seconds=5)` (average of the most recent samples). Both return `cpu_percent`, `memory_percent`, `gpu_memory_percent` and `gpu_utilization`; the GPU values are 0.0 without CUDA. `get_summary()` covers progress and time only. Read resource usage before the reporter closes: `close()` (the end of the `with` block) stops the monitor and both methods then return `{}`.

```python
from src.utils.progress import ProgressReporter
import time

with ProgressReporter(total=100, desc="Resource monitoring") as progress:
    for i in range(100):
        # Do some work
        time.sleep(0.1)
        progress.update(1)
    
    # Get the summary and the average resource usage over the last 5 seconds
    summary = progress.get_summary()
    usage = progress.get_average_resource_usage()
    
    print("\nResource Usage:")
    print(f"CPU: {usage['cpu_percent']:.1f}%")
    print(f"Memory: {usage['memory_percent']:.1f}%")
    if usage['gpu_memory_percent'] > 0:
        print(f"GPU memory: {usage['gpu_memory_percent']:.1f}%")
    
    print("\nTime Information:")
    print(f"Elapsed time: {summary['elapsed']:.2f}s")
    print(f"Items per second: {summary['completed'] / summary['elapsed']:.2f}")
```

## Integration with Transcription

The transcription pipeline reports progress through a callback rather than a progress bar. `TranscriptionService.transcribe_file()` and `Transcriber.transcribe()` take a `progress_callback(message, fraction)` and call it at each stage with a message ("Preparing audio", "Transcribing audio", ..., "Completed") and the overall fraction done, from 0.0 to 1.0. The model server uses this to update job progress, and `transcribe --progress jsonl` uses it to emit events (see below). To show the stages as a progress bar:

```python
from src.config import Config, load_env_file
from src.service import TranscriptionService
from src.utils.progress import ProgressReporter

# Read the project .env, as the command-line tools do
load_env_file()
service = TranscriptionService(Config())

with ProgressReporter(total=100, desc="Transcribing", unit="%") as progress:
    def on_progress(message, fraction):
        progress.set_description(message)
        progress.update(max(0, round(fraction * 100) - progress.completed))

    result = service.transcribe_file("path/to/video.mp4", progress_callback=on_progress)

print(f"Saved {len(result['segments'])} segments to {result['output_file']}")
if result["diarization_error"]:
    print(f"Diarization failed; no speaker labels: {result['diarization_error']}")
```

The fraction jumps between stages rather than advancing smoothly: model inference itself reports no progress.

## Structured Progress Events (JSONL)

`transcribe --progress jsonl` writes one JSON object per line to stderr instead of the usual console output, and turns logging down to warnings once it starts:

```bash
python -m scripts.transcribe transcribe path/to/video.mp4 --progress jsonl 2> events.jsonl
```

Other lines can still reach stderr: start-up log messages, warnings and errors, and a traceback if the command fails. Skip lines that don't parse as JSON.

Every event has `event` and `ts` (Unix time in seconds) and `elapsed_s` (seconds since `started`):

- `started`: `input`, `output`, `format`, `diarize`, `engine`, `model` (the model that engine loads: `PARAKEET_MODEL` for Parakeet, the Whisper model otherwise) and `language`.
- `progress`: `stage` (the message as a lowercase slug, e.g. `transcribing_audio`), `stage_label` and `message` (the original message), `progress` (0.0 to 1.0; it never goes backwards), `percent` (0 to 100), and `eta_s` (seconds, or `null` until progress reaches 0.05).
- `completed`: `output`, `segments` (the number of segments), `processing_time`, and `diarization_error` when diarization failed after transcription succeeded (the transcript is then written without speaker labels).
- `error`: `error` (the message). The command then exits with a non-zero status.

The events come from `JsonlProgressEmitter` (`src/utils/progress_events.py`), which you can pass as the `progress_callback` yourself:

```python
import sys

from src.config import Config, load_env_file
from src.service import TranscriptionService
from src.utils.progress_events import JsonlProgressEmitter

load_env_file()
service = TranscriptionService(Config())

emit = JsonlProgressEmitter(stream=sys.stdout)  # default: sys.stderr
emit.emit_started(input="path/to/video.mp4")
result = service.transcribe_file("path/to/video.mp4", progress_callback=emit)
emit.emit_completed(output=result["output_file"], segments=len(result["segments"]))
```

`emit_started()`, `emit_completed()` and `emit_error(message)` add whatever keyword fields you pass to the event.

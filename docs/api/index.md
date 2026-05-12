# API Reference

This document provides detailed information about the Whisperbox API, including classes, methods, and their parameters.

## Table of Contents

1. [Transcriber](#transcriber)
2. [Audio Processor](#audio-processor)
3. [Transcription Engine](#transcription-engine)
4. [Diarization Engine](#diarization-engine)
5. [Output Formatter](#output-formatter)
6. [Cache Manager](#cache-manager)
7. [Progress Reporter](#progress-reporter)
8. [Configuration](#configuration)

## Transcriber

The `Transcriber` class is the main entry point for the Whisperbox. It orchestrates the transcription process by coordinating the audio processing, transcription, diarization, and output formatting components.

### Class: `Transcriber`

```python
from src.transcriber import Transcriber

transcriber = Transcriber(config=None, test_mode=False)
```

#### Parameters

- `config` (Optional[Config]): Configuration object or None to use default configuration
- `test_mode` (bool): If True, use mock models for testing

#### Methods

##### `transcribe`

```python
segments = transcriber.transcribe(input_path)
```

Transcribe a video or audio file with timeouts and progress monitoring.

**Parameters:**
- `input_path` (str): Path to the video or audio file

**Returns:**
- `List[Tuple[float, float, str, str]]`: List of segments with start time, end time, text, and speaker

##### `save_transcript`

```python
transcriber.save_transcript(segments, output_path)
```

Save transcription to a file in the specified format.

**Parameters:**
- `segments` (List[Tuple[float, float, str, str]]): List of segments with start time, end time, text, and speaker
- `output_path` (str): Path to save the transcript

##### `transcribe_stream`

```python
for segment in transcriber.transcribe_stream(input_path):
    # Process segment
```

Transcribe a video or audio file in streaming mode to reduce memory usage.

**Parameters:**
- `input_path` (str): Path to the video or audio file

**Returns:**
- `Generator[Dict[str, Any], None, None]`: Generator yielding segments as they are transcribed

##### `transcribe_stream_with_diarization`

```python
for segment in transcriber.transcribe_stream_with_diarization(input_path):
    # Process segment
```

Transcribe a video or audio file in streaming mode with speaker diarization.

**Parameters:**
- `input_path` (str): Path to the video or audio file

**Returns:**
- `Generator[Dict[str, Any], None, None]`: Generator yielding segments with speaker information

## Audio Processor

The `AudioProcessor` class handles audio extraction and processing.

### Class: `AudioProcessor`

```python
from src.audio.processor import AudioProcessor

audio_processor = AudioProcessor(config)
```

#### Parameters

- `config` (Config): Configuration object

#### Methods

##### `extract_audio`

```python
audio_path = audio_processor.extract_audio(video_path)
```

Extract audio from a video file with timeout.

**Parameters:**
- `video_path` (str): Path to the video file

**Returns:**
- `str`: Path to the extracted audio file

##### `is_audio_file`

```python
is_audio = audio_processor.is_audio_file(file_path)
```

Check if the file is an audio file based on extension.

**Parameters:**
- `file_path` (str): Path to the file

**Returns:**
- `bool`: True if the file is an audio file, False otherwise

##### `get_audio_path`

```python
audio_path, needs_cleanup = audio_processor.get_audio_path(input_path)
```

Get the audio path and whether it needs cleanup.

**Parameters:**
- `input_path` (str): Path to the input file

**Returns:**
- `Tuple[str, bool]`: Path to the audio file and whether it needs cleanup

## Transcription Engine

The `TranscriptionEngine` class manages speech-to-text transcription.

### Class: `TranscriptionEngine`

```python
from src.transcription.engine import TranscriptionEngine

transcription_engine = TranscriptionEngine(config, test_mode=False)
```

#### Parameters

- `config` (Config): Configuration object
- `test_mode` (bool): If True, use mock models for testing

#### Methods

##### `transcribe`

```python
segments = transcription_engine.transcribe(audio_path)
```

Transcribe an audio file with timeout.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- `List[Dict[str, Any]]`: List of segments with start time, end time, and text

##### `transcribe_stream`

```python
for segment in transcription_engine.transcribe_stream(audio_path):
    # Process segment
```

Transcribe an audio file in streaming mode to reduce memory usage.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- `Generator[Dict[str, Any], None, None]`: Generator yielding segments as they are transcribed

## Diarization Engine

The `DiarizationEngine` class handles speaker identification.

### Class: `DiarizationEngine`

```python
from src.diarization.engine import DiarizationEngine

diarization_engine = DiarizationEngine(config, test_mode=False)
```

#### Parameters

- `config` (Config): Configuration object
- `test_mode` (bool): If True, use mock models for testing

#### Methods

##### `diarize`

```python
segments = diarization_engine.diarize(audio_path)
```

Perform speaker diarization with timeout.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- `Optional[List[Dict[str, Any]]]`: List of segments with speaker information or None if diarization is disabled

## Output Formatter

The `OutputFormatter` class formats transcripts in various output formats.

### Class: `OutputFormatter`

```python
from src.output.formatter import OutputFormatter

output_formatter = OutputFormatter(config)
```

#### Parameters

- `config` (Config): Configuration object

#### Methods

##### `save_transcript`

```python
output_formatter.save_transcript(segments, output_path)
```

Save transcription to a file in the specified format.

**Parameters:**
- `segments` (List[Tuple[float, float, str, str]]): List of segments with start time, end time, text, and speaker
- `output_path` (str): Path to save the transcript

## Cache Manager

The `CacheManager` class manages caching of audio, transcription, and diarization results.

### Class: `CacheManager`

```python
from src.cache.manager import CacheManager

cache_manager = CacheManager(config)
```

#### Parameters

- `config` (Config): Configuration object

#### Methods

##### `get_cached_audio`

```python
cached_path = cache_manager.get_cached_audio(input_path)
```

Get the cached audio path for an input file.

**Parameters:**
- `input_path` (str): Path to the input file

**Returns:**
- `Optional[str]`: Path to the cached audio file or None if not cached

##### `cache_audio`

```python
cache_manager.cache_audio(input_path, audio_path)
```

Cache an audio file for an input file.

**Parameters:**
- `input_path` (str): Path to the input file
- `audio_path` (str): Path to the audio file

##### `get_cached_transcription`

```python
segments = cache_manager.get_cached_transcription(audio_path)
```

Get the cached transcription for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- `Optional[List[Dict[str, Any]]]`: Cached transcription segments or None if not cached

##### `cache_transcription`

```python
cache_manager.cache_transcription(audio_path, segments)
```

Cache transcription segments for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `segments` (List[Dict[str, Any]]): Transcription segments

##### `get_cached_diarization`

```python
segments = cache_manager.get_cached_diarization(audio_path)
```

Get the cached diarization for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- `Optional[List[Dict[str, Any]]]`: Cached diarization segments or None if not cached

##### `cache_diarization`

```python
cache_manager.cache_diarization(audio_path, segments)
```

Cache diarization segments for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `segments` (List[Dict[str, Any]]): Diarization segments

## Progress Reporter

The `ProgressReporter` class provides progress reporting for long-running operations.

### Class: `ProgressReporter`

```python
from src.utils.progress import ProgressReporter

progress = ProgressReporter(total=100, description="Processing")
```

#### Parameters

- `total` (int): Total number of steps
- `description` (str): Description of the progress
- `unit` (str): Unit of progress (default: "it")
- `monitor_resources` (bool): Whether to monitor system resources (default: True)

#### Methods

##### `update`

```python
progress.update(n=1)
```

Update the progress by n steps.

**Parameters:**
- `n` (int): Number of steps to update

##### `set_description`

```python
progress.set_description("New description")
```

Set a new description for the progress bar.

**Parameters:**
- `description` (str): New description

##### `set_postfix`

```python
progress.set_postfix(key1="value1", key2="value2")
```

Set postfix text to display after the progress bar.

**Parameters:**
- `**kwargs`: Key-value pairs to display

##### `add_checkpoint`

```python
progress.add_checkpoint("Checkpoint 1")
```

Add a checkpoint to the progress.

**Parameters:**
- `name` (str): Name of the checkpoint

##### `get_summary`

```python
summary = progress.get_summary()
```

Get a summary of the progress, including elapsed time and resource usage.

**Returns:**
- `Dict[str, Any]`: Summary information

### Class: `MultiProgressReporter`

```python
from src.utils.progress import MultiProgressReporter

multi_progress = MultiProgressReporter()
```

#### Methods

##### `add_reporter`

```python
reporter = multi_progress.add_reporter(name="task1", total=100, description="Task 1")
```

Add a progress reporter.

**Parameters:**
- `name` (str): Name of the reporter
- `total` (int): Total number of steps
- `description` (str): Description of the progress
- `unit` (str): Unit of progress (default: "it")
- `monitor_resources` (bool): Whether to monitor system resources (default: True)

**Returns:**
- `ProgressReporter`: The created progress reporter

##### `update`

```python
multi_progress.update(name="task1", n=1)
```

Update the progress of a reporter by n steps.

**Parameters:**
- `name` (str): Name of the reporter
- `n` (int): Number of steps to update

##### `get_reporter`

```python
reporter = multi_progress.get_reporter(name="task1")
```

Get a progress reporter by name.

**Parameters:**
- `name` (str): Name of the reporter

**Returns:**
- `Optional[ProgressReporter]`: The progress reporter or None if not found

##### `get_summary`

```python
summary = multi_progress.get_summary()
```

Get a summary of all progress reporters.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Summary information for all reporters

## Configuration

The `Config` class handles configuration settings for the Whisperbox.

### Class: `Config`

```python
from src.config import Config

config = Config()
```

#### Properties

- `hf_token` (str): HuggingFace token for accessing models
- `whisper_model_size` (str): Whisper model size (tiny, base, small, medium, large-v3)
- `diarization_model` (str): Diarization model to use
- `language` (str): Target language for transcription
- `output_format` (str): Transcript format (txt, srt, vtt, json)
- `include_diarization` (bool): Whether to include speaker diarization
- `cache_enabled` (bool): Whether to enable caching
- `cache_expiration` (int): Cache expiration time in seconds
- `max_cache_size` (int): Maximum cache size in bytes
- `audio_timeout` (int): Timeout for audio extraction in seconds
- `transcribe_timeout` (int): Timeout for transcription in seconds
- `diarize_timeout` (int): Timeout for diarization in seconds
- `model_server_host` (str): Host for the model server
- `model_server_port` (int): Port for the model server 
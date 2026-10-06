# API Reference

This document provides detailed information about the Whisperbox API, including classes, methods, and their parameters.

Import paths assume the project root is on `sys.path` (run from the project root with `python -m ...`, or add the root to `sys.path` as the scripts in `scripts/` and `examples/` do).

## Table of Contents

1. [Transcriber](#transcriber)
2. [Transcription Service](#transcription-service)
3. [Audio Processor](#audio-processor)
4. [Transcription Engine](#transcription-engine)
5. [Diarization Engine](#diarization-engine)
6. [Output Formatter](#output-formatter)
7. [Cache Manager](#cache-manager)
8. [Progress Reporter](#progress-reporter)
9. [Configuration](#configuration)

## Transcriber

The `Transcriber` class is the main entry point for the Whisperbox. It orchestrates the transcription process by coordinating the audio processing, transcription, diarization, and output formatting components.

### Class: `Transcriber`

```python
from src.transcriber import Transcriber

transcriber = Transcriber(config=None, test_mode=False)
```

#### Parameters

- `config` (Optional[Config]): Configuration object or None to use default configuration (`Config()`, which reads the environment only; see [`load_env_file`](#function-load_env_file))
- `test_mode` (bool): If True, use mock models for testing

The components are available as `transcriber.audio_processor`, `transcriber.transcription_engine` (built by [`make_asr_engine`](#factory-make_asr_engine)), `transcriber.diarization_engine` and `transcriber.output_formatter`. Models load on first use, not in the constructor.

#### Methods

##### `transcribe`

```python
segments = transcriber.transcribe(input_path, progress_callback=None, include_diarization=None)
```

Transcribe a video or audio file. Audio extraction, transcription and diarization each run under their configured timeouts. With diarization on, transcription and diarization run concurrently.

**Parameters:**
- `input_path` (str): Path to the video or audio file
- `progress_callback` (Optional[Callable[[str, float], None]]): Called with a stage message and the overall fraction done (0.0 to 1.0); see [Progress Reporting Examples](../examples/progress_reporting.md#integration-with-transcription)
- `include_diarization` (Optional[bool]): Diarize this call; None uses `config.include_diarization`

**Returns:**
- `List[Tuple[float, float, str, str]]`: List of segments with start time, end time, text, and speaker (an empty string without diarization)

If the diarization model can't be loaded (for example, no `HF_TOKEN`), the call raises before transcribing. If diarization fails after transcription succeeded, the segments are returned without speaker labels and the error message is stored in `transcriber.last_diarization_error` (None otherwise).

##### `save_transcript`

```python
transcriber.save_transcript(segments, output_path)
```

Save transcription to a file in the output format the transcriber was created with (`config.output_format`).

**Parameters:**
- `segments` (List[Tuple[float, float, str, str]]): List of segments with start time, end time, text, and speaker
- `output_path` (str): Path to save the transcript

##### `transcribe_stream`

```python
for segment in transcriber.transcribe_stream(input_path, word_timestamps=False):
    # Process segment
```

Transcribe a video or audio file in streaming mode to reduce memory usage. Whisper only: if `config.transcription_engine` is not `whisper`, the generator raises `NotImplementedError` when first iterated. No diarization.

**Parameters:**
- `input_path` (str): Path to the video or audio file
- `word_timestamps` (bool): Fill each segment's `words` with per-word timing

**Returns:**
- `Generator[Dict[str, Any], None, None]`: Generator yielding segments (`start`, `end`, `text`, `words`) as they are transcribed

##### `transcribe_stream_with_diarization`

```python
for segment in transcriber.transcribe_stream_with_diarization(input_path, word_timestamps=False):
    # Process segment
```

Transcribe a video or audio file in streaming mode with speaker diarization. If `config.include_diarization` is set, the whole file is diarized first and each streamed segment is labeled with its speaker. Whisper only, like `transcribe_stream`.

**Parameters:**
- `input_path` (str): Path to the video or audio file
- `word_timestamps` (bool): Fill each segment's `words` with per-word timing

**Returns:**
- `Generator[Dict[str, Any], None, None]`: Generator yielding segments with `start`, `end`, `text`, `speaker` and `words` (without diarization, segments are yielded as from `transcribe_stream`)

## Transcription Service

The `TranscriptionService` class wraps a `Transcriber` for applications: it builds the transcriber lazily, writes the transcript, and serializes calls with a lock so one instance can be shared between threads. The command-line `transcribe` command and the model server use it.

### Class: `TranscriptionService`

```python
from src.service import TranscriptionService

service = TranscriptionService(config=None, test_mode=False, preload_models=False)
```

#### Parameters

- `config` (Optional[Config]): Configuration object or None to use `Config()`
- `test_mode` (bool): If True, use mock models for testing
- `preload_models` (bool): Load the models in the constructor (see `preload_models`)

#### Properties

- `transcriber` (Transcriber): The underlying transcriber, created on first access
- `engine_name` (str): The ASR engine actually in use, `"whisper"` or `"parakeet"`. Read from the constructed engine, so it reflects the fallback to Whisper when Parakeet was only the platform default and parakeet-mlx isn't installed.
- `model_name` (str): The model that engine loads: the Parakeet model id, or the Whisper model

#### Methods

##### `transcribe_file`

```python
result = service.transcribe_file(input_path, output_path=None, output_format=None,
                                 progress_callback=None, include_diarization=None)
```

Transcribe a file and write the transcript.

**Parameters:**
- `input_path` (str): Path to the video or audio file
- `output_path` (Optional[str]): Where to write the transcript (default: `transcripts/<input name>.<format>` in the current directory)
- `output_format` (Optional[str]): One of [`OUTPUT_FORMATS`](#constant-output_formats) (default: `config.output_format`)
- `progress_callback` (Optional[Callable[[str, float], None]]): As for `Transcriber.transcribe`; the service adds a "Saving transcript" (0.96) and a "Completed" (1.0) call
- `include_diarization` (Optional[bool]): Diarize this call; None uses `config.include_diarization`

**Returns:**
- `Dict[str, Any]` with:
  - `segments`: List of (start, end, text, speaker) tuples
  - `preview_text` (str): The transcript as written to the file
  - `output_format` (str), `output_file` (str)
  - `processing_time` (float): Seconds
  - `diarization_error` (Optional[str]): None, or the error message when diarization was requested but failed after transcription succeeded (the transcript is then written without speaker labels)

##### `transcribe_existing_audio`

```python
result = service.transcribe_existing_audio(audio_path, include_diarization=None)
```

Transcribe a file without writing a transcript. The model server's JSON `/transcribe` endpoint uses it. Despite the name, it accepts anything `Transcriber.transcribe` accepts, including video.

**Parameters:**
- `audio_path` (str): Path to the audio or video file
- `include_diarization` (Optional[bool]): Diarize this call; None uses `config.include_diarization`

**Returns:**
- `Dict[str, Any]` with `segments`, `processing_time` and `diarization_error` (as for `transcribe_file`)

##### `preload_models`

```python
service.preload_models()
```

Load the transcription model and, if `config.include_diarization` is set, the diarization model. If the diarization model fails to load, diarization is turned off for this service with a warning instead of raising.

##### `build_output_path`

```python
output_path = service.build_output_path(input_path, output_format=None)
```

Return `transcripts/<input name>.<format>`, creating the `transcripts` directory in the current directory if needed.

## Audio Processor

The `AudioProcessor` class handles audio extraction and processing. It decodes audio and video with the ffmpeg binary bundled with imageio-ffmpeg.

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

Extract audio from a video file to a temporary 16 kHz mono WAV, with timeout (`config.audio_timeout`). The caller deletes the file.

**Parameters:**
- `video_path` (str): Path to the video file

**Returns:**
- `str`: Path to the extracted audio file

##### `get_audio_path`

```python
audio_path, needs_cleanup = audio_processor.get_audio_path(input_path)
```

Get a WAV file for an input file. A `.wav` input is returned as is. Anything else is converted to a 16 kHz mono WAV: into the audio cache when caching is enabled (`needs_cleanup` is False), otherwise into a temporary file (`needs_cleanup` is True).

**Parameters:**
- `input_path` (str): Path to the input file

**Returns:**
- `Tuple[str, bool]`: Path to the audio file and whether it needs cleanup

**Raises:**
- `FileNotFoundError`: If the input file doesn't exist

##### `stream_audio_from_file`

```python
for chunk in audio_processor.stream_audio_from_file(audio_path, chunk_duration=5.0, target_sr=16000):
    # Process chunk
```

Read a WAV file in chunks, for streaming transcription.

**Parameters:**
- `audio_path` (str): Path to a WAV file (use `get_audio_path` first for other inputs)
- `chunk_duration` (float): Length of each chunk in seconds
- `target_sr` (int): Sample rate to resample to

**Returns:**
- `Iterator[np.ndarray]`: Mono float32 chunks

##### `load_audio`

```python
audio = audio_processor.load_audio(audio_path, target_sr=16000)
```

Load a whole WAV file as a mono float32 array resampled to `target_sr`.

## Transcription Engine

Speech-to-text is handled by ASR engine classes that implement the `ASREngine`
protocol. Two engines exist: `WhisperEngine` (faster-whisper) and
`ParakeetEngine` (parakeet-mlx, Apple Silicon only). Use the factory to pick the
engine configured via `TRANSCRIPTION_ENGINE` (default: parakeet on Apple Silicon,
whisper elsewhere):

### Factory: `make_asr_engine`

```python
from src.transcription.engine import make_asr_engine

engine = make_asr_engine(config, test_mode=False)
```

#### Parameters

- `config` (Config): Configuration object (`config.transcription_engine` selects the engine)
- `test_mode` (bool): If True, use mock models for testing

**Returns:**
- An `ASREngine` instance (`WhisperEngine` or `ParakeetEngine`). When Parakeet is only the platform default (`config.transcription_engine_defaulted`) and parakeet-mlx isn't installed, it returns a `WhisperEngine` with a warning; an explicitly requested Parakeet is not replaced.

**Raises:**
- `ValueError`: If the engine name is unknown

### Classes: `WhisperEngine`, `ParakeetEngine`

```python
from src.transcription.engine import WhisperEngine, ParakeetEngine

engine = WhisperEngine(config, test_mode=False)
```

`WhisperEngine` runs on CUDA when available and on the CPU otherwise (never on Apple's GPU); `FORCE_CPU` forces the CPU. `ParakeetEngine` runs on the Apple GPU through MLX, ignores `FORCE_CPU`, and detects the language itself.

#### Parameters

- `config` (Config): Configuration object
- `test_mode` (bool): If True, use mock models for testing

#### Methods

##### `ensure_model_loaded`

```python
engine.ensure_model_loaded()
```

Load the model if it isn't loaded yet (downloading it on first use). `transcribe` calls this itself.

##### `transcribe`

```python
segments = engine.transcribe(audio_path)
```

Transcribe an audio file with timeout (`config.transcribe_timeout`). Results are cached per engine and model (and, for Whisper, language and beam size) when caching is enabled.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- `List[Dict[str, Any]]`: List of segments with `start`, `end`, `text` and `words` (a list of `start`, `end`, `word` dicts; Whisper leaves it empty here)

##### `transcribe_stream`

```python
audio_path, needs_cleanup = audio_processor.get_audio_path(input_path)
for segment in engine.transcribe_stream(audio_processor.stream_audio_from_file(audio_path),
                                        word_timestamps=False):
    # Process segment
```

Transcribe a stream of audio chunks, yielding segments as they are transcribed, to reduce memory usage. This method takes audio data, not a path; to stream a file by path, use [`Transcriber.transcribe_stream`](#transcribe_stream). Whisper only —
`ParakeetEngine` raises `NotImplementedError` for streaming entry points.

**Parameters:**
- `audio_stream` (Iterator[np.ndarray]): Mono float32 audio chunks at 16 kHz, such as those from `AudioProcessor.stream_audio_from_file`
- `word_timestamps` (bool): Fill each segment's `words` with per-word timing

**Returns:**
- `Generator[Dict[str, Any], None, None]`: Generator yielding segments (`start`, `end`, `text`, `words`) as they are transcribed

## Diarization Engine

The `DiarizationEngine` class handles speaker identification with pyannote. It needs `HF_TOKEN` with access to the model named by `DIARIZATION_MODEL`.

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
segments = diarization_engine.diarize(audio_path, enabled=None)
```

Perform speaker diarization with timeout (`config.diarize_timeout`).

**Parameters:**
- `audio_path` (str): Path to the audio file
- `enabled` (Optional[bool]): Diarize this call; None uses `config.include_diarization`

**Returns:**
- `Optional[List[Dict[str, Any]]]`: List of segments with `start`, `end` and `speaker`, sorted by start time, or None if diarization is disabled

##### `ensure_model_loaded`

```python
diarization_engine.ensure_model_loaded(force=False)
```

Load the diarization model if diarization is enabled (or `force` is True) and the model isn't loaded yet. Raises if the model can't be loaded.

## Output Formatter

The `OutputFormatter` class formats transcripts in various output formats.

### Class: `OutputFormatter`

```python
from src.output.formatter import OutputFormatter

output_formatter = OutputFormatter(config)
```

#### Parameters

- `config` (Config): Configuration object

The format is taken from `config.output_format` and stored in `output_formatter.format`; assign that attribute to change it. Supported formats are listed in [`OUTPUT_FORMATS`](#constant-output_formats).

#### Methods

##### `save_transcript`

```python
output_formatter.save_transcript(segments, output_path)
```

Save transcription to a file in the current format, creating the parent directory if needed.

**Parameters:**
- `segments` (List[Tuple[float, float, str, str]]): List of segments with start time, end time, text, and speaker
- `output_path` (str): Path to save the transcript

**Raises:**
- `ValueError`: If the format is not supported

##### `format_transcript`

```python
text = output_formatter.format_transcript(segments)
```

Return the transcript in the current format as a string, without writing a file.

**Parameters:**
- `segments` (List[Tuple[float, float, str, str]]): List of segments with start time, end time, text, and speaker

**Returns:**
- `str`: The formatted transcript

## Cache Manager

The `CacheManager` class manages caching of audio, transcription, and diarization results under `~/.cache/whisperbox/{audio,transcription,diarization}`. Entries are keyed on the file's path, size and modification time, expire after `config.cache_expiration` seconds, and the oldest are removed when the cache grows past `config.max_cache_size`; this cleanup runs whenever a `CacheManager` is created. The engines and the audio processor use it when `config.cache_enabled` is set.

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

##### `audio_cache_path`

```python
cache_path = cache_manager.audio_cache_path(input_path)
```

Get the path where the extracted audio for an input file belongs in the cache (`AudioProcessor` extracts straight to it).

**Parameters:**
- `input_path` (str): Path to the input file

**Returns:**
- `str`: Path inside the audio cache

**Raises:**
- `FileNotFoundError`: If the input file doesn't exist

##### `cache_audio`

```python
cached_path = cache_manager.cache_audio(input_path, audio_path)
```

Copy an audio file into the cache for an input file.

**Parameters:**
- `input_path` (str): Path to the input file
- `audio_path` (str): Path to the audio file

**Returns:**
- `str`: Path to the cached audio file

##### `get_cached_transcription`

```python
segments = cache_manager.get_cached_transcription(audio_path, engine_id="whisper")
```

Get the cached transcription for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `engine_id` (str): Identifies the engine and the settings that change its output; entries for one ID are never returned for another

**Returns:**
- `Optional[List[Dict[str, Any]]]`: Cached transcription segments or None if not cached

##### `cache_transcription`

```python
cache_manager.cache_transcription(audio_path, transcription_results, engine_id="whisper")
```

Cache transcription segments for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `transcription_results` (List[Dict[str, Any]]): Transcription segments
- `engine_id` (str): As for `get_cached_transcription`

##### `get_cached_diarization`

```python
segments = cache_manager.get_cached_diarization(audio_path, model_id="pyannote")
```

Get the cached diarization for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `model_id` (str): The diarization model (`DIARIZATION_MODEL`); results from another model are never returned

**Returns:**
- `Optional[List[Dict[str, Any]]]`: Cached diarization segments or None if not cached

##### `cache_diarization`

```python
cache_manager.cache_diarization(audio_path, diarization_results, model_id="pyannote")
```

Cache diarization segments for an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `diarization_results` (List[Dict[str, Any]]): Diarization segments
- `model_id` (str): As for `get_cached_diarization`

##### `clear_cache`

```python
cache_manager.clear_cache(cache_type=None)
```

Delete cached files.

**Parameters:**
- `cache_type` (Optional[str]): `"audio"`, `"transcription"` or `"diarization"`; None clears all three

## Progress Reporter

The `ProgressReporter` class provides progress reporting for long-running operations. See [Progress Reporting Examples](../examples/progress_reporting.md) for runnable examples.

### Class: `ProgressReporter`

```python
from src.utils.progress import ProgressReporter

progress = ProgressReporter(total=100, desc="Processing")
```

#### Parameters

- `total` (int): Total number of steps (default: 100)
- `desc` (str): Description of the progress (default: "Processing")
- `unit` (str): Unit of progress (default: "it")
- `monitor_resources` (bool): Whether to monitor system resources (default: True)
- `log_interval` (int): Seconds between progress log messages (default: 10)
- `color` (Optional[str]): Progress bar color, passed to tqdm (default: None)

The progress bar is shown between `start()` and `close()`; using the reporter as a context manager (`with ProgressReporter(...) as progress:`) calls both.

#### Methods

##### `update`

```python
progress.update(n=1, status=None)
```

Update the progress by n steps.

**Parameters:**
- `n` (int): Number of steps to update
- `status` (Optional[str]): Text to show after the description

##### `set_description`

```python
progress.set_description("New description")
```

Set a new description for the progress bar.

**Parameters:**
- `desc` (str): New description

##### `set_postfix`

```python
progress.set_postfix(key1="value1", key2="value2")
```

Set postfix text to display after the progress bar.

**Parameters:**
- `**kwargs`: Key-value pairs to display

##### `add_checkpoint`

```python
progress.add_checkpoint("Checkpoint 1", data=None)
```

Add a checkpoint to the progress.

**Parameters:**
- `name` (str): Name of the checkpoint
- `data` (Optional[Dict[str, Any]]): Data to store with the checkpoint

##### `get_summary`

```python
summary = progress.get_summary()
```

Get a summary of the progress.

**Returns:**
- `Dict[str, Any]`: `total`, `completed`, `percent`, `elapsed` (seconds since `start()`, measured now), `elapsed_formatted`, `start_time`, `end_time` (now) and `checkpoints`

##### `get_resource_usage` / `get_average_resource_usage`

```python
usage = progress.get_resource_usage()
usage = progress.get_average_resource_usage(seconds=5)
```

Get the latest resource sample, or the average of the most recent samples (taken once a second).

**Returns:**
- `Dict[str, Any]`: `cpu_percent`, `memory_percent`, `gpu_memory_percent` and `gpu_utilization` (GPU values are 0.0 without CUDA); empty if resources aren't monitored or the reporter is closed

`get_elapsed_time()`, `get_estimated_time_remaining()` and their `get_formatted_*` variants return the timing on its own.

### Class: `MultiProgressReporter`

```python
from src.utils.progress import MultiProgressReporter

multi_progress = MultiProgressReporter()
```

The reporters share one resource monitor, which runs between `start()` and `close()` (or inside a `with` block); `close()` also closes every reporter.

#### Methods

##### `add_reporter`

```python
reporter = multi_progress.add_reporter(name="task1", total=100, desc="Task 1")
```

Add a progress reporter and start its progress bar.

**Parameters:**
- `name` (str): Name of the reporter
- `total` (int): Total number of steps (default: 100)
- `desc` (str): Description of the progress (default: "Processing")
- `unit` (str): Unit of progress (default: "it")
- `color` (Optional[str]): Progress bar color (default: None)

**Returns:**
- `ProgressReporter`: The created progress reporter

##### `update`

```python
multi_progress.update(name="task1", n=1, status=None)
```

Update the progress of a reporter by n steps.

**Parameters:**
- `name` (str): Name of the reporter
- `n` (int): Number of steps to update
- `status` (Optional[str]): Text to show after the description

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
- `Dict[str, Dict[str, Any]]`: `ProgressReporter.get_summary()` for each reporter, by name

`get_resource_usage()` and `get_average_resource_usage(seconds=5)` return the shared monitor's samples, as for `ProgressReporter`.

### Function: `create_callback_progress`

```python
from src.utils.progress import create_callback_progress

callback = create_callback_progress(on_progress, total=100)
callback(completed, status)  # calls on_progress(completed, 100, status)
```

Adapt a `(completed, total, status)` handler to a `(completed, status)` callback.

### Class: `JsonlProgressEmitter`

```python
from src.utils.progress_events import JsonlProgressEmitter

emit = JsonlProgressEmitter(stream=None)
```

A `progress_callback` for `Transcriber.transcribe` and `TranscriptionService.transcribe_file` that writes one JSON event per line to `stream` (default: `sys.stderr`). `transcribe --progress jsonl` uses it. `emit_started(**fields)`, `emit_completed(**fields)` and `emit_error(error, **fields)` write the other event types. The event schema is in [Progress Reporting Examples](../examples/progress_reporting.md#structured-progress-events-jsonl).

## Configuration

The `Config` class handles configuration settings for the Whisperbox.

### Function: `load_env_file`

```python
from src.config import load_env_file

loaded = load_env_file(env_file=None)
```

Load a `.env` file into the environment without overriding variables that are already set (exported in the shell, or set inline as in `INCLUDE_DIARIZATION=true python ...`). Every command-line entry point calls it before building a `Config`. `Config()` itself reads only the environment, so call `load_env_file()` first if your code should honor `.env`.

**Parameters:**
- `env_file` (Optional[Union[str, Path]]): The file to load (default: `.env` in the project root, whatever the current directory)

**Returns:**
- `Optional[Path]`: The file that was loaded, or None if it doesn't exist

### Constant: `OUTPUT_FORMATS`

```python
from src.config import OUTPUT_FORMATS
# ["txt", "srt", "vtt", "vtt-voice", "json", "json3", "pretty"]
```

Every format `OutputFormatter` can write. The command-line tools, the model server and `Config.validate()` take their choices from this list.

### Class: `Config`

```python
from src.config import Config

config = Config()
config = Config(output_format="srt", include_diarization=False)
```

#### Parameters

- `env_file` (Optional[str]): A `.env` file to load first, with the same rule as `load_env_file` (the environment wins)
- `**overrides`: Values that take precedence over the environment. Supported keys: `whisper_model`, `language`, `output_format`, `include_diarization`, `diarization_model`, `force_cpu`, `transcription_engine`, `parakeet_model`; other keys are ignored.

A `whisper_model` override selects the Whisper engine unless the engine was chosen explicitly (a `transcription_engine` override or `TRANSCRIPTION_ENGINE`); in that case it only logs a warning that the model has no effect. A `language` override under Parakeet logs a warning and is ignored. The command-line `--model`, `--engine` and `--language` options are passed as these overrides.

#### Properties

Each property reads the environment variable shown, unless overridden.

- `hf_token` (str, `HF_TOKEN`): HuggingFace token for accessing models
- `transcription_engine` (str, `TRANSCRIPTION_ENGINE`): ASR engine to use (`whisper` or `parakeet`; default: parakeet on Apple Silicon, whisper elsewhere)
- `transcription_engine_defaulted` (bool): True when the engine came from the platform default rather than the environment or an override
- `whisper_model_size` (str, `WHISPER_MODEL`): Whisper model: any name faster-whisper accepts, such as tiny, base, small, medium, large-v3, large-v3-turbo or distil-large-v3, a Hugging Face repo id, or a local path (default: large-v3-turbo)
- `parakeet_model` (str, `PARAKEET_MODEL`): HF model id or local path to MLX-format Parakeet weights (default: mlx-community/parakeet-tdt-0.6b-v3)
- `diarization_model` (str, `DIARIZATION_MODEL`): Diarization model to use (default: pyannote/speaker-diarization-community-1)
- `language` (str, `LANGUAGE`): Target language for Whisper (default: en; Parakeet detects the language)
- `output_format` (str, `OUTPUT_FORMAT`): Transcript format, one of `OUTPUT_FORMATS` (default: txt)
- `include_diarization` (bool, `INCLUDE_DIARIZATION`): Default for speaker diarization. The model server and `transcribe_video.py` use it; the CLI's `transcribe`, `stream` and `batch` commands diarize only with `--diarize`.
- `force_cpu` (bool, `FORCE_CPU`): Run Whisper and pyannote on the CPU even when CUDA is available; no effect on Parakeet (default: false)
- `whisper_beam_size` (int, `WHISPER_BEAM_SIZE`): Whisper beam size; 1 is greedy decoding (default: 5)
- `whisper_cpu_threads` (int, `WHISPER_CPU_THREADS`): Whisper CPU threads; 0 keeps the library default (default: 0)
- `whisper_batch_size` (int, `WHISPER_BATCH_SIZE`): Decode Whisper chunks in parallel batches of this size; 0 decodes sequentially (default: 0)
- `cache_enabled` (bool, `CACHE_ENABLED`): Whether to enable caching (default: true)
- `cache_expiration` (int, `CACHE_EXPIRATION`): Cache expiration time in seconds (default: 604800, 7 days)
- `max_cache_size` (int, `MAX_CACHE_SIZE`): Maximum cache size in bytes (default: 10 GB)
- `audio_timeout` (int, `AUDIO_TIMEOUT`): Timeout for audio extraction in seconds (default: 300)
- `transcribe_timeout` (int, `TRANSCRIBE_TIMEOUT`): Timeout for transcription in seconds (default: 3600)
- `diarize_timeout` (int, `DIARIZE_TIMEOUT`): Timeout for diarization in seconds (default: 3600)

#### Methods

- `to_dict()`: The settings as a dictionary
- `validate()`: Return False, with a log message, if diarization is on without `HF_TOKEN`, the output format or engine is unknown, or Parakeet is selected off Apple Silicon. Only `transcribe_video.py` calls it.

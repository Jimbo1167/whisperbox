#!/usr/bin/env python3
"""
Enhanced CLI for Video Transcriber using Click.

This script provides a modern command-line interface for the Video Transcriber
with features like command completion, color output, and better help messages.
"""

import os
import sys
import time
import click
import logging
from pathlib import Path
from typing import Optional, List, Tuple

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.service import TranscriptionService
from src.utils.progress_events import JsonlProgressEmitter
from src.utils.resource_monitor import ResourceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define output format options
OUTPUT_FORMATS = ['txt', 'srt', 'vtt', 'json', 'pretty']

# Define model size options
MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large']

# Progress reporting modes. ``pretty`` is the legacy click-colored output;
# ``jsonl`` emits machine-readable events on stderr for programmatic callers;
# ``none`` silences both.
PROGRESS_MODES = ['pretty', 'jsonl', 'none']

def print_version(ctx, param, value):
    """Print version information and exit."""
    if not value or ctx.resilient_parsing:
        return
    click.echo("Video Transcriber v0.2.0")
    ctx.exit()

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show version and exit.')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging.')
@click.pass_context
def cli(ctx, verbose):
    """Video Transcriber - Convert audio and video to text with speaker diarization.
    
    This tool provides various commands for transcribing audio and video files,
    with options for streaming, batch processing, and using a model server.
    """
    # Set up logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Create a context object for sharing data between commands
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path.')
@click.option('--diarize', '-d', is_flag=True, help='Include speaker diarization.')
@click.option('--model', '-m', type=click.Choice(MODEL_SIZES), help='Whisper model size.')
@click.option('--language', '-l', help='Language code (e.g., en, fr, de).')
@click.option('--format', '-f', 'output_format', type=click.Choice(OUTPUT_FORMATS),
              help='Output format.')
@click.option('--progress', 'progress_mode', type=click.Choice(PROGRESS_MODES),
              default='pretty', show_default=True,
              help='Progress reporting mode. Use "jsonl" to emit one JSON event '
                   'per line on stderr for programmatic callers.')
def transcribe(input_path, output, diarize, model, language, output_format, progress_mode):
    """Transcribe an audio or video file.

    This command transcribes the given audio or video file and saves the result
    to the specified output file. If no output file is specified, the result is
    saved to a file with the same name as the input file but with a different
    extension based on the output format.

    Examples:
        transcribe video.mp4
        transcribe audio.mp3 --output transcript.txt
        transcribe interview.wav --diarize --model medium
        transcribe meeting.mp4 --progress jsonl 2> events.jsonl
    """
    start_time = time.time()

    # Create configuration
    config_kwargs = {}
    if model:
        config_kwargs['whisper_model'] = model
    if language:
        config_kwargs['language'] = language
    if output_format:
        config_kwargs['output_format'] = output_format
    if diarize:
        config_kwargs['include_diarization'] = True

    config = Config(**config_kwargs)

    service = TranscriptionService(config)

    # Generate output path if not provided
    if not output:
        input_file = Path(input_path)
        output_dir = Path('transcripts')
        output_dir.mkdir(exist_ok=True)

        ext = output_format or config.output_format or 'txt'

        output = str(output_dir / f"{input_file.stem}.{ext}")

    jsonl_emitter: Optional[JsonlProgressEmitter] = None
    if progress_mode == 'jsonl':
        # Keep stderr parseable: silence INFO logs from the pipeline and skip
        # the human-friendly click.echo lines below.
        logging.getLogger().setLevel(logging.WARNING)
        jsonl_emitter = JsonlProgressEmitter()
        jsonl_emitter.emit_started(
            input=input_path,
            output=output,
            format=config.output_format,
            diarize=bool(config.include_diarization),
            model=config.whisper_model_size,
            language=config.language,
        )

    try:
        with ResourceMonitor() as monitor:
            if progress_mode != 'jsonl':
                click.echo(click.style(f"Transcribing {input_path}...", fg="green"))
                click.echo(click.style(f"Saving transcript to {output}...", fg="green"))
            result = service.transcribe_file(
                input_path,
                output_path=output,
                progress_callback=jsonl_emitter,
            )
    except Exception as exc:
        if jsonl_emitter is not None:
            jsonl_emitter.emit_error(str(exc))
        raise

    if jsonl_emitter is not None:
        jsonl_emitter.emit_completed(
            output=result.get("output_file", output),
            segments=len(result.get("segments", [])),
            processing_time=result.get("processing_time"),
        )
        return

    # Pretty-mode resource + timing summary (legacy behavior).
    metrics = monitor.get_average_metrics()
    click.echo(click.style("\nResource usage:", fg="blue"))
    click.echo(f"  CPU: {metrics['cpu_percent']:.1f}%")
    click.echo(f"  Memory: {metrics['memory_percent']:.1f}%")
    if metrics.get('gpu_memory_percent', 0) > 0:
        click.echo(f"  GPU Memory: {metrics['gpu_memory_percent']:.1f}%")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    click.echo(click.style(f"\nTranscription completed in {int(minutes)}m {seconds:.2f}s", fg="green"))

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path.')
@click.option('--diarize', '-d', is_flag=True, help='Include speaker diarization.')
@click.option('--words', '-w', is_flag=True, help='Include word-level timestamps.')
@click.option('--model', '-m', type=click.Choice(MODEL_SIZES), help='Whisper model size.')
@click.option('--language', '-l', help='Language code (e.g., en, fr, de).')
@click.option('--format', '-f', 'output_format', type=click.Choice(OUTPUT_FORMATS), 
              help='Output format.')
def stream(input_path, output, diarize, words, model, language, output_format):
    """Transcribe using streaming to reduce memory usage.
    
    This command transcribes the given audio or video file using streaming
    to reduce memory usage, which is useful for large files.
    
    Examples:
        stream large_video.mp4
        stream podcast.mp3 --diarize
    """
    # Import here to avoid circular imports
    from scripts.stream_transcribe import main as stream_main
    
    # Prepare arguments
    args = ['--input', input_path]
    
    if output:
        args.extend(['--output', output])
    if diarize:
        args.append('--diarize')
    if words:
        args.append('--words')
    if model:
        args.extend(['--model', model])
    if language:
        args.extend(['--language', language])
    if output_format:
        args.extend(['--format', output_format])
    
    # Run stream transcribe script
    stream_main(args)

@cli.command()
@click.argument('input_pattern', type=str)
@click.option('--output-dir', '-o', type=click.Path(), default='transcripts',
              help='Output directory for transcripts.')
@click.option('--workers', '-w', type=int, default=0,
              help='Number of worker processes (0 for auto-detection).')
@click.option('--adaptive', '-a', is_flag=True,
              help='Use adaptive worker pool that adjusts based on system load.')
@click.option('--diarize', '-d', is_flag=True, help='Include speaker diarization.')
@click.option('--streaming', '-s', is_flag=True,
              help='Use streaming transcription (reduces memory usage).')
@click.option('--model', '-m', type=click.Choice(MODEL_SIZES), help='Whisper model size.')
@click.option('--language', '-l', help='Language code (e.g., en, fr, de).')
@click.option('--format', '-f', 'output_format', type=click.Choice(OUTPUT_FORMATS), 
              help='Output format.')
def batch(input_pattern, output_dir, workers, adaptive, diarize, streaming,
          model, language, output_format):
    """Batch process multiple audio or video files.
    
    This command processes multiple files matching the given glob pattern.
    
    Examples:
        batch "videos/*.mp4"
        batch "audio/*.mp3" --diarize --workers 4
        batch "interviews/*.wav" --adaptive --streaming
    """
    # Import here to avoid circular imports
    from scripts.batch_transcribe import main as batch_main
    
    # Prepare arguments
    args = [input_pattern]
    
    args.extend(['--output-dir', output_dir])
    if workers:
        args.extend(['--workers', str(workers)])
    if adaptive:
        args.append('--adaptive')
    if diarize:
        args.append('--diarize')
    if streaming:
        args.append('--streaming')
    if model:
        args.extend(['--model', model])
    if language:
        args.extend(['--language', language])
    if output_format:
        args.extend(['--format', output_format])
    
    # Run batch transcribe script
    batch_main(args)

@cli.command()
@click.option('--host', type=str, default='localhost',
              help='Host to bind the server to.')
@click.option('--port', '-p', type=int, default=8000,
              help='Port to bind the server to.')
@click.option('--config', '-c', type=click.Path(), default='.env',
              help='Path to configuration file.')
def server(host, port, config):
    """Run a model server for persistent model instances.
    
    This command starts a server that keeps models loaded in memory
    for faster transcription requests.
    
    Examples:
        server
        server --port 8080
        server --host 0.0.0.0 --port 9000
    """
    # Import here to avoid circular imports
    from scripts.model_server import main as server_main
    
    # Prepare arguments
    args = []
    
    args.extend(['--host', host])
    args.extend(['--port', str(port)])
    if config:
        args.extend(['--config', config])
    
    # Run model server script
    server_main(args)

@cli.command()
@click.option('--server', type=str, default='http://localhost:8000',
              help='URL of the model server.')
@click.argument('command', type=click.Choice(['status', 'transcribe']))
@click.argument('args', nargs=-1)
def client(server, command, args):
    """Interact with the model server.
    
    This command allows you to interact with the model server,
    either to check its status or to transcribe files.
    
    Examples:
        client status
        client transcribe audio.mp3
    """
    # Import here to avoid circular imports
    from scripts.model_client import main as client_main
    
    # Prepare arguments
    cmd_args = ['--server', server, command]
    
    # Add any additional arguments
    cmd_args.extend(args)
    
    # Run model client script
    client_main(cmd_args)

@cli.command()
def completion():
    """Generate shell completion script.
    
    This command generates a shell completion script for the current shell.
    It supports bash, zsh, and fish shells.
    
    To install completions:
    
    For bash:
        transcribe completion > ~/.transcribe-complete.bash
        echo 'source ~/.transcribe-complete.bash' >> ~/.bashrc
    
    For zsh:
        transcribe completion > ~/.transcribe-complete.zsh
        echo 'source ~/.transcribe-complete.zsh' >> ~/.zshrc
    
    For fish:
        transcribe completion > ~/.config/fish/completions/transcribe.fish
    """
    # Detect shell
    shell = os.environ.get('SHELL', '').split('/')[-1]
    
    if shell == 'bash':
        script = os.popen(f'_TRANSCRIBE_COMPLETE=bash_source {sys.argv[0]}').read()
    elif shell == 'zsh':
        script = os.popen(f'_TRANSCRIBE_COMPLETE=zsh_source {sys.argv[0]}').read()
    elif shell == 'fish':
        script = os.popen(f'_TRANSCRIBE_COMPLETE=fish_source {sys.argv[0]}').read()
    else:
        click.echo(f"Unsupported shell: {shell}")
        return
    
    click.echo(script)

if __name__ == '__main__':
    cli(obj={}) 

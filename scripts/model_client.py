#!/usr/bin/env python3
"""
Model Client Script

This script provides a client interface to interact with the model server.
It allows sending transcription requests and viewing server status.
"""

import os
import sys
import time
import logging
import argparse
import requests

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import OUTPUT_FORMATS
from src.utils.progress import ProgressReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Seconds between job-status polls while the server transcribes.
POLL_INTERVAL = 1.0

def build_transcribe_options(args):
    """Build the form fields for a transcription request.

    The diarize field is always sent, so the server honors this request's
    choice instead of falling back to its own INCLUDE_DIARIZATION default —
    a client that didn't ask for diarization must never get (or fail on) it.
    """
    options = {'diarize': 'true' if args.diarize else 'false'}
    if args.model:
        options['model'] = args.model
    if args.language:
        options['language'] = args.language
    if args.format:
        options['format'] = args.format
    return options


def get_server_status(server_url):
    """Get the status of the model server."""
    try:
        response = requests.get(f"{server_url}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to server: {str(e)}")
        return None

def transcribe_file(server_url, file_path, options=None):
    """Send a transcription request to the server."""
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    # Create progress reporter
    progress = ProgressReporter(
        desc=f"Uploading {os.path.basename(file_path)}",
        unit="B",
        color="green",
        total=os.path.getsize(file_path)
    )
    
    try:
        # Prepare the request
        files = {'file': open(file_path, 'rb')}
        data = options or {}
        
        # Get file size for progress reporting
        file_size = os.path.getsize(file_path)
        
        with progress:
            # Custom session with progress tracking
            session = requests.Session()
            
            # Create a custom adapter to track upload progress
            original_send = session.send
            
            def send_with_progress(*args, **kwargs):
                response = original_send(*args, **kwargs)
                # Update progress based on bytes sent
                if hasattr(response.request, 'body') and response.request.body:
                    # Calculate the difference between current and previous bytes sent
                    current_bytes = len(response.request.body)
                    # Store the current position as a completed value
                    progress.completed = current_bytes
                    # Update the progress bar description
                    progress.set_description(f"Uploading {os.path.basename(file_path)} - {current_bytes/1024/1024:.1f} MB")
                return response
            
            session.send = send_with_progress
            
            # Send the request
            response = session.post(
                f"{server_url}/transcribe",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            # Update progress to show processing
            progress.set_description("Processing on server")
            
            # Check if the response is immediate or a job ID
            result = response.json()
            
            if 'job_id' in result:
                # This is an async job, poll for results
                job_id = result['job_id']
                progress.set_description(f"Processing job {job_id}")
                
                while True:
                    time.sleep(POLL_INTERVAL)
                    status_response = session.get(f"{server_url}/api/jobs/{job_id}")
                    status_response.raise_for_status()
                    job_status = status_response.json()
                    
                    if job_status['status'] == 'completed':
                        progress.set_description("Completed")
                        progress.completed = file_size
                        return job_status['result']
                    elif job_status['status'] == 'failed':
                        progress.set_description("Failed")
                        logger.error(f"Job failed: {job_status.get('error', 'Unknown error')}")
                        return None
                    else:
                        # Update progress based on job status
                        if 'progress' in job_status:
                            progress_pct = job_status['progress']
                            progress.set_postfix(progress=f"{progress_pct:.1f}%")
                            
                            # If we have detailed progress info
                            if 'current_segment' in job_status and 'total_segments' in job_status:
                                progress.set_description(
                                    f"Processing segment {job_status['current_segment']}/{job_status['total_segments']}"
                                )
            else:
                # Immediate response
                progress.set_description("Completed")
                progress.completed = file_size
                return result
                
    except requests.exceptions.HTTPError as e:
        logger.error(f"Server rejected the request: {_error_detail(e.response)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during transcription request: {str(e)}")
        return None
    finally:
        # Close the file
        if 'files' in locals() and 'file' in files:
            files['file'].close()

def _error_detail(response):
    """The server's error message from an error response, else its status line."""
    try:
        return response.json()["error"]
    except (ValueError, KeyError, TypeError):
        return f"{response.status_code} {response.reason}"

def format_segment(segment):
    """Format a server segment as ``[MM:SS.mmm --> MM:SS.mmm] (speaker) text``.

    The server sends segments as ``[start, end, text, speaker]`` lists (the
    transcriber's tuples, JSON-encoded).
    """
    start, end, text, speaker = segment

    # Format timestamp as [MM:SS.mmm]
    start_str = f"{int(start // 60):02d}:{int(start % 60):02d}.{int((start % 1) * 1000):03d}"
    end_str = f"{int(end // 60):02d}:{int(end % 60):02d}.{int((end % 1) * 1000):03d}"

    # Add speaker if available
    speaker_str = f" ({speaker})" if speaker else ""

    return f"[{start_str} --> {end_str}]{speaker_str} {text}"

def display_transcription(result):
    """Display the transcription result."""
    if not result:
        return
    
    print("\n=== Transcription Result ===")
    
    if 'segments' in result:
        for segment in result['segments']:
            print(format_segment(segment))
    else:
        # Simple text output
        print(result.get('text', 'No text available'))
    
    # Print metadata if available
    if 'metadata' in result:
        print("\n=== Metadata ===")
        for key, value in result['metadata'].items():
            print(f"{key}: {value}")
    
    print("\n=== Processing Info ===")
    print(f"Processing time: {result.get('processing_time', 'N/A')} seconds")
    if 'model' in result:
        print(f"Model: {result['model']}")
    if 'language' in result:
        print(f"Detected language: {result['language']}")
    
    # Print output file path if available
    if 'output_file' in result:
        print(f"\n=== Output File ===")
        print(f"Transcript saved to: {result['output_file']}")

def display_server_status(status):
    """Display the server status."""
    if not status:
        return
    
    print("\n=== Server Status ===")
    print(f"Status: {status.get('status', 'Unknown')}")
    uptime = status.get('uptime')
    print(f"Uptime: {uptime:.0f}s" if isinstance(uptime, (int, float)) else "Uptime: Unknown")
    
    model = status.get('model') or {}
    if model:
        print("\n=== Model ===")
        print(f"Engine: {model.get('engine', 'Unknown')}")
        print(f"Model: {model.get('model') or model.get('model_size', 'Unknown')}")
        print(f"Language: {model.get('language', 'Unknown')}")
        print(f"Device: {model.get('device', 'Unknown')}")

    stats = status.get('stats') or {}
    if stats:
        print("\n=== Requests ===")
        print(f"Requests: {stats.get('requests', 0)}")
        print(f"Successful: {stats.get('successful', 0)}")
        print(f"Failed: {stats.get('failed', 0)}")
        print(f"Average processing time: {stats.get('avg_processing_time', 0.0):.1f}s")

def main(argv=None):
    """Main function for the model client script."""
    parser = argparse.ArgumentParser(
        description="Client for interacting with the model server"
    )
    
    parser.add_argument(
        "--server", "-s",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get server status")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe an audio or video file")
    transcribe_parser.add_argument(
        "file_path",
        help="Path to the audio or video file"
    )
    transcribe_parser.add_argument(
        "--model", "-m",
        help="Model the server must be running; the request is refused otherwise"
    )
    transcribe_parser.add_argument(
        "--language", "-l",
        help="Language the server must be using; the request is refused otherwise"
    )
    transcribe_parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Include speaker diarization"
    )
    transcribe_parser.add_argument(
        "--output", "-o",
        help="Output file path (if not specified, result will be displayed)"
    )
    transcribe_parser.add_argument(
        "--format", "-f",
        choices=OUTPUT_FORMATS,
        help="Output format (default: the server's OUTPUT_FORMAT)"
    )
    
    args = parser.parse_args(argv)
    
    # Default to status if no command specified
    if not args.command:
        args.command = "status"
    
    # Execute the command
    if args.command == "status":
        status = get_server_status(args.server)
        if status:
            display_server_status(status)
        else:
            logger.error("Failed to get server status")
            return 1
    
    elif args.command == "transcribe":
        # Send transcription request
        result = transcribe_file(args.server, args.file_path, build_transcribe_options(args))

        if result:
            if result.get('diarization_error'):
                logger.warning(
                    "Diarization failed; transcript has no speaker labels: %s",
                    result['diarization_error'],
                )
            # Save to file if output path specified
            if args.output:
                try:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        # The server formatted the transcript in the requested
                        # format (preview_text); write exactly that. A result
                        # without it gets the console rendering.
                        text = result.get('preview_text')
                        if text is None:
                            text = "".join(
                                format_segment(segment) + "\n"
                                for segment in result.get('segments', [])
                            )
                        f.write(text)
                    
                    logger.info(f"Transcription saved to {args.output}")
                except Exception as e:
                    logger.error(f"Error saving output: {str(e)}")
                    return 1
            else:
                # Display the result
                display_transcription(result)
        else:
            logger.error("Transcription failed")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
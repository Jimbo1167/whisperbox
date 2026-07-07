#!/usr/bin/env python3
"""
Script to batch process multiple audio or video files for transcription.

This script allows processing multiple files in parallel, with configurable
worker count and support for both regular and streaming transcription modes.
"""

import os
import sys
import time
import glob
import argparse
import logging
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.transcriber import Transcriber
from src.utils.resource_monitor import AdaptiveWorkerPool, get_optimal_worker_count
from src.utils.progress import ProgressReporter, MultiProgressReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Worker-local Transcriber cache: model loads are the dominant per-file cost,
# so each worker (thread or process) loads its models once and reuses them for
# every file it processes instead of paying a full load per file.
_worker_state = threading.local()


def _config_fingerprint(config: Config) -> Tuple:
    return (
        config.transcription_engine,
        config.whisper_model_size,
        config.parakeet_model,
        config.diarization_model,
        config.language,
        config.include_diarization,
        config.force_cpu,
    )


def _get_worker_transcriber(config: Config) -> Transcriber:
    fingerprint = _config_fingerprint(config)
    if getattr(_worker_state, "fingerprint", None) != fingerprint:
        _worker_state.transcriber = Transcriber(config)
        _worker_state.fingerprint = fingerprint
    return _worker_state.transcriber


def process_file(
    input_path: str, 
    config: Config, 
    output_dir: str, 
    use_streaming: bool = False,
    output_format: Optional[str] = None
) -> Tuple[str, bool, float]:
    """
    Process a single file for transcription.
    
    Args:
        input_path: Path to the input file
        config: Configuration object
        output_dir: Directory to save output files
        use_streaming: Whether to use streaming transcription
        output_format: Output format override
        
    Returns:
        Tuple of (output_path, success, processing_time)
    """
    start_time = time.time()
    
    try:
        # Reuse this worker's transcriber (loads models only on first use)
        transcriber = _get_worker_transcriber(config)
        
        # Generate output path
        input_file = Path(input_path)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True)
        
        ext = output_format or config.output_format or 'txt'
            
        output_path = str(output_dir_path / f"{input_file.stem}.{ext}")
        
        # Perform transcription
        logger.info(f"Processing {input_path}...")
        
        if use_streaming:
            # Use streaming transcription
            segments = []
            for segment in transcriber.transcribe_stream_with_diarization(input_path):
                segments.append((
                    segment['start'],
                    segment['end'],
                    segment['text'],
                    segment.get('speaker', 'SPEAKER')
                ))
        else:
            # Use regular transcription
            segments = transcriber.transcribe(input_path)
        
        # Save transcript
        transcriber.save_transcript(segments, output_path)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed {input_path} in {processing_time:.2f} seconds")
        
        return output_path, True, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing {input_path}: {str(e)}")
        return "", False, processing_time

def main(args=None):
    """Main entry point for the batch transcription script."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple audio or video files for transcription"
    )
    
    parser.add_argument(
        "input_pattern", 
        help="Glob pattern for input files (e.g., 'videos/*.mp4')"
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        default="transcripts",
        help="Output directory for transcripts (default: transcripts)"
    )
    
    parser.add_argument(
        "--workers", "-w", 
        type=int, 
        default=0,
        help="Number of worker processes (0 for auto-detection, default: 0)"
    )
    
    parser.add_argument(
        "--min-workers", 
        type=int, 
        default=1,
        help="Minimum number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=None,
        help="Maximum number of worker processes (default: CPU count)"
    )
    
    parser.add_argument(
        "--adaptive", "-a", 
        action="store_true",
        help="Use adaptive worker pool that adjusts based on system load"
    )
    
    parser.add_argument(
        "--diarize", "-d", 
        action="store_true",
        help="Include speaker diarization"
    )
    
    parser.add_argument(
        "--model", "-m", 
        default=None,
        help="Whisper model size (tiny, base, small, medium, large)"
    )
    
    parser.add_argument(
        "--language", "-l", 
        default=None,
        help="Language code (e.g., en, fr, de)"
    )
    
    parser.add_argument(
        "--format", "-f", 
        choices=["txt", "srt", "vtt", "json"],
        default=None,
        help="Output format (default: from config)"
    )
    
    parser.add_argument(
        "--streaming", "-s", 
        action="store_true",
        help="Use streaming transcription (reduces memory usage)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args(args)
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find input files
    input_files = glob.glob(args.input_pattern, recursive=True)
    
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return 1
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Create configuration
    config_kwargs = {}
    if args.model:
        config_kwargs['whisper_model'] = args.model
    if args.language:
        config_kwargs['language'] = args.language
    if args.format:
        config_kwargs['output_format'] = args.format
    if args.diarize:
        config_kwargs['include_diarization'] = True
    
    config = Config(**config_kwargs)
    
    # Determine worker count
    if args.workers > 0:
        worker_count = args.workers
    else:
        worker_count = get_optimal_worker_count(
            min_workers=args.min_workers,
            max_workers=args.max_workers
        )
    
    logger.info(f"Using {worker_count} worker processes")
    
    # Process files
    start_time = time.time()
    results = []
    
    # Create progress reporter
    progress = ProgressReporter(
        total=len(input_files),
        desc="Processing files",
        unit="file",
        color="green"
    )
    
    try:
        with progress:
            if args.adaptive:
                # Use adaptive worker pool
                with AdaptiveWorkerPool(
                    min_workers=args.min_workers,
                    max_workers=args.max_workers,
                    cpu_threshold=80.0,
                    memory_threshold=80.0
                ) as pool:
                    # Submit all tasks
                    futures = []
                    for input_file in input_files:
                        future = pool.submit(
                            process_file,
                            input_file,
                            config,
                            args.output_dir,
                            args.streaming,
                            args.format
                        )
                        futures.append((future, input_file))
                    
                    # Process results as they complete
                    for future, input_file in futures:
                        try:
                            output_path, success, processing_time = future.result()
                            results.append({
                                'input': input_file,
                                'output': output_path,
                                'success': success,
                                'time': processing_time
                            })
                            
                            # Update progress
                            status = "Success" if success else "Failed"
                            progress.update(1, status)
                            progress.set_postfix(
                                success=sum(r['success'] for r in results),
                                failed=sum(not r['success'] for r in results)
                            )
                            
                        except Exception as e:
                            logger.error(f"Error processing {input_file}: {str(e)}")
                            results.append({
                                'input': input_file,
                                'output': "",
                                'success': False,
                                'time': 0,
                                'error': str(e)
                            })
                            
                            # Update progress
                            progress.update(1, "Failed")
                            progress.set_postfix(
                                success=sum(r['success'] for r in results),
                                failed=sum(not r['success'] for r in results)
                            )
            else:
                # Use standard thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    # Submit all tasks
                    futures = {
                        executor.submit(
                            process_file,
                            input_file,
                            config,
                            args.output_dir,
                            args.streaming,
                            args.format
                        ): input_file for input_file in input_files
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        input_file = futures[future]
                        try:
                            output_path, success, processing_time = future.result()
                            results.append({
                                'input': input_file,
                                'output': output_path,
                                'success': success,
                                'time': processing_time
                            })
                            
                            # Update progress
                            status = "Success" if success else "Failed"
                            progress.update(1, status)
                            progress.set_postfix(
                                success=sum(r['success'] for r in results),
                                failed=sum(not r['success'] for r in results)
                            )
                            
                        except Exception as e:
                            logger.error(f"Error processing {input_file}: {str(e)}")
                            results.append({
                                'input': input_file,
                                'output': "",
                                'success': False,
                                'time': 0,
                                'error': str(e)
                            })
                            
                            # Update progress
                            progress.update(1, "Failed")
                            progress.set_postfix(
                                success=sum(r['success'] for r in results),
                                failed=sum(not r['success'] for r in results)
                            )
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    
    # Print summary
    total_time = time.time() - start_time
    success_count = sum(r['success'] for r in results)
    failed_count = len(results) - success_count
    
    logger.info(f"\nProcessing completed in {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(input_files)}")
    
    if failed_count > 0:
        logger.warning(f"Failed to process: {failed_count}/{len(input_files)}")
        for result in results:
            if not result['success']:
                logger.warning(f"  - {result['input']}")
    
    # Get resource usage summary
    resource_summary = progress.get_average_resource_usage()
    logger.info("\nResource usage summary:")
    logger.info(f"  CPU: {resource_summary.get('cpu_percent', 0):.1f}%")
    logger.info(f"  Memory: {resource_summary.get('memory_used_gb', 0):.2f} GB")
    
    if 'gpu_memory_used_gb' in resource_summary and resource_summary['gpu_memory_used_gb'] > 0:
        logger.info(f"  GPU Memory: {resource_summary['gpu_memory_used_gb']:.2f} GB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

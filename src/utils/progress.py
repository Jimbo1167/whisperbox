"""
Progress reporting utilities for the Whisperbox.

This module provides classes for reporting progress during long-running operations,
including progress bars, estimated time remaining, and resource usage reporting.
"""

import sys
import time
import logging
from typing import Optional, Dict, Any, List, Callable
from tqdm import tqdm
from datetime import datetime, timedelta

from .resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)

class ProgressReporter:
    """
    A class for reporting progress during long-running operations.
    
    This class provides methods for creating and updating progress bars,
    estimating time remaining, and reporting resource usage.
    """
    
    def __init__(self, 
                 total: int = 100, 
                 desc: str = "Processing", 
                 unit: str = "it",
                 monitor_resources: bool = True,
                 log_interval: int = 10,
                 color: Optional[str] = None):
        """
        Initialize a progress reporter.
        
        Args:
            total: Total number of items to process
            desc: Description of the progress bar
            unit: Unit of items being processed
            monitor_resources: Whether to monitor system resources
            log_interval: How often to log progress (in seconds)
            color: Color of the progress bar (if supported)
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.monitor_resources = monitor_resources
        self.log_interval = log_interval
        self.color = color
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.completed = 0
        self.resource_monitor = None
        self.progress_bar = None
        self.checkpoints: List[Dict[str, Any]] = []
        
        # Initialize resource monitor if requested
        if self.monitor_resources:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.start()
    
    def __enter__(self):
        """Start the progress reporter when used as a context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the progress reporter when exiting the context."""
        self.close()
    
    def start(self):
        """Start the progress reporter."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.completed = 0
        
        # Create progress bar
        self.progress_bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            colour=self.color,
            file=sys.stdout
        )
        
        # Start resource monitor if requested
        if self.monitor_resources and self.resource_monitor is None:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.start()
        
        # Log start
        logger.info(f"Started {self.desc} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self
    
    def update(self, n: int = 1, status: Optional[str] = None):
        """
        Update the progress bar.
        
        Args:
            n: Number of items completed
            status: Optional status message to display
        """
        self.completed += n
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.update(n)
            
            if status:
                self.progress_bar.set_description(f"{self.desc} - {status}")
        
        # Log progress at intervals
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time
    
    def set_description(self, desc: str):
        """Set the description of the progress bar."""
        self.desc = desc
        if self.progress_bar:
            self.progress_bar.set_description(desc)
    
    def set_postfix(self, **kwargs):
        """Set the postfix of the progress bar."""
        if self.progress_bar:
            self.progress_bar.set_postfix(**kwargs)
    
    def add_checkpoint(self, name: str, data: Optional[Dict[str, Any]] = None):
        """
        Add a checkpoint to track progress at specific points.
        
        Args:
            name: Name of the checkpoint
            data: Optional data to associate with the checkpoint
        """
        checkpoint = {
            'name': name,
            'time': time.time(),
            'completed': self.completed,
            'data': data or {}
        }
        
        # Add resource metrics if available
        if self.resource_monitor:
            checkpoint['metrics'] = self.resource_monitor.get_metrics()
        
        self.checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint: {name} at {self.completed}/{self.total} items")
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """
        Get the estimated time remaining in seconds.
        
        Returns:
            Estimated time remaining in seconds, or None if not enough data
        """
        if self.completed == 0:
            return None
        
        elapsed = time.time() - self.start_time
        items_per_second = self.completed / elapsed if elapsed > 0 else 0
        
        if items_per_second == 0:
            return None
        
        remaining_items = self.total - self.completed
        return remaining_items / items_per_second
    
    def get_formatted_time_remaining(self) -> str:
        """
        Get a formatted string of the estimated time remaining.
        
        Returns:
            Formatted time remaining string
        """
        remaining_seconds = self.get_estimated_time_remaining()
        
        if remaining_seconds is None:
            return "Calculating..."
        
        remaining = timedelta(seconds=int(remaining_seconds))
        
        if remaining.days > 0:
            return f"{remaining.days}d {remaining.seconds // 3600}h {(remaining.seconds // 60) % 60}m"
        elif remaining.seconds // 3600 > 0:
            return f"{remaining.seconds // 3600}h {(remaining.seconds // 60) % 60}m {remaining.seconds % 60}s"
        elif remaining.seconds // 60 > 0:
            return f"{remaining.seconds // 60}m {remaining.seconds % 60}s"
        else:
            return f"{remaining.seconds}s"
    
    def get_elapsed_time(self) -> float:
        """
        Get the elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def get_formatted_elapsed_time(self) -> str:
        """
        Get a formatted string of the elapsed time.
        
        Returns:
            Formatted elapsed time string
        """
        elapsed_seconds = self.get_elapsed_time()
        elapsed = timedelta(seconds=int(elapsed_seconds))
        
        if elapsed.days > 0:
            return f"{elapsed.days}d {elapsed.seconds // 3600}h {(elapsed.seconds // 60) % 60}m"
        elif elapsed.seconds // 3600 > 0:
            return f"{elapsed.seconds // 3600}h {(elapsed.seconds // 60) % 60}m {elapsed.seconds % 60}s"
        elif elapsed.seconds // 60 > 0:
            return f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"
        else:
            return f"{elapsed.seconds}s"
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get the current resource usage.
        
        Returns:
            Dictionary of resource usage metrics
        """
        if self.resource_monitor:
            return self.resource_monitor.get_metrics()
        return {}
    
    def get_average_resource_usage(self, seconds: int = 5) -> Dict[str, Any]:
        """
        Get the average resource usage over the specified time period.
        
        Args:
            seconds: Number of seconds to average over
            
        Returns:
            Dictionary of average resource usage metrics
        """
        if self.resource_monitor:
            return self.resource_monitor.get_average_metrics(seconds)
        return {}
    
    def _log_progress(self):
        """Log progress information."""
        percent = (self.completed / self.total) * 100 if self.total > 0 else 0
        elapsed = self.get_formatted_elapsed_time()
        remaining = self.get_formatted_time_remaining()
        
        logger.info(
            f"Progress: {self.completed}/{self.total} ({percent:.1f}%) - "
            f"Elapsed: {elapsed} - Remaining: {remaining}"
        )
        
        # Log resource usage if available
        if self.resource_monitor:
            metrics = self.resource_monitor.get_average_metrics()
            logger.info(
                f"Resource usage: CPU: {metrics.get('cpu_percent', 0):.1f}% - "
                f"Memory: {metrics.get('memory_used_gb', 0):.2f} GB"
            )
            
            if 'gpu_memory_used_gb' in metrics and metrics['gpu_memory_used_gb'] > 0:
                logger.info(f"GPU Memory: {metrics['gpu_memory_used_gb']:.2f} GB")
    
    def close(self):
        """Close the progress reporter and clean up resources."""
        # Log final progress
        self._log_progress()
        
        # Close progress bar
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None
        
        # Stop resource monitor
        if self.resource_monitor:
            self.resource_monitor.stop()
            self.resource_monitor = None
        
        # Log completion
        logger.info(
            f"Completed {self.desc} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Total time: {self.get_formatted_elapsed_time()}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the progress.
        
        Returns:
            Dictionary with progress summary
        """
        return {
            'total': self.total,
            'completed': self.completed,
            'percent': (self.completed / self.total) * 100 if self.total > 0 else 0,
            'elapsed': self.get_elapsed_time(),
            'elapsed_formatted': self.get_formatted_elapsed_time(),
            'start_time': self.start_time,
            'end_time': time.time(),
            'checkpoints': self.checkpoints
        }


class MultiProgressReporter:
    """
    A class for reporting progress on multiple tasks simultaneously.
    
    This class manages multiple progress reporters and provides methods
    for updating and displaying progress for all tasks.
    """
    
    def __init__(self):
        """Initialize a multi-progress reporter."""
        self.reporters: Dict[str, ProgressReporter] = {}
        self.resource_monitor = ResourceMonitor()
    
    def __enter__(self):
        """Start the multi-progress reporter when used as a context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the multi-progress reporter when exiting the context."""
        self.close()
    
    def start(self):
        """Start the multi-progress reporter."""
        self.resource_monitor.start()
        return self
    
    def add_reporter(self, 
                    name: str, 
                    total: int = 100, 
                    desc: str = "Processing", 
                    unit: str = "it",
                    color: Optional[str] = None) -> ProgressReporter:
        """
        Add a new progress reporter.
        
        Args:
            name: Name of the reporter
            total: Total number of items to process
            desc: Description of the progress bar
            unit: Unit of items being processed
            color: Color of the progress bar (if supported)
            
        Returns:
            The created progress reporter
        """
        reporter = ProgressReporter(
            total=total,
            desc=desc,
            unit=unit,
            monitor_resources=False,  # We'll use the shared resource monitor
            color=color
        )
        reporter.start()
        self.reporters[name] = reporter
        return reporter
    
    def update(self, name: str, n: int = 1, status: Optional[str] = None):
        """
        Update a specific progress reporter.
        
        Args:
            name: Name of the reporter to update
            n: Number of items completed
            status: Optional status message to display
        """
        if name in self.reporters:
            self.reporters[name].update(n, status)
    
    def get_reporter(self, name: str) -> Optional[ProgressReporter]:
        """
        Get a specific progress reporter.
        
        Args:
            name: Name of the reporter to get
            
        Returns:
            The progress reporter, or None if not found
        """
        return self.reporters.get(name)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get the current resource usage.
        
        Returns:
            Dictionary of resource usage metrics
        """
        return self.resource_monitor.get_metrics()
    
    def get_average_resource_usage(self, seconds: int = 5) -> Dict[str, Any]:
        """
        Get the average resource usage over the specified time period.
        
        Args:
            seconds: Number of seconds to average over
            
        Returns:
            Dictionary of average resource usage metrics
        """
        return self.resource_monitor.get_average_metrics(seconds)
    
    def close(self):
        """Close all progress reporters and clean up resources."""
        for reporter in self.reporters.values():
            reporter.close()
        
        self.resource_monitor.stop()
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all progress reporters.
        
        Returns:
            Dictionary with progress summaries for all reporters
        """
        return {name: reporter.get_summary() for name, reporter in self.reporters.items()}


def create_callback_progress(callback: Callable[[int, int, Optional[str]], None], 
                            total: int, 
                            desc: str = "Processing") -> Callable[[int, Optional[str]], None]:
    """
    Create a progress callback function that can be used with libraries that support callbacks.
    
    Args:
        callback: Function to call with progress updates (completed, total, status)
        total: Total number of items to process
        desc: Description of the progress
        
    Returns:
        A function that can be used as a callback
    """
    def progress_callback(completed: int, status: Optional[str] = None):
        callback(completed, total, status)
    
    return progress_callback 
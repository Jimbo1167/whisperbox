"""
Cache manager for the Whisperbox.

This module provides a cache manager class that handles caching of audio files,
transcription results, and diarization results.
"""

import os
import json
import time
import hashlib
import logging
import shutil
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages caching for the Whisperbox.
    
    This class provides methods for caching and retrieving:
    - Extracted audio files
    - Transcription results
    - Diarization results
    
    It also handles cache invalidation based on file modification times and
    configurable cache expiration periods.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the cache manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set up cache directory
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperbox")
        
        # Create subdirectories for different cache types
        self.audio_cache_dir = os.path.join(self.cache_dir, "audio")
        self.transcription_cache_dir = os.path.join(self.cache_dir, "transcription")
        self.diarization_cache_dir = os.path.join(self.cache_dir, "diarization")
        
        # Create directories if they don't exist
        for directory in [self.audio_cache_dir, self.transcription_cache_dir, self.diarization_cache_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Cache expiration in seconds (default: 7 days)
        self.cache_expiration = config.cache_expiration if hasattr(config, "cache_expiration") else 7 * 24 * 60 * 60
        
        # Maximum cache size in bytes (default: 10GB)
        self.max_cache_size = config.max_cache_size if hasattr(config, "max_cache_size") else 10 * 1024 * 1024 * 1024
        
        # Clean up cache on initialization if needed
        self._cleanup_cache()
        
        logger.info(f"Cache manager initialized with cache directory: {self.cache_dir}")
    
    def _generate_cache_key(self, file_path: str, prefix: str = "") -> str:
        """
        Generate a unique cache key for a file.
        
        Args:
            file_path: Path to the file
            prefix: Optional prefix for the cache key
            
        Returns:
            A unique cache key based on the file path and modification time
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return None
            
        # Get file stats
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_mtime = file_stats.st_mtime
        
        # Create a unique identifier based on file path, size, and modification time
        unique_id = f"{file_path}_{file_size}_{file_mtime}"
        
        # Generate a hash of the unique identifier
        hash_obj = hashlib.md5(unique_id.encode())
        hash_str = hash_obj.hexdigest()
        
        # Add prefix if provided
        if prefix:
            hash_str = f"{prefix}_{hash_str}"
        
        return hash_str
    
    def _get_cache_path(self, cache_key: str, cache_type: str) -> str:
        """
        Get the path to a cached file.
        
        Args:
            cache_key: The cache key
            cache_type: The type of cache (audio, transcription, diarization)
            
        Returns:
            Path to the cached file
        """
        if cache_type == "audio":
            return os.path.join(self.audio_cache_dir, f"{cache_key}.wav")
        elif cache_type == "transcription":
            return os.path.join(self.transcription_cache_dir, f"{cache_key}.json")
        elif cache_type == "diarization":
            return os.path.join(self.diarization_cache_dir, f"{cache_key}.json")
        else:
            raise ValueError(f"Invalid cache type: {cache_type}")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """
        Check if a cached file is valid (exists and not expired).
        
        Args:
            cache_path: Path to the cached file
            
        Returns:
            True if the cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
        
        # Check if the cache has expired
        cache_mtime = os.path.getmtime(cache_path)
        current_time = time.time()
        
        return (current_time - cache_mtime) < self.cache_expiration
    
    def _cleanup_cache(self):
        """
        Clean up the cache by removing expired files and ensuring the cache size
        is within limits.
        """
        logger.info("Cleaning up cache...")
        
        # Remove expired files
        for cache_dir in [self.audio_cache_dir, self.transcription_cache_dir, self.diarization_cache_dir]:
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Check if the file has expired
                file_mtime = os.path.getmtime(file_path)
                current_time = time.time()
                
                if (current_time - file_mtime) > self.cache_expiration:
                    logger.debug(f"Removing expired cache file: {file_path}")
                    os.remove(file_path)
        
        # Check total cache size and remove oldest files if needed
        total_size = self._get_cache_size()
        
        if total_size > self.max_cache_size:
            logger.info(f"Cache size ({total_size / 1024 / 1024:.2f} MB) exceeds limit ({self.max_cache_size / 1024 / 1024:.2f} MB), removing oldest files")
            
            # Get all cache files with their modification times
            cache_files = []
            for cache_dir in [self.audio_cache_dir, self.transcription_cache_dir, self.diarization_cache_dir]:
                for filename in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, filename)
                    
                    # Skip directories
                    if os.path.isdir(file_path):
                        continue
                    
                    file_mtime = os.path.getmtime(file_path)
                    file_size = os.path.getsize(file_path)
                    
                    cache_files.append((file_path, file_mtime, file_size))
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Remove files until we're under the limit
            for file_path, _, file_size in cache_files:
                if total_size <= self.max_cache_size:
                    break
                
                logger.debug(f"Removing cache file to free space: {file_path}")
                os.remove(file_path)
                total_size -= file_size
    
    def _get_cache_size(self) -> int:
        """
        Get the total size of the cache in bytes.
        
        Returns:
            Total size of the cache in bytes
        """
        total_size = 0
        
        for cache_dir in [self.audio_cache_dir, self.transcription_cache_dir, self.diarization_cache_dir]:
            for filename in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                total_size += os.path.getsize(file_path)
        
        return total_size
    
    def get_cached_audio(self, input_path: str) -> Optional[str]:
        """
        Get the path to a cached audio file if it exists.
        
        Args:
            input_path: Path to the input file (audio or video)
            
        Returns:
            Path to the cached audio file if it exists, None otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(input_path, prefix="audio")
        
        # Return None if file doesn't exist
        if cache_key is None:
            return None
            
        cache_path = self._get_cache_path(cache_key, "audio")
        
        # Check if the cache is valid
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached audio file: {cache_path}")
            return cache_path
        
        return None
    
    def cache_audio(self, input_path: str, audio_path: str) -> str:
        """
        Cache an audio file.
        
        Args:
            input_path: Path to the input file (audio or video)
            audio_path: Path to the extracted audio file
            
        Returns:
            Path to the cached audio file
        """
        # Generate cache key
        cache_key = self._generate_cache_key(input_path, prefix="audio")
        cache_path = self._get_cache_path(cache_key, "audio")
        
        # Copy the audio file to the cache
        shutil.copy2(audio_path, cache_path)
        
        logger.info(f"Cached audio file: {cache_path}")
        
        return cache_path
    
    def get_cached_transcription(self, audio_path: str, engine_id: str = "whisper") -> Optional[List[Dict[str, Any]]]:
        """Get cached transcription results if they exist.

        Args:
            audio_path: Path to the audio file
            engine_id: ASR engine identifier (e.g. "whisper-large-v3-turbo",
                "parakeet-<slug>"). Different engines must not share cache entries.
        """
        prefix = f"transcription-{engine_id}"
        cache_key = self._generate_cache_key(audio_path, prefix=prefix)
        if cache_key is None:
            return None

        cache_path = self._get_cache_path(cache_key, "transcription")
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached transcription results: {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def cache_transcription(
        self,
        audio_path: str,
        transcription_results: List[Dict[str, Any]],
        engine_id: str = "whisper",
    ) -> None:
        """Cache transcription results.

        Args:
            audio_path: Path to the audio file
            transcription_results: Transcription results to cache
            engine_id: ASR engine identifier (must match the value passed to
                get_cached_transcription).
        """
        prefix = f"transcription-{engine_id}"
        cache_key = self._generate_cache_key(audio_path, prefix=prefix)
        cache_path = self._get_cache_path(cache_key, "transcription")
        try:
            with open(cache_path, "w") as f:
                json.dump(transcription_results, f)
            logger.info(f"Cached transcription results: {cache_path}")
        except Exception as e:
            logger.warning(f"Error caching transcription results: {e}")
    
    def get_cached_diarization(self, audio_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached diarization results if they exist.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Cached diarization results if they exist, None otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(audio_path, prefix="diarization")
        
        # Return None if file doesn't exist
        if cache_key is None:
            return None
            
        cache_path = self._get_cache_path(cache_key, "diarization")
        
        # Check if the cache is valid
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached diarization results: {cache_path}")
            
            # Load the diarization results from the cache
            with open(cache_path, "r") as f:
                diarization_results = json.load(f)
            
            return diarization_results
        
        return None
    
    def cache_diarization(self, audio_path: str, diarization_results: List[Dict[str, Any]]) -> None:
        """
        Cache diarization results.
        
        Args:
            audio_path: Path to the audio file
            diarization_results: Diarization results to cache
        """
        # Generate cache key
        cache_key = self._generate_cache_key(audio_path, prefix="diarization")
        cache_path = self._get_cache_path(cache_key, "diarization")
        
        # Save the diarization results to the cache
        try:
            with open(cache_path, "w") as f:
                json.dump(diarization_results, f)
            
            logger.info(f"Cached diarization results: {cache_path}")
        except Exception as e:
            logger.warning(f"Error caching diarization results: {e}")
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            cache_type: Optional type of cache to clear (audio, transcription, diarization)
                       If None, all caches will be cleared
        """
        if cache_type is None or cache_type == "audio":
            logger.info("Clearing audio cache...")
            for filename in os.listdir(self.audio_cache_dir):
                file_path = os.path.join(self.audio_cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        if cache_type is None or cache_type == "transcription":
            logger.info("Clearing transcription cache...")
            for filename in os.listdir(self.transcription_cache_dir):
                file_path = os.path.join(self.transcription_cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        if cache_type is None or cache_type == "diarization":
            logger.info("Clearing diarization cache...")
            for filename in os.listdir(self.diarization_cache_dir):
                file_path = os.path.join(self.diarization_cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        logger.info("Cache cleared successfully") 
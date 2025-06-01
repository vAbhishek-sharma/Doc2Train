# utils/cache.py
"""
Complete enhanced caching system for Doc2Train
Configuration-aware caching with compression and cleanup
"""

import os
import json
import hashlib
import time
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

class CacheManager:
    """
    Enhanced cache manager with configuration awareness and optimization
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager

        Args:
            cache_dir: Base cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.extraction_cache_dir = self.cache_dir / "extraction"
        self.lock = threading.Lock()

        # Create cache directories
        self.extraction_cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache settings
        self.use_compression = True
        self.max_cache_size_gb = 5.0  # 5GB max cache size
        self.cache_expiry_days = 30   # 30 days expiry

    def _get_cache_key(self, file_path: str, config: Dict) -> str:
        """
        Generate cache key based on file and configuration

        Args:
            file_path: Path to original file
            config: Processing configuration

        Returns:
            Unique cache key
        """
        try:
            file_stat = os.stat(file_path)

            # Include relevant config in cache key
            cache_relevant_config = {
                'start_page': config.get('start_page', 1),
                'end_page': config.get('end_page'),
                'skip_pages': config.get('skip_pages', []),
                'min_image_size': config.get('min_image_size', 1000),
                'min_text_length': config.get('min_text_length', 100),
                'header_regex': config.get('header_regex', ''),
                'skip_single_color': config.get('skip_single_color', False),
                'use_ocr': config.get('use_ocr', True),
                'quality_threshold': config.get('quality_threshold', 0.7)
            }

            # Create cache key from file info and config
            key_data = {
                'file_path': str(file_path),
                'file_size': file_stat.st_size,
                'file_mtime': file_stat.st_mtime,
                'config': cache_relevant_config
            }

            key_string = json.dumps(key_data, sort_keys=True)
            cache_key = hashlib.sha256(key_string.encode()).hexdigest()

            return cache_key

        except Exception as e:
            # Fallback to simple hash
            fallback_key = hashlib.md5(f"{file_path}{time.time()}".encode()).hexdigest()
            return fallback_key

    def _get_cache_file_path(self, cache_key: str, compressed: bool = None) -> Path:
        """Get cache file path"""
        if compressed is None:
            compressed = self.use_compression

        extension = '.cache.gz' if compressed else '.cache'
        return self.extraction_cache_dir / f"{cache_key}{extension}"

    def load_from_cache(self, file_path: str, config: Dict) -> Optional[Dict]:
        """
        Load cached extraction results

        Args:
            file_path: Path to original file
            config: Processing configuration

        Returns:
            Cached data dictionary or None if not found/invalid
        """
        try:
            cache_key = self._get_cache_key(file_path, config)

            # Try compressed cache first
            cache_file = self._get_cache_file_path(cache_key, True)
            if not cache_file.exists():
                # Try uncompressed cache
                cache_file = self._get_cache_file_path(cache_key, False)
                if not cache_file.exists():
                    return None

            # Load cache data
            with self.lock:
                if cache_file.name.endswith('.gz'):
                    with gzip.open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                else:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)

            # Validate cache data
            if not self._validate_cache_data(cache_data, file_path):
                # Invalid cache, remove it
                cache_file.unlink()
                return None

            # Update access time for LRU cleanup
            cache_file.touch()

            if config.get('verbose'):
                print(f"   📌 Cache hit: {Path(file_path).name}")

            return cache_data

        except Exception as e:
            if config.get('verbose'):
                print(f"   ⚠️ Cache load error: {e}")
            return None

    def save_to_cache(self, file_path: str, text: str, images: List[Dict],
                     config: Dict, processing_time: float):
        """
        Save extraction results to cache

        Args:
            file_path: Path to original file
            text: Extracted text
            images: Extracted images (without binary data for size efficiency)
            config: Processing configuration
            processing_time: Time taken to process
        """
        try:
            cache_key = self._get_cache_key(file_path, config)
            cache_file = self._get_cache_file_path(cache_key, self.use_compression)

            # Prepare cache data (remove binary image data for efficiency)
            cache_images = []
            for img in images:
                cache_img = {k: v for k, v in img.items() if k != 'data'}
                cache_img['has_data'] = 'data' in img
                cache_images.append(cache_img)

            file_stat = os.stat(file_path)

            cache_data = {
                'version': '2.0',
                'file_path': str(file_path),
                'file_size': file_stat.st_size,
                'file_mtime': file_stat.st_mtime,
                'text': text,
                'images': cache_images,
                'config_hash': hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                'processing_time': processing_time,
                'cached_at': time.time(),
                'cache_key': cache_key
            }

            # Save to cache
            with self.lock:
                if self.use_compression:
                    with gzip.open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)

            if config.get('verbose'):
                print(f"   💾 Cached: {Path(file_path).name}")

        except Exception as e:
            if config.get('verbose'):
                print(f"   ⚠️ Cache save error: {e}")

    def is_cached(self, file_path: str, config: Dict) -> bool:
        """
        Check if file has valid cached results

        Args:
            file_path: Path to original file
            config: Processing configuration

        Returns:
            True if cached
        """
        try:
            cache_key = self._get_cache_key(file_path, config)

            # Check both compressed and uncompressed
            for compressed in [True, False]:
                cache_file = self._get_cache_file_path(cache_key, compressed)
                if cache_file.exists():
                    return True

            return False

        except Exception:
            return False

    def _validate_cache_data(self, cache_data: Dict, file_path: str) -> bool:
        """Validate cached data is still valid"""
        try:
            # Check cache version
            if cache_data.get('version') != '2.0':
                return False

            # Check file hasn't changed
            if not os.path.exists(file_path):
                return False

            file_stat = os.stat(file_path)
            if (cache_data.get('file_size') != file_stat.st_size or
                cache_data.get('file_mtime') != file_stat.st_mtime):
                return False

            # Check cache age
            cache_age_days = (time.time() - cache_data.get('cached_at', 0)) / (24 * 3600)
            if cache_age_days > self.cache_expiry_days:
                return False

            # Check required fields
            required_fields = ['text', 'images', 'processing_time']
            if not all(field in cache_data for field in required_fields):
                return False

            return True

        except Exception:
            return False

    def clear_cache(self, file_path: str = None, config: Dict = None):
        """
        Clear cache entries

        Args:
            file_path: Specific file to clear (None for all)
            config: Configuration for specific cache entry
        """
        try:
            if file_path and config:
                # Clear specific cache entry
                cache_key = self._get_cache_key(file_path, config)
                for compressed in [True, False]:
                    cache_file = self._get_cache_file_path(cache_key, compressed)
                    if cache_file.exists():
                        cache_file.unlink()
                        print(f"🗑️ Cleared cache for {Path(file_path).name}")

            elif file_path:
                # Clear all cache entries for a file (all configs)
                file_name = Path(file_path).name
                cleared = 0

                for cache_file in self.extraction_cache_dir.glob("*.cache*"):
                    try:
                        # Try to load and check if it's for this file
                        if cache_file.name.endswith('.gz'):
                            with gzip.open(cache_file, 'rb') as f:
                                cache_data = pickle.load(f)
                        else:
                            with open(cache_file, 'r') as f:
                                cache_data = json.load(f)

                        if cache_data.get('file_path') == str(file_path):
                            cache_file.unlink()
                            cleared += 1
                    except:
                        continue

                if cleared > 0:
                    print(f"🗑️ Cleared {cleared} cache entries for {file_name}")

            else:
                # Clear all cache
                import shutil
                if self.extraction_cache_dir.exists():
                    shutil.rmtree(self.extraction_cache_dir)
                    self.extraction_cache_dir.mkdir(parents=True, exist_ok=True)
                    print("🗑️ Cleared all extraction cache")

        except Exception as e:
            print(f"❌ Error clearing cache: {e}")

    def cleanup_cache(self, max_size_gb: float = None, max_age_days: int = None):
        """
        Clean up cache based on size and age limits

        Args:
            max_size_gb: Maximum cache size in GB
            max_age_days: Maximum age in days
        """
        if max_size_gb is None:
            max_size_gb = self.max_cache_size_gb
        if max_age_days is None:
            max_age_days = self.cache_expiry_days

        try:
            cache_files = list(self.extraction_cache_dir.glob("*.cache*"))

            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in cache_files)
            total_size_gb = total_size / (1024 ** 3)

            if total_size_gb <= max_size_gb and max_age_days <= 0:
                return  # No cleanup needed

            # Get file info with access times
            file_info = []
            current_time = time.time()

            for cache_file in cache_files:
                stat = cache_file.stat()
                age_days = (current_time - stat.st_atime) / (24 * 3600)

                file_info.append({
                    'path': cache_file,
                    'size': stat.st_size,
                    'age_days': age_days,
                    'access_time': stat.st_atime
                })

            # Remove old files first
            removed_files = 0
            freed_size = 0

            if max_age_days > 0:
                for info in file_info:
                    if info['age_days'] > max_age_days:
                        info['path'].unlink()
                        removed_files += 1
                        freed_size += info['size']
                        total_size_gb -= info['size'] / (1024 ** 3)

            # If still over size limit, remove least recently used files
            if total_size_gb > max_size_gb:
                # Sort by access time (oldest first)
                file_info = [f for f in file_info if f['path'].exists()]  # Only existing files
                file_info.sort(key=lambda x: x['access_time'])

                for info in file_info:
                    if total_size_gb <= max_size_gb:
                        break

                    info['path'].unlink()
                    removed_files += 1
                    freed_size += info['size']
                    total_size_gb -= info['size'] / (1024 ** 3)

            if removed_files > 0:
                print(f"🧹 Cache cleanup: Removed {removed_files} files, freed {freed_size / (1024**2):.1f} MB")

        except Exception as e:
            print(f"❌ Cache cleanup error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.extraction_cache_dir.glob("*.cache*"))

            if not cache_files:
                return {
                    'cache_entries': 0,
                    'total_size_mb': 0,
                    'total_size_gb': 0,
                    'cache_directory': str(self.extraction_cache_dir)
                }

            # Calculate sizes
            total_size = sum(f.stat().st_size for f in cache_files)
            total_size_mb = total_size / (1024 ** 2)
            total_size_gb = total_size / (1024 ** 3)

            # Analyze cache entries
            file_types = {}
            compressed_count = 0

            for cache_file in cache_files[:100]:  # Sample first 100 for performance
                try:
                    if cache_file.name.endswith('.gz'):
                        compressed_count += 1
                        with gzip.open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                    else:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)

                    file_path = cache_data.get('file_path', '')
                    ext = Path(file_path).suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1

                except:
                    continue

            # Calculate average sizes
            avg_size_mb = total_size_mb / len(cache_files) if cache_files else 0

            return {
                'cache_entries': len(cache_files),
                'total_size_mb': round(total_size_mb, 2),
                'total_size_gb': round(total_size_gb, 3),
                'avg_size_mb': round(avg_size_mb, 2),
                'compressed_entries': compressed_count,
                'compression_ratio': round(compressed_count / len(cache_files), 2) if cache_files else 0,
                'file_types': file_types,
                'cache_directory': str(self.extraction_cache_dir),
                'max_size_gb': self.max_cache_size_gb,
                'expiry_days': self.cache_expiry_days
            }

        except Exception as e:
            return {
                'error': str(e),
                'cache_directory': str(self.extraction_cache_dir)
            }

    def optimize_cache(self):
        """Optimize cache by compressing uncompressed files and removing duplicates"""
        try:
            optimized = 0
            space_saved = 0

            # Find uncompressed cache files
            uncompressed_files = list(self.extraction_cache_dir.glob("*.cache"))

            for cache_file in uncompressed_files:
                try:
                    # Load uncompressed data
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)

                    # Save as compressed
                    compressed_file = cache_file.with_suffix('.cache.gz')
                    with gzip.open(compressed_file, 'wb') as f:
                        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # Calculate space saved
                    original_size = cache_file.stat().st_size
                    compressed_size = compressed_file.stat().st_size
                    space_saved += original_size - compressed_size

                    # Remove original
                    cache_file.unlink()
                    optimized += 1

                except Exception as e:
                    continue

            if optimized > 0:
                print(f"📦 Cache optimization: Compressed {optimized} files, saved {space_saved / (1024**2):.1f} MB")

            # Run cleanup
            self.cleanup_cache()

        except Exception as e:
            print(f"❌ Cache optimization error: {e}")

# Global cache manager instance
_global_cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    return _global_cache_manager

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return _global_cache_manager.get_cache_stats()

def clear_cache(file_path: str = None, config: Dict = None):
    """Clear cache entries"""
    _global_cache_manager.clear_cache(file_path, config)

def cleanup_cache(max_size_gb: float = None, max_age_days: int = None):
    """Clean up cache based on size and age limits"""
    _global_cache_manager.cleanup_cache(max_size_gb, max_age_days)

def optimize_cache():
    """Optimize cache by compressing and cleaning up"""
    _global_cache_manager.optimize_cache()

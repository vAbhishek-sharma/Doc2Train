# processors/base_processor.py
"""
Complete base processor class with plugin architecture
Provides common functionality for all document processors
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Type
from pathlib import Path
import time
import hashlib
import json
import re
import regex
import unicodedata
from utils.progress import start_file_processing, complete_file_processing, add_processing_error
from utils.cache import CacheManager
from utils.validation import validate_extraction_quality
_PROCESSOR_PLUGINS: Dict[str, Type["BaseProcessor"]] = {}

class BaseProcessor(ABC):
    """
    Base class for all document processors with complete functionality
    Provides plugin architecture, caching, validation, and common features
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize processor with configuration

        Args:
            config: Processing configuration dictionary
        """
        self.config = config or {}
        self.cache_manager = CacheManager()
        self.supported_extensions = []
        self.processor_name = self.__class__.__name__
        self.stats = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_text_chars': 0,
            'total_images': 0,
            'total_processing_time': 0.0
        }

    @abstractmethod
    def extract_content_impl(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Abstract method for content extraction - must be implemented by subclasses

        Args:
            file_path: Path to file to process

        Returns:
            Tuple of (text_content, list_of_images)
        """
        pass

    def extract_content(self, file_path: str, use_cache: bool = True) -> Tuple[str, List[Dict]]:
        """
        Main content extraction method with full pipeline

        Args:
            file_path: Path to file to process
            use_cache: Whether to use cached results

        Returns:
            Tuple of (text_content, list_of_images)
        """
        file_name = Path(file_path).name
        start_time = time.time()

        # Update progress
        start_file_processing(file_name)

        try:
            # Pre-processing validation
            if not self._validate_file_before_processing(file_path):
                raise ValueError(f"File validation failed: {file_path}")

            # Check cache first
            if use_cache and self.config.get('use_cache', True):
                cached_result = self._load_from_cache(file_path)
                if cached_result:
                    processing_time = time.time() - start_time
                    self._update_stats(True, len(cached_result['text']), len(cached_result['images']), processing_time)
                    complete_file_processing(file_name, len(cached_result['text']),
                                           len(cached_result['images']), processing_time, True, self.processor_name)
                    return cached_result['text'], cached_result['images']

            # Extract content using implementation
            text, images = self.extract_content_impl(file_path)

            # Post-processing
            text = self._apply_text_filters(text)
            images = self._apply_image_filters(images)

            # Validate extraction quality
            if not self._validate_extraction_quality(text, images):
                if not self.config.get('allow_low_quality', False):
                    raise ValueError("Content quality below threshold")

            processing_time = time.time() - start_time

            # Cache results
            if use_cache and self.config.get('use_cache', True):
                self._save_to_cache(file_path, text, images, processing_time)

            # Update statistics
            self._update_stats(True, len(text), len(images), processing_time)

            # Update progress
            complete_file_processing(file_name, len(text), len(images), processing_time, True, self.processor_name)

            return text, images

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, 0, 0, processing_time)

            error_msg = f"Error in {self.processor_name}: {str(e)}"
            add_processing_error(file_name, error_msg)

            if self.config.get('fail_on_error', False):
                raise

            complete_file_processing(file_name, 0, 0, processing_time, False, self.processor_name)
            return "", []

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about file without processing

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        path = Path(file_path)

        try:
            stat = path.stat()
            info = {
                'name': path.name,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'extension': path.suffix.lower(),
                'processor': self.processor_name,
                'modified': stat.st_mtime,
                'is_cached': self._is_cached(file_path),
                'supported': self.supports_file(file_path),
                'estimated_processing_time': self._estimate_processing_time(file_path)
            }

            # Add processor-specific info
            processor_info = self._get_processor_specific_info(file_path)
            info.update(processor_info)

            return info

        except Exception as e:
            return {
                'name': path.name,
                'error': str(e),
                'supported': False
            }

    def supports_file(self, file_path: str) -> bool:
        """
        Check if this processor supports the given file

        Args:
            file_path: Path to file

        Returns:
            True if supported
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions

    def _validate_file_before_processing(self, file_path: str) -> bool:
        """Validate file before processing"""
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            return False

        # Check file size
        max_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB default
        if path.stat().st_size > max_size:
            raise ValueError(f"File too large: {path.stat().st_size} bytes > {max_size} bytes")

        # Check if file is supported
        if not self.supports_file(file_path):
            return False

        return True

    def _apply_text_filters(self, text: str) -> str:
        """Apply text filtering based on configuration"""
        if not text:
            return text

        # Apply minimum length filter first
        min_length = self.config.get('min_text_length', 0)
        if len(text.strip()) < min_length:
            return ""

        # Apply header/footer removal
        if self.config.get('header_regex'):
            text = self._remove_headers_footers(text, self.config['header_regex'])

        # Apply additional text cleaning
        text = self._clean_text(text)

        return text

    def _apply_image_filters(self, images: List[Dict]) -> List[Dict]:
        """Apply image filtering based on configuration"""
        if not images:
            return images

        filtered = []
        min_size = self.config.get('min_image_size', 0)
        skip_single_color_images = self.config.get('skip_single_color_images', False)

        for img in images:
            # Size filter
            if 'dimensions' in img:
                width, height = img['dimensions']
                if width * height < min_size:
                    if self.config.get('verbose'):
                        print(f"   Skipping small image: {width}x{height} ({width*height} px)")
                    continue

            # Single color filter
            if skip_single_color_images and img.get('is_single_color', False):
                if self.config.get('verbose'):
                    print(f"   Skipping single-color image")
                continue

            # Quality filter
            quality_threshold = self.config.get('quality_threshold', 0.0)
            if img.get('quality_score', 1.0) < quality_threshold:
                if self.config.get('verbose'):
                    print(f"   Skipping low-quality image: {img.get('quality_score', 0):.2f}")
                continue

            filtered.append(img)

        if len(filtered) < len(images) and self.config.get('verbose'):
            print(f"   Filtered images: {len(images)} → {len(filtered)}")

        return filtered

    def _remove_headers_footers(self, text: str, header_regex: str) -> str:
        """Remove headers and footers using regex pattern"""
        try:
            lines = text.split('\n')
            filtered_lines = []
            pattern = re.compile(header_regex, re.IGNORECASE)

            for line in lines:
                if not pattern.search(line.strip()):
                    filtered_lines.append(line)

            filtered_text = '\n'.join(filtered_lines)

            if self.config.get('verbose'):
                removed_lines = len(lines) - len(filtered_lines)
                if removed_lines > 0:
                    print(f"   Removed {removed_lines} header/footer lines")

            return filtered_text

        except Exception as e:
            if self.config.get('verbose'):
                print(f"   Warning: Header regex failed: {e}")
            return text



    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        import unicodedata

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)  # Reduce multiple spaces/tabs to 1

        # Remove control characters but keep basic punctuation and letters
        # Using simpler regex without Unicode categories
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\'/@#$%^&*+=<>|\\`~\n]', '', text)

        # Optional: Replace common ligatures
        ligature_map = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '"': "'",
            '"': '"',
            '"': '"',
            '–': '-',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            '–': '-',     # en dash
            '—': '-',     # em dash
            '…': '...',   # ellipsis
        }
        for bad, good in ligature_map.items():
            text = text.replace(bad, good)

        return text.strip()




    def _validate_extraction_quality(self, text: str, images: List[Dict]) -> bool:
        """Validate extraction quality"""
        return validate_extraction_quality(text, images, self.config)

    def _estimate_processing_time(self, file_path: str) -> float:
        """Estimate processing time for a file"""
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            # Base estimate - subclasses should override for better accuracy
            return file_size_mb * 1.0  # 1 second per MB
        except:
            return 5.0  # Default estimate

    def _get_processor_specific_info(self, file_path: str) -> Dict[str, Any]:
        """Get processor-specific file information - override in subclasses"""
        return {}

    ### Cache functions ###
    def _load_from_cache(self, file_path: str) -> Optional[Dict]:
        """Load cached extraction results"""
        return self.cache_manager.load_from_cache(file_path, self.config)

    def _save_to_cache(self, file_path: str, text: str, images: List[Dict], processing_time: float):
        """Save extraction results to cache"""
        self.cache_manager.save_to_cache(file_path, text, images, self.config, processing_time)

    def _is_cached(self, file_path: str) -> bool:
        """Check if file has cached results"""
        return self.cache_manager.is_cached(file_path, self.config)

    ### Stat Functions ###
    def _update_stats(self, success: bool, text_chars: int, image_count: int, processing_time: float):
        """Update processor statistics"""
        self.stats['files_processed'] += 1
        if success:
            self.stats['files_successful'] += 1
            self.stats['total_text_chars'] += text_chars
            self.stats['total_images'] += image_count
        else:
            self.stats['files_failed'] += 1

        self.stats['total_processing_time'] += processing_time

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        stats = self.stats.copy()

        # Calculate derived statistics
        if self.stats['files_processed'] > 0:
            stats['success_rate'] = self.stats['files_successful'] / self.stats['files_processed']
            stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['files_processed']
        else:
            stats['success_rate'] = 0.0
            stats['avg_processing_time'] = 0.0

        if self.stats['files_successful'] > 0:
            stats['avg_text_per_file'] = self.stats['total_text_chars'] / self.stats['files_successful']
            stats['avg_images_per_file'] = self.stats['total_images'] / self.stats['files_successful']
        else:
            stats['avg_text_per_file'] = 0.0
            stats['avg_images_per_file'] = 0.0

        return stats


def register_processor(name: str, cls: Type[BaseProcessor]):
    _PROCESSOR_PLUGINS[name] = cls

def get_processor(name: str) -> Optional[Type[BaseProcessor]]:
    return _PROCESSOR_PLUGINS.get(name)

def list_processors() -> List[str]:
    return List(_PROCESSOR_PLUGINS)

class ProcessorRegistry:
    """
    Registry for document processors with plugin support
    Manages processor discovery and instantiation
    """

    def __init__(self):
        self.processors = {}
        self.plugin_processors = {}
        self._register_builtin_processors()

    def _register_builtin_processors(self):
        """Register built-in processors"""
        # Import and register built-in processors
        try:
            from .pdf_processor import PDFProcessor
            self.register('pdf', ['.pdf'], PDFProcessor)
        except ImportError:
            pass

        try:
            from .text_processor import TextProcessor
            self.register('text', ['.txt', '.srt', '.vtt'], TextProcessor)
        except ImportError:
            pass

        try:
            from .epub_processor import EPUBProcessor
            self.register('epub', ['.epub'], EPUBProcessor)
        except ImportError:
            pass

        try:
            from .image_processor import ImageProcessor
            self.register('image', ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'], ImageProcessor)
        except ImportError:
            pass

    def register(self, name: str, extensions: List[str], processor_class):
        """
        Register a processor for given extensions

        Args:
            name: Processor name
            extensions: List of file extensions (with dots)
            processor_class: Processor class
        """
        for ext in extensions:
            self.processors[ext.lower()] = {
                'name': name,
                'class': processor_class,
                'extensions': extensions
            }

    def register_plugin(self, plugin_path: str):
        """
        Register a processor plugin from file

        Args:
            plugin_path: Path to plugin file
        """
        try:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)

            # Look for processor classes in the plugin
            for attr_name in dir(plugin_module):
                attr = getattr(plugin_module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseProcessor) and
                    attr != BaseProcessor):

                    # Get processor info
                    processor_instance = attr()
                    name = getattr(processor_instance, 'processor_name', attr_name)
                    extensions = getattr(processor_instance, 'supported_extensions', [])

                    if extensions:
                        self.register(name, extensions, attr)
                        self.plugin_processors[name] = plugin_path
                        print(f"✅ Loaded plugin processor: {name} for {extensions}")

        except Exception as e:
            print(f"❌ Failed to load plugin {plugin_path}: {e}")

    def discover_plugins(self, plugin_dir: str):
        """
        Discover and register all plugins in a directory

        Args:
            plugin_dir: Directory containing plugin files
        """
        if not Path(plugin_dir).exists():
            return

        for plugin_file in Path(plugin_dir).glob("*.py"):
            if plugin_file.name != "__init__.py":
                self.register_plugin(str(plugin_file))

    def get_processor(self, file_path: str, config: Optional[Dict] = None):
        """
        Get appropriate processor for file

        Args:
            file_path: Path to file
            config: Processing configuration

        Returns:
            Processor instance
        """
        extension = Path(file_path).suffix.lower()
        processor_info = self.processors.get(extension)

        if processor_info:
            processor_class = processor_info['class']
            return processor_class(config)

        raise ValueError(f"No processor found for extension: {extension}")

    def get_processor_by_name(self, name: str, config: Optional[Dict] = None):
        """
        Get processor by name

        Args:
            name: Processor name
            config: Processing configuration

        Returns:
            Processor instance
        """
        for processor_info in self.processors.values():
            if processor_info['name'] == name:
                processor_class = processor_info['class']
                return processor_class(config)

        raise ValueError(f"No processor found with name: {name}")

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported extensions"""
        return list(self.processors.keys())

    def get_processor_info(self) -> Dict[str, Dict]:
        """Get information about all registered processors"""
        info = {}

        # Group by processor name
        for ext, processor_info in self.processors.items():
            name = processor_info['name']
            if name not in info:
                info[name] = {
                    'name': name,
                    'extensions': [],
                    'class': processor_info['class'].__name__,
                    'is_plugin': name in self.plugin_processors
                }
            info[name]['extensions'].append(ext)

        return info

    def list_processors(self):
        """Print list of all registered processors"""
        info = self.get_processor_info()

        print("📋 Registered Processors:")
        for name, details in info.items():
            plugin_marker = " (Plugin)" if details['is_plugin'] else ""
            print(f"   {name}{plugin_marker}:")
            print(f"     Extensions: {', '.join(details['extensions'])}")
            print(f"     Class: {details['class']}")

    # Global processor registry instance
_processor_registry = ProcessorRegistry()

def get_processor_registry() -> ProcessorRegistry:
    """Get the global processor registry"""
    return _processor_registry

def register_processor(name: str, extensions: List[str], processor_class):
    """Register a processor class globally"""
    _processor_registry.register(name, extensions, processor_class)

def get_processor_for_file(file_path: str, config: Optional[Dict] = None):
    """Get processor instance for file"""
    return _processor_registry.get_processor(file_path, config)

def get_supported_extensions() -> List[str]:
    """Get list of all supported file extensions"""
    return _processor_registry.get_supported_extensions()

def discover_plugins(plugin_dir: str):
    """Discover and register plugins from directory"""
    _processor_registry.discover_plugins(plugin_dir)

def list_all_processors():
    """List all registered processors"""
    _processor_registry.list_processors()



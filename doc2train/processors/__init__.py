# processors/__init__.py
"""
Document processors for Doc2Train v2.0 Enhanced
Now with proper class-based processors and registry system
"""

# Import the base processor and registry
from .base_processor import (
    BaseProcessor, get_processor_registry,
    register_processor, get_processor_for_file, get_supported_extensions,
    discover_plugins, list_all_processors
)

# Import all processor classes
try:
    from .pdf_processor import PDFProcessor
except ImportError as e:
    print(f"Warning: PDFProcessor not available: {e}")
    PDFProcessor = None

try:
    from .text_processor import TextProcessor
except ImportError as e:
    print(f"Warning: TextProcessor not available: {e}")
    TextProcessor = None

try:
    from .epub_processor import EPUBProcessor
except ImportError as e:
    print(f"Warning: EPUBProcessor not available: {e}")
    EPUBProcessor = None

try:
    from .image_processor import ImageProcessor
except ImportError as e:
    print(f"Warning: ImageProcessor not available: {e}")
    ImageProcessor = None

# Import backward compatibility functions
from .pdf_processor import extract_pdf_content
from .text_processor import extract_text_content
from .epub_processor import extract_epub_content
from .image_processor import extract_image_content

# Register all available processors
# def _register_all_processors():
#     """Register all available processor classes"""
#     registry = get_processor_registry()

#     if PDFProcessor:
#         registry.register('pdf', ['.pdf'], PDFProcessor)

#     if TextProcessor:
#         registry.register('text', ['.txt', '.srt', '.vtt'], TextProcessor)

#     if EPUBProcessor:
#         registry.register('epub', ['.epub'], EPUBProcessor)

#     if ImageProcessor:
#         registry.register('image', ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'], ImageProcessor)

# # Auto-register processors on import
# _register_all_processors()

__all__ = [
    # Base processor and registry
    'BaseProcessor', 'get_processor_registry',
    'register_processor', 'get_processor_for_file', 'get_supported_extensions',
    'discover_plugins', 'list_all_processors', 'extract_content_with_processor',

    # Processor classes
    'PDFProcessor', 'TextProcessor', 'EPUBProcessor', 'ImageProcessor',

    # Backward compatibility functions
    'extract_pdf_content', 'extract_text_content', 'extract_epub_content', 'extract_image_content'
]

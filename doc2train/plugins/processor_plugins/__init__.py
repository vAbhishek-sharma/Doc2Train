# plugins/processors_plugins/__init__.py

# Expose all Processor plugins here:
from .sample_custom_video_processor import VideoProcessor
from .epub_processor import EPUBProcessor
from .image_processor import ImageProcessor
from .pdf_processor import PDFProcessor
from .text_processor import TextProcessor
__all__ = [
    "VideoProcessor",
    "PDFProcessor",
    "ImageProcessor",
    "EPUBProcessor",
    "TextProcessor"
]

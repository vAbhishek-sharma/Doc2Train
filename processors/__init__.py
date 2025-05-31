# processors/__init__.py
"""
Document processors for Doc2Train v2.0
"""

from .pdf_processor import extract_pdf_content
from .text_processor import extract_text_content
from .image_processor import extract_image_content
from .epub_processor import extract_epub_content

__all__ = [
    'extract_pdf_content',
    'extract_text_content',
    'extract_image_content',
    'extract_epub_content'
]

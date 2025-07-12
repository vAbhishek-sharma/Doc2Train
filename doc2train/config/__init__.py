# config/__init__.py
"""
Configuration for Doc2Train
"""

from .settings import *

__all__ = [
    'SUPPORTED_FORMATS',
    'LLM_PROVIDERS',
    'GENERATORS',
    'OUTPUT_FORMATS',
    'CHUNK_SIZE',
    'MAX_WORKERS',
    'DEFAULT_PROVIDER',
    'DEFAULT_GENERATORS',
    'OUTPUT_DIR',
    'CACHE_DIR',
    'USE_CACHE',
    'USE_OCR',
    'EXTRACT_IMAGES'
]

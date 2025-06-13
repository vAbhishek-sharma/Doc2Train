# core/extractor.py
"""
Simple content extractor - handles all document types
Extract text and images from any supported file format
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional  # Added this line
from config.settings import *
def extract_content(file_path: str, use_cache: bool = True) -> Tuple[str, List[Dict]]:
    """
    Extract text and images from any supported file

    Args:
        file_path: Path to the file to process
        use_cache: Whether to use cached results

    Returns:
        Tuple of (text_content, list_of_images)
    """
    if not is_supported_format(file_path):
        raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")

    # Check cache first
    if use_cache and USE_CACHE:
        cached_result = _load_from_cache(file_path)
        if cached_result:
            print(f"📌 Loading from cache: {Path(file_path).name}")
            return cached_result['text'], cached_result['images']

    # Extract content based on file type
    processor_name = get_processor_for_file(file_path)

    if processor_name == 'pdf_processor':
        text, images = _extract_pdf(file_path)
    elif processor_name == 'text_processor':
        text, images = _extract_text_file(file_path)
    elif processor_name == 'epub_processor':
        text, images = _extract_epub(file_path)
    elif processor_name == 'image_processor':
        text, images = _extract_image(file_path)
    else:
        raise ValueError(f"No processor found for {file_path}")

    # Cache the results
    if USE_CACHE:
        _save_to_cache(file_path, text, images)

    print(f"✅ Extracted from {Path(file_path).name}: {len(text)} chars, {len(images)} images")
    return text, images

def _extract_pdf(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract text and images from PDF"""
    from processors.pdf_processor import extract_pdf_content
    return extract_pdf_content(file_path)

def _extract_text_file(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract text from text-based files (TXT, SRT, VTT)"""
    from processors.text_processor import extract_text_content
    return extract_text_content(file_path)

def _extract_epub(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract text and images from EPUB"""
    from processors.epub_processor import extract_epub_content
    return extract_epub_content(file_path)

def _extract_image(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract text from image using OCR"""
    from processors.image_processor import extract_image_content
    return extract_image_content(file_path)

def _get_cache_path(file_path: str) -> str:
    """Get cache file path for a given input file"""
    # Create hash of file path + modification time for cache key
    file_stat = os.stat(file_path)
    cache_key = hashlib.md5(
        f"{file_path}{file_stat.st_mtime}{file_stat.st_size}".encode()
    ).hexdigest()

    cache_dir = Path(CACHE_DIR) / "extracted"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return str(cache_dir / f"{cache_key}.json")

def _load_from_cache(file_path: str) -> Optional[Dict]:
    """Load cached extraction results"""
    cache_path = _get_cache_path(file_path)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if DEBUG:
                print(f"Cache read error: {e}")

    return None

def _save_to_cache(file_path: str, text: str, images: List[Dict]):
    """Save extraction results to cache"""
    cache_path = _get_cache_path(file_path)

    cache_data = {
        'file_path': file_path,
        'text': text,
        'images': images,
        'extracted_at': os.path.getctime(cache_path) if os.path.exists(cache_path) else None
    }

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        if DEBUG:
            print(f"Cache write error: {e}")

def extract_batch(file_paths: List[str], use_cache: bool = True) -> Dict[str, Tuple[str, List[Dict]]]:
    """
    Extract content from multiple files

    Args:
        file_paths: List of file paths to process
        use_cache: Whether to use cached results

    Returns:
        Dictionary mapping file paths to (text, images) tuples
    """
    results = {}

    for file_path in file_paths:
        try:
            text, images = extract_content(file_path, use_cache)
            results[file_path] = (text, images)
        except Exception as e:
            print(f"❌ Error extracting {file_path}: {e}")
            results[file_path] = ("", [])  # Empty result for failed files

    return results

def get_supported_files(directory: str) -> List[str]:
    """
    Get all supported files from a directory (recursively)

    Args:
        directory: Directory path to scan

    Returns:
        List of supported file paths
    """
    supported_files = []
    directory_path = Path(directory)

    if directory_path.is_file():
        # Single file
        if is_supported_format(str(directory_path)):
            supported_files.append(str(directory_path))
    elif directory_path.is_dir():
        # Directory - scan recursively
        for ext in SUPPORTED_FORMATS.keys():
            pattern = f"**/*{ext}"
            files = list(directory_path.glob(pattern))
            supported_files.extend([str(f) for f in files])

    return sorted(supported_files)

def get_file_info(file_path: str) -> Dict:
    """
    Get basic information about a file

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    stat = path.stat()

    return {
        'name': path.name,
        'size_mb': stat.st_size / (1024 * 1024),
        'extension': path.suffix.lower(),
        'processor': get_processor_for_file(file_path),
        'modified': stat.st_mtime,
        'is_cached': _load_from_cache(file_path) is not None
    }

def clear_cache(file_path: str = None):
    """
    Clear extraction cache

    Args:
        file_path: Specific file to clear cache for, or None to clear all
    """
    if file_path:
        # Clear cache for specific file
        cache_path = _get_cache_path(file_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"🗑️  Cleared cache for {Path(file_path).name}")
    else:
        # Clear all cache
        cache_dir = Path(CACHE_DIR) / "extracted"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("🗑️  Cleared all extraction cache")

def validate_extraction(text: str, images: List[Dict]) -> bool:
    """
    Validate extracted content meets minimum quality requirements

    Args:
        text: Extracted text content
        images: List of extracted images

    Returns:
        True if content meets quality requirements
    """
    # Check text length
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    # Check for reasonable text (not just gibberish)
    words = text.split()
    if len(words) < 10:  # At least 10 words
        return False

    # Check average word length (detect OCR garbage)
    avg_word_length = sum(len(word) for word in words) / len(words)
    if avg_word_length > 15:  # Suspiciously long words might be OCR errors
        return False

    return True

# Utility functions for working with extracted content
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to split
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within overlap range
            search_start = max(start, end - overlap)
            sentence_end = text.rfind('.', search_start, end)

            if sentence_end > search_start:
                end = sentence_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks

def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count (for cost estimation)

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for English
    return len(text) // 4

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            text, images = extract_content(file_path)
            print(f"Text length: {len(text)}")
            print(f"Images found: {len(images)}")
            print(f"First 200 chars: {text[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python core/extractor.py <file_path>")

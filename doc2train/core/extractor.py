# core/extractor.py

"""
Universal content extractor for Doc2Train (PLUGIN-BASED)
- Dynamically selects processor plugins based on file extension.
- Adds flexible, configurable text chunking strategies.
- Supports caching, validation, and batch extraction.

To add new processors: just drop into processor_plugins/ and they’ll be picked up.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from doc2train.config.settings import *
import ipdb
# Import plugin registry for processors
from doc2train.core.registries.processor_registry import get_processor_for_file, get_supported_extensions_dict

# ------------------------
# EXTRACT CONTENT (SMART PLUGIN ROUTING)
# ------------------------

def extract_content(file_path: str, use_cache: bool = True, config: dict = None) -> Tuple[str, List[Dict]]:
    """
    Extract text and images from any supported file using plugin-based processors.

    Args:
        file_path: Path to the file to process.
        use_cache: Whether to use cached results.
        config: Optional processor-specific config.

    Returns:
        Tuple of (text_content, list_of_images)
    """
    ext = Path(file_path).suffix.lower()
    # Check if supported
    supported_exts = get_supported_extensions_dict()
    if not any(ext in [e.lower() for e in exts] for exts in supported_exts.values()):
        raise ValueError(f"Unsupported file format: {ext}")

    # Check cache first
    if use_cache and USE_CACHE:
        cached_result = _load_from_cache(file_path)
        if cached_result:
            print(f"📌 Loading from cache: {Path(file_path).name}")
            return cached_result['text'], cached_result['images']

    # Find processor plugin
    try:
        processor = get_processor_for_file(file_path, config=config)
    except Exception as e:
        raise ValueError(f"No processor found for {file_path}: {e}")
    # Extract content (every processor exposes extract(file_path))
    modalities = processor.extract(file_path)

    # Cache results
    if USE_CACHE:
        _save_to_cache(file_path, modalities)

    print(f"✅ Extracted from {Path(file_path).name}: {len(modalities['text'])} chars, {len(modalities['images'])} images")

    return modalities

# ------------------------
# CACHING UTILS
# ------------------------

def _get_cache_path(file_path: str) -> str:
    """Get cache file path for a given input file"""
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

def _save_to_cache(file_path: str, modalities: dict, extracted_at: Optional[str|int]):
    """Save extraction results to cache"""
    cache_path = _get_cache_path(file_path)
    cache_data = {
        'file_path': file_path,
        'text': modalities['text'],
        'images': modalities['images'],
        'extracted_at': os.path.getctime(cache_path) if os.path.exists(cache_path) else None
    }
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        if DEBUG:
            print(f"Cache write error: {e}")

# ------------------------
# BATCH EXTRACTION
# ------------------------

def extract_batch(file_paths: List[str], use_cache: bool = True, config: dict = None) -> Dict[str, Tuple[str, List[Dict]]]:
    """
    Extract content from multiple files using smart plugin routing.

    Args:
        file_paths: List of file paths to process
        use_cache: Whether to use cached results
        config: Optional processor config

    Returns:
        Dictionary mapping file paths to (text, images) tuples
    """
    results = {}
    for file_path in file_paths:
        try:
            text, images = extract_content(file_path, use_cache, config=config)
            results[file_path] = (text, images)
        except Exception as e:
            print(f"❌ Error extracting {file_path}: {e}")
            results[file_path] = ("", [])  # Empty result for failed files
    return results

# ------------------------
# SMART FILE DISCOVERY
# ------------------------

def get_supported_files(directory: str) -> List[str]:
    """
    Get all supported files from a directory (recursively)
    Args:
        directory: Directory path to scan
    Returns:
        List of supported file paths
    """
    supported_exts = set()
    ext_dict = get_supported_extensions_dict()
    for ext_list in ext_dict.values():
        supported_exts.update([e.lower() for e in ext_list])
    directory_path = Path(directory)
    files = []
    if directory_path.is_file():
        ext = directory_path.suffix.lower()
        if ext in supported_exts:
            files.append(str(directory_path))
    elif directory_path.is_dir():
        for ext in supported_exts:
            pattern = f"**/*{ext}"
            files.extend([str(f) for f in directory_path.glob(pattern)])
    return sorted(files)

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
    ext = path.suffix.lower()
    return {
        'name': path.name,
        'size_mb': stat.st_size / (1024 * 1024),
        'extension': ext,
        'processor': None,  # Optionally: get_processor_for_file(file_path).__class__.__name__
        'modified': stat.st_mtime,
        'is_cached': _load_from_cache(file_path) is not None
    }

# ------------------------
# CACHE CLEAR & VALIDATION
# ------------------------

def clear_cache(file_path: str = None):
    """
    Clear extraction cache
    Args:
        file_path: Specific file to clear cache for, or None to clear all
    """
    if file_path:
        cache_path = _get_cache_path(file_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"🗑️  Cleared cache for {Path(file_path).name}")
    else:
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
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    words = text.split()
    if len(words) < 10:
        return False
    avg_word_length = sum(len(word) for word in words) / len(words)
    if avg_word_length > 15:
        return False
    return True

# ------------------------
# ADVANCED CHUNKING
# ------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, strategy: str = "default", **kwargs) -> List[str]:
    """
    Split text into chunks using multiple strategies.

    Args:
        text: Text to split
        chunk_size: Max size (chars or lines, depending on strategy)
        overlap: Overlap between chunks (default strategy only)
        strategy: One of ["default", "paragraph", "lines", "separator", "tokens"]
        kwargs: Extra args for some strategies
    Returns:
        List of text chunks
    """
    if strategy == "default":
        # Original overlapping chunking (break at sentence if possible)
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to break at sentence boundary
            if end < len(text):
                search_start = max(start, end - overlap)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > search_start:
                    end = sentence_end + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    elif strategy == "paragraph":
        # Split on double newlines as paragraphs, then merge to max size
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        curr = ""
        for p in paragraphs:
            if len(curr) + len(p) + 2 > chunk_size:
                if curr:
                    chunks.append(curr.strip())
                curr = p
            else:
                curr += ("\n\n" if curr else "") + p
        if curr:
            chunks.append(curr.strip())
        return chunks

    elif strategy == "lines":
        # Split by lines, then group every chunk_size lines
        lines = text.splitlines()
        lines_per_chunk = kwargs.get("lines_per_chunk", 20)
        chunks = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk = "\n".join(lines[i:i+lines_per_chunk])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    elif strategy == "separator":
        # Split by a custom separator (e.g., "###", "\n---\n")
        sep = kwargs.get("separator", "\n---\n")
        pieces = text.split(sep)
        return [p.strip() for p in pieces if p.strip()]

    elif strategy == "tokens":
        # Split by estimated token count
        tokens_per_chunk = kwargs.get("tokens_per_chunk", 200)
        # Approximate: 1 token = 4 chars
        chunk_size = tokens_per_chunk * 4
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap, strategy="default")

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count (for cost estimation)
    Args:
        text: Text to count tokens for
    Returns:
        Estimated token count
    """
    return len(text) // 4

# ------------------------
# CLI Test Harness
# ------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Doc2Train universal extractor")
    parser.add_argument("file_path", help="File to extract")
    parser.add_argument("--chunk-strategy", default="default", choices=["default", "paragraph", "lines", "separator", "tokens"], help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size (chars or lines)")
    parser.add_argument("--lines-per-chunk", type=int, default=20, help="Lines per chunk (if using lines strategy)")
    parser.add_argument("--separator", type=str, default="\n---\n", help="Custom separator (if using separator strategy)")
    args = parser.parse_args()

    try:
        text, images = extract_content(args.file_path)
        print(f"Text length: {len(text)}")
        print(f"Images found: {len(images)}")
        print(f"First 200 chars: {text[:200]}...")

        # Show chunked text
        kwargs = {}
        if args.chunk_strategy == "lines":
            kwargs["lines_per_chunk"] = args.lines_per_chunk
        if args.chunk_strategy == "separator":
            kwargs["separator"] = args.separator

        chunks = chunk_text(text, chunk_size=args.chunk_size, strategy=args.chunk_strategy, **kwargs)
        print(f"Total chunks: {len(chunks)} (strategy: {args.chunk_strategy})")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"--- Chunk {i} ---\n{chunk[:200]}...\n")

    except Exception as e:
        print(f"Error: {e}")

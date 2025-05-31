# processors/text_processor.py
"""
Simple text processor - handles TXT, SRT, VTT files
"""

import re
import webvtt
from typing import Tuple, List, Dict
from pathlib import Path

def extract_text_content(file_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text content from text-based files

    Args:
        file_path: Path to text file

    Returns:
        Tuple of (text_content, empty_images_list)
    """
    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.srt':
            return _extract_srt(file_path), []
        elif file_ext == '.vtt':
            return _extract_vtt(file_path), []
        else:  # .txt and other text files
            return _extract_txt(file_path), []

    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")
        return "", []

def _extract_txt(file_path: str) -> str:
    """Extract text from plain text file"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # If all encodings fail, read as binary and decode with errors='ignore'
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')

def _extract_srt(file_path: str) -> str:
    """Extract text from SRT subtitle file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove subtitle numbers and timestamps
    lines = content.split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines, numbers, and timestamp lines
        if (line and
            not re.match(r'^\d+$', line) and
            not re.match(r'^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$', line)):
            text_lines.append(line)

    return ' '.join(text_lines)

def _extract_vtt(file_path: str) -> str:
    """Extract text from VTT subtitle file"""
    try:
        vtt = webvtt.read(file_path)
        text_parts = []

        for caption in vtt:
            # Clean up the text (remove formatting tags)
            clean_text = re.sub(r'<[^>]+>', '', caption.text)
            text_parts.append(clean_text)

        return ' '.join(text_parts)

    except Exception as e:
        print(f"Error reading VTT file: {e}")
        # Fallback to simple text extraction
        return _extract_txt(file_path)

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        text_path = sys.argv[1]

        print(f"Processing text file: {text_path}")
        text, images = extract_text_content(text_path)

        print(f"Text length: {len(text)} characters")
        print(f"First 200 characters: {text[:200]}...")
    else:
        print("Usage: python processors/text_processor.py <text_file>")

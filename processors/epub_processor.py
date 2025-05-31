# processors/epub_processor.py
"""
Simple EPUB processor - handles EPUB ebook files
"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Tuple, List, Dict

def extract_epub_content(file_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text content from EPUB file

    Args:
        file_path: Path to EPUB file

    Returns:
        Tuple of (text_content, images_list)
    """
    try:
        book = epub.read_epub(file_path)
        text_content = ""
        images = []

        # Extract text from all chapters
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')

                # Extract text
                chapter_text = soup.get_text(separator=' ', strip=True)
                text_content += chapter_text + "\n\n"

            elif item.get_type() == ebooklib.ITEM_IMAGE:
                # Extract images
                image_info = {
                    'filename': item.get_name(),
                    'data': item.get_content(),
                    'context': f"Image from EPUB: {item.get_name()}",
                    'ocr_text': ""  # Could add OCR here if needed
                }
                images.append(image_info)

        return text_content.strip(), images

    except Exception as e:
        print(f"Error processing EPUB {file_path}: {e}")
        return "", []

def get_epub_info(file_path: str) -> Dict:
    """Get basic information about an EPUB file"""
    try:
        book = epub.read_epub(file_path)

        # Get metadata
        metadata = {}
        for meta in book.get_metadata('DC', 'title'):
            metadata['title'] = meta[0]
        for meta in book.get_metadata('DC', 'creator'):
            metadata['author'] = meta[0]
        for meta in book.get_metadata('DC', 'language'):
            metadata['language'] = meta[0]

        # Count items
        chapter_count = 0
        image_count = 0

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_count += 1
            elif item.get_type() == ebooklib.ITEM_IMAGE:
                image_count += 1

        info = {
            'metadata': metadata,
            'chapter_count': chapter_count,
            'image_count': image_count,
            'spine_length': len(book.spine)
        }

        return info

    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        epub_path = sys.argv[1]

        print(f"Processing EPUB: {epub_path}")

        # Get basic info
        info = get_epub_info(epub_path)
        print(f"EPUB info: {info}")

        # Extract content
        text, images = extract_epub_content(epub_path)

        print(f"Text length: {len(text)} characters")
        print(f"Images found: {len(images)}")

        if text:
            print(f"First 200 characters: {text[:200]}...")
    else:
        print("Usage: python processors/epub_processor.py <epub_file>")

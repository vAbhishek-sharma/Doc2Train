# processors/image_processor.py
"""
Simple image processor - handles PNG, JPG, etc.
"""

import os
from PIL import Image
from typing import Tuple, List, Dict
from config.settings import USE_OCR

def extract_image_content(file_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text from image using OCR

    Args:
        file_path: Path to image file

    Returns:
        Tuple of (ocr_text, image_data_list)
    """
    try:
        # Load image
        image = Image.open(file_path)

        # Perform OCR if enabled
        ocr_text = ""
        if USE_OCR:
            ocr_text = _perform_ocr(image)

        # Read image data for potential vision LLM processing
        with open(file_path, 'rb') as f:
            image_data = f.read()

        image_info = {
            'path': file_path,
            'data': image_data,
            'ocr_text': ocr_text,
            'context': f"Image file: {os.path.basename(file_path)}",
            'dimensions': image.size
        }

        return ocr_text, [image_info]

    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return "", []

def _perform_ocr(image: Image.Image) -> str:
    """Perform OCR on PIL Image"""
    try:
        import pytesseract

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform OCR
        text = pytesseract.image_to_string(image)
        return text.strip()

    except ImportError:
        print("pytesseract not available for OCR")
        return ""
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def get_image_info(file_path: str) -> Dict:
    """Get basic information about an image file"""
    try:
        image = Image.open(file_path)

        info = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        }

        return info

    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        print(f"Processing image: {image_path}")

        # Get basic info
        info = get_image_info(image_path)
        print(f"Image info: {info}")

        # Extract content
        text, images = extract_image_content(image_path)

        print(f"OCR text length: {len(text)} characters")
        print(f"Images found: {len(images)}")

        if text:
            print(f"OCR text: {text[:200]}...")
    else:
        print("Usage: python processors/image_processor.py <image_file>")

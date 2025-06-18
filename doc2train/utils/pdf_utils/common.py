# processors/pdf_utils/common.py
"""
Common PDF utility functions for image extraction, OCR, and quality assessment
"""

import fitz  # PyMuPDF
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple



def perform_ocr_on_page(page, config=None) -> str:
    """
    Perform OCR on a PDF page

    Args:
        page: PyMuPDF page object
        config: Optional configuration dict

    Returns:
        Extracted text from OCR
    """
    config = config or {}

    try:
        import pytesseract

        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))

        # Configure OCR options
        ocr_config = config.get('ocr_config', '--psm 6')

        # Perform OCR
        ocr_text = pytesseract.image_to_string(pil_image, config=ocr_config)

        # Clean up
        del pix

        return ocr_text.strip()

    except ImportError:
        # pytesseract not available
        return ""
    except Exception as e:
        # OCR failed
        return ""


def assess_pdf_image_quality(page) -> Dict:
    """
    Assess the quality and characteristics of images on a PDF page

    Args:
        page: PyMuPDF page object

    Returns:
        Dictionary with quality assessment
    """
    assessment = {
        'total_images': 0,
        'high_quality_images': 0,
        'medium_quality_images': 0,
        'low_quality_images': 0,
        'average_resolution': 0,
        'has_vector_graphics': False,
        'dominant_image_type': 'none'
    }

    try:
        images = page.get_images(full=True)
        assessment['total_images'] = len(images)

        if not images:
            return assessment

        resolutions = []
        quality_scores = []

        for img in images:
            try:
                # Get image dimensions
                img_rect = page.get_image_bbox(img)
                if img_rect:
                    width = img_rect.width
                    height = img_rect.height
                    resolution = width * height
                    resolutions.append(resolution)

                    # Simple quality assessment based on size
                    if resolution > 500000:  # High quality
                        assessment['high_quality_images'] += 1
                        quality_scores.append(3)
                    elif resolution > 100000:  # Medium quality
                        assessment['medium_quality_images'] += 1
                        quality_scores.append(2)
                    else:  # Low quality
                        assessment['low_quality_images'] += 1
                        quality_scores.append(1)
            except:
                continue

        # Calculate averages
        if resolutions:
            assessment['average_resolution'] = sum(resolutions) / len(resolutions)

        # Determine dominant image type
        if assessment['high_quality_images'] > assessment['medium_quality_images'] + assessment['low_quality_images']:
            assessment['dominant_image_type'] = 'high_quality'
        elif assessment['medium_quality_images'] > assessment['low_quality_images']:
            assessment['dominant_image_type'] = 'medium_quality'
        elif assessment['low_quality_images'] > 0:
            assessment['dominant_image_type'] = 'low_quality'

        # Check for vector graphics (simplified check)
        # This is a basic heuristic - real vector detection would be more complex
        if len(images) > 0 and assessment['average_resolution'] > 1000000:
            assessment['has_vector_graphics'] = True

    except Exception as e:
        # Return default assessment on error
        pass

    return assessment


def extract_pdf_metadata(file_path: str) -> Dict:
    """
    Extract metadata from PDF file

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with PDF metadata
    """
    metadata = {}

    try:
        doc = fitz.open(file_path)

        # Basic document info
        metadata.update({
            'page_count': len(doc),
            'pdf_version': getattr(doc, 'pdf_version', 'unknown'),
            'is_pdf': doc.is_pdf,
            'is_encrypted': doc.is_encrypted,
            'can_save_incrementally': doc.can_save_incrementally()
        })

        # Document metadata
        doc_metadata = doc.metadata
        if doc_metadata:
            metadata.update({
                'title': doc_metadata.get('title', ''),
                'author': doc_metadata.get('author', ''),
                'subject': doc_metadata.get('subject', ''),
                'creator': doc_metadata.get('creator', ''),
                'producer': doc_metadata.get('producer', ''),
                'creation_date': doc_metadata.get('creationDate', ''),
                'modification_date': doc_metadata.get('modDate', '')
            })

        # Calculate document statistics
        total_text_length = 0
        total_images = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            total_text_length += len(page_text)
            total_images += len(page.get_images())

        metadata.update({
            'total_text_length': total_text_length,
            'total_images': total_images,
            'avg_text_per_page': total_text_length / len(doc) if len(doc) > 0 else 0,
            'avg_images_per_page': total_images / len(doc) if len(doc) > 0 else 0
        })

        doc.close()

    except Exception as e:
        metadata['error'] = str(e)

    return metadata


def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a file is a valid PDF

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        doc = fitz.open(file_path)

        if not doc.is_pdf:
            doc.close()
            return False, "File is not a valid PDF"

        if len(doc) == 0:
            doc.close()
            return False, "PDF has no pages"

        if doc.is_encrypted:
            doc.close()
            return False, "PDF is encrypted and cannot be processed"

        doc.close()
        return True, "Valid PDF file"

    except Exception as e:
        return False, f"Error validating PDF: {str(e)}"


def get_pdf_text_sample(file_path: str, max_chars: int = 1000) -> str:
    """
    Get a sample of text from the PDF for preview

    Args:
        file_path: Path to PDF file
        max_chars: Maximum number of characters to return

    Returns:
        Sample text from the PDF
    """
    try:
        doc = fitz.open(file_path)
        sample_text = ""

        # Extract text from first few pages until we have enough
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            sample_text += page_text

            if len(sample_text) >= max_chars:
                break

        doc.close()

        # Truncate to max_chars and clean up
        if len(sample_text) > max_chars:
            sample_text = sample_text[:max_chars] + "..."

        return sample_text.strip()

    except Exception as e:
        import traceback
        print(f"🐛 DEBUG - Full traceback for PDF processing:")
        traceback.print_exc()
        return f"Error extracting text sample: {str(e)}"

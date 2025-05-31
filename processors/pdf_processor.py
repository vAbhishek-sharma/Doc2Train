# processors/pdf_processor_fixed.py
"""
Fixed PDF processor - handles the bbox issue
"""

import fitz  # PyMuPDF
import io
from PIL import Image
from typing import Tuple, List, Dict
from config.settings import USE_OCR, EXTRACT_IMAGES

def extract_pdf_content(file_path: str) -> Tuple[str, List[Dict]]:
    """
    Simple, working PDF extraction that avoids bbox issues

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple of (text_content, list_of_images)
    """
    try:
        doc = fitz.open(file_path)
        text_content = ""
        images = []

        print(f"📄 Processing PDF: {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            page_text = page.get_text()

            # If no text and OCR is enabled, try OCR
            if not page_text.strip() and USE_OCR:
                page_text = _ocr_page(page)

            text_content += page_text + "\n\n"

            # Extract images if enabled (with fixed approach)
            if EXTRACT_IMAGES:
                page_images = _extract_page_images_safe(page, page_num, doc)
                images.extend(page_images)

        doc.close()

        print(f"✅ Extracted {len(text_content)} characters and {len(images)} images")
        return text_content.strip(), images

    except Exception as e:
        print(f"❌ Error processing PDF {file_path}: {e}")
        return "", []

def _extract_page_images_safe(page, page_num: int, doc) -> List[Dict]:
    """Extract images from a PDF page - safe version that avoids bbox issues"""
    images = []

    try:
        image_list = page.get_images()
        print(f"🖼️  Page {page_num + 1}: Found {len(image_list)} images")

        for img_index, img in enumerate(image_list):
            try:
                # Get image data using xref
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha < 4:  # Valid image (GRAY or RGB)
                    # Convert to bytes
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_data = pix.tobytes("png")

                    # Get page text as context (avoid bbox issues)
                    page_text = page.get_text()

                    # Try OCR on image if enabled
                    ocr_text = ""
                    if USE_OCR:
                        ocr_text = _ocr_image_data(img_data)

                    # Create image info without problematic bbox
                    images.append({
                        'page_num': page_num + 1,
                        'image_index': img_index,
                        'data': img_data,
                        'context': page_text[:500] + "..." if len(page_text) > 500 else page_text,
                        'ocr_text': ocr_text,
                        'dimensions': (pix.width, pix.height),
                        'size_bytes': len(img_data)
                    })

                    print(f"✅ Extracted image {img_index + 1}: {pix.width}x{pix.height}")

                pix = None  # Clean up

            except Exception as e:
                print(f"⚠️  Error extracting image {img_index} from page {page_num}: {e}")
                continue

    except Exception as e:
        print(f"❌ Error processing images on page {page_num}: {e}")

    return images

def _ocr_page(page) -> str:
    """Perform OCR on a PDF page"""
    try:
        import pytesseract

        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
        img_data = pix.tobytes("png")

        # OCR the image
        pil_image = Image.open(io.BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(pil_image)

        pix = None
        return ocr_text

    except ImportError:
        print("⚠️  pytesseract not available for OCR")
        return ""
    except Exception as e:
        print(f"⚠️  OCR failed: {e}")
        return ""

def _ocr_image_data(img_data: bytes) -> str:
    """Perform OCR on image data"""
    try:
        import pytesseract

        pil_image = Image.open(io.BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(pil_image)
        return ocr_text.strip()

    except ImportError:
        return ""
    except Exception as e:
        print(f"⚠️  Image OCR failed: {e}")
        return ""

def get_pdf_info(file_path: str) -> Dict:
    """Get basic information about a PDF file"""
    try:
        doc = fitz.open(file_path)

        info = {
            'page_count': len(doc),
            'metadata': doc.metadata,
            'has_text': False,
            'has_images': False
        }

        # Quick check for content
        for page_num in range(min(3, len(doc))):  # Check first 3 pages
            page = doc[page_num]

            if page.get_text().strip():
                info['has_text'] = True

            if page.get_images():
                info['has_images'] = True

            if info['has_text'] and info['has_images']:
                break

        doc.close()
        return info

    except Exception as e:
        return {'error': str(e)}

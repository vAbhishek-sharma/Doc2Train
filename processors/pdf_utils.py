# processors/pdf_utils.py
"""PDF processing utilities - safe image extraction and OCR"""

import fitz  # PyMuPDF
import io
from typing import List, Dict
from PIL import Image
from config import settings as config


def extract_page_images_safe(page, page_num: int, doc) -> List[Dict]:
    """Extract images from a PDF page - safe version with optional OCR and filtering"""
    images = []
    min_image_size = 1000  # pixel threshold

    try:
        image_list = page.get_images(full=True)
        if getattr(config, 'VERBOSE', False) and image_list:
            print(f"🖼️  Page {page_num}: Found {len(image_list)} images")

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # Check if it's a valid image (not CMYK or too complex)
                if pix.n - pix.alpha < 4:
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_size = pix.width * pix.height
                    if image_size < min_image_size:
                        if getattr(config, 'VERBOSE', False):
                            print(f"   ⏭️ Skipping small image: {pix.width}x{pix.height}")
                        del pix
                        continue

                    img_data = pix.tobytes("png")
                    page_text = page.get_text()

                    ocr_text = ""
                    if getattr(config, 'USE_OCR', True):
                        ocr_text = perform_ocr_on_image_data(img_data)

                    image_info = {
                        'page_num': page_num,
                        'image_index': img_index,
                        'data': img_data,
                        'context': page_text[:500] + "..." if len(page_text) > 500 else page_text,
                        'ocr_text': ocr_text,
                        'dimensions': (pix.width, pix.height),
                        'size_bytes': len(img_data),
                        'quality_score': assess_pdf_image_quality(pix)
                    }

                    images.append(image_info)

                    if getattr(config, 'VERBOSE', False):
                        print(f"✅ Extracted image {img_index + 1}: {pix.width}x{pix.height}")

                del pix

            except Exception as e:
                if getattr(config, 'VERBOSE', False):
                    print(f"⚠️  Error extracting image {img_index} from page {page_num}: {e}")
                continue

    except Exception as e:
        if getattr(config, 'VERBOSE', False):
            print(f"❌ Error processing images on page {page_num}: {e}")

    return images


def perform_ocr_on_image_data(img_data: bytes) -> str:
    """Perform OCR on image data (PNG bytes)"""
    try:
        import pytesseract
        pil_image = Image.open(io.BytesIO(img_data))
        return pytesseract.image_to_string(pil_image)

    except Exception as e:
        if getattr(config, 'VERBOSE', False):
            print(f"⚠️  OCR failed on image data: {e}")
        return ""


def perform_ocr_on_page(page) -> str:
    """Render page as image and apply OCR"""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
        img_data = pix.tobytes("png")
        del pix
        return perform_ocr_on_image_data(img_data)

    except Exception as e:
        if getattr(config, 'VERBOSE', False):
            print(f"⚠️  OCR failed on page: {e}")
        return ""


def assess_pdf_image_quality(pix) -> float:
    """Estimate image quality based on size"""
    try:
        width, height = pix.width, pix.height
        area = width * height
        size_score = min(1.0, area / 100_000)  # Normalize to 100K pixels
        return size_score
    except:
        return 0.5  # Fallback quality

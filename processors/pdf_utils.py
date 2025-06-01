# processors/pdf_utils.py - NEW FILE (extracted from pdf_processor.py)
"""PDF processing utilities - extracted to reduce file size"""

from typing import List, Dict

def extract_page_images_safe( page, page_num: int, doc) -> List[Dict]:
    """Extract images from a PDF page - safe version"""
    images = []
    min_image_size = self.config.get('min_image_size', 1000)

    try:
        image_list = page.get_images()
        if self.config.get('verbose') and image_list:
            print(f"🖼️  Page {page_num}: Found {len(image_list)} images")

        for img_index, img in enumerate(image_list):
            try:
                # Get image data using xref
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha < 4:  # Valid image (GRAY or RGB)
                    # Convert to bytes
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Check size filter
                    image_size = pix.width * pix.height
                    if image_size < min_image_size:
                        if self.config.get('verbose'):
                            print(f"   ⏭️ Skipping small image: {pix.width}x{pix.height}")
                        pix = None
                        continue

                    img_data = pix.tobytes("png")

                    # Get page text as context
                    page_text = page.get_text()

                    # Try OCR on image if enabled
                    ocr_text = ""
                    if self.config.get('use_ocr', True):
                        ocr_text = self._ocr_image_data(img_data)

                    # Create image info
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

                    if self.config.get('verbose'):
                        print(f"✅ Extracted image {img_index + 1}: {pix.width}x{pix.height}")

                pix = None  # Clean up

            except Exception as e:
                if self.config.get('verbose'):
                    print(f"⚠️  Error extracting image {img_index} from page {page_num}: {e}")
                continue

    except Exception as e:
        if self.config.get('verbose'):
            print(f"❌ Error processing images on page {page_num}: {e}")

    return images


def perform_ocr_on_page( page ) -> str:
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
        if self.config.get('verbose'):
            print("⚠️  pytesseract not available for OCR")
        return ""
    except Exception as e:
        if self.config.get('verbose'):
            print(f"⚠️  OCR failed: {e}")
        return ""


def assess_pdf_image_quality( pix ) -> float:
    """Assess image quality (0.0 to 1.0)"""
    try:
        # Basic quality assessment based on size and color diversity
        width, height = pix.width, pix.height
        area = width * height

        # Size score (larger = better, up to a point)
        size_score = min(1.0, area / 100000)  # Normalize to 100K pixels

        # TODO: Add more sophisticated quality metrics
        # - Color diversity
        # - Sharpness
        # - Contrast

        return size_score

    except:
        return 0.5  # Default moderate quality

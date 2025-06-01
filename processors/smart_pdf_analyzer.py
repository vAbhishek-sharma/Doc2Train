# processors/smart_pdf_analyzer.py
"""
Smart PDF Content Analyzer - Detects text, images, and mixed content in PDFs
Determines the best processing strategy for each document
"""

import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pdb
class PDFContentType(Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MIXED_CONTENT = "mixed_content"
    SCANNED_DOCUMENT = "scanned_document"
    EMPTY = "empty"

@dataclass
class PDFAnalysis:
    content_type: PDFContentType
    total_pages: int
    text_pages: int
    image_pages: int
    mixed_pages: int
    text_coverage: float  # % of pages with meaningful text
    image_coverage: float  # % of pages with images
    average_text_per_page: float
    total_images: int
    processing_strategy: str
    confidence: float

class SmartPDFAnalyzer:
    """Analyzes PDF content to determine optimal processing strategy"""

    def __init__(self):
        self.min_text_threshold = 50  # Minimum chars to consider "text content"
        self.min_words_threshold = 10  # Minimum words to consider meaningful text

    def analyze_pdf(self, file_path: str) -> PDFAnalysis:
        """
        Comprehensive PDF analysis to determine content type and processing strategy

        Args:
            file_path: Path to PDF file

        Returns:
            PDFAnalysis object with detailed content analysis
        """
        try:
            doc = fitz.open(file_path)

            # Initialize counters
            total_pages = len(doc)
            text_pages = 0
            image_pages = 0
            mixed_pages = 0
            total_text_length = 0
            total_images = 0

            page_analysis = []

            # Analyze each page
            for page_num in range(total_pages):
                page = doc[page_num]
                page_info = self._analyze_page(page)
                page_analysis.append(page_info)

                # Update counters
                if page_info['has_meaningful_text'] and page_info['has_images']:
                    mixed_pages += 1
                elif page_info['has_meaningful_text']:
                    text_pages += 1
                elif page_info['has_images']:
                    image_pages += 1

                total_text_length += page_info['text_length']
                total_images += page_info['image_count']

            doc.close()

            # Calculate metrics
            text_coverage = (text_pages + mixed_pages) / total_pages if total_pages > 0 else 0
            image_coverage = (image_pages + mixed_pages) / total_pages if total_pages > 0 else 0
            avg_text_per_page = total_text_length / total_pages if total_pages > 0 else 0

            # Determine content type and strategy
            content_type, strategy, confidence = self._determine_content_type(
                text_coverage, image_coverage, mixed_pages, total_pages, avg_text_per_page
            )

            return PDFAnalysis(
                content_type=content_type,
                total_pages=total_pages,
                text_pages=text_pages,
                image_pages=image_pages,
                mixed_pages=mixed_pages,
                text_coverage=text_coverage,
                image_coverage=image_coverage,
                average_text_per_page=avg_text_per_page,
                total_images=total_images,
                processing_strategy=strategy,
                confidence=confidence
            )

        except Exception as e:
            print(f"Error analyzing PDF {file_path}: {e}")
            return PDFAnalysis(
                content_type=PDFContentType.EMPTY,
                total_pages=0,
                text_pages=0,
                image_pages=0,
                mixed_pages=0,
                text_coverage=0.0,
                image_coverage=0.0,
                average_text_per_page=0.0,
                total_images=0,
                processing_strategy="error",
                confidence=0.0
            )

    def _analyze_page(self, page) -> Dict:
        """Analyze a single page for text and image content"""

        # Extract text
        text = page.get_text()
        text_length = len(text.strip())
        word_count = len(text.strip().split()) if text.strip() else 0

        # Check for meaningful text (not just whitespace, page numbers, etc.)
        has_meaningful_text = (
            text_length >= self.min_text_threshold and
            word_count >= self.min_words_threshold and
            self._is_meaningful_text(text)
        )

        # Check for images
        images = page.get_images()
        image_count = len(images)
        has_images = image_count > 0

        # Analyze image types and sizes
        significant_images = 0
        for img in images:
            try:
                # Get image dimensions from the page
                img_rect = page.get_image_bbox(img)
                img_area = (img_rect.width * img_rect.height) if img_rect else 0

                # Consider images significant if they're large enough
                # (filters out small icons, logos, etc.)
                if img_area > 10000:  # Adjust threshold as needed
                    significant_images += 1
            except:
                significant_images += 1  # Count it if we can't analyze

        return {
            'text_length': text_length,
            'word_count': word_count,
            'has_meaningful_text': has_meaningful_text,
            'has_images': has_images,
            'image_count': image_count,
            'significant_images': significant_images,
            'text_to_image_ratio': text_length / max(image_count, 1)
        }

    def _is_meaningful_text(self, text: str) -> bool:
        """Check if text contains meaningful content (not just headers, page numbers, etc.)"""

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if len(lines) < 2:  # Very short content
            return False

        # Check for common non-meaningful patterns
        meaningful_lines = 0
        for line in lines:
            # Skip page numbers, headers, single words, etc.
            if (len(line) > 10 and
                not line.isdigit() and
                len(line.split()) > 2 and
                not self._is_header_footer(line)):
                meaningful_lines += 1

        # At least 50% of lines should be meaningful
        return meaningful_lines / len(lines) > 0.5

    def _is_header_footer(self, line: str) -> bool:
        """Detect common header/footer patterns"""
        line_lower = line.lower().strip()

        # Common header/footer patterns
        header_footer_patterns = [
            'page ', 'chapter ', 'section ',
            'confidential', 'proprietary',
            'copyright', '©', '®', '™'
        ]

        # Very short lines are likely headers/footers
        if len(line.split()) <= 3:
            return True

        # Check for common patterns
        for pattern in header_footer_patterns:
            if pattern in line_lower:
                return True

        return False

    def _determine_content_type(self, text_coverage: float, image_coverage: float,
                               mixed_pages: int, total_pages: int, avg_text_per_page: float) -> Tuple[PDFContentType, str, float]:
        """Determine content type and optimal processing strategy"""

        # Calculate confidence based on how clear the content type is
        confidence = 0.0

        # Mixed content detection
        if mixed_pages / total_pages > 0.3:  # 30%+ pages have both text and images
            content_type = PDFContentType.MIXED_CONTENT
            strategy = "mixed_processing"  # Process both text and images with context linking
            confidence = min(0.9, mixed_pages / total_pages + 0.3)

        # Primarily text with some images
        elif text_coverage > 0.7 and image_coverage > 0.1:
            content_type = PDFContentType.MIXED_CONTENT
            strategy = "text_primary_mixed"  # Focus on text, but process images for context
            confidence = text_coverage * 0.8

        # Text-only document
        elif text_coverage > 0.8 and image_coverage < 0.2:
            content_type = PDFContentType.TEXT_ONLY
            strategy = "text_only"  # Standard text extraction
            confidence = text_coverage

        # Image-heavy document
        elif image_coverage > 0.6 and text_coverage < 0.3:
            content_type = PDFContentType.IMAGE_ONLY
            strategy = "image_heavy"  # OCR + image analysis focus
            confidence = image_coverage

        # Scanned document (low text coverage, high image coverage, low avg text)
        elif text_coverage < 0.5 and image_coverage > 0.5 and avg_text_per_page < 100:
            content_type = PDFContentType.SCANNED_DOCUMENT
            strategy = "ocr_primary"  # Heavy OCR processing
            confidence = 0.7

        # Empty or unprocessable
        elif text_coverage < 0.1 and image_coverage < 0.1:
            content_type = PDFContentType.EMPTY
            strategy = "skip"  # Don't process
            confidence = 0.9

        # Default to mixed if unclear
        else:
            content_type = PDFContentType.MIXED_CONTENT
            strategy = "conservative_mixed"  # Process everything carefully
            confidence = 0.5

        return content_type, strategy, confidence

# Integration with main PDF processor
def analyze_and_extract_pdf(file_path: str) -> Tuple[str, List[Dict], PDFAnalysis]:
    """
    Analyze PDF and extract content using optimal strategy

    Returns:
        Tuple of (text_content, images_list, analysis_result)
    """
    pdb.set_trace()
    # First, analyze the PDF
    analyzer = SmartPDFAnalyzer()
    analysis = analyzer.analyze_pdf(file_path)

    print(f"📊 PDF Analysis: {analysis.content_type.value} "
          f"({analysis.confidence:.2f} confidence)")
    print(f"📄 {analysis.text_pages} text pages, "
          f"🖼️  {analysis.image_pages} image pages, "
          f"📄🖼️  {analysis.mixed_pages} mixed pages")

    # Extract content based on strategy
    if analysis.processing_strategy == "skip":
        return "", [], analysis

    elif analysis.processing_strategy == "text_only":
        text, images = extract_text_focused(file_path)

    elif analysis.processing_strategy == "image_heavy":
        text, images = extract_image_focused(file_path)

    elif analysis.processing_strategy == "ocr_primary":
        text, images = extract_with_heavy_ocr(file_path)

    elif analysis.processing_strategy in ["mixed_processing", "text_primary_mixed", "conservative_mixed"]:
        text, images = extract_mixed_content(file_path, analysis)

    else:
        # Fallback to standard extraction
        from processors.pdf_processor import extract_pdf_content
        text, images = extract_pdf_content(file_path)

    return text, images, analysis

def extract_mixed_content(file_path: str, analysis: PDFAnalysis, verbose: bool = False) -> Tuple[str, List[Dict]]:
    """Extract content from mixed text/image PDFs with context linking"""

    doc = fitz.open(file_path)
    text_content = ""
    images = []

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            page_text = page.get_text()

            # Extract images with enhanced context
            page_images = page.get_images(full=True)

            if page_images and page_text.strip():
                for img_index, img in enumerate(page_images):
                    try:
                        # Skip if img doesn't have enough data
                        if len(img) < 7:
                            if verbose:
                                print(f"⚠️ Skipping incomplete image ref on page {page_num + 1}")
                            continue

                        xref = img[0]

                        # Try primary and fallback bbox extraction
                        try:
                            img_rect = page.get_image_bbox(img)
                        except Exception:
                            img_rects = page.get_image_rects(xref)
                            img_rect = img_rects[0] if img_rects else fitz.Rect(0, 0, page.rect.width, page.rect.height)

                        # Extract image data
                        pix = fitz.Pixmap(doc, xref)

                        # Optional: skip small images (e.g., icons)
                        if pix.width * pix.height < 10000:
                            if verbose:
                                print(f"⏭️ Skipping small image ({pix.width}x{pix.height}) on page {page_num + 1}")
                            del pix
                            continue

                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")

                            enhanced_context = {
                                'text_before': extract_text_before_image(page, img_rect),
                                'text_after': extract_text_after_image(page, img_rect),
                                'full_page_text': page_text,
                                'page_type': 'mixed_content',
                                'position_in_page': get_image_position(img_rect, page.rect)
                            }

                            images.append({
                                'page_num': page_num + 1,
                                'image_index': img_index,
                                'data': img_data,
                                'format': 'png',
                                'context': enhanced_context,
                                'bbox': [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1],
                                'content_type': 'mixed',
                                'dimensions': (pix.width, pix.height)
                            })

                            if verbose:
                                print(f"✅ Extracted image {img_index + 1} on page {page_num + 1}: {pix.width}x{pix.height}")

                        del pix

                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error extracting image {img_index} on page {page_num + 1}: {e}")
                        continue

            # Add page text with placeholder
            if page_text.strip():
                if page_images:
                    enhanced_text = insert_image_placeholders(page_text, page_images, page)
                    text_content += enhanced_text + "\n\n"
                else:
                    text_content += page_text + "\n\n"

        except Exception as e:
            if verbose:
                print(f"❌ Error processing page {page_num + 1}: {e}")
            continue

    doc.close()
    return text_content.strip(), images

def extract_text_before_image(page, img_rect) -> str:
    """Extract text that appears before an image on the page"""
    # Create a rectangle for text above the image
    text_rect = fitz.Rect(0, 0, page.rect.width, img_rect.y0)
    return page.get_text(clip=text_rect).strip()

def extract_text_after_image(page, img_rect) -> str:
    """Extract text that appears after an image on the page"""
    # Create a rectangle for text below the image
    text_rect = fitz.Rect(0, img_rect.y1, page.rect.width, page.rect.height)
    return page.get_text(clip=text_rect).strip()

def get_image_position(img_rect, page_rect) -> str:
    """Determine the relative position of an image on the page"""
    img_center_x = (img_rect.x0 + img_rect.x1) / 2
    img_center_y = (img_rect.y0 + img_rect.y1) / 2

    page_width = page_rect.width
    page_height = page_rect.height

    # Determine horizontal position
    if img_center_x < page_width / 3:
        h_pos = "left"
    elif img_center_x > 2 * page_width / 3:
        h_pos = "right"
    else:
        h_pos = "center"

    # Determine vertical position
    if img_center_y < page_height / 3:
        v_pos = "top"
    elif img_center_y > 2 * page_height / 3:
        v_pos = "bottom"
    else:
        v_pos = "middle"

    return f"{v_pos}_{h_pos}"

def insert_image_placeholders(text: str, images: List, page) -> str:
    """Insert image placeholders in text for better context understanding"""

    # This is a simplified approach - in practice, you'd want more sophisticated
    # text-image alignment based on actual positions

    if not images:
        return text

    # Simple approach: mention images at the end of text
    placeholder = f"\n[Document contains {len(images)} image(s) on this page]\n"
    return text + placeholder

# Helper extraction functions for different content types
def extract_text_focused(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract from text-focused PDFs"""
    from processors.pdf_processor import extract_pdf_content
    return extract_pdf_content(file_path, skip_analysis=True)


def extract_image_focused(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract from image-heavy PDFs with minimal text"""
    # Focus on OCR and image analysis
    doc = fitz.open(file_path)
    text_content = ""
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Try text extraction first
        page_text = page.get_text()
        if not page_text.strip():
            # Use OCR if no text
            try:
                import pytesseract
                from PIL import Image
                import io

                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                page_text = pytesseract.image_to_string(pil_image)
                pix = None
            except:
                page_text = ""

        text_content += page_text + "\n\n"

        # Extract all images
        page_images = page.get_images()
        for img_index, img in enumerate(page_images):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if pix.n - pix.alpha < 4:
                    img_data = pix.tobytes("png")

                    images.append({
                        'page_num': page_num + 1,
                        'image_index': img_index,
                        'data': img_data,
                        'context': page_text,
                        'content_type': 'image_focused'
                    })

                pix = None
            except:
                continue

    doc.close()
    return text_content.strip(), images

def extract_with_heavy_ocr(file_path: str) -> Tuple[str, List[Dict]]:
    """Extract from scanned documents using heavy OCR"""
    # Implementation for scanned documents
    return extract_image_focused(file_path)  # Similar approach with enhanced OCR

if __name__ == "__main__":
    # Test the analyzer
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]

        analyzer = SmartPDFAnalyzer()
        analysis = analyzer.analyze_pdf(pdf_path)

        print(f"📄 PDF Analysis Results for: {pdf_path}")
        print(f"Content Type: {analysis.content_type.value}")
        print(f"Processing Strategy: {analysis.processing_strategy}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Pages: {analysis.total_pages} total, {analysis.mixed_pages} mixed")
        print(f"Text Coverage: {analysis.text_coverage:.2f}")
        print(f"Image Coverage: {analysis.image_coverage:.2f}")
        print(f"Total Images: {analysis.total_images}")
    else:
        print("Usage: python smart_pdf_analyzer.py <pdf_file>")

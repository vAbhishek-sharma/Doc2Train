# processors/pdf_utils/analyzer.py
"""
Smart PDF Content Analyzer - Detects text, images, and mixed content in PDFs
Determines the best processing strategy for each document
"""

import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


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

    def __init__(self, config=None):
        # Allow configuration override
        config = config or {}
        self.min_text_threshold = config.get('min_text_threshold', 50)
        self.min_words_threshold = config.get('min_words_threshold', 10)
        self.min_lines_threshold = config.get('min_lines_threshold', 2)
        self.verbose = config.get('verbose', False)

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

            if self.verbose:
                print(f"🔍 Analyzing PDF: {file_path} ({total_pages} pages)")

            # Analyze each page
            for page_num in range(total_pages):
                page = doc[page_num]
                page_info = self._analyze_page(page)

                # Update counters
                if page_info['has_meaningful_text'] and page_info['has_images']:
                    mixed_pages += 1
                elif page_info['has_meaningful_text']:
                    text_pages += 1
                elif page_info['has_images']:
                    image_pages += 1

                total_text_length += page_info['text_length']
                total_images += page_info['image_count']

                if self.verbose and page_num < 5:  # Show first 5 pages analysis
                    print(f"   Page {page_num + 1}: Text={page_info['text_length']} chars, "
                          f"Images={page_info['image_count']}, "
                          f"Meaningful={'Yes' if page_info['has_meaningful_text'] else 'No'}")

            doc.close()

            # Calculate metrics
            text_coverage = (text_pages + mixed_pages) / total_pages if total_pages > 0 else 0
            image_coverage = (image_pages + mixed_pages) / total_pages if total_pages > 0 else 0
            avg_text_per_page = total_text_length / total_pages if total_pages > 0 else 0

            # Determine content type and strategy
            content_type, strategy, confidence = self._determine_content_type(
                text_coverage, image_coverage, mixed_pages, total_pages, avg_text_per_page
            )

            if self.verbose:
                print(f"📊 Analysis complete: {content_type.value} ({confidence:.2f} confidence)")

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
            if self.verbose:
                print(f"❌ Error analyzing PDF {file_path}: {e}")
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

        if len(lines) < self.min_lines_threshold:  # Very short content
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


if __name__ == "__main__":
    # Test the analyzer
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]

        analyzer = SmartPDFAnalyzer(config={'verbose': True})
        analysis = analyzer.analyze_pdf(pdf_path)

        print(f"\n📄 PDF Analysis Results for: {pdf_path}")
        print(f"Content Type: {analysis.content_type.value}")
        print(f"Processing Strategy: {analysis.processing_strategy}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Pages: {analysis.total_pages} total, {analysis.mixed_pages} mixed")
        print(f"Text Coverage: {analysis.text_coverage:.2f}")
        print(f"Image Coverage: {analysis.image_coverage:.2f}")
        print(f"Total Images: {analysis.total_images}")
    else:
        print("Usage: python analyzer.py <pdf_file>")

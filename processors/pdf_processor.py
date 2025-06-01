# processors/pdf_processor.py
"""
Class-based PDF processor with Smart PDF Analysis integration
"""

import fitz  # PyMuPDF
import io
from PIL import Image
from typing import Tuple, List, Dict
from pathlib import Path

from .base_processor import BaseProcessor
from .smart_pdf_analyzer import SmartPDFAnalyzer, analyze_and_extract_pdf
from .pdf_utils import extract_page_images_safe, perform_ocr_on_page, assess_pdf_image_quality

class PDFProcessor(BaseProcessor):
    """PDF processor with smart analysis and full BaseProcessor functionality"""

    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.pdf']
        self.processor_name = "PDFProcessor"
        self.analyzer = SmartPDFAnalyzer()

    def extract_content_impl(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text and images from PDF using smart analysis

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (text_content, list_of_images)
        """
        try:
            # Use smart analyzer to determine optimal processing strategy
            if self.config.get('use_smart_analysis', True):
                text, images, analysis = analyze_and_extract_pdf(file_path)

                # Store analysis results for debugging/stats
                if self.config.get('verbose'):
                    print(f"📊 PDF Analysis: {analysis.content_type.value} "
                          f"({analysis.confidence:.2f} confidence)")
                    print(f"📄 Strategy: {analysis.processing_strategy}")
                    print(f"📄 Pages: {analysis.text_pages} text, "
                          f"🖼️ {analysis.image_pages} image, "
                          f"📄🖼️ {analysis.mixed_pages} mixed")

                # Apply configuration filters
                text = self._apply_text_filters(text)
                images = self._apply_image_filters(images)

                return text, images
            else:
                # Fallback to basic extraction
                return self._basic_pdf_extraction(file_path)

        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {e}")

    def _basic_pdf_extraction(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Basic PDF extraction without smart analysis (fallback)"""
        doc = fitz.open(file_path)
        text_content = ""
        images = []

        # Get page range from config
        start_page = self.config.get('start_page', 1) - 1  # Convert to 0-based
        end_page = self.config.get('end_page', len(doc))
        skip_pages = set(self.config.get('skip_pages', []))

        if self.config.get('verbose'):
            print(f"📄 Basic PDF processing: {len(doc)} pages (range: {start_page+1}-{end_page})")

        for page_num in range(max(0, start_page), min(len(doc), end_page)):
            # Skip pages if specified
            if (page_num + 1) in skip_pages:
                if self.config.get('verbose'):
                    print(f"   ⏭️ Skipping page {page_num + 1}")
                continue

            page = doc[page_num]

            # Extract text
            page_text = page.get_text()

            # If no text and OCR is enabled, try OCR
            if not page_text.strip() and self.config.get('use_ocr', True):
                page_text = perform_ocr_on_page(page)

            text_content += page_text + "\n\n"

            # Extract images if enabled
            if self.config.get('extract_images', True):
                page_images = extract_page_images_safe(page, page_num + 1, doc)
                images.extend(page_images)

        doc.close()
        return text_content.strip(), images

    def get_pdf_analysis(self, file_path: str):
        """Get detailed PDF analysis without processing"""
        return self.analyzer.analyze_pdf(file_path)

    def _ocr_image_data(self, img_data: bytes) -> str:
        """Perform OCR on image data"""
        try:
            import pytesseract

            pil_image = Image.open(io.BytesIO(img_data))
            ocr_text = pytesseract.image_to_string(pil_image)
            return ocr_text.strip()

        except ImportError:
            return ""
        except Exception as e:
            if self.config.get('verbose'):
                print(f"⚠️  Image OCR failed: {e}")
            return ""

    def _get_processor_specific_info(self, file_path: str) -> Dict:
        """Get PDF-specific file information with smart analysis"""
        try:
            # Use smart analyzer for comprehensive info
            analysis = self.analyzer.analyze_pdf(file_path)

            info = {
                'page_count': analysis.total_pages,
                'content_type': analysis.content_type.value,
                'processing_strategy': analysis.processing_strategy,
                'confidence': analysis.confidence,
                'text_pages': analysis.text_pages,
                'image_pages': analysis.image_pages,
                'mixed_pages': analysis.mixed_pages,
                'text_coverage': analysis.text_coverage,
                'image_coverage': analysis.image_coverage,
                'avg_text_per_page': analysis.average_text_per_page,
                'total_images': analysis.total_images
            }

            # Add basic PDF metadata
            try:
                doc = fitz.open(file_path)
                info.update({
                    'metadata': doc.metadata,
                    'pdf_version': getattr(doc, 'pdf_version', 'unknown')
                })
                doc.close()
            except:
                pass

            return info

        except Exception as e:
            return {'error': str(e)}

    def _estimate_processing_time(self, file_path: str) -> float:
        """Estimate processing time for PDF using smart analysis"""
        try:
            analysis = self.analyzer.analyze_pdf(file_path)

            # Base time depends on content type
            if analysis.content_type.value == 'text_only':
                time_per_page = 0.2
            elif analysis.content_type.value == 'image_only':
                time_per_page = 1.0
            elif analysis.content_type.value == 'scanned_document':
                time_per_page = 2.0  # OCR is slow
            else:  # mixed content
                time_per_page = 0.8

            # Adjust for features
            if self.config.get('extract_images', True):
                time_per_page += 0.3

            if self.config.get('use_ocr', True):
                time_per_page += 0.5

            return analysis.total_pages * time_per_page

        except:
            return 10.0  # Default estimate


# Keep backward compatibility function
def extract_pdf_content(file_path: str) -> Tuple[str, List[Dict]]:
    """Backward compatibility function"""
    processor = PDFProcessor()
    return processor.extract_content_impl(file_path)

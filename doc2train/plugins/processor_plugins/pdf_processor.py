# processors/pdf_processor.py
"""
PDF processor handles the extraction of the data
"""

import fitz  # PyMuPDF
import io
from PIL import Image
from typing import Tuple, List, Dict
from pathlib import Path
from doc2train.plugins.processor_plugins.base_processor import BaseProcessor
from doc2train.utils.pdf_utils.analyzer import SmartPDFAnalyzer
from doc2train.utils.pdf_utils.common import  perform_ocr_on_page
from doc2train.utils.pdf_utils.extraction import extract_page_images_safe
import ipdb

class PDFProcessor(BaseProcessor):
    """PDF processor with smart analysis and full BaseProcessor functionality"""
    supported_extensions = ['.pdf']
    priority ='10'
    description = ''
    version = '1.0.0'
    author = 'doc2train'
    processor_name = 'PDFProcessor'
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.pdf']
        self.processor_name = "PDFProcessor"
        # Import analyzer here to avoid circular import
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
                # Import here to avoid circular import
                from doc2train.utils.pdf_utils.extraction import analyze_and_extract_pdf
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

                return {"text": text, "images": images}
            else:
                # Fallback to basic extraction
                return self._basic_pdf_extraction(file_path)

        except Exception as e:
            import traceback
            print(f"🐛 DEBUG - Full traceback for PDF processing:")
            traceback.print_exc()
            raise Exception(f"Error processing PDF {file_path}: {e}")

    def _basic_pdf_extraction(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Basic PDF extraction without smart analysis (fallback)"""

        import os
        doc = fitz.open(file_path)
        text_content = ""
        images = []

        # Output directory logic (configurable)
        base_output_dir = self.config.get('output_dir', 'output')
        images_subdir = os.path.join(base_output_dir, "images", Path(file_path).stem)
        os.makedirs(images_subdir, exist_ok=True)

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
                page_images = self._extract_page_images_with_base(page, page_num + 1, doc, file_path, images_subdir)
                images.extend(page_images)

        doc.close()
        return text_content.strip(), images

    #TO DELETE
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
            import traceback
            print(f"🐛 DEBUG - Full traceback for PDF processing:")
            traceback.print_exc()
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
            import traceback
            print(f"🐛 DEBUG - Full traceback for PDF processing:")
            traceback.print_exc()
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
            import traceback
            print(f"🐛 DEBUG - Full traceback for PDF processing:")
            traceback.print_exc()
            return 10.0  # Default estimate


# Simple extraction function for backward compatibility
def extract_pdf_content(file_path: str, skip_analysis=False) -> Tuple[str, List[Dict]]:
    """Simple extraction function"""
    processor = PDFProcessor(config={'use_smart_analysis': not skip_analysis})
    return processor.extract_content_impl(file_path)

def _extract_page_images_with_base(self, page, page_num: int, doc, file_path: str, images_subdir: str) -> List[Dict]:
    """
    Save images to images_subdir, returning metadata dicts with file paths (never raw bytes).
    """
    from pathlib import Path
    images = []
    try:
        page_images = page.get_images(full=True)

        for img_index, img in enumerate(page_images):
            try:
                if len(img) < 7:
                    continue
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width * pix.height < 1000:
                        del pix
                        continue
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        try:
                            img_rect = page.get_image_bbox(img)
                            bbox = [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1]
                        except:
                            bbox = None

                        base_name = f"page{page_num}_img{img_index+1}"
                        extra = {
                            'page_num': page_num,
                            'image_index': img_index,
                            'format': 'png',
                            'dimensions': (pix.width, pix.height),
                            'bbox': bbox,
                            'xref': xref
                        }
                        img_info = self._save_and_record_image(
                            img_data,
                            output_dir=images_subdir,
                            base_name=base_name,
                            extra=extra
                        )
                        images.append(img_info)
                    del pix
                except Exception:
                    continue
            except Exception:
                continue
    except Exception:
        pass
    return images

# processors/epub_processor.py
"""
Class-based EPUB processor for ebook files
"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Tuple, List, Dict
from pathlib import Path

from doc2train.utils.image_save import save_image_data
from doc2train.plugins.processor_plugins.base_processor import BaseProcessor
class EPUBProcessor(BaseProcessor):
    """EPUB processor with full BaseProcessor functionality"""
    supported_extensions = ['.epub']
    priority ='10'
    description = ''
    version = '1.0.0'
    author = 'doc2train'
    processor_name = 'EPUBProcessor'
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.epub']
        self.processor_name = "EPUBProcessor"

    def extract_content_impl(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text content and images from EPUB file

        Args:
            file_path: Path to EPUB file

        Returns:
            Tuple of (text_content, images_list)
        """
        try:
            from pathlib import Path
            import os
            book = epub.read_epub(file_path)
            text_content = ""
            images = []

            # Output directory logic (configurable)
            base_output_dir = self.config.get('output_dir', 'output')
            images_subdir = os.path.join(base_output_dir, "images", Path(file_path).stem)
            os.makedirs(images_subdir, exist_ok=True)

            # Get page range constraints if specified
            start_page = self.config.get('start_page', 1)
            end_page = self.config.get('end_page')
            current_chapter = 0

            if self.config.get('verbose'):
                print(f"📚 Processing EPUB: {Path(file_path).name}")

            # Extract text from chapters and images
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    current_chapter += 1

                    # Apply page range (treating chapters as pages for EPUB)
                    if current_chapter < start_page:
                        continue
                    if end_page and current_chapter > end_page:
                        break

                    # Skip if in skip list
                    skip_pages = set(self.config.get('skip_pages', []))
                    if current_chapter in skip_pages:
                        if self.config.get('verbose'):
                            print(f"   ⏭️ Skipping chapter {current_chapter}")
                        continue

                    # Parse HTML content
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(item.get_content(), 'html.parser')

                    # Extract text
                    chapter_text = soup.get_text(separator=' ', strip=True)

                    # Apply minimum length filter
                    min_length = self.config.get('min_text_length', 0)
                    if len(chapter_text.strip()) >= min_length:
                        text_content += f"\n\n=== Chapter {current_chapter} ===\n\n"
                        text_content += chapter_text + "\n\n"

                        if self.config.get('verbose'):
                            print(f"   ✅ Chapter {current_chapter}: {len(chapter_text)} chars")

                elif item.get_type() == ebooklib.ITEM_IMAGE and self.config.get('extract_images', True):
                    image_data = item.get_content()
                    min_size = self.config.get('min_image_size', 1000)
                    if len(image_data) >= min_size:  # Basic size check

                        dimensions = self._get_image_dimensions(image_data)
                        if dimensions:
                            width, height = dimensions
                            if width * height >= min_size:
                                ocr_text = ""
                                if self.config.get('use_ocr', True):
                                    ocr_text = self._ocr_image_data(item)

                                # Compose a DRY base name
                                base_name = f"chapter{current_chapter}_{item.get_name()}"

                                img_path = None
                                if self.config.get('save_images', False):
                                    img_path = save_image_data(
                                        image_data,
                                        output_dir=images_subdir,
                                        base_name=base_name
                                    )

                                image_info = {
                                    'filename': item.get_name(),
                                    'context': f"Image from EPUB: {item.get_name()}",
                                    'ocr_text': ocr_text,
                                    'dimensions': dimensions,
                                    'size_bytes': len(image_data),
                                    'quality_score': self._assess_image_quality(image_data, dimensions),
                                    'file_path': img_path
                                }
                                images.append(image_info)
                                if self.config.get('verbose'):
                                    print(f"   🖼️ Image: {item.get_name()} ({width}x{height})")

            if self.config.get('verbose'):
                print(f"✅ Extracted {len(text_content)} characters and {len(images)} images")

            return {"text": text_content.strip(), "images": images}

        except Exception as e:
            raise Exception(f"Error processing EPUB {file_path}: {e}")

    def _get_image_dimensions(self, image_data: bytes) -> tuple:
        """Get image dimensions from image data"""
        try:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_data))
            return image.size
        except:
            return None

    def _ocr_image_data(self, item: dict) -> str:
        """Perform OCR on image data.
        If include_vision=True, try the LLM vision API first; otherwise (or on failure) fall back to Tesseract."""
        from doc2train.core.llm_client import call_vision_llm
        image_data = item.get_content()
        # 1) LLM-based OCR
        if self.config.get('include_vision', False):
            try:
                return call_vision_llm(
                    prompt=(
                        "Extract and return only the raw text content from the provided image. "
                        "Do not include any commentary, labels, formatting instructions, or metadata—just the text."
                    ),
                    images=[item],
                    config=self.config
                ).strip()
            except Exception as e:
                if self.config.get('verbose'):
                    print(f"⚠️  Vision OCR failed, falling back to Tesseract: {e}")

        # 2) Fallback: local Tesseract OCR (if configured)
        if not self.config.get('use_ocr', True):
            return ""

        try:
            import pytesseract
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_data))
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            text = pytesseract.image_to_string(img)
            return text.strip()

        except ImportError:
            if self.config.get('verbose'):
                print("⚠️  pytesseract not installed.")
            return ""
        except Exception as e:
            if self.config.get('verbose'):
                print(f"⚠️  Image OCR failed: {e}")
            return ""

    def _assess_image_quality(self, image_data: bytes, dimensions: tuple) -> float:
        """Assess image quality (0.0 to 1.0)"""
        try:
            if not dimensions:
                return 0.3

            width, height = dimensions
            area = width * height

            # Size score
            size_score = min(1.0, area / 100000)  # Normalize to 100K pixels

            # File size score (indicates compression quality)
            bytes_per_pixel = len(image_data) / area if area > 0 else 0
            size_quality = min(1.0, bytes_per_pixel / 3.0)  # Good quality ~3 bytes/pixel

            return (size_score + size_quality) / 2

        except:
            return 0.5

    def _get_processor_specific_info(self, file_path: str) -> Dict:
        """Get EPUB-specific file information"""
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
            for meta in book.get_metadata('DC', 'publisher'):
                metadata['publisher'] = meta[0]

            # Count items
            chapter_count = 0
            image_count = 0
            total_size = 0

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_count += 1
                    total_size += len(item.get_content())
                elif item.get_type() == ebooklib.ITEM_IMAGE:
                    image_count += 1

            info = {
                'metadata': metadata,
                'chapter_count': chapter_count,
                'image_count': image_count,
                'spine_length': len(book.spine),
                'total_content_size': total_size,
                'estimated_pages': total_size // 2000  # Rough estimate: 2000 chars per page
            }

            return info

        except Exception as e:
            return {'error': str(e)}

    def _estimate_processing_time(self, file_path: str) -> float:
        """Estimate processing time for EPUB"""
        try:
            book = epub.read_epub(file_path)

            chapter_count = 0
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_count += 1

            # Base time per chapter
            time_per_chapter = 0.2

            # Add time for image processing if enabled
            if self.config.get('extract_images', True):
                time_per_chapter += 0.1

            return chapter_count * time_per_chapter

        except:
            return 5.0  # Default estimate


# Keep backward compatibility function
def extract_epub_content(file_path: str) -> Tuple[str, List[Dict]]:
    """Backward compatibility function"""
    processor = EPUBProcessor()
    return processor.extract_content_impl(file_path)

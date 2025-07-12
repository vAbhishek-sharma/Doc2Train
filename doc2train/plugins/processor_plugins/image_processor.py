# processors/image_processor.py
"""
Class-based image processor for PNG, JPG, etc.
"""

import os
from PIL import Image
from typing import Tuple, List, Dict
from pathlib import Path

from doc2train.plugins.processor_plugins.base_processor import BaseProcessor

class ImageProcessor(BaseProcessor):
    """Image processor with full BaseProcessor functionality"""
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp']
    priority ='10'
    description = ''
    version = '1.0.0'
    author = 'doc2train'
    processor_name = 'ImageProcessor'
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp']
        self.processor_name = "ImageProcessor"

    def extract_content_impl(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text from image using OCR (either vision-LLM or local Tesseract)

        Args:
            file_path: Path to image file

        Returns:
            Dict with keys "text" (the OCR result) and "images" (a list of info dicts)
        """
        from doc2train.core.llm_client import call_vision_llm
        try:
            # Load image
            image = Image.open(file_path)
            # Filter out very small images
            min_size = self.config.get('min_image_size', 1000)
            area = image.width * image.height
            if area < min_size:
                if self.config.get('verbose'):
                    print(f"⏭️ Skipping small image: {image.width}x{image.height} ({area} px)")
                return {"text": "", "images": []}

            # Read raw bytes once (for vision LLM + quality check)
            with open(file_path, "rb") as f:
                image_data = f.read()

            # 1) Try vision-LLM OCR if requested
            ocr_text = ""
            if self.config.get("include_vision", False):
                ocr_text = call_vision_llm(
                    prompt=(
                        "Extract and return only the raw text content from the provided image. "
                        "Do not include any commentary, labels, formatting instructions, or metadata—just the text."
                    ),
                    images=[file_path],
                    config=self.config
                ).strip()
                if self.config.get("verbose"):
                    print(f"🔮 Vision OCR succeeded: {len(ocr_text)} chars")

            # 2) Fallback to your existing Tesseract OCR
            if not ocr_text and self.config.get("use_ocr", True):
                ocr_text = self._perform_ocr(image)

            # Assess image quality
            quality_score = self._assess_image_quality(image, image_data)
            thresh = self.config.get("quality_threshold", 0.0)
            if quality_score < thresh:
                if self.config.get("verbose"):
                    print(f"⏭️ Skipping low-quality image: {quality_score:.2f} < {thresh}")
                return {"text": "", "images": []}

            # Optionally skip single-color images
            if self.config.get("skip_single_color_images", False) and self._is_single_color(image):
                if self.config.get("verbose"):
                    print("⏭️ Skipping single-color image")
                return {"text": "", "images": []}

            # Gather metadata
            image_info = {
                "path": file_path,
                "file_path": file_path,
                "ocr_text": ocr_text,
                "context": f"Image file: {os.path.basename(file_path)}",
                "dimensions": (image.width, image.height),
                "format": image.format,
                "mode": image.mode,
                "size_bytes": len(image_data),
                "quality_score": quality_score,
                "is_single_color": self._is_single_color(image),
            }

            if self.config.get("verbose"):
                print(f"✅ Processed image: {image.width}x{image.height}, {len(ocr_text)} OCR chars")

            return {"text": ocr_text, "images": [image_info]}

        except Exception as e:
            raise Exception(f"Error processing image {file_path}: {e}")

    def _perform_ocr(self, image: Image.Image) -> str:
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
            if self.config.get('verbose'):
                print("⚠️  pytesseract not available for OCR")
            return ""
        except Exception as e:
            if self.config.get('verbose'):
                print(f"⚠️  OCR failed: {e}")
            return ""

    def _assess_image_quality(self, image: Image.Image, image_data: bytes) -> float:
        """Assess image quality (0.0 to 1.0)"""
        try:
            width, height = image.size
            area = width * height

            # Size score (larger = better, up to a point)
            size_score = min(1.0, area / 100000)  # Normalize to 100K pixels

            # File size score (indicates compression quality)
            bytes_per_pixel = len(image_data) / area if area > 0 else 0

            # Good quality images have reasonable bytes per pixel
            if image.mode == 'RGB':
                ideal_bpp = 3.0  # 3 bytes per pixel for RGB
            elif image.mode == 'RGBA':
                ideal_bpp = 4.0  # 4 bytes per pixel for RGBA
            else:
                ideal_bpp = 1.0  # 1 byte per pixel for grayscale

            compression_score = min(1.0, bytes_per_pixel / ideal_bpp)

            # Color diversity score
            try:
                colors = image.getcolors(maxcolors=256*256*256)
                if colors:
                    unique_colors = len(colors)
                    diversity_score = min(1.0, unique_colors / 1000)  # Normalize to 1000 colors
                else:
                    diversity_score = 1.0  # Too many colors to count = high diversity
            except:
                diversity_score = 0.5  # Default

            # Weighted average
            quality = (size_score * 0.4 + compression_score * 0.3 + diversity_score * 0.3)
            return quality

        except:
            return 0.5  # Default moderate quality

    def _is_single_color(self, image: Image.Image) -> bool:
        """Check if image is essentially a single color"""
        try:
            # Convert to RGB for consistent analysis
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get colors (limit to avoid memory issues)
            colors = image.getcolors(maxcolors=100)

            if colors is None:
                # Too many colors to count
                return False

            if len(colors) <= 3:
                # Very few colors - likely single color or simple pattern
                return True

            # Check if one color dominates (>95% of pixels)
            total_pixels = image.width * image.height
            for count, color in colors:
                if count / total_pixels > 0.95:
                    return True

            return False

        except:
            return False  # If we can't determine, assume it's not single color

    def _get_processor_specific_info(self, file_path: str) -> Dict:
        """Get image-specific file information"""
        try:
            image = Image.open(file_path)

            info = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'area_pixels': image.width * image.height,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
                'estimated_colors': self._estimate_color_count(image)
            }

            # Add EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                try:
                    exif = image._getexif()
                    info['has_exif'] = True
                    info['exif_keys'] = list(exif.keys()) if exif else []
                except:
                    info['has_exif'] = False

            return info

        except Exception as e:
            return {'error': str(e)}

    def _estimate_color_count(self, image: Image.Image) -> str:
        """Estimate the number of unique colors in the image"""
        try:
            # Sample the image for performance
            if image.width * image.height > 100000:  # If larger than 100K pixels
                # Resize for sampling
                sample_image = image.copy()
                sample_image.thumbnail((300, 300))
            else:
                sample_image = image

            colors = sample_image.getcolors(maxcolors=256*256*256)

            if colors is None:
                return "many (>16M)"
            elif len(colors) > 10000:
                return f"many (~{len(colors)//1000}K)"
            else:
                return str(len(colors))

        except:
            return "unknown"

    def _estimate_processing_time(self, file_path: str) -> float:
        """Estimate processing time for image"""
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

            # Base time for image loading
            base_time = file_size_mb * 0.1

            # Add OCR time if enabled
            if self.config.get('use_ocr', True):
                base_time += file_size_mb * 2.0  # OCR is much slower

            return base_time

        except:
            return 2.0  # Default estimate

#To Remove if not used
# Keep backward compatibility function
def extract_image_content(file_path: str) -> Tuple[str, List[Dict]]:
    """Backward compatibility function"""
    processor = ImageProcessor()
    return processor.extract_content_impl(file_path)

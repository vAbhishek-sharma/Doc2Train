# processors/pdf_utils/extraction.py
"""
PDF content extraction strategies for different document types
"""


import fitz
import io
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from typing import Dict, List, Tuple
from doc2train.utils.pdf_utils.analyzer import SmartPDFAnalyzer, PDFAnalysis
from doc2train.utils.common import save_image_data

def get_ocr_config(config: Dict, quality_level: str = "medium") -> str:
    """Get OCR config for pytesseract based on quality"""
    quality_configs = {
        "light": "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
        "medium": "--psm 6",
        "heavy": "--psm 6 --oem 3 -c tessedit_do_invert=0"
    }
    return config.get('ocr_config') or quality_configs.get(quality_level, quality_configs["medium"])

def perform_ocr_on_image_data(img_data: bytes, config: Optional[Dict] = None) -> str:
    config = config or {}
    if not config.get('use_ocr', True):
        return ""
    try:
        import pytesseract
        pil_image = Image.open(io.BytesIO(img_data))
        if pil_image.mode not in ('RGB', 'L'):
            pil_image = pil_image.convert('RGB')
        ocr_quality = config.get('ocr_quality', 'medium')
        ocr_config = get_ocr_config(config, ocr_quality)
        ocr_text = pytesseract.image_to_string(pil_image, config=ocr_config)
        return ocr_text.strip()
    except ImportError:
        if config.get('verbose'):
            print("⚠️  pytesseract not available for OCR")
        return ""
    except Exception as e:
        if config.get('verbose'):
            print(f"⚠️  OCR failed: {e}")
        return ""

def perform_ocr_on_extracted_images(images: List[Dict], config: Optional[Dict] = None) -> List[Dict]:
    config = config or {}
    if not config.get('ocr_images', True):
        return images
    updated_images = []
    for img in images:
        updated_img = img.copy()
        img_data = img.get('data')
        if not img_data and img.get('file_path'):
            try:
                with open(img['file_path'], 'rb') as f:
                    img_data = f.read()
            except:
                img_data = None
        updated_img['ocr_text'] = perform_ocr_on_image_data(img_data, config) if img_data else ""
        updated_images.append(updated_img)
    return updated_images

def combine_text_and_image_content(page_text: str, images: List[Dict], config: Optional[Dict] = None) -> str:
    config = config or {}
    if not config.get('include_image_text_in_content', False):
        return page_text
    image_texts = [img.get('ocr_text', '') for img in images if img.get('ocr_text')]
    if image_texts:
        return page_text + "\n\n" + "\n\n".join(image_texts)
    return page_text




def analyze_and_extract_pdf(file_path: str, config=None) -> Tuple[str, List[Dict], PDFAnalysis]:
    """
    Analyze PDF and extract content using optimal strategy

    Args:
        file_path: Path to PDF file
        config: Optional configuration dict

    Returns:
        Tuple of (text_content, images_list, analysis_result)
    """
    config = config or {}
    verbose = config.get('verbose', False)

    # First, analyze the PDF
    analyzer = SmartPDFAnalyzer(config=config)
    analysis = analyzer.analyze_pdf(file_path)

    if verbose:
        print(f"📊 PDF Analysis: {analysis.content_type.value} "
              f"({analysis.confidence:.2f} confidence)")
        print(f"📄 {analysis.text_pages} text pages, "
              f"🖼️  {analysis.image_pages} image pages, "
              f"📄🖼️  {analysis.mixed_pages} mixed pages")

    # Extract content based on strategy
    if analysis.processing_strategy == "skip":
        return "", [], analysis

    elif analysis.processing_strategy == "text_only":
        text, images = extract_text_focused(file_path, config)

    elif analysis.processing_strategy == "image_heavy":
        text, images = extract_image_focused(file_path, config)

    elif analysis.processing_strategy == "ocr_primary":
        text, images = extract_with_heavy_ocr(file_path, config)

    elif analysis.processing_strategy in ["mixed_processing", "text_primary_mixed", "conservative_mixed"]:
        text, images = extract_mixed_content(file_path, analysis, config)

    else:
        # Fallback to basic extraction
        text, images = extract_basic_content(file_path, config)

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
                            if img_rect is None:
                                # Fallback to image rects if bbox returns None
                                img_rects = page.get_image_rects(xref)
                                img_rect = img_rects[0] if img_rects else fitz.Rect(0, 0, page.rect.width, page.rect.height)
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


# Helper functions for context extraction
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


def extract_enhanced_mixed_content(file_path: str, analysis: PDFAnalysis, config=None) -> Tuple[str, List[Dict]]:
    """
    Enhanced mixed content extraction with better text-image correlation
    """
    config = config or {}
    verbose = config.get('verbose', False)

    doc = fitz.open(file_path)
    text_content = ""
    images = []

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            page_text = page.get_text()

            # Get all text blocks with positioning
            text_blocks = page.get_text("dict")
            page_images = page.get_images(full=True)

            # Process images with spatial context
            for img_index, img in enumerate(page_images):
                try:
                    if len(img) < 7:
                        continue

                    xref = img[0]

                    # Get image position
                    try:
                        img_rect = page.get_image_bbox(img)
                    except:
                        continue

                    # Extract image
                    pix = fitz.Pixmap(doc, xref)

                    if pix.width * pix.height < 10000:  # Skip small images
                        del pix
                        continue

                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")

                        # Find nearby text blocks
                        nearby_text = find_nearby_text_blocks(text_blocks, img_rect)

                        enhanced_context = {
                            'text_before': extract_text_before_image(page, img_rect),
                            'text_after': extract_text_after_image(page, img_rect),
                            'nearby_text_blocks': nearby_text,
                            'full_page_text': page_text,
                            'page_type': 'enhanced_mixed_content',
                            'position_in_page': get_image_position(img_rect, page.rect)
                        }

                        images.append({
                            'page_num': page_num + 1,
                            'image_index': img_index,
                            'data': img_data,
                            'format': 'png',
                            'context': enhanced_context,
                            'bbox': [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1],
                            'content_type': 'enhanced_mixed',
                            'dimensions': (pix.width, pix.height)
                        })

                        if verbose:
                            print(f"✅ Enhanced extraction - image {img_index + 1} on page {page_num + 1}")

                    del pix

                except Exception as e:
                    if verbose:
                        print(f"⚠️ Error in enhanced extraction: {e}")
                    continue

            # Add processed text
            if page_text.strip():
                processed_text = process_text_with_image_markers(page_text, page_images, page)
                text_content += processed_text + "\n\n"

        except Exception as e:
            if verbose:
                print(f"❌ Error processing page {page_num + 1}: {e}")
            continue

    doc.close()
    return text_content.strip(), images


def find_nearby_text_blocks(text_blocks: Dict, img_rect) -> List[str]:
    """
    Find text blocks that are spatially near an image
    """
    nearby_text = []

    try:
        for block in text_blocks.get("blocks", []):
            if "lines" in block:
                block_rect = fitz.Rect(block["bbox"])

                # Check if text block is near the image
                if rectangles_are_nearby(block_rect, img_rect, threshold=50):
                    block_text = ""
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")

                    if block_text.strip():
                        nearby_text.append(block_text.strip())
    except:
        pass

    return nearby_text


def rectangles_are_nearby(rect1, rect2, threshold=50) -> bool:
    """Check if two rectangles are within threshold distance"""
    try:
        # Calculate minimum distance between rectangles
        dx = max(0, max(rect1.x0 - rect2.x1, rect2.x0 - rect1.x1))
        dy = max(0, max(rect1.y0 - rect2.y1, rect2.y0 - rect1.y1))
        distance = (dx**2 + dy**2)**0.5
        return distance <= threshold
    except:
        return False


def process_text_with_image_markers(text: str, images: List, page) -> str:
    """
    Process text and insert contextual image markers
    """
    if not images:
        return text

    # For now, simple approach - could be enhanced with NLP
    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        processed_lines.append(line)

        # Insert image markers for significant content breaks
        if len(line.strip()) > 50 and line.strip().endswith('.'):
            # This is a potential place where an image might be referenced
            pass

    # Add summary at the end
    if images:
        image_summary = f"\n[This page contains {len(images)} image(s) with contextual relationships]\n"
        return '\n'.join(processed_lines) + image_summary

    return '\n'.join(processed_lines)

#To UPDATE FUTURE SUPPORT
def extract_table_focused_content(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Extract content optimized for tables and structured data, plus optional image OCR.
    """
    config = config or {}
    verbose = config.get('verbose', False)

    doc = fitz.open(file_path)
    text_content = ""
    images = []

    if verbose:
        print("🔍 Using table-focused extraction")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Try to find tables
        try:
            tables = page.find_tables()
            if tables:
                if verbose:
                    print(f"   Found {len(tables)} table(s) on page {page_num + 1}")

                for table in tables:
                    try:
                        table_data = table.extract()
                        if table_data:
                            table_text = format_table_as_text(table_data)
                            text_content += f"\n[TABLE on page {page_num + 1}]\n{table_text}\n[/TABLE]\n\n"
                    except:
                        bbox = table.bbox
                        table_rect = fitz.Rect(bbox)
                        table_text = page.get_text(clip=table_rect)
                        text_content += f"\n[TABLE on page {page_num + 1}]\n{table_text}\n[/TABLE]\n\n"
        except:
            pass

        # Regular text extraction for non-table content
        regular_text = page.get_text()
        text_content += regular_text + "\n\n"

        # Extract images
        page_images = []
        if config.get('extract_images', True):
            page_images = extract_page_images_safe(page, page_num + 1, doc, file_path, config.get('save_images', False))
            if config.get('ocr_images', False):
                page_images = perform_ocr_on_extracted_images(page_images, config)
            images.extend(page_images)
            if config.get('include_image_text_in_content', False):
                regular_text = combine_text_and_image_content(regular_text, page_images, config)

    doc.close()
    return text_content.strip(), images


def format_table_as_text(table_data: List[List]) -> str:
    """Format table data as readable text"""
    if not table_data:
        return ""

    formatted_rows = []
    for row in table_data:
        # Clean and join cells
        clean_cells = [str(cell).strip() if cell else "" for cell in row]
        formatted_row = " | ".join(clean_cells)
        formatted_rows.append(formatted_row)

    return "\n".join(formatted_rows)


def extract_form_focused_content(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Extract content optimized for forms and fillable fields
    """
    config = config or {}
    verbose = config.get('verbose', False)

    doc = fitz.open(file_path)
    text_content = ""
    images = []

    if verbose:
        print("📝 Using form-focused extraction")

    # Check if document has form fields
    has_forms = False
    try:
        if hasattr(doc, 'form_fields') and doc.form_fields():
            has_forms = True
            if verbose:
                print(f"   Document has {len(doc.form_fields())} form fields")
    except:
        pass

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract regular text
        page_text = page.get_text()

        # If forms detected, try to extract field information
        if has_forms:
            try:
                widgets = page.widgets()
                if widgets:
                    form_text = "\n[FORM FIELDS]\n"
                    for widget in widgets:
                        field_name = getattr(widget, 'field_name', 'Unknown')
                        field_value = getattr(widget, 'field_value', '')
                        field_type = getattr(widget, 'field_type', 'Unknown')

                        form_text += f"Field: {field_name} (Type: {field_type})"
                        if field_value:
                            form_text += f" = {field_value}"
                        form_text += "\n"

                    form_text += "[/FORM FIELDS]\n\n"
                    page_text = form_text + page_text
            except:
                pass

        text_content += page_text + "\n\n"

        # Extract images
        if config.get('extract_images', True):
            page_images = extract_page_images_safe(page, page_num + 1, doc, file_path)
            images.extend(page_images)

    doc.close()
    return text_content.strip(), images

def extract_text_focused(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Extract from text-focused PDFs: primary text, light image OCR fallback
    """
    config = config or {}
    verbose = config.get('verbose', False)
    try:
        doc = fitz.open(file_path)
        text_content = ""
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            # OCR fallback if little/no text
            if not page_text.strip() and config.get('use_ocr', True):
                if verbose:
                    print(f"   📝 OCR fallback for page {page_num + 1}")
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                img_data = pix.tobytes("png")
                page_text = perform_ocr_on_image_data(img_data, config)
            # Extract images
            page_images = []
            if config.get('extract_images', True):
                page_images = extract_page_images_safe(page, page_num + 1, doc, file_path, config.get('save_images', False))
                if config.get('ocr_images', False):
                    page_images = perform_ocr_on_extracted_images(page_images, config)
                images.extend(page_images)
                page_text = combine_text_and_image_content(page_text, page_images, config)
            text_content += page_text + "\n\n"
        doc.close()
        return text_content.strip(), images
    except Exception as e:
        if config.get('debug', False):
            import traceback
            print(f"🐛 DEBUG - Error in extract_text_focused:")
            traceback.print_exc()
        return "", []

def extract_image_focused(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Extract from image-heavy PDFs: aggressive OCR on all images
    """
    config = config or {}
    verbose = config.get('verbose', False)
    doc = fitz.open(file_path)
    text_content = ""
    images = []
    config = config.copy()
    config['ocr_images'] = True
    config['ocr_quality'] = config.get('ocr_quality', 'medium')
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if not page_text.strip() and config.get('use_ocr', True):
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img_data = pix.tobytes("png")
            page_text = perform_ocr_on_image_data(img_data, config)
        page_images = extract_page_images_safe(page, page_num + 1, doc, file_path, config.get('save_images', False))
        page_images = perform_ocr_on_extracted_images(page_images, config)
        images.extend(page_images)
        page_text = combine_text_and_image_content(page_text, page_images, config)
        text_content += page_text + "\n\n"
    doc.close()
    return text_content.strip(), images


def extract_with_heavy_ocr(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Extract from scanned documents with maximum OCR power
    """
    config = config or {}
    verbose = config.get('verbose', False)
    doc = fitz.open(file_path)
    text_content = ""
    images = []
    heavy_config = config.copy()
    heavy_config['ocr_quality'] = 'heavy'
    heavy_config['ocr_images'] = True
    heavy_config['use_ocr'] = True
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Always do page OCR (max quality)
        if verbose:
            print(f"   📄 Heavy OCR on page {page_num + 1}")
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_data = pix.tobytes("png")
        page_ocr_text = perform_ocr_on_image_data(img_data, heavy_config)
        page_images = extract_page_images_safe(page, page_num + 1, doc, file_path, config.get('save_images', False))
        page_images = perform_ocr_on_extracted_images(page_images, heavy_config)
        images.extend(page_images)
        page_text = combine_text_and_image_content(page_ocr_text, page_images, heavy_config)
        text_content += page_text + "\n\n"
    doc.close()
    return text_content.strip(), images


def extract_basic_content(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Basic extraction with optional image OCR (reliable fallback)
    """
    config = config or {}
    verbose = config.get('verbose', False)
    doc = fitz.open(file_path)
    text_content = ""
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if not page_text.strip() and config.get('use_ocr', True):
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img_data = pix.tobytes("png")
            page_text = perform_ocr_on_image_data(img_data, config)
        page_images = []
        if config.get('extract_images', True):
            page_images = extract_page_images_safe(page, page_num + 1, doc, file_path, config.get('save_images', False))
            if config.get('ocr_images', False):
                page_images = perform_ocr_on_extracted_images(page_images, config)
            page_text = combine_text_and_image_content(page_text, page_images, config)
            images.extend(page_images)
        text_content += page_text + "\n\n"
    doc.close()
    return text_content.strip(), images

def extract_page_images_safe(page, page_num: int, doc, file_path: str, save_images: bool) -> List[Dict]:
    images = []
    try:
        page_images = page.get_images(full=True)
        for img_index, img in enumerate(page_images):
            try:
                if len(img) < 7:
                    continue
                xref = img[0]
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
                    base_name = f"{Path(file_path).stem}_page{page_num}_img{img_index+1}"
                    img_path = None
                    if save_images:
                        img_path = save_image_data(img_data, "output/images", base_name)
                    images.append({
                        'page_num': page_num,
                        'image_index': img_index,
                        'file_path': img_path,
                        'data': img_data,
                        'format': 'png',
                        'dimensions': (pix.width, pix.height),
                        'bbox': bbox,
                        'xref': xref
                    })
                del pix
            except Exception:
                continue
    except Exception:
        pass
    return images

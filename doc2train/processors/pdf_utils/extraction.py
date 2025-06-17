# processors/pdf_utils/extraction.py
"""
PDF content extraction strategies for different document types
"""

import fitz  # PyMuPDF
from typing import Dict, List, Tuple
from .analyzer import SmartPDFAnalyzer, PDFAnalysis


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


def extract_table_focused_content(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """
    Extract content optimized for tables and structured data
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
                        # Extract table as structured data
                        table_data = table.extract()
                        if table_data:
                            # Convert table to formatted text
                            table_text = format_table_as_text(table_data)
                            text_content += f"\n[TABLE on page {page_num + 1}]\n{table_text}\n[/TABLE]\n\n"
                    except:
                        # Fallback to regular text extraction for this area
                        bbox = table.bbox
                        table_rect = fitz.Rect(bbox)
                        table_text = page.get_text(clip=table_rect)
                        text_content += f"\n[TABLE on page {page_num + 1}]\n{table_text}\n[/TABLE]\n\n"
        except:
            # Tables not supported or error occurred
            pass

        # Regular text extraction for non-table content
        regular_text = page.get_text()
        text_content += regular_text + "\n\n"

        # Extract images
        if config.get('extract_images', True):
            page_images = extract_page_images_safe(page, page_num + 1, doc)
            images.extend(page_images)

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
            page_images = extract_page_images_safe(page, page_num + 1, doc)
            images.extend(page_images)

    doc.close()
    return text_content.strip(), images

def extract_text_focused(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """Extract from text-focused PDFs"""
    try:
        return extract_basic_content(file_path, config)
    except Exception as e:
        if config and config.get('debug', False):
            import traceback
            print(f"🐛 DEBUG - Error in extract_text_focused:")
            traceback.print_exc()
        # Return empty result instead of None
        return "", []

def extract_image_focused(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """Extract from image-heavy PDFs with minimal text"""
    config = config or {}
    verbose = config.get('verbose', False)

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
                        'content_type': 'image_focused',
                        'dimensions': (pix.width, pix.height)
                    })

                    if verbose:
                        print(f"✅ Extracted image {img_index + 1} on page {page_num + 1}")

                pix = None
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error extracting image {img_index} on page {page_num + 1}: {e}")
                continue

    doc.close()
    return text_content.strip(), images


def extract_with_heavy_ocr(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """Extract from scanned documents using heavy OCR"""
    config = config or {}
    verbose = config.get('verbose', False)

    if verbose:
        print("🔍 Using heavy OCR processing for scanned document")

    # For now, use image-focused approach with enhanced OCR
    return extract_image_focused(file_path, config)


def extract_basic_content(file_path: str, config=None) -> Tuple[str, List[Dict]]:
    """Basic extraction without smart analysis"""
    config = config or {}
    verbose = config.get('verbose', False)

    from .common import perform_ocr_on_page

    doc = fitz.open(file_path)
    text_content = ""
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text
        page_text = page.get_text()

        # If no text and OCR is enabled, try OCR
        if not page_text.strip() and config.get('use_ocr', True):
            page_text = perform_ocr_on_page(page, config)

        text_content += page_text + "\n\n"

        # Extract images if enabled
        if config.get('extract_images', True):
            page_images = extract_page_images_safe(page, page_num + 1, doc)
            images.extend(page_images)

    doc.close()
    return text_content.strip(), images

def extract_page_images_safe(page, page_num: int, doc) -> List[Dict]:
    """
    Safely extract images from a PDF page with error handling

    Args:
        page: PyMuPDF page object
        page_num: Page number (1-based)
        doc: PyMuPDF document object

    Returns:
        List of image dictionaries
    """
    images = []

    try:
        page_images = page.get_images(full=True)

        for img_index, img in enumerate(page_images):
            try:
                # Skip if img doesn't have enough data
                if len(img) < 7:
                    continue

                xref = img[0]

                # Extract image data safely
                try:
                    pix = fitz.Pixmap(doc, xref)

                    # Skip very small images (likely icons/decorations)
                    if pix.width * pix.height < 1000:
                        del pix
                        continue

                    # Convert to PNG format
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")

                        # Get image position if possible
                        try:
                            img_rect = page.get_image_bbox(img)
                            bbox = [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1]
                        except:
                            bbox = None

                        images.append({
                            'page_num': page_num,
                            'image_index': img_index,
                            'data': img_data,
                            'format': 'png',
                            'dimensions': (pix.width, pix.height),
                            'bbox': bbox,
                            'xref': xref
                        })

                    del pix

                except Exception as e:
                    # Skip problematic images
                    continue

            except Exception as e:
                # Skip this image and continue
                continue

    except Exception as e:
        # Return empty list if page processing fails
        pass

    return images


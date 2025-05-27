#file_processor.py
from datetime import datetime
import json
import os
import pandas as pd
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import pdb
import cv2
import re
import pytesseract
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from data_structurer import process_content
import webvtt
import glob
from pdf2image import convert_from_path


def read_pdf(file_path, start_page=1, end_page=None, use_ocr=False):
    """Extract text from a PDF file, using OCR if required, and save it for future use."""
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    cache_dir = "extracted_text_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{base_filename}.txt")

    if os.path.exists(cache_file):
        print(f"📌 Loading extracted text from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # Read PDF
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError("⚠️ No pages found in the PDF.")
    except Exception as e:
        print(f"🚨 Error reading {file_path}: {e}")
        return ""

    end_page = min(end_page or total_pages, total_pages)
    text = []

    for i in range(start_page - 1, end_page):
        try:
            page = reader.pages[i]
            extracted_text = page.extract_text()

            if extracted_text and not extracted_text.isspace():
                print(f"✅ Extracted text from page {i+1} (without OCR).")
                text.append(extracted_text)
            elif use_ocr:
                print(f"⚡ Applying OCR on page {i+1} of {file_path}...")

                images = convert_from_path(file_path, first_page=i + 1, last_page=i + 1)
                for img in images:
                    gray = cv2.cvtColor(cv2.cvtColor(img.convert("RGB"), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                    ocr_text = pytesseract.image_to_string(gray)

                    if ocr_text.strip():
                        print(f"✅ OCR extracted text from page {i+1}.")
                    else:
                        print(f"⚠️ OCR did not extract any text from page {i+1}.")

                    text.append(ocr_text)
        except Exception as e:
            print(f"🚨 Error processing page {i+1}: {e}")

    final_text = "\n\n".join(text).strip()

    if final_text:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"✅ Extracted text saved to cache: {cache_file}")

    return final_text if final_text else f"⚠️ Failed to extract text from {file_path}."

def read_epub(file_path):
    """Extract text from an EPUB file"""
    book = epub.read_epub(file_path)
    text = ""

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text() + "\n\n"

    return text.strip()

def read_text(file_path):
    """Read content from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_srt(file_path):
    """Extract text from an SRT subtitle file, removing timestamps"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    text = []
    for line in lines:
        if not re.match(r'^[0-9]+$', line.strip()) and not re.match(r'^[0-9:,--> ]+$', line.strip()):
            text.append(line.strip())

    return "\n".join(text)

def read_vtt(file_path):
    """Extract text from a VTT subtitle file, removing timestamps"""
    text = ""
    for caption in webvtt.read(file_path):
        text += caption.text + "\n"

    return text.strip()

def process_file(input_file, provider, model, start_page=1, end_page=None, continue_process=False, use_ocr=False):
    split_methods = os.getenv('SPLITTING_METHOD', 'smart_splitting').split(',')

    if continue_process:
        continue_stalled_process(input_file, provider, model, start_page=1, end_page=None)
    else:
        for index,method in enumerate(split_methods):
            process_new_file_or_folder(input_file, provider, model, start_page=1, end_page=None, split_type=method, use_ocr=use_ocr)

def process_new_file_or_folder(input_file, provider, model, start_page=1, end_page=None, split_type='smart_splitting', use_ocr=False):
    """Process PDF/EPUB/TXT/SRT/VTT and apply OCR if necessary"""
    if input_file.lower().endswith('.pdf'):
        content = read_pdf(input_file, start_page, end_page, use_ocr=use_ocr)
    elif input_file.lower().endswith('.epub'):
        content = read_epub(input_file)
    elif input_file.lower().endswith('.srt'):
        content = read_srt(input_file)
    elif input_file.lower().endswith('.vtt'):
        content = read_vtt(input_file)
    else:
        content = read_text(input_file)

    # Split content into sections based on method
    if split_type == 'smart_splitting':
        sections = split_text_smart(content)
    elif split_type == 'chunk_splitting':
        sections = split_text_with_overlap(content)
    else:
        sections = split_text_section_based(content)

    print(f"\nProcessing {len(sections)} sections...")

    # Create timestamp and output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_filename}_{split_type}_structured_{timestamp}_.csv"

    # Initialize DataFrame with all sections
    section_data = [
        {
            'index': i + 1,
            'section_data': section,
            'json_data': None,
            'text_data': None
        }
        for i, section in enumerate(sections)
    ]
    df = pd.DataFrame(section_data)

    # Save initial DataFrame with sections
    df.to_csv(output_file, index=False)

    test_mode = os.getenv('TEST_MODE', 'False').lower() in ('true', '1', 'yes')
    structure_type = os.getenv('STRUCTURE_TYPE', 'json')

    # Process each section and update the corresponding row
    for i, section in enumerate(sections, 1):
        print(f"Processing section {i}/{len(sections)}")

        # For testing, use dummy content
        structured_content = process_content(section, provider, model, input_file, structure_type)

        # Read existing DataFrame
        df = pd.read_csv(output_file)

        # Update only the processed content columns
        if structured_content and structure_type == 'text':
            df.at[i-1, 'text_data'] = structured_content
        else:
            try:
                if isinstance(structured_content, list) and len(structured_content) > 0:
                    structured_content = structured_content[0]
                json_data = json.loads(structured_content)
                df.at[i-1, 'json_data'] = str(json_data)
            except json.JSONDecodeError:
                print("Warning: Response is not valid JSON. Saving as text.")
                df.at[i-1, 'text_data'] = structured_content

        # Save updated DataFrame
        df.to_csv(output_file, index=False)

        if test_mode:
            print("✅ TEST_MODE enabled: Stopping after first loop iteration.")
            break


def continue_stalled_process(input_file, provider, model, start_page=1, end_page=None):
    # Check if the file is a CSV
    if not input_file.lower().endswith('.csv'):
        print(f"Error: {input_file} is not a CSV file. Skipping.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Validate column names
    expected_columns = ['section_data', 'section_data', 'json_data', 'text_data']
    if not all(col in df.columns for col in expected_columns):
        print("Error: CSV file does not have the expected columns.")
        return

    # Process sections that have not been processed
    structure_type = os.getenv('STRUCTURE_TYPE', 'json')
    test_mode = os.getenv('TEST_MODE', 'False').lower() in ('true', '1', 'yes')

    for i, row in df.iterrows():
        # Check if both json_data and text_data are empty
        if pd.isna(row['json_data']) and pd.isna(row['text_data']):
            print(f"Processing unprocessed section {i + 1}")

            # Process the section
            structured_content = process_content(row['section_data'], provider, model, input_file, structure_type)

            # Update the row with processed content
            if structured_content and structure_type == 'text':
                df.at[i, 'text_data'] = structured_content
            else:
                try:
                    if isinstance(structured_content, list) and len(structured_content) > 0:
                        structured_content = structured_content[0]
                    json_data = json.loads(structured_content)
                    df.at[i, 'json_data'] = str(json_data)
                except json.JSONDecodeError:
                    print(f"Warning: Response for section {i + 1} is not valid JSON. Saving as text.")
                    df.at[i, 'text_data'] = structured_content

            # Save updated DataFrame
            df.to_csv(input_file, index=False)

            if test_mode:
                print("✅ TEST_MODE enabled: Stopping after first unprocessed section.")
                break


def split_text_with_overlap(text, chunk_size=4000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlapping for context retention
    return chunks

def split_text_smart(text, max_len=4000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_len:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def split_text_section_based(content):
    [s.strip() for s in content.split("\n\n") if s.strip()]

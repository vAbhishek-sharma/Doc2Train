# processors/text_processor.py
"""
Class-based text processor for TXT, SRT, VTT files
"""

import re
import webvtt
from typing import Tuple, List, Dict
from pathlib import Path

from .base_processor import BaseProcessor

class TextProcessor(BaseProcessor):
    """Text processor with full BaseProcessor functionality"""

    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.txt', '.srt', '.vtt']
        self.processor_name = "TextProcessor"

    def extract_content_impl(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text content from text-based files

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (text_content, empty_images_list)
        """
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.srt':
                text = self._extract_srt(file_path)
            elif file_ext == '.vtt':
                text = self._extract_vtt(file_path)
            else:  # .txt and other text files
                text = self._extract_txt(file_path)

            if self.config.get('verbose'):
                print(f"✅ Extracted {len(text)} characters from {Path(file_path).name}")

            return text, []

        except Exception as e:
            raise Exception(f"Error processing text file {file_path}: {e}")

    def _extract_txt(self, file_path: str) -> str:
        """Extract text from plain text file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, read as binary and decode with errors='ignore'
        try:
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')
        except Exception as e:
            raise Exception(f"Cannot read file with any encoding: {e}")

    def _extract_srt(self, file_path: str) -> str:
        """Extract text from SRT subtitle file"""
        content = self._extract_txt(file_path)

        # Remove subtitle numbers and timestamps
        lines = content.split('\n')
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines, numbers, and timestamp lines
            if (line and
                not re.match(r'^\d+$', line) and
                not re.match(r'^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$', line)):
                text_lines.append(line)

        return ' '.join(text_lines)

    def _extract_vtt(self, file_path: str) -> str:
        """Extract text from VTT subtitle file"""
        try:
            vtt = webvtt.read(file_path)
            text_parts = []

            for caption in vtt:
                # Clean up the text (remove formatting tags)
                clean_text = re.sub(r'<[^>]+>', '', caption.text)
                text_parts.append(clean_text)

            return ' '.join(text_parts)

        except Exception as e:
            if self.config.get('verbose'):
                print(f"⚠️  VTT parsing failed, falling back to plain text: {e}")
            # Fallback to simple text extraction
            return self._extract_txt(file_path)

    def _get_processor_specific_info(self, file_path: str) -> Dict:
        """Get text-specific file information"""
        try:
            path = Path(file_path)
            ext = path.suffix.lower()

            info = {
                'file_type': ext,
                'encoding': self._detect_encoding(file_path)
            }

            # Get line count and basic stats
            try:
                text = self._extract_txt(file_path)
                lines = text.split('\n')
                words = text.split()

                info.update({
                    'line_count': len(lines),
                    'word_count': len(words),
                    'char_count': len(text),
                    'estimated_reading_time_minutes': len(words) / 200  # ~200 words per minute
                })

                # Subtitle-specific info
                if ext in ['.srt', '.vtt']:
                    info['is_subtitle_file'] = True
                    if ext == '.srt':
                        info['subtitle_entries'] = self._count_srt_entries(text)
                    elif ext == '.vtt':
                        info['subtitle_entries'] = self._count_vtt_entries(file_path)

            except Exception as e:
                info['text_analysis_error'] = str(e)

            return info

        except Exception as e:
            return {'error': str(e)}

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Try to read some content
                return encoding
            except UnicodeDecodeError:
                continue

        return 'unknown'

    def _count_srt_entries(self, content: str) -> int:
        """Count subtitle entries in SRT content"""
        try:
            # Count timestamp lines (indicates subtitle entries)
            timestamp_pattern = r'^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$'
            lines = content.split('\n')
            return len([line for line in lines if re.match(timestamp_pattern, line.strip())])
        except:
            return 0

    def _count_vtt_entries(self, file_path: str) -> int:
        """Count subtitle entries in VTT file"""
        try:
            vtt = webvtt.read(file_path)
            return len(vtt)
        except:
            return 0

    def _estimate_processing_time(self, file_path: str) -> float:
        """Estimate processing time for text files"""
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            # Text files are very fast to process
            return file_size_mb * 0.1  # 0.1 seconds per MB
        except:
            return 1.0  # Default estimate


# Keep backward compatibility function
def extract_text_content(file_path: str) -> Tuple[str, List[Dict]]:
    """Backward compatibility function"""
    processor = TextProcessor()
    return processor.extract_content_impl(file_path)

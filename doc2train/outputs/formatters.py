# outputs/formatters.py
"""
Complete output formatters for different data types and formats
Handles format-specific output generation with templates and validation
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from abc import ABC, abstractmethod

from outputs.base_formatters import BaseFormatter

class JSONLFormatter(BaseFormatter):
    """JSONL (JSON Lines) formatter - one JSON object per line"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_name = "jsonl"

    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations as JSONL"""
        lines = []
        for conv in conversations:
            lines.append(json.dumps(conv, ensure_ascii=False))
        return '\n'.join(lines)

    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs as JSONL"""
        lines = []
        for qa in qa_pairs:
            lines.append(json.dumps(qa, ensure_ascii=False))
        return '\n'.join(lines)

    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embeddings as JSONL"""
        lines = []
        for emb in embeddings:
            lines.append(json.dumps(emb, ensure_ascii=False))
        return '\n'.join(lines)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries as JSONL"""
        lines = []
        for summary in summaries:
            lines.append(json.dumps(summary, ensure_ascii=False))
        return '\n'.join(lines)

    def get_file_extension(self) -> str:
        return ".jsonl"

class JSONFormatter(BaseFormatter):
    """Standard JSON formatter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_name = "json"

    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations as JSON"""
        return json.dumps(conversations, indent=2, ensure_ascii=False)

    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs as JSON"""
        return json.dumps(qa_pairs, indent=2, ensure_ascii=False)

    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embeddings as JSON"""
        return json.dumps(embeddings, indent=2, ensure_ascii=False)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries as JSON"""
        return json.dumps(summaries, indent=2, ensure_ascii=False)

    def get_file_extension(self) -> str:
        return ".json"

class CSVFormatter(BaseFormatter):
    """CSV formatter with flattened data structure"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_name = "csv"

    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations as CSV"""
        if not conversations:
            return ""

        # Flatten conversation structure for CSV
        flattened = []
        for conv in conversations:
            if 'messages' in conv:
                for i, message in enumerate(conv['messages']):
                    flat_row = {
                        'conversation_id': conv.get('id', ''),
                        'source_file': conv.get('source_file', ''),
                        'message_index': i,
                        'role': message.get('role', ''),
                        'content': message.get('content', ''),
                        'conversation_length': len(conv['messages'])
                    }
                    flattened.append(flat_row)

        return self._dict_list_to_csv(flattened)

    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs as CSV"""
        if not qa_pairs:
            return ""

        # Flatten Q&A structure
        flattened = []
        for qa in qa_pairs:
            flat_row = {
                'question': qa.get('question', ''),
                'answer': qa.get('answer', ''),
                'source_file': qa.get('source_file', ''),
                'question_length': len(qa.get('question', '')),
                'answer_length': len(qa.get('answer', ''))
            }
            flattened.append(flat_row)

        return self._dict_list_to_csv(flattened)

    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embeddings as CSV"""
        if not embeddings:
            return ""

        return self._dict_list_to_csv(embeddings)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries as CSV"""
        if not summaries:
            return ""

        return self._dict_list_to_csv(summaries)

    def _dict_list_to_csv(self, dict_list: List[Dict]) -> str:
        """Convert list of dictionaries to CSV string"""
        if not dict_list:
            return ""

        import io
        output = io.StringIO()

        # Get all field names
        fieldnames = set()
        for item in dict_list:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for item in dict_list:
            # Flatten any nested objects to JSON strings
            flat_item = {}
            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    flat_item[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flat_item[key] = value
            writer.writerow(flat_item)

        return output.getvalue()

    def get_file_extension(self) -> str:
        return ".csv"

class TextFormatter(BaseFormatter):
    """Human-readable text formatter"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_name = "txt"

    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations as readable text"""
        output = []
        output.append("CONVERSATION TRAINING DATA")
        output.append("=" * 50)
        output.append(f"Total conversations: {len(conversations)}")
        output.append("")

        for i, conv in enumerate(conversations, 1):
            output.append(f"CONVERSATION {i}")
            output.append("-" * 30)

            if 'messages' in conv:
                for message in conv['messages']:
                    role = message.get('role', 'unknown').upper()
                    content = message.get('content', '')
                    output.append(f"{role}: {content}")
                    output.append("")

            if 'source_file' in conv:
                output.append(f"Source: {conv['source_file']}")

            output.append("=" * 50)
            output.append("")

        return '\n'.join(output)

    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs as readable text"""
        output = []
        output.append("QUESTION & ANSWER TRAINING DATA")
        output.append("=" * 50)
        output.append(f"Total Q&A pairs: {len(qa_pairs)}")
        output.append("")

        for i, qa in enumerate(qa_pairs, 1):
            output.append(f"Q&A PAIR {i}")
            output.append("-" * 20)
            output.append(f"QUESTION: {qa.get('question', '')}")
            output.append("")
            output.append(f"ANSWER: {qa.get('answer', '')}")
            output.append("")

            if 'source_file' in qa:
                output.append(f"Source: {qa['source_file']}")

            output.append("=" * 50)
            output.append("")

        return '\n'.join(output)

    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embeddings as readable text"""
        output = []
        output.append("EMBEDDING TRAINING DATA")
        output.append("=" * 50)
        output.append(f"Total embedding pairs: {len(embeddings)}")
        output.append("")

        for i, emb in enumerate(embeddings, 1):
            output.append(f"EMBEDDING PAIR {i}")
            output.append("-" * 25)
            output.append(f"Sentence 1: {emb.get('sentence1', '')}")
            output.append(f"Sentence 2: {emb.get('sentence2', '')}")
            output.append(f"Similarity: {emb.get('similarity', 'N/A')}")
            output.append("")

            if 'source_file' in emb:
                output.append(f"Source: {emb['source_file']}")

            output.append("=" * 50)
            output.append("")

        return '\n'.join(output)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries as readable text"""
        output = []
        output.append("SUMMARY TRAINING DATA")
        output.append("=" * 50)
        output.append(f"Total summaries: {len(summaries)}")
        output.append("")

        for i, summary in enumerate(summaries, 1):
            output.append(f"SUMMARY {i}")
            output.append("-" * 15)
            output.append(f"SUMMARY: {summary.get('summary', '')}")
            output.append("")

            if 'original_text' in summary:
                original = summary['original_text']
                if len(original) > 200:
                    original = original[:200] + "..."
                output.append(f"ORIGINAL (excerpt): {original}")
                output.append("")

            if 'source_file' in summary:
                output.append(f"Source: {summary['source_file']}")

            output.append("=" * 50)
            output.append("")

        return '\n'.join(output)

    def get_file_extension(self) -> str:
        return ".txt"

class XMLFormatter(BaseFormatter):
    """XML formatter for structured data"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_name = "xml"

    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations as XML"""
        root = ET.Element("conversations")
        root.set("count", str(len(conversations)))

        for i, conv in enumerate(conversations):
            conv_elem = ET.SubElement(root, "conversation")
            conv_elem.set("id", str(i + 1))

            if 'source_file' in conv:
                conv_elem.set("source", conv['source_file'])

            if 'messages' in conv:
                messages_elem = ET.SubElement(conv_elem, "messages")
                for j, message in enumerate(conv['messages']):
                    msg_elem = ET.SubElement(messages_elem, "message")
                    msg_elem.set("index", str(j))
                    msg_elem.set("role", message.get('role', ''))
                    msg_elem.text = message.get('content', '')

        return self._prettify_xml(root)

    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs as XML"""
        root = ET.Element("qa_pairs")
        root.set("count", str(len(qa_pairs)))

        for i, qa in enumerate(qa_pairs):
            qa_elem = ET.SubElement(root, "qa_pair")
            qa_elem.set("id", str(i + 1))

            if 'source_file' in qa:
                qa_elem.set("source", qa['source_file'])

            question_elem = ET.SubElement(qa_elem, "question")
            question_elem.text = qa.get('question', '')

            answer_elem = ET.SubElement(qa_elem, "answer")
            answer_elem.text = qa.get('answer', '')

        return self._prettify_xml(root)

    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embeddings as XML"""
        root = ET.Element("embeddings")
        root.set("count", str(len(embeddings)))

        for i, emb in enumerate(embeddings):
            emb_elem = ET.SubElement(root, "embedding_pair")
            emb_elem.set("id", str(i + 1))
            emb_elem.set("similarity", str(emb.get('similarity', '')))

            if 'source_file' in emb:
                emb_elem.set("source", emb['source_file'])

            sent1_elem = ET.SubElement(emb_elem, "sentence1")
            sent1_elem.text = emb.get('sentence1', '')

            sent2_elem = ET.SubElement(emb_elem, "sentence2")
            sent2_elem.text = emb.get('sentence2', '')

        return self._prettify_xml(root)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries as XML"""
        root = ET.Element("summaries")
        root.set("count", str(len(summaries)))

        for i, summary in enumerate(summaries):
            summary_elem = ET.SubElement(root, "summary")
            summary_elem.set("id", str(i + 1))

            if 'source_file' in summary:
                summary_elem.set("source", summary['source_file'])

            summary_text_elem = ET.SubElement(summary_elem, "summary_text")
            summary_text_elem.text = summary.get('summary', '')

            if 'original_text' in summary:
                original_elem = ET.SubElement(summary_elem, "original_text")
                original_elem.text = summary['original_text']

        return self._prettify_xml(root)

    def _prettify_xml(self, elem) -> str:
        """Create pretty-printed XML string"""
        import xml.dom.minidom
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def get_file_extension(self) -> str:
        return ".xml"

class MarkdownFormatter(BaseFormatter):
    """Markdown formatter for documentation-style output"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_name = "markdown"

    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations as Markdown"""
        output = []
        output.append("# Conversation Training Data")
        output.append("")
        output.append(f"**Total conversations:** {len(conversations)}")
        output.append("")

        for i, conv in enumerate(conversations, 1):
            output.append(f"## Conversation {i}")
            output.append("")

            if 'messages' in conv:
                for message in conv['messages']:
                    role = message.get('role', 'unknown').title()
                    content = message.get('content', '')
                    output.append(f"**{role}:** {content}")
                    output.append("")

            if 'source_file' in conv:
                output.append(f"*Source: {conv['source_file']}*")
                output.append("")

            output.append("---")
            output.append("")

        return '\n'.join(output)

    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs as Markdown"""
        output = []
        output.append("# Q&A Training Data")
        output.append("")
        output.append(f"**Total Q&A pairs:** {len(qa_pairs)}")
        output.append("")

        for i, qa in enumerate(qa_pairs, 1):
            output.append(f"## Q&A Pair {i}")
            output.append("")
            output.append(f"**Q:** {qa.get('question', '')}")
            output.append("")
            output.append(f"**A:** {qa.get('answer', '')}")
            output.append("")

            if 'source_file' in qa:
                output.append(f"*Source: {qa['source_file']}*")
                output.append("")

            output.append("---")
            output.append("")

        return '\n'.join(output)

    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embeddings as Markdown"""
        output = []
        output.append("# Embedding Training Data")
        output.append("")
        output.append(f"**Total embedding pairs:** {len(embeddings)}")
        output.append("")

        for i, emb in enumerate(embeddings, 1):
            output.append(f"## Embedding Pair {i}")
            output.append("")
            output.append(f"**Sentence 1:** {emb.get('sentence1', '')}")
            output.append("")
            output.append(f"**Sentence 2:** {emb.get('sentence2', '')}")
            output.append("")
            output.append(f"**Similarity:** {emb.get('similarity', 'N/A')}")
            output.append("")

            if 'source_file' in emb:
                output.append(f"*Source: {emb['source_file']}*")
                output.append("")

            output.append("---")
            output.append("")

        return '\n'.join(output)

    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summaries as Markdown"""
        output = []
        output.append("# Summary Training Data")
        output.append("")
        output.append(f"**Total summaries:** {len(summaries)}")
        output.append("")

        for i, summary in enumerate(summaries, 1):
            output.append(f"## Summary {i}")
            output.append("")
            output.append(f"**Summary:** {summary.get('summary', '')}")
            output.append("")

            if 'original_text' in summary:
                output.append(f"**Original Text:**")
                output.append("```")
                original = summary['original_text']
                if len(original) > 500:
                    original = original[:500] + "..."
                output.append(original)
                output.append("```")
                output.append("")

            if 'source_file' in summary:
                output.append(f"*Source: {summary['source_file']}*")
                output.append("")

            output.append("---")
            output.append("")

        return '\n'.join(output)

    def get_file_extension(self) -> str:
        return ".md"

# Formatter factory
class FormatterFactory:
    """Factory for creating appropriate formatters"""

    _formatters = {
        'jsonl': JSONLFormatter,
        'json': JSONFormatter,
        'csv': CSVFormatter,
        'txt': TextFormatter,
        'xml': XMLFormatter,
        'markdown': MarkdownFormatter,
        'md': MarkdownFormatter
    }

    @classmethod
    def create_formatter(cls, format_name: str, config: Dict[str, Any]) -> BaseFormatter:
        """
        Create formatter for specified format

        Args:
            format_name: Name of the format
            config: Configuration dictionary

        Returns:
            Formatter instance
        """
        formatter_class = cls._formatters.get(format_name.lower())
        if not formatter_class:
            raise ValueError(f"Unsupported format: {format_name}")

        return formatter_class(config)

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported format names"""
        return list(cls._formatters.keys())

    @classmethod
    def register_formatter(cls, format_name: str, formatter_class):
        """Register a new formatter class"""
        cls._formatters[format_name.lower()] = formatter_class

# Utility functions
def format_data(data: List[Dict], data_type: str, format_name: str, config: Dict[str, Any]) -> str:
    """
    Format data using specified formatter

    Args:
        data: Data to format
        data_type: Type of data (conversations, qa_pairs, etc.)
        format_name: Output format name
        config: Configuration

    Returns:
        Formatted string
    """
    formatter = FormatterFactory.create_formatter(format_name, config)

    if data_type == 'conversations':
        return formatter.format_conversations(data)
    elif data_type == 'qa_pairs':
        return formatter.format_qa_pairs(data)
    elif data_type == 'embeddings':
        return formatter.format_embeddings(data)
    elif data_type == 'summaries':
        return formatter.format_summaries(data)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def get_file_extension(format_name: str, config: Dict[str, Any]) -> str:
    """Get file extension for format"""
    try:
        formatter = FormatterFactory.create_formatter(format_name, config)
        return formatter.get_file_extension()
    except ValueError:
        return ".txt"  # Default fallback

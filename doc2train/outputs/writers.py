# outputs/writers.py
"""
Complete output writers system for Doc2Train v2.0 Enhanced
Handles all output formats and per-file saving with templates
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import pandas as pd
import ipdb
from doc2train.outputs.writer_plugin_manager import WriterPluginManager

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
from doc2train.outputs.base_writer import BaseWriter

class OutputWriter:
    """
    Complete output writer with multiple format support and templates
    """

    def __init__(self, config: Dict[str, Any]):
            """Initialize output writer"""
            self.config = config
            self.output_dir = Path(config.get('output_dir', 'output'))
            self.config = config
            self.plugin_mgr = WriterPluginManager(config)
            # now plugins are available by name:
            # example: self.plugin_mgr.get('csv_writer')
            # NEW: Parse multiple output formats
            output_format = config.get('output_format', 'jsonl')
            if isinstance(output_format, str) and ',' in output_format:
                # Handle "jsonl, json, csv" format
                self.output_formats = [fmt.strip() for fmt in output_format.split(',')]
            elif isinstance(output_format, list):
                # Handle ["jsonl", "json"] format
                self.output_formats = output_format
            else:
                # Single format
                self.output_formats = [output_format]

            print(f"📝 Output formats: {', '.join(self.output_formats)}")

            # Keep backward compatibility
            self.output_format = self.output_formats[0]  # Primary format
            self.template_file = config.get('output_template')

            # Create output directories
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_extraction_results(self, results: Dict[str, Any]):
        """
        Save extraction results to files

        Args:
            results: Processing results dictionary
        """
        # Save extracted content
        extracted_dir = self.output_dir / "extracted"
        extracted_dir.mkdir(exist_ok=True)

        if 'extracted_data' in results:
            self._save_extracted_data(results['extracted_data'], extracted_dir)

        # Save processing summary
        self._save_processing_summary(results)

    def save_generated_data(self, generated_data: Dict[str, Dict[str, List]],
                          file_mapping: Optional[Dict[str, str]] = None):
        """
        Save generated training data to organized output files

        Args:
            generated_data: Dictionary mapping file paths to generated data
            file_mapping: Optional mapping of file paths to output names
        """
        # Organize data by type
        organized_data = self._organize_generated_data(generated_data, file_mapping)

        # Save each data type
        for data_type, items in organized_data.items():
            if items:
                self._save_data_type(data_type, items)

    def save_per_file_data(self, file_output_dir: Path, file_data: Dict[str, List]):
        """
        Save data for individual file (fault-tolerant per-file saving)

        Args:
            file_output_dir: Directory for this file's output
            file_data: Generated data for this file
        """
        file_output_dir.mkdir(parents=True, exist_ok=True)

        for data_type, items in file_data.items():
            if items:
                output_file = file_output_dir / f"{data_type}.{self.output_format}"
                self._write_data_file(output_file, items, data_type)

    def _save_extracted_data(self, extracted_data: Dict[str, tuple], output_dir: Path):
        """Save raw extracted data"""
        output_formats = self.output_formats
        for file_path, (text, images) in extracted_data.items():
            file_name = Path(file_path).stem

            # Save text
            if text:
                # Convert text to list of dict for compatibility with _write_data_file
                text_items = [{
                    'content': text,
                    'source_file': file_path,
                    'extraction_time': time.time(),
                    'character_count': len(text)
                }]

                # Save in each requested format
                for fmt in output_formats:
                    output_file = output_dir / f"{file_name}.{fmt}"

                    # Temporarily set the format and use existing method
                    original_format = self.output_format
                    self.output_format = fmt

                    try:
                        if fmt == 'txt':
                            # For txt, keep original behavior (raw text)
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(text)
                        else:
                            # Use existing _write_data_file for other formats
                            self._write_data_file(output_file, text_items, 'extracted')
                    finally:
                        # Restore original format
                        self.output_format = original_format

            # Save image metadata
            if images:
                images_file = output_dir / f"{file_name}_images.json"
                # Remove binary data for JSON storage
                json_safe_images = []
                for img in images:
                    img_copy = {k: v for k, v in img.items() if k != 'data'}
                    img_copy['has_data'] = 'data' in img
                    json_safe_images.append(img_copy)

                with open(images_file, 'w', encoding='utf-8') as f:
                    json.dump(json_safe_images, f, indent=2, default=str)

    def _organize_generated_data(self, generated_data: Dict[str, Dict[str, List]],
                                file_mapping: Optional[Dict[str, str]] = None) -> Dict[str, List]:
        """Organize generated data by type across all files"""
        organized = {}

        for file_path, file_data in generated_data.items():
            # Get display name for file
            source_name = file_mapping.get(file_path, Path(file_path).name) if file_mapping else Path(file_path).name

            for data_type, items in file_data.items():
                if data_type not in organized:
                    organized[data_type] = []

                # Add source information to each item
                for item in items:
                    if isinstance(item, dict):
                        item['source_file'] = source_name
                        item['source_path'] = file_path

                organized[data_type].extend(items)

        return organized

    def _save_data_type(self, data_type: str, items: List[Dict]):
        """Save a specific data type to file, in **all** configured formats."""
        type_dir = self.output_dir / data_type
        type_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())

        for fmt in self.output_formats:
            out = type_dir / f"{data_type}_{timestamp}.{fmt}"
            plugin_cls = self.plugin_mgr.get(fmt)

            if plugin_cls:
                # custom plugin writer
                writer = plugin_cls(self.config)
                writer.write(out, items, data_type)
                print(f"💾 Saved {len(items)} '{data_type}' via plugin '{fmt}' → {out.name}")
            else:
                # fallback to built-in
                self.output_format = fmt
                self._write_data_file(out, items, data_type)
                print(f"💾 Saved {len(items)} '{data_type}' → {out.name}")

    def _write_data_file(self, output_file: Path, items: List[Dict], data_type: str):
        """Write data to file in specified format"""
        try:
            if self.output_format == 'jsonl':
                self._write_jsonl(output_file, items)
            elif self.output_format == 'json':
                self._write_json(output_file, items)
            elif self.output_format == 'csv':
                self._write_csv(output_file, items, data_type)
            elif self.output_format == 'txt':
                self._write_txt(output_file, items, data_type)
            else:
                raise ValueError(f"Unsupported output format: {self.output_format}")

        except Exception as e:
            print(f"❌ Error writing {output_file}: {e}")
            # Fallback to original single format method
            self._write_data_file_original(output_file, items, data_type)

    def _write_jsonl(self, output_file: Path, items: List[Dict]):
        """Write data in JSONL format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _write_json(self, output_file: Path, items: List[Dict]):
        """Write data in JSON format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False, default=str)

    def _write_csv(self, output_file: Path, items: List[Dict], data_type: str):
        """Write data in CSV format"""
        if not items:
            return

        # Flatten nested dictionaries for CSV
        flattened_items = []
        for item in items:
            flat_item = self._flatten_dict(item)
            flattened_items.append(flat_item)

        # Use pandas for better CSV handling
        try:
            df = pd.DataFrame(flattened_items)
            df.to_csv(output_file, index=False, encoding='utf-8')
        except Exception:
            # Fallback to manual CSV writing
            self._write_csv_manual(output_file, flattened_items)

    def _write_txt(self, output_file: Path, items: List[Dict], data_type: str):
        """Write data in human-readable text format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {data_type.upper()} DATA\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total items: {len(items)}\n\n")

            for i, item in enumerate(items, 1):
                f.write(f"## Item {i}\n")

                if data_type == 'conversations':
                    self._write_conversation_txt(f, item)
                elif data_type == 'qa_pairs':
                    self._write_qa_txt(f, item)
                elif data_type == 'summaries':
                    self._write_summary_txt(f, item)
                else:
                    # Generic format
                    for key, value in item.items():
                        f.write(f"**{key}**: {value}\n")

                f.write("\n" + "-" * 50 + "\n\n")

    def _write_conversation_txt(self, f, item: Dict):
        """Write conversation in readable text format"""
        if 'messages' in item:
            f.write("**Conversation:**\n\n")
            for i, message in enumerate(item['messages'], 1):
                role = message.get('role', 'unknown').title()
                content = message.get('content', '')
                f.write(f"{role}: {content}\n\n")

        if 'source_file' in item:
            f.write(f"*Source: {item['source_file']}*\n")

    def _write_qa_txt(self, f, item: Dict):
        """Write Q&A pair in readable text format"""
        question = item.get('question', '')
        answer = item.get('answer', '')

        f.write(f"**Question:** {question}\n\n")
        f.write(f"**Answer:** {answer}\n\n")

        if 'source_file' in item:
            f.write(f"*Source: {item['source_file']}*\n")

    def _write_summary_txt(self, f, item: Dict):
        """Write summary in readable text format"""
        summary = item.get('summary', '')
        original = item.get('original_text', '')

        f.write(f"**Summary:** {summary}\n\n")

        if original:
            f.write(f"**Original Text (excerpt):** {original[:200]}...\n\n")

        if 'source_file' in item:
            f.write(f"*Source: {item['source_file']}*\n")

    def _write_csv_manual(self, output_file: Path, items: List[Dict]):
        """Manual CSV writing as fallback"""
        if not items:
            return

        # Get all possible field names
        fieldnames = set()
        for item in items:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item)

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV output"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _save_processing_summary(self, results: Dict[str, Any]):
        """Save processing summary with metadata"""
        summary = {
            'processing_summary': {
                'timestamp': time.time(),
                'readable_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'mode': results.get('mode'),
                'files_processed': results.get('files_processed', 0),
                'successful': results.get('successful', 0),
                'failed': results.get('failed', 0),
                'total_processing_time': results.get('total_processing_time', 0),
                'total_text_chars': results.get('total_text_chars', 0),
                'total_images': results.get('total_images', 0)
            },
            'configuration': results.get('config_used', {}),
            'errors': results.get('errors', []),
            'output_info': {
                'output_directory': str(self.output_dir),
                'output_format': self.output_format,
                'files_created': []
            }
        }

        # List output files created
        for item in self.output_dir.rglob('*'):
            if item.is_file() and item.name != 'summary.json':
                relative_path = item.relative_to(self.output_dir)
                summary['output_info']['files_created'].append(str(relative_path))

        # Save summary
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"📊 Processing summary saved to {summary_file}")

class TemplateProcessor:
    """
    Template processor for custom output formats
    """

    def __init__(self, template_file: Optional[str] = None):
        self.template = None
        if template_file:
            self.load_template(template_file)

    def load_template(self, template_file: str):
        """Load output template from file"""
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                self.template = f.read()
        except Exception as e:
            print(f"❌ Error loading template {template_file}: {e}")

    def apply_template(self, data: Dict[str, Any]) -> str:
        """Apply template to data"""
        if not self.template:
            return json.dumps(data, indent=2)

        try:
            # Simple template substitution
            output = self.template
            for key, value in data.items():
                placeholder = f"{{{{{key}}}}}"
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                output = output.replace(placeholder, str(value))

            return output
        except Exception as e:
            print(f"❌ Template application error: {e}")
            return json.dumps(data, indent=2)

class OutputValidator:
    """
    Validator for output data quality and format compliance
    """

    @staticmethod
    def validate_conversations(conversations: List[Dict]) -> Dict[str, Any]:
        """Validate conversation data quality"""
        validation_results = {
            'total_conversations': len(conversations),
            'valid_conversations': 0,
            'invalid_conversations': 0,
            'issues': []
        }

        for i, conv in enumerate(conversations):
            issues = []

            # Check required fields
            if 'messages' not in conv:
                issues.append("Missing 'messages' field")
            elif not isinstance(conv['messages'], list):
                issues.append("'messages' must be a list")
            elif len(conv['messages']) < 2:
                issues.append("Conversation must have at least 2 messages")
            else:
                # Validate each message
                for j, msg in enumerate(conv['messages']):
                    if 'role' not in msg:
                        issues.append(f"Message {j} missing 'role'")
                    if 'content' not in msg:
                        issues.append(f"Message {j} missing 'content'")
                    elif len(msg['content'].strip()) < 10:
                        issues.append(f"Message {j} content too short")

            if issues:
                validation_results['invalid_conversations'] += 1
                validation_results['issues'].append({
                    'conversation_index': i,
                    'issues': issues
                })
            else:
                validation_results['valid_conversations'] += 1

        return validation_results

    @staticmethod
    def validate_qa_pairs(qa_pairs: List[Dict]) -> Dict[str, Any]:
        """Validate Q&A pairs data quality"""
        validation_results = {
            'total_pairs': len(qa_pairs),
            'valid_pairs': 0,
            'invalid_pairs': 0,
            'issues': []
        }

        for i, pair in enumerate(qa_pairs):
            issues = []

            # Check required fields
            if 'question' not in pair:
                issues.append("Missing 'question' field")
            elif len(pair['question'].strip()) < 5:
                issues.append("Question too short")

            if 'answer' not in pair:
                issues.append("Missing 'answer' field")
            elif len(pair['answer'].strip()) < 10:
                issues.append("Answer too short")

            if issues:
                validation_results['invalid_pairs'] += 1
                validation_results['issues'].append({
                    'pair_index': i,
                    'issues': issues
                })
            else:
                validation_results['valid_pairs'] += 1

        return validation_results

    @staticmethod
    def validate_embeddings(embeddings: List[Dict]) -> Dict[str, Any]:
        """Validate embedding pairs data quality"""
        validation_results = {
            'total_pairs': len(embeddings),
            'valid_pairs': 0,
            'invalid_pairs': 0,
            'issues': []
        }

        for i, pair in enumerate(embeddings):
            issues = []

            # Check required fields
            required_fields = ['sentence1', 'sentence2', 'similarity']
            for field in required_fields:
                if field not in pair:
                    issues.append(f"Missing '{field}' field")

            # Validate similarity score
            if 'similarity' in pair:
                similarity = pair['similarity']
                if not isinstance(similarity, (int, float)):
                    issues.append("Similarity must be a number")
                elif not 0 <= similarity <= 1:
                    issues.append("Similarity must be between 0 and 1")

            # Validate sentences
            for field in ['sentence1', 'sentence2']:
                if field in pair and len(pair[field].strip()) < 5:
                    issues.append(f"{field} too short")

            if issues:
                validation_results['invalid_pairs'] += 1
                validation_results['issues'].append({
                    'pair_index': i,
                    'issues': issues
                })
            else:
                validation_results['valid_pairs'] += 1

        return validation_results

class OutputManager:
    """
    High-level output manager that coordinates all output operations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.writer = OutputWriter(config)
        self.validator = OutputValidator()
        self.template_processor = TemplateProcessor(config.get('output_template'))

    def save_all_results(self, results: Dict[str, Any]):
        """Save all processing results with validation"""
        print("💾 Saving processing results...")

        # Save extraction results
        if 'extracted_data' in results:
            self.writer.save_extraction_results(results)

        # Save generated data with validation
        if 'generated_data' in results:
            self._save_and_validate_generated_data(results['generated_data'])

        # Create comprehensive report
        self._create_comprehensive_report(results)

    def _save_and_validate_generated_data(self, generated_data: Dict[str, Dict[str, List]]):
        """Save generated data with quality validation"""
        # Organize and save data
        self.writer.save_generated_data(generated_data)

        # Validate data quality
        validation_report = self._validate_all_generated_data(generated_data)

        # Save validation report
        validation_file = self.writer.output_dir / 'validation_report.json'
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)

        print(f"📋 Data validation report saved to {validation_file}")

        # Print validation summary
        self._print_validation_summary(validation_report)

    def _validate_all_generated_data(self, generated_data: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        """Validate all generated data types"""
        # Organize data by type
        organized_data = self.writer._organize_generated_data(generated_data)

        validation_report = {
            'timestamp': time.time(),
            'validation_results': {}
        }

        # Validate each data type
        for data_type, items in organized_data.items():
            if data_type == 'conversations':
                validation_report['validation_results'][data_type] = self.validator.validate_conversations(items)
            elif data_type == 'qa_pairs':
                validation_report['validation_results'][data_type] = self.validator.validate_qa_pairs(items)
            elif data_type == 'embeddings':
                validation_report['validation_results'][data_type] = self.validator.validate_embeddings(items)
            else:
                # Generic validation for other types
                validation_report['validation_results'][data_type] = {
                    'total_items': len(items),
                    'validation_method': 'generic'
                }

        return validation_report

    def _print_validation_summary(self, validation_report: Dict[str, Any]):
        """Print validation summary to console"""
        print(f"\n📊 Data Quality Validation Summary:")

        for data_type, results in validation_report['validation_results'].items():
            if 'valid_conversations' in results:
                total = results['total_conversations']
                valid = results['valid_conversations']
                print(f"   {data_type}: {valid}/{total} valid ({valid/total*100:.1f}%)")
            elif 'valid_pairs' in results:
                total = results['total_pairs']
                valid = results['valid_pairs']
                print(f"   {data_type}: {valid}/{total} valid ({valid/total*100:.1f}%)")
            else:
                total = results.get('total_items', 0)
                print(f"   {data_type}: {total} items")

    def _create_comprehensive_report(self, results: Dict[str, Any]):
        """Create comprehensive processing report"""
        report = {
            'doc2train_version': '2.0_enhanced',
            'processing_report': {
                'timestamp': time.time(),
                'readable_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'mode': results.get('mode'),
                'configuration': results.get('config_used', {}),
                'performance': {
                    'files_processed': results.get('files_processed', 0),
                    'successful': results.get('successful', 0),
                    'failed': results.get('failed', 0),
                    'success_rate': results.get('successful', 0) / max(results.get('files_processed', 1), 1),
                    'total_processing_time': results.get('total_processing_time', 0),
                    'avg_time_per_file': results.get('total_processing_time', 0) / max(results.get('files_processed', 1), 1)
                },
                'content_statistics': {
                    'total_text_chars': results.get('total_text_chars', 0),
                    'total_images': results.get('total_images', 0),
                    'avg_text_per_file': results.get('total_text_chars', 0) / max(results.get('successful', 1), 1),
                    'avg_images_per_file': results.get('total_images', 0) / max(results.get('successful', 1), 1)
                },
                'errors': results.get('errors', []),
                'output_files': []
            }
        }

        # List all created files
        for file_path in self.writer.output_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.writer.output_dir)
                file_info = {
                    'path': str(relative_path),
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'type': file_path.suffix.lower()
                }
                report['processing_report']['output_files'].append(file_info)

        # Save comprehensive report
        report_file = self.writer.output_dir / 'comprehensive_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"📋 Comprehensive report saved to {report_file}")

# to be removed
def create_output_manager(config: Dict[str, Any]) -> OutputManager:
    """Create output manager with configuration"""
    return OutputManager(config)

# to be removed
def save_extraction_only_results(file_paths: List[str], extracted_data: Dict[str, tuple], config: Dict[str, Any]):
    """Save extraction-only results"""
    manager = OutputManager(config)
    results = {
        'mode': 'extract-only',
        'extracted_data': extracted_data,
        'files_processed': len(file_paths),
        'successful': len([d for d in extracted_data.values() if d[0] or d[1]]),
        'failed': len([d for d in extracted_data.values() if not d[0] and not d[1]]),
        'config_used': config
    }
    manager.writer.save_extraction_results(results)

# to be removed
def save_generated_training_data(generated_data: Dict[str, Dict[str, List]], config: Dict[str, Any]):
    """Save generated training data with validation"""
    manager = OutputManager(config)
    manager._save_and_validate_generated_data(generated_data)

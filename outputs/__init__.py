# outputs/__init__.py
"""
Complete outputs package for Doc2Train v2.0 Enhanced
Output handling and formatting components
"""

from .writers import (
    OutputWriter, TemplateProcessor, OutputValidator, OutputManager,
    create_output_manager, save_extraction_only_results, save_generated_training_data
)

from .formatters import (
    BaseFormatter, JSONLFormatter, JSONFormatter, CSVFormatter, TextFormatter,
    XMLFormatter, MarkdownFormatter, FormatterFactory,
    format_data, get_file_extension
)

__all__ = [
    # Writers
    'OutputWriter', 'TemplateProcessor', 'OutputValidator', 'OutputManager',
    'create_output_manager', 'save_extraction_only_results', 'save_generated_training_data',

    # Formatters
    'BaseFormatter', 'JSONLFormatter', 'JSONFormatter', 'CSVFormatter', 'TextFormatter',
    'XMLFormatter', 'MarkdownFormatter', 'FormatterFactory',
    'format_data', 'get_file_extension'
]

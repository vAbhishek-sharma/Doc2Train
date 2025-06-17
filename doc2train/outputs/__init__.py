"""
Outputs package for Doc2Train v2.0 Enhanced
All writer, formatter, and manager classes for handling output generation
"""

# Base formatter API
from .base_formatters import (
    BaseFormatter,
    register_formatter,
    get_formatter,
    list_formatters
)

# Base writer API
from .base_writer import (
    BaseWriter,
    register_writer,
    get_writer,
    list_writers
)

# Concrete formatters
from .formatters import (
    JSONLFormatter,
    JSONFormatter,
    CSVFormatter,
    MarkdownFormatter,
    FormatterFactory,
    format_data,
    get_file_extension
)

# Concrete writers and managers
from .writers import (
    OutputWriter,
    OutputValidator,
    OutputManager,
    create_output_manager,
    save_extraction_only_results,
    save_generated_training_data
)

# Plugin managers
from .formatter_plugin_manager import FormatterPluginManager
from .writer_plugin_manager import WriterPluginManager

__all__ = [
    # Base formatter API
    'BaseFormatter', 'register_formatter', 'get_formatter', 'list_formatters',
    # Base writer API
    'BaseWriter', 'register_writer', 'get_writer', 'list_writers',
    # Concrete formatters
    'JSONLFormatter', 'JSONFormatter', 'CSVFormatter', 'MarkdownFormatter',
    'FormatterFactory', 'format_data', 'get_file_extension',
    # Concrete writers and managers
    'OutputWriter', 'OutputValidator', 'OutputManager',
    'create_output_manager', 'save_extraction_only_results', 'save_generated_training_data',
    # Plugin managers
    'FormatterPluginManager', 'WriterPluginManager'
]

# outputs/formatters.py
"""
Unified output formatters for different data types and formats
- Delegates to formatter plugins (jsonl, csv, txt, etc.)
- Smart and explicit API for formatting
"""

from typing import Any, Optional

def format_data(data, data_type, format_name, config=None):
    """
    Format data using a specific formatter plugin (by name).
    """
    from doc2train.core.registries.formatter_registry import get_formatter
    fmt_cls = get_formatter(format_name)
    if not fmt_cls:
        raise ValueError(f"Formatter {format_name} not found")
    return fmt_cls(config).format(data, data_type)

def smart_format_data(data, data_type, output_file: Optional[str] = None, config: Optional[dict] = None):
    """
    Format data using the best available formatter plugin.
    Selection order:
      1. config['output_format'] (if set)
      2. file extension (from output_file)
      3. data_type (if formatter exists)
      4. fallback: 'jsonl'
    """
    from pathlib import Path
    from doc2train.core.registries.formatter_registry import get_formatter
    format_name = None

    # 1. Config override
    if config and config.get("output_format"):
        format_name = config["output_format"]

    # 2. Guess by file extension if not set
    if not format_name and output_file:
        ext = Path(output_file).suffix.lower().lstrip('.')
        if get_formatter(ext):
            format_name = ext

    # 3. Guess by data_type if not set
    if not format_name and data_type and get_formatter(data_type):
        format_name = data_type

    # 4. Fallback to jsonl
    if not format_name:
        format_name = "jsonl"

    fmt_cls = get_formatter(format_name)
    if not fmt_cls:
        raise ValueError(f"Formatter {format_name} not found")
    return fmt_cls(config).format(data, data_type)

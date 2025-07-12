# cli/__init__.py
"""
Complete CLI package for Doc2Train Enhanced
Command line interface components
"""

from .args import (
    create_enhanced_parser, parse_skip_pages, args_to_config, get_examples_text
)

from .commands import (
    execute_processing_command, execute_validate_command, execute_benchmark_command,
    execute_cache_command, execute_info_command,
    route_command, print_command_help, handle_keyboard_interrupt, format_command_results
)

__all__ = [
    # Argument parsing
    'create_enhanced_parser', 'parse_skip_pages', 'args_to_config', 'get_examples_text',

    # Command execution
    'execute_processing_command', 'execute_validate_command', 'execute_benchmark_command',
    'execute_cache_command', 'execute_info_command',
    'route_command', 'print_command_help', 'handle_keyboard_interrupt', 'format_command_results'
]

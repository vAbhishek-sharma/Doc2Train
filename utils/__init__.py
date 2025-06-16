"""
Complete utilities package for Doc2Train v2.0 Enhanced
All utility modules for enterprise document processing
"""

# Progress tracking
from .progress import (
    ProgressTracker, ProgressDisplay, PerformanceMonitor,
    initialize_progress, start_file_processing, complete_file_processing,
    add_processing_error, update_progress_display, show_completion_summary,
    get_processing_stats, get_performance_report
)

# Validation utilities
from .validation import (
    validate_input_and_files, validate_extraction_quality, validate_configuration,
    validate_system_requirements, validate_and_report_system, create_validation_report,
    assess_text_quality, validate_image_quality, is_supported_file, find_supported_files
)

# Caching system
from .cache import (
    CacheManager, get_cache_manager, get_cache_stats,
    clear_cache, cleanup_cache, optimize_cache
)

# File utilities
from .files import (
    get_supported_files, get_file_info, calculate_file_hash, find_duplicate_files,
    organize_files_by_type, create_directory_structure, safe_copy_file, safe_move_file,
    cleanup_empty_directories, get_directory_size, find_files_by_pattern,
    backup_file, validate_file_permissions, ensure_directory_exists,
    get_file_encoding, list_files_with_info, cleanup_temp_files
)

# Config loader
from .config_loader import (
    ConfigLoader,
    get_config_loader,
    load_config_from_yaml,
    validate_config
)

# Plugin loader
from .plugin_loader import load_plugins_from_dirs

# Process management
from .process import (
    ProcessManager, SystemMonitor, TaskQueue, ProcessLimiter, GracefulShutdown,
    get_system_info, optimize_worker_count, wait_for_resources
)

__all__ = [
    # Progress tracking
    'ProgressTracker', 'ProgressDisplay', 'PerformanceMonitor',
    'initialize_progress', 'start_file_processing', 'complete_file_processing',
    'add_processing_error', 'update_progress_display', 'show_completion_summary',
    'get_processing_stats', 'get_performance_report',

    # Validation
    'validate_input_and_files', 'validate_extraction_quality', 'validate_configuration',
    'validate_system_requirements', 'validate_and_report_system', 'create_validation_report',
    'assess_text_quality', 'validate_image_quality', 'is_supported_file', 'find_supported_files',

    # Caching
    'CacheManager', 'get_cache_manager', 'get_cache_stats',
    'clear_cache', 'cleanup_cache', 'optimize_cache',

    # File utilities
    'get_supported_files', 'get_file_info', 'calculate_file_hash', 'find_duplicate_files',
    'organize_files_by_type', 'create_directory_structure', 'safe_copy_file', 'safe_move_file',
    'cleanup_empty_directories', 'get_directory_size', 'find_files_by_pattern',
    'backup_file', 'validate_file_permissions', 'ensure_directory_exists',
    'get_file_encoding', 'list_files_with_info', 'cleanup_temp_files',

    # Config loader
    'ConfigLoader', 'get_config_loader', 'load_config_from_yaml', 'validate_config',

    # Plugin loader
    'load_plugins_from_dirs',

    # Process management
    'ProcessManager', 'SystemMonitor', 'TaskQueue', 'ProcessLimiter', 'GracefulShutdown',
    'get_system_info', 'optimize_worker_count', 'wait_for_resources'
]

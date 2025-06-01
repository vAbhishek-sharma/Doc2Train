# cli/commands.py
"""
Complete CLI commands system for Doc2Train v2.0 Enhanced
Handles all command execution with proper error handling and validation
"""

import time
from typing import Dict, List, Any
from pathlib import Path

from core.pipeline import ProcessingPipeline, PerformanceBenchmark
from utils.validation import validate_and_report_system, create_validation_report
from utils.cache import get_cache_stats, cleanup_cache, optimize_cache
from processors.base_processor import list_all_processors, discover_plugins
from outputs.writers import OutputManager

def execute_processing_command(args, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the main processing command with all enhancements

    Args:
        args: Command line arguments
        file_paths: List of files to process
        config: Processing configuration

    Returns:
        Processing results
    """
    # Handle special commands first
    if args.validate_only:
        return execute_validate_command(args, file_paths, config)

    if args.benchmark:
        return execute_benchmark_command(args, file_paths, config)

    # Execute main processing pipeline
    try:
        pipeline = ProcessingPipeline(config)
        results = pipeline.process_files(file_paths, args)

        # Save results using output manager
        output_manager = OutputManager(config)
        output_manager.save_all_results(results)

        return results

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'command': 'process'
        }

def execute_validate_command(args, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute validation-only command"""
    print("🔍 Running comprehensive validation...")

    # System validation
    system_valid = validate_and_report_system()

    # Create detailed validation report
    validation_report = create_validation_report(file_paths, config)

    # Save validation report
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    report_file = output_dir / 'validation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, default=str)

    # Print summary
    print(f"\n📋 Validation Summary:")
    print(f"   System requirements: {'✅ Passed' if system_valid else '❌ Failed'}")

    file_validation = validation_report['file_validation']
    print(f"   Files found: {file_validation['total_files']}")
    print(f"   Supported files: {file_validation['supported_files']}")
    print(f"   Total size: {file_validation['total_size_mb']:.1f} MB")

    if file_validation['unsupported_files'] > 0:
        print(f"   ⚠️ Unsupported files: {file_validation['unsupported_files']}")

    if file_validation['large_files'] > 0:
        print(f"   ⚠️ Large files: {file_validation['large_files']}")

    config_errors = validation_report['config_validation']['errors']
    if config_errors:
        print(f"   ❌ Configuration errors: {len(config_errors)}")
        for error in config_errors:
            print(f"      • {error}")
    else:
        print(f"   ✅ Configuration valid")

    print(f"\n📄 Detailed report saved to: {report_file}")

    return {
        'success': system_valid and len(config_errors) == 0,
        'command': 'validate',
        'validation_report': validation_report,
        'system_valid': system_valid
    }

def execute_benchmark_command(args, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute performance benchmark command"""
    print("📊 Running performance benchmark...")

    # Limit files for benchmark
    benchmark_files = file_paths[:6]  # Use max 6 files for benchmark

    try:
        benchmark = PerformanceBenchmark(config)
        benchmark_results = benchmark.run_benchmark(benchmark_files)

        # Save benchmark results
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        benchmark_file = output_dir / 'benchmark_results.json'
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, default=str)

        print(f"\n📄 Benchmark results saved to: {benchmark_file}")

        return {
            'success': True,
            'command': 'benchmark',
            'benchmark_results': benchmark_results
        }

    except Exception as e:
        return {
            'success': False,
            'command': 'benchmark',
            'error': str(e)
        }

def execute_cache_command(args) -> Dict[str, Any]:
    """Execute cache-related commands"""
    if hasattr(args, 'cache_action'):
        if args.cache_action == 'stats':
            return execute_cache_stats_command()
        elif args.cache_action == 'cleanup':
            return execute_cache_cleanup_command(args)
        elif args.cache_action == 'optimize':
            return execute_cache_optimize_command()
        elif args.cache_action == 'clear':
            return execute_cache_clear_command(args)

    return {'success': False, 'error': 'Unknown cache command'}

def execute_cache_stats_command() -> Dict[str, Any]:
    """Show cache statistics"""
    print("📊 Cache Statistics:")

    stats = get_cache_stats()

    print(f"   Cache entries: {stats['cache_entries']}")
    print(f"   Total size: {stats['total_size_mb']:.1f} MB ({stats['total_size_gb']:.2f} GB)")

    if stats['cache_entries'] > 0:
        print(f"   Average size: {stats['avg_size_mb']:.2f} MB per entry")
        print(f"   Compression ratio: {stats['compression_ratio']:.0%}")

        if stats['file_types']:
            print(f"   File types:")
            for ext, count in stats['file_types'].items():
                print(f"     {ext}: {count} files")

    print(f"   Cache directory: {stats['cache_directory']}")
    print(f"   Max size limit: {stats['max_size_gb']:.1f} GB")
    print(f"   Expiry: {stats['expiry_days']} days")

    return {
        'success': True,
        'command': 'cache_stats',
        'stats': stats
    }

def execute_cache_cleanup_command(args) -> Dict[str, Any]:
    """Clean up cache"""
    print("🧹 Cleaning up cache...")

    max_size_gb = getattr(args, 'max_cache_size', 5.0)
    max_age_days = getattr(args, 'max_cache_age', 30)

    cleanup_cache(max_size_gb, max_age_days)

    return {
        'success': True,
        'command': 'cache_cleanup'
    }

def execute_cache_optimize_command() -> Dict[str, Any]:
    """Optimize cache"""
    print("📦 Optimizing cache...")

    optimize_cache()

    return {
        'success': True,
        'command': 'cache_optimize'
    }

def execute_cache_clear_command(args) -> Dict[str, Any]:
    """Clear cache"""
    print("🗑️ Clearing cache...")

    from utils.cache import clear_cache

    file_path = getattr(args, 'cache_file', None)
    clear_cache(file_path)

    return {
        'success': True,
        'command': 'cache_clear'
    }

def execute_info_command(args) -> Dict[str, Any]:
    """Execute info/status commands"""
    print("ℹ️ Doc2Train v2.0 Enhanced Information:")

    # System info
    print(f"\n🖥️ System Information:")
    import sys
    print(f"   Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   Available memory: {memory_gb:.1f} GB")
    except ImportError:
        print(f"   Memory: Unable to detect")

    # Processor info
    print(f"\n📋 Available Processors:")
    list_all_processors()

    # Cache info
    print(f"\n💾 Cache Information:")
    stats = get_cache_stats()
    print(f"   Cache entries: {stats['cache_entries']}")
    print(f"   Cache size: {stats['total_size_mb']:.1f} MB")

    # LLM providers
    print(f"\n🤖 LLM Providers:")
    try:
        from core.llm_client import get_available_providers
        providers = get_available_providers()
        if providers:
            for provider in providers:
                print(f"   ✅ {provider}")
        else:
            print(f"   ❌ No providers configured")
    except Exception as e:
        print(f"   ❌ Error checking providers: {e}")

    return {
        'success': True,
        'command': 'info'
    }

def execute_plugin_command(args) -> Dict[str, Any]:
    """Execute plugin-related commands"""
    if hasattr(args, 'plugin_action'):
        if args.plugin_action == 'list':
            return execute_plugin_list_command()
        elif args.plugin_action == 'discover':
            return execute_plugin_discover_command(args)

    return {'success': False, 'error': 'Unknown plugin command'}

def execute_plugin_list_command() -> Dict[str, Any]:
    """List available plugins"""
    print("🔌 Available Processors (including plugins):")
    list_all_processors()

    return {
        'success': True,
        'command': 'plugin_list'
    }

def execute_plugin_discover_command(args) -> Dict[str, Any]:
    """Discover plugins in directory"""
    plugin_dir = getattr(args, 'plugin_dir', 'plugins')

    print(f"🔍 Discovering plugins in: {plugin_dir}")

    if not Path(plugin_dir).exists():
        print(f"❌ Plugin directory not found: {plugin_dir}")
        return {
            'success': False,
            'command': 'plugin_discover',
            'error': f'Plugin directory not found: {plugin_dir}'
        }

    discover_plugins(plugin_dir)

    print(f"✅ Plugin discovery complete")

    return {
        'success': True,
        'command': 'plugin_discover',
        'plugin_dir': plugin_dir
    }

# Command router
def route_command(args, file_paths: List[str] = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Route command to appropriate handler

    Args:
        args: Command line arguments
        file_paths: List of files (for processing commands)
        config: Processing configuration (for processing commands)

    Returns:
        Command results
    """
    # Special commands that don't need file processing
    if hasattr(args, 'command'):
        if args.command == 'info':
            return execute_info_command(args)
        elif args.command == 'cache':
            return execute_cache_command(args)
        elif args.command == 'plugin':
            return execute_plugin_command(args)

    # Processing commands
    if file_paths is not None and config is not None:
        return execute_processing_command(args, file_paths, config)

    return {
        'success': False,
        'error': 'Invalid command or missing parameters'
    }

# Utility functions for command line tools
def print_command_help():
    """Print help for available commands"""
    help_text = """
🚀 Doc2Train v2.0 Enhanced - Available Commands

PROCESSING COMMANDS:
  python main.py <input> --mode extract-only    # Extract text and images only
  python main.py <input> --mode generate        # Extract + generate training data
  python main.py <input> --mode full            # Complete processing pipeline
  python main.py <input> --validate-only        # Validate input and configuration
  python main.py <input> --benchmark            # Run performance benchmark

CACHE MANAGEMENT:
  python main.py --cache-stats                   # Show cache statistics
  python main.py --cache-cleanup                # Clean up old cache entries
  python main.py --cache-optimize               # Optimize cache storage
  python main.py --cache-clear                  # Clear all cache

SYSTEM INFO:
  python main.py --info                         # Show system information
  python main.py --list-processors              # List available processors
  python main.py --list-providers               # List available LLM providers

PLUGIN MANAGEMENT:
  python main.py --discover-plugins <dir>       # Discover plugins in directory
  python main.py --list-plugins                 # List loaded plugins

EXAMPLES:
  # Basic extraction with progress
  python main.py documents/ --mode extract-only --show-progress

  # Advanced processing with quality control
  python main.py documents/ --mode generate --min-image-size 5000 --threads 8

  # Page range control for PDFs
  python main.py document.pdf --start-page 5 --end-page 50 --skip-pages "1,2,10-15"

  # System validation and benchmarking
  python main.py documents/ --validate-only
  python main.py documents/ --benchmark --test-mode
"""
    print(help_text)

def handle_keyboard_interrupt(results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle keyboard interrupt gracefully"""
    print("\n⚠️ Processing interrupted by user")

    if results and results.get('files_completed', 0) > 0:
        print(f"📊 Partial completion: {results['files_completed']}/{results.get('files_total', 0)} files")
        print(f"💾 Results saved up to interruption point")

    return {
        'success': False,
        'interrupted': True,
        'partial_results': results
    }

def format_command_results(results: Dict[str, Any]) -> str:
    """Format command results for display"""
    if not results.get('success', False):
        if results.get('interrupted'):
            return "⚠️ Processing was interrupted"
        else:
            error = results.get('error', 'Unknown error')
            return f"❌ Command failed: {error}"

    command = results.get('command', 'process')

    if command == 'process':
        files_processed = results.get('files_processed', 0)
        successful = results.get('successful', 0)
        return f"✅ Processed {successful}/{files_processed} files successfully"

    elif command == 'validate':
        return "✅ Validation completed - check report for details"

    elif command == 'benchmark':
        return "✅ Benchmark completed - check results for performance data"

    elif command in ['cache_stats', 'cache_cleanup', 'cache_optimize', 'cache_clear']:
        return f"✅ Cache operation completed"

    elif command == 'info':
        return "✅ System information displayed"

    else:
        return f"✅ Command '{command}' completed successfully"

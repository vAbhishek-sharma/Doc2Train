# cli/commands.py
"""
Complete CLI commands system for Doc2Train Enhanced
Handles all command execution with proper error handling and validation
"""

from datetime import datetime
import time
from typing import Dict, List, Any
from pathlib import Path


from doc2train.core.llm_client import test_provider
from doc2train.core.pipeline import ProcessingPipeline, PerformanceBenchmark, create_processing_pipeline
from doc2train.utils.validation import validate_and_report_system, create_validation_report
from doc2train.utils.cache import get_cache_stats, cleanup_cache, optimize_cache
from doc2train.core.registries.processor_registry import (
    list_all_processors,
    get_supported_extensions_dict as get_processor_supported_exts
)
from doc2train.core.registries.llm_registry import (
    list_llm_plugins,
    get_available_providers,
    get_llm_plugin
)
from doc2train.core.writers import OutputManager

# --- Processing Commands ---

def execute_processing_command(config: Dict[str, Any], file_paths: List[str]) -> Dict[str, Any]:
    """Execute the main processing command with all enhancements"""
    if config.get('validate_only'):
        return execute_validate_command(config, file_paths)

    try:
        pipeline = create_processing_pipeline(config)
        results = pipeline.process_files(file_paths)


        if not config.get('benchmark'):
            OutputManager(config).save_all_results(results, 'final_summary')

        try:
            pipeline.cleanup_after_processing()
        except Exception as e:
            print(f"⚠️ Cache cleanup warning: {e}")

        return results
    except Exception as e:
        return {'success': False, 'error': str(e), 'command': 'process'}


def execute_validate_command(config: Dict[str, Any], file_paths: List[str]) -> Dict[str, Any]:
    """Execute validation-only command"""
    print("🔍 Running comprehensive validation...")

    system_valid = validate_and_report_system()
    validation_report = create_validation_report(file_paths, config)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    report_file = output_dir / 'validation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, default=str)

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
        print("   ✅ Configuration valid")

    print(f"\n📄 Detailed report saved to: {report_file}")

    return {
        'success': system_valid and len(config_errors) == 0,
        'command': 'validate',
        'validation_report': validation_report,
        'system_valid': system_valid
    }


def execute_benchmark_command(config: Dict[str, Any], file_paths: List[str]) -> Dict[str, Any]:
    """Execute performance benchmark command"""
    print("📊 Running performance benchmark...")
    benchmark_files = file_paths[:6]

    try:
        benchmark = PerformanceBenchmark(config)
        benchmark_results = benchmark.process_files(benchmark_files)

        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        benchmark_file = output_dir / 'benchmark_results.json'
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, default=str)

        print(f"\n📄 Benchmark results saved to: {benchmark_file}")
        return {'success': True, 'command': 'benchmark', 'benchmark_results': benchmark_results}
    except Exception as e:
        return {'success': False, 'command': 'benchmark', 'error': str(e)}


def execute_cache_command(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute cache-related commands"""
    action = config.get('cache_action')
    if action == 'stats':
        return execute_cache_stats_command()
    elif action == 'cleanup':
        return execute_cache_cleanup_command(config)
    elif action == 'optimize':
        return execute_cache_optimize_command()
    elif action == 'clear':
        return execute_cache_clear_command(config)
    return {'success': False, 'error': 'Unknown cache command'}


def execute_cache_stats_command() -> Dict[str, Any]:
    """Show cache statistics"""
    print("📊 Cache Statistics:")
    stats = get_cache_stats()
    print(f"   Cache entries: {stats['cache_entries']}")
    print(f"   Total size: {stats['total_size_mb']:.1f} MB")
    if stats['cache_entries'] > 0:
        print(f"   Average size: {stats['avg_size_mb']:.2f} MB per entry")
        print(f"   Compression ratio: {stats['compression_ratio']:.0%}")
        if stats['file_types']:
            print("   File types:")
            for ext, count in stats['file_types'].items():
                print(f"     {ext}: {count} files")
    print(f"   Cache directory: {stats['cache_directory']}")
    print(f"   Max size limit: {stats['max_size_gb']:.1f} GB")
    print(f"   Expiry: {stats['expiry_days']} days")
    return {'success': True, 'command': 'cache_stats', 'stats': stats}


def execute_cache_cleanup_command(config: Dict[str, Any]) -> Dict[str, Any]:
    """User-initiated cache cleanup"""
    return ProcessingPipeline.perform_cache_cleanup(config=config, force_clear_if_needed=False)


def execute_cache_optimize_command() -> Dict[str, Any]:
    """Optimize cache"""
    print("📦 Optimizing cache...")
    optimize_cache()
    return {'success': True, 'command': 'cache_optimize'}


def execute_cache_clear_command(config: Dict[str, Any]) -> Dict[str, Any]:
    """Clear cache"""
    print("🗑️ Clearing cache...")
    from doc2train.utils.cache import clear_cache
    clear_cache(config.get('cache_file', None))
    return {'success': True, 'command': 'cache_clear'}


def execute_info_command(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute info/status commands"""
    print("ℹ️ Doc2Train Information:")

    # System info
    print(f"\n🖥️ System Information:")
    import sys
    print(f"   Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    try:
        import psutil
        mem_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   Available memory: {mem_gb:.1f} GB")
    except ImportError:
        print("   Memory: Unable to detect")

    # Supported extensions
    print("\n📦 Available Processors:")
    for proc, exts in get_processor_supported_exts().items():
        print(f"   {proc}: {', '.join(exts)}")

    # Cache info
    print(f"\n💾 Cache Information:")
    stats = get_cache_stats()
    print(f"   Cache entries: {stats['cache_entries']}")
    print(f"   Cache size: {stats['total_size_mb']:.1f} MB")

    # LLM providers
    print(f"\n🤖 LLM Providers:")
    try:
        providers = get_available_providers()
        if providers:
            print()
            for provider in providers:
                plugin_cls = get_llm_plugin(provider)
                cfg_ok = plugin_cls.configured() if hasattr(plugin_cls, "configured") else True
                live_ok = test_provider(provider)
                status_cfg = "✅" if cfg_ok else "❌"
                status_rch = "✅" if live_ok else "❌"
                types = getattr(plugin_cls, "supported_types", ["text"])
                vision = getattr(plugin_cls, "supports_vision", False)
                types_str = ", ".join(types)
                if vision and "image" not in types:
                    types_str += ", image"
                print(f"   {provider:15}  Configured: {status_cfg}   Reachable: {status_rch}   Types: {types_str}")
        else:
            print("   ❌ No providers found")
    except Exception as e:
        print(f"   ❌ Error checking providers: {e}")

    return {'success': True, 'command': 'info'}


def execute_plugin_list_command() -> Dict[str, Any]:
    """List available plugins"""
    print("🔌 Available Processors (including plugins):")
    list_all_processors()

    print("\n🤖 Available LLM Plugins:")

    for name in list_llm_plugins():
        plugin_cls = get_llm_plugin(name)
        cfg_ok = plugin_cls.configured() if hasattr(plugin_cls, "configured") else True
        live_ok = test_provider(name)
        status_cfg = "✅" if cfg_ok else "❌"
        status_rch = "✅" if live_ok else "❌"
        types = getattr(plugin_cls, "supported_types", ["text"])
        vision = getattr(plugin_cls, "supports_vision", False)
        types_str = ", ".join(types)
        if vision and "image" not in types:
            types_str += ", image"
        print(f"   {name:15}  Configured: {status_cfg}   Reachable: {status_rch}   Types: {types_str}")

    return {'success': True, 'command': 'plugin_list'}

# Command router
def route_command(config: Dict[str, Any], file_paths: List[str] = None) -> Dict[str, Any]:
    """
    Route command to appropriate handler

    Args:
        config: Processing configuration (merged CLI + config file)
        file_paths: List of files (for processing commands)

    Returns:
        Command results
    """
    command = config.get('command')

    # Special commands that don't need file processing
    if command == 'info':
        return execute_info_command(config)
    elif command == 'cache':
        return execute_cache_command(config)

    # Processing commands
    if file_paths is not None:
        return execute_processing_command(config, file_paths)

    return {
        'success': False,
        'error': 'Invalid command or missing parameters'
    }

# Utility functions for command line tools
def print_command_help():
    """Print help for available commands"""
    help_text = """
🚀 Doc2Train Enhanced - Available Commands

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


def execute_list_providers_command() -> Dict[str, Any]:
    """List all available LLM providers"""
    print("🤖 Available LLM Providers:")

    try:
        providers = get_available_providers()

        if providers:
            print()
            for provider in providers:
                plugin_cls = get_llm_plugin(provider)
                supported_types = plugin_cls.get("supported_types", ["text"])
                vision = plugin_cls.get("supports_vision", False)
                types_str = ", ".join(supported_types)
                if vision and "image" not in supported_types:
                    types_str += ", image"
                status = "✅" if getattr(plugin_cls, "configured", lambda: True)() else "❌"
                print(f"   {status} {provider}: {types_str}")
        else:
            print("   No providers available")

    except Exception as e:
        print(f"   ❌ Error listing providers: {e}")

    return {'success': True, 'command': 'list_providers'}

def execute_direct_media_command(args) -> Dict[str, Any]:
    """Execute direct media processing command"""
    input_path = args.input_path

    if not Path(input_path).exists():
        return {'success': False, 'error': f'Input file not found: {input_path}'}

    # Check if it's a supported media file
    supported_media = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp', '.mp4', '.avi', '.mov']
    if not any(input_path.lower().endswith(ext) for ext in supported_media):
        return {'success': False, 'error': f'Unsupported media format: {input_path}'}

    print(f"🎬 Processing media directly: {input_path}")

    try:
        from doc2train.core.plugin_managers.llm_plugin_manager import process_media_directly

        provider = getattr(args, 'provider', None)
        prompt = getattr(args, 'media_prompt', None)

        result = process_media_directly(
            media_path=input_path,
            provider=provider,
            prompt=prompt
        )

        print(f"\n📄 Analysis Result:")
        print(f"{result}\n")

        # Save result if output specified
        if hasattr(args, 'output_dir') and args.output_dir:
            output_file = Path(args.output_dir) / f"{Path(input_path).stem}_analysis.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Direct Media Analysis\n")
                f.write(f"File: {input_path}\n")
                f.write(f"Provider: {provider or 'auto-detected'}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                f.write(result)

            print(f"💾 Result saved to: {output_file}")

        return {
            'success': True,
            'command': 'direct_media',
            'result': result,
            'input_file': input_path
        }

    except Exception as e:
        print(f"❌ Direct media processing failed: {e}")
        return {'success': False, 'error': str(e)}

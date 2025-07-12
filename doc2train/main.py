#!/usr/bin/env python3
# main.py - Doc2Train v2.0 Enhanced Complete Implementation
"""
Doc2Train v2.0 Enhanced - Enterprise Document Processing
Convert documents to AI training data at god speed!

Complete implementation with all enhanced features:
- Real-time progress tracking with ETA
- Page range control and skip functionality
- Multi-threading for performance
- Smart content filtering and quality control
- Plugin architecture for extensibility
- Fault-tolerant processing with per-file saving
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from doc2train.core.llm_client import estimate_cost, estimate_tokens
from doc2train.utils.config_loader import get_config_loader, validate_config
from doc2train.cli.commands import  execute_direct_media_command, execute_list_providers_command, route_command
from doc2train.core.plugin_setup import set_plugins
from doc2train.cli.commands import execute_list_providers_command, execute_info_command, route_command


# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import core functionality
    from doc2train.utils.files import get_supported_files
    from doc2train.core.registries.processor_registry import get_processor_for_file, _PROCESSOR_REGISTRY
    from doc2train.core.pipeline import ProcessingPipeline

    # Import utilities
    from doc2train.utils.progress import ProgressTracker, ProgressDisplay
    from doc2train.utils.validation import validate_input_and_files
    from doc2train.utils.cache import CacheManager
    from doc2train.utils.config import *  # Import all config variables
    from doc2train.cli.commands import execute_processing_command
    # Import CLI components
    from doc2train.cli.args import create_enhanced_parser, parse_skip_pages, args_to_config

    # Import output handling
    from doc2train.core.writers import OutputWriter

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in place")
    print("Missing dependencies? Try: pip install -r requirements.txt")
    sys.exit(1)

# Global progress tracking
progress_tracker = ProgressTracker()
progress_display = ProgressDisplay()

def print_banner():
    """Print enhanced Doc2Train banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                       Doc2Train v2.0 Enhanced                    ║
║               🚀 Enterprise Document Processing 🚀              ║
║          Real-time • Parallel • Fault-tolerant • Smart           ║
║                     Plugin Architecture Ready                    ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Enhanced main entry point with YAML config support and checkpoint resume"""
    # Print banner
    print_banner()

    # Parse enhanced arguments
    parser = create_enhanced_parser()
    args = parser.parse_args()

    # 2. Handle config-related utility commands

    try:

        if getattr(args, 'show_config', False):
            show_current_config()
            return 0

        if getattr(args, 'save_config', False):
            save_config_from_args(args)
            return 0


        # Validate arguments

        #merge args and config file into single config param
        config_file = getattr(args, 'config_file', 'config.yaml')
        if config_file and Path(config_file).exists():
            config_loader = get_config_loader(config_file)
            config_loader.update_from_args(args)
            config = config_loader.get_processing_config()
            print(f"⚙️ Using config: {config_file}")
        else:
            config = args_to_config(args)

        # 1) Validate the merged CLI+YAML config
        validate_config(config)
        # 2) Discover & register **all** plugins (LLM, Processor, Writer, Formatter)
        set_plugins(config)

        # 3) Now handle any plugin-related commands (list, discover, etc.)
        handle_plugin_commands(config)
        # 3. Handle resume logic (may need merged config)
        if config.get('resume_from_checkpoint'):
            return resume_from_checkpoint(config['resume_from_checkpoint'], config)


        # Show async/sync mode
        if config.get('use_async', True):
            concurrent_calls = config.get('max_concurrent_calls', 5)
            print(f"🚀 Async mode: {concurrent_calls} concurrent LLM calls")
        else:
            print(f"📋 Sync mode: Sequential LLM processing")

        # Show auto-stop settings if enabled
        auto_stop_settings = []
        if config.get('auto_stop_on_quota_exceeded', True):
            auto_stop_settings.append("quota exceeded")
        if config.get('auto_stop_on_consecutive_failures'):
            auto_stop_settings.append(f"{config['auto_stop_on_consecutive_failures']} consecutive failures")
        if config.get('auto_stop_after_time'):
            auto_stop_settings.append(f"{config['auto_stop_after_time']} minutes")
        if config.get('auto_stop_after_files'):
            auto_stop_settings.append(f"{config['auto_stop_after_files']} files")

        if auto_stop_settings:
            print(f"⏸️  Auto-stop enabled: {', '.join(auto_stop_settings)}")

        #To rename later
        # Validate input and get files (keep existing logic)
        # TO DO: Need to confirm if its holds value here in the flow?
        # if not validate_input_and_files(config):
        #     return 1

        execute_info_command(config)
        # Get supported files
        #TO UPDATE: To add support for files from plugin processor folder
        supported_files = get_supported_files(config.get('input_path'))
        if not supported_files:
            print(f"❌ Error: No supported files found in '{config.get('input_path')}'")
            print(f"Supported extensions: {supported_files}")
            return 1

        # Set input path in config

        # Limit files in test mode
        if config.get('test_mode'):
            supported_files = supported_files[:TEST_MAX_FILES]
            print(f"🧪 Test mode: Processing only {len(supported_files)} files")

        # Show processing plan
        show_processing_plan(config, supported_files)

        # Perform dry run if requested
        if config.get('dry_run'):
            perform_dry_run(supported_files, config)
            return 0

        # Confirm processing (unless in test mode)
        if not config.get('test_mode') and not config.get('show_progress'):
            response = input("🚀 Ready to start processing? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                print("Processing cancelled.")
                return 0


        # Initialize progress tracking
        progress_tracker.initialize(len(supported_files))
        progress_display.set_show_progress(config.get('show_progress', True))

        results = route_command(config, supported_files)
        if not results.get("success"):
            print(f"\n❌ Error encountered during processing: {results.get('error')}")
            if config.get('verbose', False):
                print(f"\n📄 Traceback:\n{results.get('traceback', 'No traceback available')}")
            if 'errors' in results:
                for err in results['errors']:
                    print(f"\n🚩 File: {err.get('file')}\nError: {err.get('error')}")

        #  Check if processing was auto-stopped
        if results.get('auto_stopped', False):
            print(f"\n⏸️  Processing auto-stopped: {results.get('stop_reason', 'Unknown reason')}")
            checkpoint_file = Path(config['output_dir']) / 'checkpoint.json'
            if checkpoint_file.exists():
                print(f"💾 Checkpoint saved: {checkpoint_file}")
                print(f"🔄 To continue: python main.py --resume-from-checkpoint {checkpoint_file}")

            # Show partial results
            show_results(results, config )
            return 2  # Special exit code for auto-stop
        else:
            # Show enhanced results
            show_results(results, config)
            return 0 if results.get('success', True) else 1

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")

        #  Save checkpoint on manual interruption
        try:
            checkpoint_data = {
                'timestamp': time.time(),
                'stop_reason': 'User interruption',
                'interrupted': True,
                'config': config if 'config' in locals() else args_to_config(args)
            }

            output_dir = config.get('output_dir')
            checkpoint_file = Path(output_dir) / 'interrupted_checkpoint.json'

            with open(checkpoint_file, 'w') as f:
                import json
                json.dump(checkpoint_data, f, indent=2, default=str)

            print(f"💾 Interruption checkpoint saved: {checkpoint_file}")

        except Exception as e:
            print(f"⚠️  Could not save interruption checkpoint: {e}")

        # Show partial results if any
        if progress_tracker.get_completed_count() > 0:
            print(f"📊 Partial completion: {progress_tracker.get_completed_count()}/{progress_tracker.get_total_count()} files")
        return 130
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if config.get('verbose', False) if 'config' in locals() else getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1



def show_processing_plan(config: Dict, files: List[str]):
    """Show processing plan for all dataset domains (text, media, audio, etc.)"""

    print(f"\n📋 Processing Plan:")
    print(f"   Mode:    {config.get('mode', 'N/A')}")
    print(f"   Files:   {len(files)}")
    print(f"   Threads: {config.get('threads', 'N/A')}")
    print(f"   Output:  {config.get('output_dir', 'output')}")

    # — Page range & skips —
    start_page = config.get('start_page', 1)
    end_page   = config.get('end_page')
    if start_page > 1 or end_page:
        pr = f"{start_page}-{end_page or 'end'}"
        print(f"   Page range: {pr}")
    if skips := config.get('skip_pages', []):
        print(f"   Skip pages: {', '.join(map(str, skips))}")

    # — Quality filters —
    qf = []
    if config.get('min_image_size', 1000) > 1000:
        qf.append(f"images ≥{config['min_image_size']}px")
    if config.get('min_text_length', 100) > 100:
        qf.append(f"text ≥{config['min_text_length']} chars")
    if config.get('skip_single_color_images'):
        qf.append("skip solid-color images")
    if regex := config.get('header_regex'):
        qf.append(f"header regex: {regex}")
    if qf:
        print(f"   Quality filters: {', '.join(qf)}")

    # — Performance options —
    po = []
    for opt, label in [
        ('save_per_file',      "save per file"),
        ('show_progress',      "real-time progress"),
        ('save_images',        "save images"),
    ]:
        if config.get(opt):
            po.append(label)
    if po:
        print(f"   Options: {', '.join(po)}")

    # — Dataset domains —
    print("\n🚀 Dataset configuration summary:")
    dataset_cfg = config.get('dataset', {})

    for domain, dc in dataset_cfg.items():
        gens = dc.get('generators', [])
        fmts = dc.get('formatters', [])
        print(f"\n • {domain.capitalize()} dataset:")
        if gens:
            print(f"    • Generators: {', '.join(gens)}")
        if fmts:
            print(f"    • Formatters: {', '.join(fmts)}")

        # text-specific details
        if domain == 'text':
            cs  = dc.get('chunk_size')
            ov  = dc.get('overlap')
            if cs and ov is not None:
                print(f"    • Chunk size/overlap: {cs}/{ov}")

            # cost estimate for text
            try:
                total_bytes = sum(Path(f).stat().st_size for f in files)
                approx_tokens = total_bytes // 2  # rough bytes→tokens
                tokens = estimate_tokens(" " * approx_tokens, model=config['llm']['model'])
                cost = estimate_cost(
                    text=" " * approx_tokens,
                    provider=config['llm']['provider'],
                    model=config['llm']['model'],
                    output_tokens=approx_tokens
                )
                print(f"    • Estimated tokens: ~{tokens}")
                print(f"    • Estimated text cost: ${cost:.4f}")
            except Exception:
                print("    • Cost estimate: unavailable")

    # vision flag (if still relevant)
    if config.get('include_vision'):
        print("\n • Vision processing: enabled")

    print()  # blank line


#To update  : Need to use the processor registry and add estimated file processing time to processors plugins
def perform_dry_run(files: List[str], config: Dict):
    """Enhanced dry run with detailed analysis"""
    print("🔍 Enhanced Dry Run - Detailed Analysis:\n")
    total_size = 0
    processing_time_estimate = 0

    for file_path in files:
        path = Path(file_path)
        size_mb = path.stat().st_size / (1024 * 1024)
        total_size += size_mb

        # Get processor info
        try:
            processor = get_processor_for_file(file_path)
            processor_name = processor.processor_name
        except:
            processor_name = "Unknown"

        # Estimate processing time
        if path.suffix.lower() == '.pdf':
            estimated_time = size_mb * 2.0  # PDF processing is complex
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            estimated_time = size_mb * 0.5  # Images are faster
        else:
            estimated_time = size_mb * 0.1  # Text files are fastest

        processing_time_estimate += estimated_time

        print(f"  📄 {path.name}")
        print(f"     Size: {size_mb:.1f} MB")
        print(f"     Processor: {processor_name}")
        print(f"     Est. time: {estimated_time:.1f}s")

        # Show what would be extracted
        if config.get('start_page', 1) > 1 or config.get('end_page'):
            print(f"     Pages: {config.get('start_page', 1)}-{config.get('end_page', 'end')}")
        if config.get('skip_pages'):
            print(f"     Skip: {config['skip_pages']}")

        print()

    # Apply threading speedup estimate
    threads = config.get('threads', 1)
    if threads > 1:
        speedup_factor = min(threads, 4) * 0.7  # Diminishing returns
        processing_time_estimate /= speedup_factor
        print(f"🧵 Threading speedup applied: {speedup_factor:.1f}x")

    print(f"📊 Summary:")
    print(f"   Total: {len(files)} files, {total_size:.1f} MB")
    print(f"   Estimated time: {processing_time_estimate:.1f}s ({processing_time_estimate/60:.1f} minutes)")
    print(f"   Threads: {threads}")
    print("Use without --dry-run to actually process files")

def show_results(results: Dict[str, Any], config: Dict):
    """Display enhanced processing results"""

    print(f"\n🎉 Processing completed!")

    # Timing summary
    if 'total_processing_time' in results:
        total_time = results['total_processing_time']
        print(f"⏱️ Total time: {format_time(total_time)}")

    # Basic file summary
    print(f"📁 Output directory: {results.get('output_dir', config.get('output_dir', 'output'))}")
    print(f"📄 Files processed: {results.get('files_processed', 0)}")
    print(f"✅ Successful: {results.get('successful', 0)}")

    if results.get('failed', 0) > 0:
        print(f"❌ Failed: {results['failed']}")

    # Performance metrics
    if results.get('successful', 0) > 0 and 'total_processing_time' in results:
        avg_time = results['total_processing_time'] / results['files_processed']
        print(f"📊 Average time per file: {avg_time:.1f}s")

        if config.get('threads', 1) > 1:
            sequential_estimate = avg_time * results['files_processed']
            speedup = sequential_estimate / results['total_processing_time']
            print(f"🚀 Threading speedup: {speedup:.1f}x")

    # Content stats
    if results.get('total_text_chars', 0) > 0:
        print(f"\n📝 Content Statistics:")
        print(f"   Text extracted: {results['total_text_chars']:,} characters")
        print(f"   Images extracted: {results.get('total_images', 0):,}")

        if results.get('successful', 0) > 0:
            avg_text_per_file = results['total_text_chars'] / results['successful']
            avg_images_per_file = results.get('total_images', 0) / results['successful']
            print(f"   Average per file: {avg_text_per_file:,.0f} chars, {avg_images_per_file:.1f} images")

    # Configuration summary
    print(f"\n⚙️ Configuration Used:")
    print(f"   Threads: {config.get('threads', 'N/A')}")

    start_page = config.get('start_page', 1)
    end_page = config.get('end_page')
    if start_page > 1 or end_page:
        page_range = f"{start_page}-{end_page or 'end'}"
        print(f"   Page range: {page_range}")

    skip_pages = config.get('skip_pages', [])
    if skip_pages:
        print(f"   Skipped pages: {', '.join(map(str, skip_pages))}")

    print(f"   Quality filters: min_image={config.get('min_image_size', 1000)}px, min_text={config.get('min_text_length', 100)} chars")

    if config.get('save_per_file'):
        print(f"   ✅ Per-file saving enabled (fault-tolerant)")

    # Recent errors
    if results.get('errors'):
        print(f"\n⚠️ Errors encountered:")
        for error in results['errors'][-3:]:  # Show last 3 errors
            print(f"   ❌ {error.get('file', 'Unknown')}: {error.get('error', 'Unknown error')}")

    # Next steps
    print(f"\n💡 Next Steps:")
    mode = config.get('mode', 'extract-only')
    if mode == 'extract-only':
        print("   - Review extracted text files")
        print("   - Run with --mode generate to create training data")
    else:
        print("   - Review generated training data")
        print("   - Use for fine-tuning your AI models")
        print("   - Consider adjusting quality thresholds if needed")

    if config.get('save_per_file'):
        print(f"   - Individual file results saved in: {config.get('output_dir', 'output')}/per_file/")

    print(f"\n📖 Documentation: Check README.md for advanced usage examples")

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

#To refactor  # to add replace it inside the pipeline as its should be mode
def resume_from_checkpoint(checkpoint_file: str, args) -> int:
    """Resume processing from a checkpoint file"""
    try:
        if not Path(checkpoint_file).exists():
            print(f"❌ Checkpoint file not found: {checkpoint_file}")
            return 1

        with open(checkpoint_file, 'r') as f:
            import json
            checkpoint_data = json.load(f)

        print(f"🔄 Resuming from checkpoint: {checkpoint_file}")
        print(f"📊 Previous stop reason: {checkpoint_data.get('stop_reason', 'Unknown')}")

        # Handle different checkpoint types
        if 'remaining_files' in checkpoint_data:
            remaining_files = checkpoint_data['remaining_files']
            print(f"📄 Remaining files: {len(remaining_files)}")

            if not remaining_files:
                print("✅ All files already processed!")
                return 0

            # Continue processing with remaining files
            config = checkpoint_data.get('config', args_to_config(args))

            # Update args input_path to process remaining files
            # Create temporary file list or use the remaining files directly
            temp_dir = Path('temp_resume')
            temp_dir.mkdir(exist_ok=True)

            # For simplicity, just process the remaining files directly
            pipeline = ProcessingPipeline(config)
            results = pipeline.process_files(remaining_files, args)

            show_results(results, args, config)

            # Clean up checkpoint file on successful completion
            if results.get('success', True) and not results.get('auto_stopped', False):
                try:
                    Path(checkpoint_file).unlink()
                    print(f"🗑️  Checkpoint file cleaned up: {checkpoint_file}")
                except:
                    pass

            return 0 if results.get('success', True) else 1

        elif checkpoint_data.get('interrupted', False):
            print("📄 Resuming from user interruption...")
            print("ℹ️  Use normal processing command to restart from the beginning")
            print("   or use --mode resume to continue from cache")
            return 0

        else:
            print("⚠️  Unknown checkpoint format")
            return 1

    except Exception as e:
        print(f"❌ Error resuming from checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1

#TO refactor
#  Add config display function
def show_current_config():
    """Show current configuration"""
    try:
        config_loader = get_config_loader()
        config = config_loader.get_processing_config()

        print("⚙️  Current Configuration:")
        print("=" * 50)

        sections = {
            'Basic Settings': ['mode', 'output_dir', 'output_format'],
            'Processing': ['use_async', 'threads', 'max_workers', 'batch_size', 'use_cache'],
            'Quality Control': ['min_image_size', 'min_text_length', 'quality_threshold'],
            'Features': ['extract_images', 'use_ocr', 'include_vision', 'use_smart_analysis'],
            'LLM Settings': ['provider', 'model', 'max_concurrent_calls'],
            'Auto-Stop': ['auto_stop_on_quota_exceeded', 'auto_stop_on_consecutive_failures', 'auto_stop_after_time']
        }

        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                value = config.get(key, 'Not set')
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error displaying config: {e}")

#TO refactor
#  Add config saving function
def save_config_from_args(args):
    """Save current args as config file"""
    try:
        config = args_to_config(args)
        config_loader = get_config_loader()

        # Update config with args
        for key, value in config.items():
            config_loader.set(key, value)

        config_loader.save_config()
        print(f"✅ Configuration saved to: {config_loader.config_file}")

    except Exception as e:
        print(f"❌ Error saving config: {e}")


def handle_plugin_commands(config) -> bool:
    """
     Handle plugin-related commands

    Returns:config.get('use_async', True)
        True if a plugin command was executed (should exit)
    """

    if config.get('list_providers', True):

        execute_list_providers_command()
        return True

    #TO UPDATE: to print list of plugins
    # List plugins
    if config.get('list_plugins', True):

        from doc2train.core.registries.llm_registry import list_llm_plugins
        list_llm_plugins()
        return True

    # TO UPDATE shift the method to commands.py
    # Provider capabilities
    if config.get('provider_capabilities', True):

        execute_provider_capabilities_command()
        return True

    return False

#to remove
def handle_direct_media_processing(args) -> bool:
    """
     Handle direct media processing

    Returns:
        True if direct media processing was executed (should exit)
    """
    if hasattr(args, 'direct_media') and args.direct_media:
        result = execute_direct_media_command(args)
        if not result['success']:
            print(f"❌ {result['error']}")
            sys.exit(1)
        return True

    return False


def execute_provider_capabilities_command() -> Dict[str, Any]:
    """Show detailed capabilities of all providers"""
    print("🔍 Provider Capabilities:")

    try:
        from core.llm_client import get_available_providers
        from doc2train.core.plugin_managers.llm_plugin_manager import get_plugin_manager

        providers = get_available_providers()
        plugin_manager = get_plugin_manager()

        for provider in providers:
            print(f"\n📊 {provider.upper()}:")

            # Get detailed info
            if provider in ['openai', 'deepseek', 'local']:
                # Built-in provider
                from config.settings import LLM_PROVIDERS
                config = LLM_PROVIDERS.get(provider, {})
                models = config.get('models', {})

                print(f"   Type: Built-in")
                print(f"   Text Model: {models.get('text', 'N/A')}")
                print(f"   Vision Model: {models.get('vision', 'N/A')}")
                print(f"   Configured: {'✅' if config.get('api_key') else '❌'}")
            else:
                # Plugin provider
                info = plugin_manager.get_provider_info(provider)
                if info:
                    print(f"   Type: Plugin")
                    print(f"   Text Support: {'✅' if info['capabilities']['text'] else '❌'}")
                    print(f"   Vision Support: {'✅' if info['capabilities']['vision'] else '❌'}")
                    print(f"   Streaming: {'✅' if info['capabilities']['streaming'] else '❌'}")
                    print(f"   Configured: {'✅' if info['config_valid'] else '❌'}")

                    if info['models']:
                        print(f"   Available Models:")
                        for model, details in info['models'].items():
                            print(f"     - {model} ({details.get('type', 'text')})")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    return {'success': True, 'command': 'provider_capabilities'}



if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


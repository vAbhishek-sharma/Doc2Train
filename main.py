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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import core functionality
    from utils.files import get_supported_files
    from processors import get_processor_for_file, get_supported_extensions
    from core.pipeline import ProcessingPipeline

    # Import utilities
    from utils.progress import ProgressTracker, ProgressDisplay
    from utils.validation import validate_input_enhanced
    from utils.cache import CacheManager
    from utils.config import *  # Import all config variables

    # Import CLI components
    from cli.args import create_enhanced_parser, parse_skip_pages, args_to_config, validate_args_enhanced

    # Import output handling
    from outputs.writers import OutputWriter

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
║                       Doc2Train v2.0 Enhanced                   ║
║               🚀 Enterprise Document Processing 🚀              ║
║          Real-time • Parallel • Fault-tolerant • Smart          ║
║                     Plugin Architecture Ready                    ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Enhanced main entry point with complete feature set"""

    # Print banner
    print_banner()

    # Parse enhanced arguments
    parser = create_enhanced_parser()
    args = parser.parse_args()

    try:
        # Validate arguments
        validate_args_enhanced(args)

        # Validate input and get files
        if not validate_input_enhanced(args):
            return 1

        # Get supported files
        supported_files = get_supported_files(args.input_path)
        if not supported_files:
            print(f"❌ Error: No supported files found in '{args.input_path}'")
            print(f"Supported extensions: {', '.join(get_supported_extensions())}")
            return 1

        # Apply configuration overrides
        apply_config_overrides(args)

        # Convert args to processing config
        config = args_to_config(args)

        # Limit files in test mode
        if args.test_mode:
            supported_files = supported_files[:TEST_MAX_FILES]
            print(f"🧪 Test mode: Processing only {len(supported_files)} files")

        # Show processing plan
        show_processing_plan_enhanced(args, supported_files, config)

        # Perform dry run if requested
        if args.dry_run:
            perform_enhanced_dry_run(supported_files, config)
            return 0

        # Confirm processing (unless in test mode)
        if not args.test_mode and not args.show_progress:
            response = input("🚀 Ready to start processing? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                print("Processing cancelled.")
                return 0

        # Initialize progress tracking
        progress_tracker.initialize(len(supported_files))
        progress_display.set_show_progress(args.show_progress)

        # Execute enhanced processing
        pipeline = ProcessingPipeline(config)
        results = pipeline.process_files(supported_files, args)

        # Show enhanced results
        show_results_enhanced(results, args, config)

        return 0 if results.get('success', True) else 1

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")

        # Show partial results if any
        if progress_tracker.get_completed_count() > 0:
            print(f"📊 Partial completion: {progress_tracker.get_completed_count()}/{progress_tracker.get_total_count()} files")
            if args.save_per_file:
                print(f"💾 Completed files saved in: {args.output_dir}/per_file/")

        return 130
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def show_processing_plan_enhanced(args, files: List[str], config: Dict):
    """Show enhanced processing plan"""

    print(f"\n📋 Enhanced Processing Plan:")
    print(f"   Mode: {args.mode}")
    print(f"   Files: {len(files)}")
    print(f"   Threads: {args.threads}")
    print(f"   Output: {args.output_dir}")

    # Page control
    if args.start_page > 1 or args.end_page:
        page_range = f"{args.start_page}"
        if args.end_page:
            page_range += f"-{args.end_page}"
        else:
            page_range += "-end"
        print(f"   Page range: {page_range}")

    if args.skip_pages:
        print(f"   Skip pages: {args.skip_pages}")

    # Quality filters
    quality_filters = []
    if args.min_image_size > 1000:
        quality_filters.append(f"images ≥{args.min_image_size}px")
    if args.min_text_length > MIN_TEXT_LENGTH:
        quality_filters.append(f"text ≥{args.min_text_length} chars")
    if args.skip_single_color:
        quality_filters.append("skip solid colors")
    if args.header_regex:
        quality_filters.append(f"header regex: {args.header_regex}")

    if quality_filters:
        print(f"   Quality filters: {', '.join(quality_filters)}")

    # Performance options
    performance_opts = []
    if args.save_per_file:
        performance_opts.append("save per file")
    if args.show_progress:
        performance_opts.append("real-time progress")
    if args.show_images:
        performance_opts.append("show images")

    if performance_opts:
        print(f"   Options: {', '.join(performance_opts)}")

    if args.mode != 'extract-only':
        print(f"   Generators: {', '.join(args.type)}")

        # Estimate costs
        try:
            from core.llm_client import estimate_cost
            total_size = sum(Path(f).stat().st_size for f in files)
            total_text = total_size // 2  # Rough text estimate
            estimated_cost = estimate_cost(" " * total_text, 'general')
            print(f"   Estimated cost: ${estimated_cost:.4f}")
        except:
            pass

    if args.include_vision:
        print(f"   Vision processing: enabled")

    print()

def perform_enhanced_dry_run(files: List[str], config: Dict):
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

def show_results_enhanced(results: Dict[str, Any], args, config: Dict):
    """Display enhanced processing results"""

    print(f"\n🎉 Processing completed!")

    if 'total_processing_time' in results:
        total_time = results['total_processing_time']
        print(f"⏱️ Total time: {format_time(total_time)}")

    print(f"📁 Output directory: {results.get('output_dir', args.output_dir)}")
    print(f"📄 Files processed: {results.get('files_processed', 0)}")
    print(f"✅ Successful: {results.get('successful', 0)}")

    if results.get('failed', 0) > 0:
        print(f"❌ Failed: {results['failed']}")

    # Performance metrics
    if results.get('successful', 0) > 0 and 'total_processing_time' in results:
        avg_time = results['total_processing_time'] / results['files_processed']
        print(f"📊 Average time per file: {avg_time:.1f}s")

        if args.threads > 1:
            sequential_estimate = avg_time * results['files_processed']
            speedup = sequential_estimate / results['total_processing_time']
            print(f"🚀 Threading speedup: {speedup:.1f}x")

    # Content statistics
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
    print(f"   Threads: {args.threads}")
    if hasattr(args, 'start_page') and (args.start_page > 1 or args.end_page):
        page_range = f"{args.start_page}-{args.end_page or 'end'}"
        print(f"   Page range: {page_range}")
    if hasattr(args, 'skip_pages') and args.skip_pages:
        print(f"   Skipped pages: {args.skip_pages}")

    print(f"   Quality filters: min_image={args.min_image_size}px, min_text={args.min_text_length} chars")

    if args.save_per_file:
        print(f"   ✅ Per-file saving enabled (fault-tolerant)")

    # Show errors if any
    if results.get('errors'):
        print(f"\n⚠️ Errors encountered:")
        for error in results['errors'][-3:]:  # Show last 3 errors
            print(f"   ❌ {error.get('file', 'Unknown')}: {error.get('error', 'Unknown error')}")

    # Show next steps
    print(f"\n💡 Next Steps:")
    if args.mode == 'extract-only':
        print("   - Review extracted text files")
        print("   - Run with --mode generate to create training data")
    else:
        print("   - Review generated training data")
        print("   - Use for fine-tuning your AI models")
        print("   - Consider adjusting quality thresholds if needed")

    if args.save_per_file:
        print(f"   - Individual file results saved in: {args.output_dir}/per_file/")

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

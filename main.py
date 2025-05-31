#!/usr/bin/env python3
# main.py
"""
Doc2Train v2.0 - Simple Main Interface
Convert documents to AI training data at god speed!

Usage:
    python main.py docs/ --mode extract-only
    python main.py file.pdf --mode generate --type conversations
    python main.py docs/ --mode full --include-vision
    python main.py docs/ --resume
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.extractor import extract_content, extract_batch, get_supported_files
    from core.generator import generate_training_data, generate_batch
    from core.processor import process_files, save_results, get_processing_summary
    from config.settings import *
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in place")
    sys.exit(1)

def print_banner():
    """Print Doc2Train banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                          Doc2Train v2.0                         ║
║               🚀 Documents to AI Training Data 🚀               ║
║                      Simple • Fast • Scalable                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Doc2Train v2.0 - Convert documents to AI training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract text only (no LLM costs):
    python main.py documents/ --mode extract-only

  Generate conversations:
    python main.py documents/ --mode generate --type conversations

  Full processing with images:
    python main.py documents/ --mode full --include-vision

  Resume failed processing:
    python main.py documents/ --resume

  Test with small sample:
    python main.py documents/ --test-mode
        """
    )

    # Required argument
    parser.add_argument(
        'input_path',
        help='File or directory to process'
    )

    # Processing mode
    parser.add_argument(
        '--mode',
        choices=['extract-only', 'generate', 'full', 'resume'],
        default='extract-only',
        help='Processing mode (default: extract-only)'
    )

    # Generation types
    parser.add_argument(
        '--type',
        nargs='+',
        choices=['conversations', 'embeddings', 'qa_pairs', 'summaries'],
        default=['conversations'],
        help='Types of training data to generate (default: conversations)'
    )

    # Feature flags
    parser.add_argument(
        '--include-vision',
        action='store_true',
        help='Process images with vision LLMs'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip caching (process everything fresh)'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Process only a small sample for testing'
    )

    # Configuration overrides
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=CHUNK_SIZE,
        help=f'Text chunk size (default: {CHUNK_SIZE})'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=MAX_WORKERS,
        help=f'Max parallel workers (default: {MAX_WORKERS})'
    )

    parser.add_argument(
        '--provider',
        choices=['openai', 'deepseek', 'local'],
        help='LLM provider to use'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--format',
        choices=OUTPUT_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f'Output format (default: {DEFAULT_OUTPUT_FORMAT})'
    )

    # Debugging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without processing'
    )

    return parser.parse_args()

def validate_input(args) -> bool:
    """Validate input arguments and environment"""

    # Check if input path exists
    if not Path(args.input_path).exists():
        print(f"❌ Error: Path '{args.input_path}' does not exist")
        return False

    # Check for supported files
    supported_files = get_supported_files(args.input_path)
    if not supported_files:
        print(f"❌ Error: No supported files found in '{args.input_path}'")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}")
        return False

    print(f"✅ Found {len(supported_files)} supported files")

    # Check API configuration for LLM modes
    if args.mode in ['generate', 'full']:
        from core.llm_client import get_available_providers

        providers = get_available_providers()
        if not providers:
            print("❌ Error: No LLM providers configured")
            print("Set up API keys in .env file for LLM processing")
            return False

        print(f"✅ Available LLM providers: {', '.join(providers)}")

    return True

def apply_config_overrides(args):
    """Apply command line argument overrides to config"""
    global CHUNK_SIZE, MAX_WORKERS, OUTPUT_DIR, USE_CACHE, TEST_MODE, VERBOSE

    # Update global config with command line args
    CHUNK_SIZE = args.chunk_size
    MAX_WORKERS = args.max_workers
    OUTPUT_DIR = args.output_dir
    USE_CACHE = not args.no_cache
    TEST_MODE = args.test_mode
    VERBOSE = args.verbose

    # Set provider override
    if args.provider:
        os.environ['DEFAULT_PROVIDER'] = args.provider

    # Enable test mode if requested
    if args.test_mode:
        os.environ['TEST_MODE'] = 'true'
        print("🧪 Test mode enabled - processing small sample only")

def show_processing_plan(args, files: List[str]):
    """Show what will be processed"""

    print(f"\n📋 Processing Plan:")
    print(f"   Mode: {args.mode}")
    print(f"   Files: {len(files)}")
    print(f"   Output: {args.output_dir}")

    if args.mode != 'extract-only':
        print(f"   Generators: {', '.join(args.type)}")

        # Estimate costs
        from core.llm_client import estimate_cost

        total_size = sum(Path(f).stat().st_size for f in files)
        total_text = total_size // 2  # Rough text estimate

        estimated_cost = estimate_cost(" " * total_text, 'general')
        print(f"   Estimated cost: ${estimated_cost:.4f}")

    if args.include_vision:
        print(f"   Vision processing: enabled")

    print()

def perform_dry_run(files: List[str]):
    """Show what would be processed in a dry run"""
    print("🔍 Dry Run - Files that would be processed:\n")

    total_size = 0
    for file_path in files:
        path = Path(file_path)
        size_mb = path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        processor = get_processor_for_file(file_path)

        print(f"  📄 {path.name}")
        print(f"     Size: {size_mb:.1f} MB")
        print(f"     Processor: {processor}")
        print()

    print(f"📊 Total: {len(files)} files, {total_size:.1f} MB")
    print("Use without --dry-run to actually process files")

def main():
    """Main entry point"""

    # Print banner
    print_banner()

    # Parse arguments
    args = parse_arguments()

    # Validate input
    if not validate_input(args):
        return 1

    # Apply configuration overrides
    apply_config_overrides(args)

    # Get files to process
    supported_files = get_supported_files(args.input_path)

    if args.test_mode:
        # Limit files in test mode
        supported_files = supported_files[:TEST_MAX_FILES]
        print(f"🧪 Test mode: Processing only {len(supported_files)} files")

    # Show processing plan
    show_processing_plan(args, supported_files)

    # Perform dry run if requested
    if args.dry_run:
        perform_dry_run(supported_files)
        return 0

    # Confirm processing (unless in test mode)
    if not args.test_mode:
        response = input("🚀 Ready to start processing? [Y/n]: ").strip().lower()
        if response in ['n', 'no']:
            print("Processing cancelled.")
            return 0

    # Start processing
    start_time = time.time()
    print("🚀 Starting Doc2Train processing...\n")

    try:
        if args.mode == 'extract-only':
            results = process_extract_only(supported_files, args)
        elif args.mode == 'generate':
            results = process_generate(supported_files, args)
        elif args.mode == 'full':
            results = process_full(supported_files, args)
        elif args.mode == 'resume':
            results = process_resume(supported_files, args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Show results
        show_results(results, processing_time, args)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def process_extract_only(files: List[str], args) -> Dict[str, Any]:
    """Process files in extract-only mode"""

    print("📄 Extracting content from files...")

    # Extract content from all files
    extracted_data = extract_batch(files, use_cache=USE_CACHE)

    # Save extracted content
    output_dir = Path(args.output_dir) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'mode': 'extract-only',
        'files_processed': len(files),
        'successful': 0,
        'failed': 0,
        'output_dir': str(output_dir)
    }

    for file_path, (text, images) in extracted_data.items():
        try:
            file_name = Path(file_path).stem

            # Save text
            if text:
                text_file = output_dir / f"{file_name}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)

            # Save images info
            if images:
                images_file = output_dir / f"{file_name}_images.json"
                import json
                with open(images_file, 'w', encoding='utf-8') as f:
                    json.dump(images, f, indent=2, default=str)

            results['successful'] += 1
            print(f"✅ Extracted: {Path(file_path).name}")

        except Exception as e:
            results['failed'] += 1
            print(f"❌ Failed: {Path(file_path).name} - {e}")

    return results

def process_generate(files: List[str], args) -> Dict[str, Any]:
    """Process files in generate mode"""

    print("🤖 Extracting content and generating training data...")

    # Extract content
    extracted_data = extract_batch(files, use_cache=USE_CACHE)

    # Generate training data
    generated_data = {}
    for file_path, (text, images) in extracted_data.items():
        if text.strip():  # Only process files with content
            print(f"🔄 Processing {Path(file_path).name}...")

            # Generate based on requested types
            data = generate_training_data(
                text,
                generators=args.type,
                images=images if args.include_vision else None
            )

            generated_data[file_path] = data

    # Save results
    results = save_generated_data(generated_data, args)
    results['mode'] = 'generate'

    return results

def process_full(files: List[str], args) -> Dict[str, Any]:
    """Process files in full mode"""

    print("🚀 Full processing - extract, generate, and process images...")

    # Use all available generators in full mode
    args.type = list(GENERATORS.keys())
    args.include_vision = True

    return process_generate(files, args)

def process_resume(files: List[str], args) -> Dict[str, Any]:
    """Resume processing from where it left off"""

    print("🔄 Resuming processing...")

    # Check what's already been processed
    output_dir = Path(args.output_dir)
    processed_files = set()

    if output_dir.exists():
        # Look for existing output files
        for output_file in output_dir.rglob("*.jsonl"):
            # Extract original filename from output file
            # This is a simple approach - could be more sophisticated
            pass

    # Filter out already processed files
    remaining_files = [f for f in files if f not in processed_files]

    if not remaining_files:
        print("✅ All files already processed!")
        return {'mode': 'resume', 'files_processed': 0, 'message': 'No files to resume'}

    print(f"📄 Resuming processing for {len(remaining_files)} files...")

    # Process remaining files
    return process_generate(remaining_files, args)

def save_generated_data(generated_data: Dict[str, Dict], args) -> Dict[str, Any]:
    """Save generated training data to files"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'files_processed': len(generated_data),
        'successful': 0,
        'failed': 0,
        'output_files': [],
        'stats': {}
    }

    # Save each data type separately
    for data_type in ['conversations', 'embeddings', 'qa_pairs', 'summaries', 'image_descriptions']:

        type_dir = output_dir / data_type
        type_dir.mkdir(exist_ok=True)

        all_items = []

        # Collect all items of this type
        for file_path, file_data in generated_data.items():
            if data_type in file_data:
                items = file_data[data_type]

                # Add source file info to each item
                for item in items:
                    item['source_file'] = str(Path(file_path).name)

                all_items.extend(items)

        if all_items:
            # Save based on format
            if args.format == 'jsonl':
                output_file = type_dir / f"{data_type}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in all_items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

            elif args.format == 'csv':
                output_file = type_dir / f"{data_type}.csv"
                import pandas as pd

                # Flatten nested data for CSV
                flattened_items = []
                for item in all_items:
                    flat_item = {}
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            flat_item[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            flat_item[key] = value
                    flattened_items.append(flat_item)

                df = pd.DataFrame(flattened_items)
                df.to_csv(output_file, index=False)

            results['output_files'].append(str(output_file))
            results['stats'][data_type] = len(all_items)

            print(f"💾 Saved {len(all_items)} {data_type} to {output_file.name}")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    results['successful'] = len([f for f in generated_data.values() if f])
    results['output_dir'] = str(output_dir)

    return results

def show_results(results: Dict[str, Any], processing_time: float, args):
    """Display processing results"""

    print(f"\n🎉 Processing completed in {processing_time:.1f} seconds!")
    print(f"📁 Output directory: {results['output_dir']}")
    print(f"📄 Files processed: {results['files_processed']}")

    if 'successful' in results:
        print(f"✅ Successful: {results['successful']}")

    if 'failed' in results:
        print(f"❌ Failed: {results['failed']}")

    if 'stats' in results:
        print(f"\n📊 Generated Training Data:")
        for data_type, count in results['stats'].items():
            print(f"   {data_type}: {count} items")

    # Show next steps
    print(f"\n💡 Next Steps:")
    if args.mode == 'extract-only':
        print("   - Review extracted text files")
        print("   - Run with --mode generate to create training data")
    else:
        print("   - Review generated training data")
        print("   - Use for fine-tuning your AI models")
        print("   - Consider adjusting quality thresholds if needed")

    print(f"\n📖 Documentation: Check README.md for usage examples")

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

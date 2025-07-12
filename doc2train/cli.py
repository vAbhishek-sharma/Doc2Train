# TO REUPDATE
#  doc2train/cli.py - Enhanced CLI Entry Point
"""
Enhanced CLI entry point for pip-installed package
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Doc2Train - Convert documents to AI training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  doc2train document.pdf                           # Extract text only
  doc2train document.pdf --ai                      # Generate training data
  doc2train image.jpg --analyze                    # Analyze image directly
  doc2train folder/ --ai --provider anthropic      # Use specific provider

Commands:
  doc2train-api                                    # Start REST API server
  doc2train-web                                    # Start web interface
"""
    )

    # Positional argument
    parser.add_argument('input_path', nargs='?', help='File or directory to process')

    # Simple processing modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--extract', action='store_true', help='Extract text only (default)')
    mode_group.add_argument('--ai', action='store_true', help='Generate AI training data')
    mode_group.add_argument('--analyze', action='store_true', help='Analyze media directly')
    mode_group.add_argument('--summary', action='store_true', help='Generate summaries only')

    # Provider options
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google', 'deepseek', 'local'],
                       help='AI provider to use')
    parser.add_argument('--api-key', help='API key for the provider')

    # Output options
    parser.add_argument('-o', '--output-dir', default='output', help='Output directory')
    parser.add_argument('--format', choices=['jsonl', 'csv', 'txt'], default='jsonl',
                       help='Output format')

    # Service commands
    parser.add_argument('--start-api', action='store_true', help='Start API server')
    parser.add_argument('--start-web', action='store_true', help='Start web interface')
    parser.add_argument('--version', action='version', version='Doc2Train 2.0.0')

    # Info commands
    parser.add_argument('--list-providers', action='store_true', help='List available providers')
    parser.add_argument('--info', action='store_true', help='Show system information')

    args = parser.parse_args()

    # Handle service commands
    if args.start_api:
        from .api import start_server
        start_server()
        return

    if args.start_web:
        from .web import start_ui
        start_ui()
        return

    # Handle info commands
    if args.list_providers:
        from .core.llm_client import get_available_providers
        providers = get_available_providers()
        print("Available providers:", ", ".join(providers))
        return

    if args.info:
        print("Doc2Train.0")
        print("Python package for converting documents to AI training data")
        return

    # Import processing functionality
    try:
        from .main import ProcessingPipeline
        from .utils.validation import validate_input_and_files
    except ImportError:
        print("❌ Error importing Doc2Train modules")
        print("Try: pip install --upgrade doc2train")
        sys.exit(1)

    # Determine processing mode
    if args.ai:
        mode = 'generate'
        generators = ['conversations', 'qa_pairs']
    elif args.analyze:
        mode = 'direct_to_llm'
        generators = []
    elif args.summary:
        mode = 'generate'
        generators = ['summaries']
    else:
        mode = 'extract_only'
        generators = []

    # Validate input
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ Input not found: {args.input_path}")
        sys.exit(1)

    # Configure processing
    config = {
        'mode': mode,
        'provider': args.provider or 'openai',
        'generators': generators,
        'output_dir': args.output_dir,
        'output_format': args.format,
        'show_progress': True,
        'verbose': True
    }

    if args.api_key:
        import os
        os.environ['OPENAI_API_KEY'] = args.api_key

    print(f"🚀 Processing: {args.input_path}")
    print(f"📁 Output: {args.output_dir}")

    try:
        # Run processing pipeline
        pipeline = ProcessingPipeline(config)

        if input_path.is_file():
            result = pipeline.process_files([str(input_path)], args)
        else:
            from .utils.files import get_supported_files
            files = get_supported_files(str(input_path))
            result = pipeline.process_files(files, args)

        print("✅ Processing completed!")
        print(f"📊 Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

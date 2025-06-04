# cli/args.py
"""
Complete enhanced command line argument parsing for Doc2Train
All enterprise features with comprehensive validation and Smart PDF Analysis
"""

import argparse
from typing import List, Dict, Any
from pathlib import Path

def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create the complete enhanced argument parser with Smart PDF Analysis"""
    parser = argparse.ArgumentParser(
        description="Doc2Train v2.0 Enhanced - Enterprise document processing with Smart PDF Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_examples_text()
    )

    # Required arguments
    parser.add_argument(
        'input_path',
        help='File or directory to process'
    )

    # Processing mode - NOW INCLUDES ANALYZE MODE
    parser.add_argument(
        '--mode',
        choices=['extract-only', 'generate', 'full', 'resume', 'analyze'],
        default='extract-only',
        help='Processing mode (default: extract-only). Use "analyze" for PDF content analysis only.'
    )

    # Generation types
    parser.add_argument(
        '--type',
        nargs='+',
        choices=['conversations', 'embeddings', 'qa_pairs', 'summaries'],
        default=['conversations'],
        help='Types of training data to generate (default: conversations)'
    )

    # Enhanced page control features
    page_group = parser.add_argument_group('📄 Page Control')
    page_group.add_argument(
        '--start-page',
        type=int,
        default=1,
        help='Start processing from this page (default: 1)'
    )
    page_group.add_argument(
        '--end-page',
        type=int,
        help='Stop processing at this page (default: last page)'
    )
    page_group.add_argument(
        '--skip-pages',
        type=str,
        help='Pages to skip (e.g., "1,2,5-10,15")'
    )

    # Enhanced quality control
    quality_group = parser.add_argument_group('🎯 Quality Control')
    quality_group.add_argument(
        '--min-image-size',
        type=int,
        default=1000,
        help='Minimum image size in pixels (default: 1000)'
    )
    quality_group.add_argument(
        '--min-text-length',
        type=int,
        default=100,
        help='Minimum text length in characters (default: 100)'
    )
    quality_group.add_argument(
        '--skip-single-color',
        action='store_true',
        help='Skip single-color images (backgrounds, separators)'
    )
    quality_group.add_argument(
        '--header-regex',
        type=str,
        help='Regex pattern to detect and remove headers/footers'
    )
    quality_group.add_argument(
        '--quality-threshold',
        type=float,
        default=0.7,
        help='Quality threshold for content filtering (0.0-1.0, default: 0.7)'
    )

    # Enhanced performance options
    perf_group = parser.add_argument_group('⚡ Performance')
    perf_group.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of parallel threads (default: 4)'
    )
    perf_group.add_argument(
        '--save-per-file',
        action='store_true',
        help='Save output after each file (fault-tolerant)'
    )
    perf_group.add_argument(
        '--show-progress',
        action='store_true',
        help='Show real-time processing progress with ETA'
    )
    perf_group.add_argument(
        '--no-cache',
        action='store_true',
        help='Skip caching (process everything fresh)'
    )
    perf_group.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of files to process in each batch (default: 10)'
    )

    # Feature flags - ENHANCED WITH SMART PDF ANALYSIS
    feature_group = parser.add_argument_group('🚀 Features')
    feature_group.add_argument(
        '--include-vision',
        action='store_true',
        help='Process images with vision LLMs'
    )
    feature_group.add_argument(
        '--use-ocr',
        action='store_true',
        default=True,
        help='Enable OCR for image text extraction (default: enabled)'
    )
    feature_group.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR processing'
    )

    # NEW: Smart PDF Analysis options
    feature_group.add_argument(
        '--smart-pdf-analysis',
        action='store_true',
        default=True,
        help='Use smart PDF analysis for optimal processing strategy (default: enabled)'
    )
    feature_group.add_argument(
        '--no-smart-analysis',
        action='store_true',
        help='Disable smart PDF analysis (use basic extraction)'
    )


    # Configuration overrides
    config_group = parser.add_argument_group('⚙️ Configuration')
    config_group.add_argument(
        '--chunk-size',
        type=int,
        default=4000,
        help='Text chunk size for processing (default: 4000)'
    )
    config_group.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Overlap between text chunks (default: 200)'
    )
    config_group.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of worker processes (default: 4)'
    )
    config_group.add_argument(
        '--provider',
        choices=['openai', 'deepseek', 'local'],
        help='LLM provider to use (openai, deepseek, local)'
    )
    config_group.add_argument(
        '--model',
        type=str,
        help='Specific model to use (e.g., gpt-4o-mini, deepseek-r1)'
    )

    config_group = parser.add_argument_group('📁 Configuration')
    config_group.add_argument(
        '--config-file',
        default='config.yaml',
        help='YAML configuration file (default: config.yaml)'
    )

    config_group.add_argument(
        '--sync',
        action='store_true',
        help='Use synchronous LLM processing (slower but more stable)'
    )

    config_group.add_argument(
        '--async-calls',
        type=int,
        help='Number of concurrent async LLM calls (default: 5)'
    )

    config_group.add_argument(
        '--prompt-style',
        choices=['default', 'detailed', 'concise', 'academic', 'casual', 'creative', 'professional'],
        help='Prompt style (overrides config.yaml)'
    )

    config_group.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration and exit'
    )

    config_group.add_argument(
        '--save-config',
        action='store_true',
        help='Save current settings to config.yaml and exit'
    )

    # Output options
    output_group = parser.add_argument_group('📤 Output')
    output_group.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output)'
    )
    output_group.add_argument(
        '--format',
        choices=['jsonl', 'csv', 'json', 'txt'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    output_group.add_argument(
        '--output-template',
        type=str,
        help='Custom output template file'
    )

    # Debug and preview options
    debug_group = parser.add_argument_group('🔍 Debug & Preview')
    debug_group.add_argument(
        '--show-images',
        action='store_true',
        help='Show detailed image extraction information'
    )
    debug_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output for debugging'
    )
    debug_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    debug_group.add_argument(
        '--test-mode',
        action='store_true',
        help='Process only a small sample for testing (3 files max)'
    )
    debug_group.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarks and show detailed timing'
    )
    debug_group.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration and input, don\'t process'
    )

    debug_group.add_argument(
        '--clear-cache-after',
        action='store_true',
        help='Clear cache after processing completion'
    )

    # Advanced features
    advanced_group = parser.add_argument_group('🔬 Advanced')
    advanced_group.add_argument(
        '--plugin-dir',
        type=str,
        help='Directory containing custom processor plugins'
    )
    advanced_group.add_argument(
        '--config-file',
        type=str,
        help='Load configuration from YAML/JSON file'
    )
    advanced_group.add_argument(
        '--resume-from',
        type=str,
        help='Resume processing from specific checkpoint file'
    )
    advanced_group.add_argument(
        '--max-file-size',
        type=str,
        default='100MB',
        help='Maximum file size to process (e.g., 100MB, 1GB)'
    )
    advanced_group.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per file in seconds (default: 300)'
    )

    return parser

def parse_skip_pages(skip_pages_str: str) -> List[int]:
    """
    Parse skip pages string into list of page numbers

    Args:
        skip_pages_str: String like "1,2,5-10,15"

    Returns:
        List of page numbers to skip
    """
    if not skip_pages_str:
        return []

    skip_pages = []
    parts = skip_pages_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like "5-10"
            try:
                start, end = part.split('-')
                skip_pages.extend(range(int(start), int(end) + 1))
            except ValueError:
                raise ValueError(f"Invalid page range: {part}")
        else:
            # Single page
            try:
                skip_pages.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")

    return sorted(list(set(skip_pages)))  # Remove duplicates and sort

def parse_file_size(size_str: str) -> int:
    """
    Parse file size string to bytes

    Args:
        size_str: Size string like "100MB", "1GB", "500KB"

    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()

    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes
        return int(size_str)

def args_to_config(args) -> Dict[str, Any]:
    """
    Convert parsed arguments to comprehensive configuration dictionary

    Args:
        args: Parsed arguments from argparse

    Returns:
        Configuration dictionary for processors and pipeline
    """
    config = {
        # Processing mode and types
        'mode': args.mode,
        'generators': args.type,
        'include_vision': args.include_vision,

        # Page control
        'start_page': args.start_page,
        'end_page': args.end_page,
        'skip_pages': parse_skip_pages(args.skip_pages) if args.skip_pages else [],

        # Quality control
        'min_image_size': args.min_image_size,
        'min_text_length': args.min_text_length,
        'skip_single_color': args.skip_single_color,
        'header_regex': args.header_regex,
        'quality_threshold': args.quality_threshold,

        # Performance settings
        'threads': args.threads,
        'max_workers': args.max_workers,
        'batch_size': args.batch_size,
        'save_per_file': args.save_per_file,
        'show_progress': args.show_progress,
        'use_cache': not args.no_cache,

        # Text processing
        'chunk_size': args.chunk_size,
        'overlap': args.overlap,

        # Features - INCLUDING SMART PDF ANALYSIS
        'use_ocr': args.use_ocr and not args.no_ocr,
        'extract_images': True,  # Always extract images
        'use_smart_analysis': args.smart_pdf_analysis and not args.no_smart_analysis,

        # LLM settings
        'provider': args.provider,
        'model': args.model,

        # Output settings
        'output_dir': args.output_dir,
        'output_format': args.format,
        'output_template': args.output_template,

        # Debug settings
        'show_images': args.show_images,
        'verbose': args.verbose,
        'test_mode': args.test_mode,
        'benchmark': args.benchmark,
        'dry_run': args.dry_run,
        'validate_only': args.validate_only,

        # Advanced settings
        'plugin_dir': args.plugin_dir,
        'config_file': args.config_file,
        'resume_from': args.resume_from,
        'max_file_size': parse_file_size(args.max_file_size),
        'timeout': args.timeout,

        # Derived settings
        'fail_on_error': not args.save_per_file,  # More tolerant with per-file saving
        'allow_low_quality': args.test_mode,      # More lenient in test mode
        'clear_cache_after_run': args.clear_cache_after if hasattr(args, 'clear_cache_after') else False,
        # NEW: Async/sync control
        'use_async': not getattr(args, 'sync', False),  # Default async unless --sync
        'max_concurrent_calls': getattr(args, 'async_calls', 5),

        # NEW: Custom prompts from style
        'custom_prompts': _get_prompts_for_style(getattr(args, 'prompt_style', 'default')),
    }

    return config

def get_examples_text() -> str:
    """Get comprehensive CLI examples with Smart PDF Analysis"""
    return """
🚀 Enhanced Examples with Smart PDF Analysis:

Basic Usage:
  python main.py documents/ --mode extract-only --show-progress
  python main.py file.pdf --mode generate --type conversations

PDF Analysis (NEW):
  python main.py document.pdf --mode analyze
  python main.py documents/ --mode analyze --verbose
  python main.py mixed_pdfs/ --mode extract-only --smart-pdf-analysis

Page Control:
  python main.py document.pdf --start-page 5 --end-page 50
  python main.py document.pdf --skip-pages "1,2,10-15,20"
  python main.py book.pdf --start-page 3 --skip-pages "1,2" --header-regex "Page \\d+"

Performance Optimization:
  python main.py documents/ --threads 8 --save-per-file
  python main.py documents/ --batch-size 20 --max-workers 6
  python main.py documents/ --show-progress --benchmark

Quality Control:
  python main.py documents/ --min-image-size 5000 --skip-single-color
  python main.py documents/ --min-text-length 200 --quality-threshold 0.8
  python main.py documents/ --header-regex "CONFIDENTIAL|Page \\d+"

Smart PDF Processing (NEW):
  python main.py scanned_docs/ --smart-pdf-analysis --use-ocr
  python main.py mixed_pdfs/ --mode extract-only --smart-pdf-analysis --verbose
  python main.py documents/ --no-smart-analysis  # Use basic extraction

Advanced Processing:
  python main.py documents/ --mode full --include-vision --provider openai
  python main.py documents/ --mode generate --type conversations,qa_pairs,summaries
  python main.py documents/ --config-file my_config.yaml --plugin-dir ./plugins

Debug & Testing:
  python main.py documents/ --dry-run --show-images --verbose
  python main.py documents/ --test-mode --benchmark
  python main.py documents/ --validate-only

Smart Analysis Examples:
  python main.py research_papers/ --mode analyze --output-dir analysis_results
  python main.py textbooks/ --smart-pdf-analysis --verbose --show-progress
  python main.py scanned_docs/ --mode extract-only --smart-pdf-analysis --use-ocr
    """

def validate_args_enhanced(args) -> bool:
    """
    Enhanced argument validation with detailed error messages

    Args:
        args: Parsed arguments from argparse

    Returns:
        True if valid, raises ValueError with details if invalid
    """
    errors = []

    # Validate input path
    if not Path(args.input_path).exists():
        errors.append(f"Input path does not exist: {args.input_path}")

    # Validate page ranges
    if args.start_page < 1:
        errors.append(f"start-page must be >= 1, got {args.start_page}")

    if args.end_page and args.end_page < args.start_page:
        errors.append(f"end-page ({args.end_page}) must be >= start-page ({args.start_page})")

    # Validate skip pages format
    if args.skip_pages:
        try:
            skip_pages = parse_skip_pages(args.skip_pages)
            if args.start_page in skip_pages:
                errors.append(f"Cannot skip start-page {args.start_page}")
        except ValueError as e:
            errors.append(f"Invalid skip-pages format: {e}")

    # Validate thread count
    if args.threads < 1 or args.threads > 64:
        errors.append(f"threads must be between 1 and 64, got {args.threads}")

    if args.max_workers < 1 or args.max_workers > 32:
        errors.append(f"max-workers must be between 1 and 32, got {args.max_workers}")

    # Validate quality thresholds
    if args.min_image_size < 0:
        errors.append(f"min-image-size must be >= 0, got {args.min_image_size}")

    if args.min_text_length < 0:
        errors.append(f"min-text-length must be >= 0, got {args.min_text_length}")

    if not 0.0 <= args.quality_threshold <= 1.0:
        errors.append(f"quality-threshold must be between 0.0 and 1.0, got {args.quality_threshold}")

    # Validate chunk settings
    if args.chunk_size < 100:
        errors.append(f"chunk-size must be >= 100, got {args.chunk_size}")

    if args.overlap < 0 or args.overlap >= args.chunk_size:
        errors.append(f"overlap must be >= 0 and < chunk-size, got {args.overlap}")

    # Validate timeout
    if args.timeout < 10:
        errors.append(f"timeout must be >= 10 seconds, got {args.timeout}")

    # Validate file size
    try:
        parse_file_size(args.max_file_size)
    except ValueError:
        errors.append(f"Invalid max-file-size format: {args.max_file_size}")

    # Validate config file if provided
    if args.config_file and not Path(args.config_file).exists():
        errors.append(f"Config file does not exist: {args.config_file}")

    # Validate plugin directory if provided
    if args.plugin_dir and not Path(args.plugin_dir).is_dir():
        errors.append(f"Plugin directory does not exist: {args.plugin_dir}")

    # Validate resume file if provided
    if args.resume_from and not Path(args.resume_from).exists():
        errors.append(f"Resume file does not exist: {args.resume_from}")

    # Validate output template if provided
    if args.output_template and not Path(args.output_template).exists():
        errors.append(f"Output template file does not exist: {args.output_template}")

    # Check for conflicting options
    if args.no_cache and args.resume_from:
        errors.append("Cannot use --no-cache with --resume-from")

    if args.dry_run and args.save_per_file:
        errors.append("Cannot use --dry-run with --save-per-file")

    if args.test_mode and args.resume_from:
        errors.append("Cannot use --test-mode with --resume-from")

    # NEW: Validate Smart PDF Analysis options
    if args.smart_pdf_analysis and args.no_smart_analysis:
        errors.append("Cannot use both --smart-pdf-analysis and --no-smart-analysis")

    # Validate LLM requirements
    if args.mode in ['generate', 'full']:
        if not args.provider:
            # Will use default provider, just warn
            pass
        elif args.provider == 'local' and not args.model:
            errors.append("Local provider requires --model to be specified")

    # Validate analyze mode
    if args.mode == 'analyze':
        # Check if input contains PDF files
        input_path = Path(args.input_path)
        if input_path.is_file() and input_path.suffix.lower() != '.pdf':
            errors.append("Analyze mode currently only supports PDF files")

    if errors:
        error_msg = "❌ Argument validation failed:\n" + "\n".join(f"  • {error}" for error in errors)
        raise ValueError(error_msg)

    return True

def _get_prompts_for_style(style: str) -> Dict[str, str]:
    """Get prompts for a specific style"""
    styles = {
        'detailed': {
            'conversations': "Create comprehensive, detailed conversations with thorough explanations and multiple follow-up questions based on this content.",
            'qa_pairs': "Generate detailed questions with comprehensive, well-explained answers based on this content.",
            'summaries': "Create detailed summaries that cover all important aspects thoroughly based on this content."
        },
        'concise': {
            'conversations': "Create brief, focused conversations that get straight to the point based on this content.",
            'qa_pairs': "Generate clear, direct questions with concise but complete answers based on this content.",
            'summaries': "Create brief summaries focusing only on the most essential points from this content."
        },
        'academic': {
            'conversations': "Create scholarly conversations with proper academic discourse based on this content.",
            'qa_pairs': "Generate academic-style questions with evidence-based, well-researched answers based on this content.",
            'summaries': "Create academic summaries with proper structure and formal language based on this content."
        },
        'casual': {
            'conversations': "Create friendly, casual conversations using everyday language based on this content.",
            'qa_pairs': "Generate approachable questions with easy-to-understand answers based on this content.",
            'summaries': "Create informal summaries using simple, conversational language based on this content."
        },
        'creative': {
            'conversations': "Create imaginative conversations using analogies, stories, and creative examples based on this content.",
            'qa_pairs': "Generate creative questions that encourage thinking outside the box based on this content.",
            'summaries': "Create engaging summaries with creative language and interesting perspectives based on this content."
        },
        'professional': {
            'conversations': "Create professional discussions with industry-specific terminology based on this content.",
            'qa_pairs': "Generate professional-level questions with expert insights based on this content.",
            'summaries': "Create business-focused summaries with actionable insights based on this content."
        }
    }

    return styles.get(style, {
        'conversations': "Create a natural conversation between a user and an AI assistant based on this content.",
        'qa_pairs': "Generate questions and answers based on this content.",
        'summaries': "Create a summary of this content."
    })

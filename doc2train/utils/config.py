#TO be Deprecated and removed
# doc2train/utils/config.py
"""
Configuration management utilities
Handles environment variables, validation, and config overrides
"""

import os
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CORE SETTINGS (moved from config/settings.py)
# =============================================================================

# Processing modes
MODES = {
    'extract_only': 'Just extract text/images, no LLM processing',
    'generate': 'Extract + generate training data with LLMs',
    'full': 'Everything - extract, generate, process images with vision LLMs',
    'resume': 'Continue from where you left off'
}

DEFAULT_MODE = 'extract_only'

# Supported file types
SUPPORTED_FORMATS = {
    '.pdf': 'pdf_processor',
    '.txt': 'text_processor',
    '.srt': 'text_processor',
    '.vtt': 'text_processor',
    '.epub': 'epub_processor',
    '.png': 'image_processor',
    '.jpg': 'image_processor',
    '.jpeg': 'image_processor',
    '.bmp': 'image_processor',
    '.tiff': 'image_processor'
}

# Text processing
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 4000))
OVERLAP = int(os.getenv('OVERLAP', 200))
EXTRACT_IMAGES = os.getenv('EXTRACT_IMAGES', 'true').lower() == 'true'
USE_OCR = os.getenv('USE_OCR', 'true').lower() == 'true'

# LLM Providers
LLM_PROVIDERS = {
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'base_url': 'https://api.openai.com/v1',
        'models': {
            'text': 'gpt-4o-mini',
            'vision': 'gpt-4o'
        }
    },
    'deepseek': {
        'api_key': os.getenv('DEEPSEEK_API_KEY'),
        'base_url': 'https://api.deepseek.com',
        'models': {
            'text': 'deepseek-r1',
            'vision': None
        }
    },
    'local': {
        'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
        'models': {
            'text': os.getenv('LOCAL_TEXT_MODEL', 'llama2'),
            'vision': os.getenv('LOCAL_VISION_MODEL', None)
        }
    }
}

DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'openai')
FALLBACK_PROVIDER = os.getenv('FALLBACK_PROVIDER', 'deepseek')

# Available generators
GENERATORS = {
    'conversations': 'Generate multi-turn conversations',
    'embeddings': 'Generate semantic similarity pairs',
    'qa_pairs': 'Generate question-answer pairs',
    'summaries': 'Generate document summaries'
}

DEFAULT_GENERATORS = os.getenv('DEFAULT_GENERATORS', 'conversations,qa_pairs').split(',')

# Output formats
OUTPUT_FORMATS = ['jsonl', 'csv', 'txt']
DEFAULT_OUTPUT_FORMAT = os.getenv('DEFAULT_OUTPUT_FORMAT', 'jsonl')

# Processing settings
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))

# Quality control
QUALITY_THRESHOLD = float(os.getenv('QUALITY_THRESHOLD', 0.7))
MIN_TEXT_LENGTH = int(os.getenv('MIN_TEXT_LENGTH', 100))
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 50000))

# Caching
USE_CACHE = os.getenv('USE_CACHE', 'true').lower() == 'true'
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

# Debug settings
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
VERBOSE = os.getenv('VERBOSE', 'false').lower() == 'true'
TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'

# Test mode settings
TEST_MAX_FILES = 3
TEST_MAX_CHUNKS = 2

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_provider_config(provider_name: str) -> Dict:
    """Get configuration for a specific LLM provider"""
    return LLM_PROVIDERS.get(provider_name, {})

def get_model_for_task(task: str) -> str:
    """Get the best model for a specific task"""
    # Smart routing logic
    routing = {
        'conversations': f"{DEFAULT_PROVIDER}/text",
        'embeddings': f"{DEFAULT_PROVIDER}/text",
        'qa_pairs': f"{DEFAULT_PROVIDER}/text",
        'summaries': f"{DEFAULT_PROVIDER}/text",
        'image_descriptions': f"{DEFAULT_PROVIDER}/vision",
        'fallback': f"{FALLBACK_PROVIDER}/text"
    }

    route = routing.get(task, routing['fallback'])
    provider, model_type = route.split('/')

    provider_config = get_provider_config(provider)
    return provider_config.get('models', {}).get(model_type)

def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported"""
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS

def get_processor_for_file(file_path: str) -> str:
    """Get the appropriate processor for a file"""
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_FORMATS.get(ext)

def apply_config_overrides(args):
    """Apply command line argument overrides to global config"""
    global CHUNK_SIZE, MAX_WORKERS, OUTPUT_DIR, USE_CACHE, TEST_MODE, VERBOSE
    global EXTRACT_IMAGES, USE_OCR

    # Update global config with command line args
    if hasattr(args, 'chunk_size'):
        CHUNK_SIZE = args.chunk_size
    if hasattr(args, 'max_workers'):
        MAX_WORKERS = args.max_workers
    if hasattr(args, 'output_dir'):
        OUTPUT_DIR = args.output_dir
    if hasattr(args, 'no_cache'):
        USE_CACHE = not args.no_cache
    if hasattr(args, 'test_mode'):
        TEST_MODE = args.test_mode
    if hasattr(args, 'verbose'):
        VERBOSE = args.verbose

    # Set provider override
    if hasattr(args, 'provider') and args.provider:
        os.environ['DEFAULT_PROVIDER'] = args.provider

    # Enable test mode if requested
    if TEST_MODE:
        os.environ['TEST_MODE'] = 'true'

def validate_environment(args) -> List[str]:
    """
    Validate environment configuration

    Args:
        args: Parsed command line arguments

    Returns:
        List of validation issues/warnings
    """
    issues = []

    # Check API keys for LLM modes
    if args.mode in ['generate', 'full']:
        if DEFAULT_PROVIDER == 'openai' and not LLM_PROVIDERS['openai']['api_key']:
            issues.append("OpenAI API key not set - LLM features will not work")

        if FALLBACK_PROVIDER == 'deepseek' and not LLM_PROVIDERS['deepseek']['api_key']:
            issues.append("DeepSeek API key not set (fallback provider)")

    # Check directories
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        issues.append(f"Could not create directories: {e}")

    # Check OCR dependencies if needed
    if USE_OCR:
        try:
            import pytesseract
        except ImportError:
            issues.append("pytesseract not installed - OCR features disabled")

    return issues

def get_system_prompts() -> Dict[str, str]:
    """Get system prompts for different generators"""
    return {
        'conversations': os.getenv('CONVERSATION_PROMPT',
            "You are an expert educator. Create natural, multi-turn conversations that help people learn. "
            "Generate realistic user questions and helpful, detailed AI responses."
        ),
        'embeddings': os.getenv('EMBEDDING_PROMPT',
            "Create pairs of sentences with similar meanings but different wording. "
            "Focus on semantic similarity for training embedding models."
        ),
        'qa_pairs': os.getenv('QA_PROMPT',
            "Generate clear, specific questions that can be answered from the given content. "
            "Provide complete, accurate answers based on the source material."
        ),
        'summaries': os.getenv('SUMMARY_PROMPT',
            "Create concise, informative summaries that capture the key points and main ideas. "
            "Focus on the most important information."
        )
    }

def get_config_summary() -> Dict[str, Any]:
    """Get summary of current configuration"""
    return {
        'modes': list(MODES.keys()),
        'supported_formats': list(SUPPORTED_FORMATS.keys()),
        'providers': list(LLM_PROVIDERS.keys()),
        'generators': list(GENERATORS.keys()),
        'output_formats': OUTPUT_FORMATS,
        'cache_enabled': USE_CACHE,
        'ocr_enabled': USE_OCR,
        'images_enabled': EXTRACT_IMAGES,
        'max_workers': MAX_WORKERS,
        'chunk_size': CHUNK_SIZE,
        'test_mode': TEST_MODE
    }

def create_default_env_file():
    """Create a default .env file if it doesn't exist"""
    env_file = Path('.env')

    if env_file.exists():
        return False

    env_content = """# Doc2Train v2.0 Enhanced Configuration
# Edit this file with your actual API keys and settings

# API Keys (Required for LLM processing)
OPENAI_API_KEY=your-openai-api-key-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Default Settings
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
CHUNK_SIZE=4000
OVERLAP=200
MAX_WORKERS=4

# Features
EXTRACT_IMAGES=true
USE_OCR=true
USE_CACHE=true
QUALITY_THRESHOLD=0.7

# Output
OUTPUT_DIR=output
DEFAULT_OUTPUT_FORMAT=jsonl

# Debug
DEBUG=false
VERBOSE=false
TEST_MODE=false
"""

    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        return True
    except Exception:
        return False

# Initialize configuration
def init_config():
    """Initialize configuration on module import"""
    # Create .env file if it doesn't exist
    if create_default_env_file():
        print("📝 Created default .env file - please edit with your API keys")

    # Create required directories
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        if DEBUG:
            print(f"Warning: Could not create directories: {e}")

# Auto-initialize on import
init_config()

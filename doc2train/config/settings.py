#TO be Deprecated and removed
# config/settings.py
"""
Simple configuration for Doc2Train v2.0
Everything you need to customize is here in one place
"""

import os
from typing import List, Dict, Any  # Added this line
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CORE SETTINGS
# =============================================================================

# Processing modes
MODES = {
    'extract_only': 'Just extract text/images, no LLM processing',
    'generate': 'Extract + generate training data with LLMs',
    'full': 'Everything - extract, generate, process images with vision LLMs',
    'resume': 'Continue from where you left off'
}

# Default processing mode
DEFAULT_MODE = 'extract_only'

# =============================================================================
# INPUT SETTINGS
# =============================================================================

# Text processing
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 4000))
OVERLAP = int(os.getenv('OVERLAP', 200))
EXTRACT_IMAGES = os.getenv('EXTRACT_IMAGES', 'true').lower() == 'true'
USE_OCR = os.getenv('USE_OCR', 'true').lower() == 'true'

# =============================================================================
# LLM PROVIDERS
# =============================================================================

# Available LLM providers
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
            'vision': None  # DeepSeek doesn't have vision models yet
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

# Default provider selection
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'openai')
FALLBACK_PROVIDER = os.getenv('FALLBACK_PROVIDER', 'deepseek')

# Smart routing - use best/cheapest model for each task
LLM_ROUTING = {
    'conversations': f"{DEFAULT_PROVIDER}/text",
    'embeddings': f"{DEFAULT_PROVIDER}/text",
    'qa_pairs': f"{DEFAULT_PROVIDER}/text",
    'summaries': f"{DEFAULT_PROVIDER}/text",
    'image_descriptions': f"{DEFAULT_PROVIDER}/vision",
    'fallback': f"{FALLBACK_PROVIDER}/text"
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Available generators
GENERATORS = {
    'conversations': 'Generate multi-turn conversations',
    'embeddings': 'Generate semantic similarity pairs',
    'qa_pairs': 'Generate question-answer pairs',
    'summaries': 'Generate document summaries'
}

# Default generators to use
DEFAULT_GENERATORS = os.getenv('DEFAULT_GENERATORS', 'conversations,qa_pairs').split(',')

# Output formats
OUTPUT_FORMATS = ['jsonl', 'csv', 'txt']
DEFAULT_OUTPUT_FORMAT = os.getenv('DEFAULT_OUTPUT_FORMAT', 'jsonl')

# Output structure
OUTPUT_STRUCTURE = {
    'extracted': 'Raw extracted text and images',
    'conversations': 'Generated conversational training data',
    'embeddings': 'Semantic similarity pairs',
    'qa_pairs': 'Question-answer pairs',
    'summaries': 'Document summaries',
    'images': 'Extracted images with descriptions'
}

# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

# Parallel processing
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

# =============================================================================
# CUSTOM PROMPTS
# =============================================================================

# System prompts for different generators
SYSTEM_PROMPTS = {
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

# =============================================================================
# DEBUGGING & DEVELOPMENT
# =============================================================================

# Debug settings
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
VERBOSE = os.getenv('VERBOSE', 'false').lower() == 'true'
TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'

# Test mode settings (process only small samples)
TEST_MAX_FILES = 3
TEST_MAX_CHUNKS = 2

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'doc2train.log')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_provider_config(provider_name):
    """Get configuration for a specific LLM provider"""
    return LLM_PROVIDERS.get(provider_name, {})

def get_model_for_task(task):
    """Get the best model for a specific task"""
    route = LLM_ROUTING.get(task, LLM_ROUTING['fallback'])
    provider, model_type = route.split('/')

    provider_config = get_provider_config(provider)
    return provider_config.get('models', {}).get(model_type)

def is_supported_format(file_path):
    """Check if file format is supported"""
    from pathlib import Path
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS

def get_processor_for_file(file_path):
    """Get the appropriate processor for a file"""
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_FORMATS.get(ext)

def validate_config():
    """Validate configuration and warn about missing settings"""
    issues = []

    # Check API keys
    if DEFAULT_PROVIDER == 'openai' and not LLM_PROVIDERS['openai']['api_key']:
        issues.append("OpenAI API key not set")

    if FALLBACK_PROVIDER == 'deepseek' and not LLM_PROVIDERS['deepseek']['api_key']:
        issues.append("DeepSeek API key not set (fallback)")

    # Check directories
    import os
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return issues

# Validate on import
if __name__ != '__main__':
    validation_issues = validate_config()
    if validation_issues and not TEST_MODE:
        print("⚠️  Configuration warnings:")
        for issue in validation_issues:
            print(f"  - {issue}")
        print("  Set up your .env file with API keys for full functionality")

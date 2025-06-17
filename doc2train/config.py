#config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'  # Enables test mode

MODELS = {
    'openai': {
        'gpt-4o': {
            'base_url': 'https://api.openai.com/v1',
            'max_tokens': 8000,  # Increased limit
            'temperature': 0.7
        },
        'gpt-4o-mini': {
            'base_url': 'https://api.openai.com/v1',
            'max_tokens': 16384,  # Increased limit for GPT-4o Mini
            'temperature': 0.7
        },
        'gpt-3.5-turbo': {
            'base_url': 'https://api.openai.com/v1',
            'max_tokens': 4096,  # Increased limit
            'temperature': 0.7
        }
    },
    'deepseek': {
        'deepseek-r1': {
            'base_url': 'https://api.deepseek.com',
            'max_tokens': 8192,
            'temperature': 0.7
        }
    }
}

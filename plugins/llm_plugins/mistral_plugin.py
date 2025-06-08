# plugins/llm_plugins/mistral_plugin.py
import requests
import json
import os
from typing import Dict, Any, List, Union, Optional
from .base_llm_plugin import BaseLLMPlugin

class MistralPlugin(BaseLLMPlugin):
    """Mistral AI provider plugin"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.provider_name = 'mistral'
        self.api_key = self.config.get('api_key') or os.getenv('MISTRAL_API_KEY')
        self.base_url = self.config.get('base_url', 'https://api.mistral.ai/v1')

        self.capabilities = {
            'text': True,
            'vision': False,  # Mistral doesn't have vision yet
            'streaming': True,
            'function_calling': True
        }

        self.supported_models = {
            'mistral-large-latest': {
                'type': 'text',
                'max_tokens': 8192,
                'context_window': 128000
            },
            'mistral-medium-latest': {
                'type': 'text',
                'max_tokens': 8192,
                'context_window': 32000
            }
        }

    def call_text_model(self, prompt: str, model: str = None, **kwargs) -> str:
        """Call Mistral text model"""
        model = model or 'mistral-large-latest'

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }

        response = requests.post(f'{self.base_url}/chat/completions',
                               headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"Mistral API error: {response.status_code} - {response.text}")

        result = response.json()
        return result['choices'][0]['message']['content']

    def get_available_models(self) -> Dict[str, Dict]:
        """Get available Mistral models"""
        return self.supported_models

    def validate_config(self) -> bool:
        """Validate Mistral configuration"""
        return bool(self.api_key)

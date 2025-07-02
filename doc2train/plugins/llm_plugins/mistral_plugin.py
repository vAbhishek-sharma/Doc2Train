from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
import os
import requests
from typing import Dict, Any, List, Union, Optional

class MistralPlugin(BaseLLMPlugin):
    provider_name = "mistral"
    priority = 10
    supported_types = ["text"]
    supports_vision = False
    description = "Mistral LLM"
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get('api_key') or os.getenv('MISTRAL_API_KEY')
        self.base_url = self.config.get('base_url', 'https://api.mistral.ai/v1')

    @classmethod
    def configured(cls):
        return bool(os.getenv('MISTRAL_API_KEY'))

    @staticmethod
    def _get_api_key() -> str:
        """Get API key from environment"""
        return os.getenv('MISTRAL_API_KEY')

    @staticmethod
    def _get_base_url() -> str:
        """Get base URL for API"""
        return 'https://api.mistral.ai/v1'

    def call_text_model(self, prompt, model=None, **kwargs):
        model = model or self.get_default_model()
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 2048),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None, model: str = None, **kwargs) -> str:
        """Mistral doesn't support vision currently, raise NotImplementedError"""
        raise NotImplementedError(f"{self.provider_name} does not support vision models")

    @staticmethod
    def call(prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Static method to call Mistral text endpoint."""
        model = model or MistralPlugin.get_default_model()
        api_key = MistralPlugin._get_api_key()
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not found in environment")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 2048),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{MistralPlugin._get_base_url()}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text}")
        return resp.json()['choices'][0]['message']['content']

    @staticmethod
    def call_vision(prompt: str, images: List[Union[str, bytes]], model: Optional[str] = None, **kwargs) -> str:
        """Mistral doesn't support vision currently, raise NotImplementedError"""
        raise NotImplementedError("Mistral does not support vision models")

    @staticmethod
    def get_default_model():
        return 'mistral-large-latest'

    @classmethod
    def supported_models(cls):
        return {
            'mistral-large-latest': {
                'type': 'text',
                'max_tokens': 32_000,
                'cost': {'input_per_1k': 1.2, 'output_per_1k': 2.4}
            },
            'mistral-medium-latest': {
                'type': 'text',
                'max_tokens': 32_000,
                'cost': {'input_per_1k': 0.8, 'output_per_1k': 1.6}
            },
            'mistral-small-latest': {
                'type': 'text',
                'max_tokens': 32_000,
                'cost': {'input_per_1k': 0.4, 'output_per_1k': 0.8}
            }
        }

    def get_available_models(self):
        return self.supported_models()

    def validate_config(self):
        return bool(self.api_key)

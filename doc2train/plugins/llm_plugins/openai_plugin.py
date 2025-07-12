from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
import os
import requests
from typing import Dict, Any, List, Union, Optional

class OpenAIPlugin(BaseLLMPlugin):
    provider_name = "openai"
    priority = 10
    supported_types = ["text", "image"]
    supports_vision = True
    description = "OpenAI GPT (vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')

    @classmethod
    def configured(cls):
        return bool(os.getenv('OPENAI_API_KEY'))

    @staticmethod
    def _get_api_key() -> str:
        """Get API key from environment"""
        return os.getenv('OPENAI_API_KEY')

    @staticmethod
    def _get_base_url() -> str:
        """Get base URL for API"""
        return 'https://api.openai.com/v1'

    def call_text_model(self, prompt, model=None, **kwargs):
        model = model or self.get_default_model()
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None, model: str = None, **kwargs) -> str:
        """Call OpenAI vision endpoint."""
        model = model or self.get_default_model()

        content = []
        # Add images
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img

            # Determine image type
            image_type = 'jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                image_type = 'png'

            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/{image_type};base64,{self._encode_image(image_data=img_data)}"
                }
            })

        # Add text prompt
        content.append({'type': 'text', 'text': prompt})

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    @staticmethod
    def call(prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Static method to call OpenAI text endpoint."""
        model = model or OpenAIPlugin.get_default_model()
        api_key = OpenAIPlugin._get_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{OpenAIPlugin._get_base_url()}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
        return resp.json()['choices'][0]['message']['content']

    @staticmethod
    def call_vision(prompt: str, images: List[Union[str, bytes]], model: Optional[str] = None, **kwargs) -> str:
        """Static method to call OpenAI vision endpoint."""
        model = model or OpenAIPlugin.get_default_model()
        api_key = OpenAIPlugin._get_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        content = []
        # Add images
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img

            # Determine image type
            image_type = 'jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                image_type = 'png'

            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/{image_type};base64,{BaseLLMPlugin._encode_image(image_data=img_data)}"
                }
            })

        # Add text prompt
        content.append({'type': 'text', 'text': prompt})

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{OpenAIPlugin._get_base_url()}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
        return resp.json()['choices'][0]['message']['content']

    @staticmethod
    def get_default_model():
        return 'gpt-4o'

    @classmethod
    def supported_models(cls):
        return {
            'gpt-4o': {
                'type': 'multimodal',
                'max_tokens': 128_000,
                'context_window': 128_000,
                'cost': {'input_per_1k': 5.0, 'output_per_1k': 15.0}
            },
            'gpt-4o-mini': {
                'type': 'multimodal',
                'max_tokens': 128_000,
                'context_window': 128_000,
                'cost': {'input_per_1k': 0.15, 'output_per_1k': 0.6}
            },
            'gpt-4-turbo': {
                'type': 'multimodal',
                'max_tokens': 128_000,
                'context_window': 128_000,
                'cost': {'input_per_1k': 10.0, 'output_per_1k': 30.0}
            },
            'gpt-3.5-turbo': {
                'type': 'text',
                'max_tokens': 16_000,
                'context_window': 16_000,
                'cost': {'input_per_1k': 1.0, 'output_per_1k': 2.0}
            }
        }

    def get_available_models(self):
        return self.supported_models()

    def validate_config(self):
        return bool(self.api_key)

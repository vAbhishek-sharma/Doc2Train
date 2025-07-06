from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
import os
import requests
from typing import Dict, Any, List, Union, Optional

class OpenRouterPlugin(BaseLLMPlugin):
    provider_name = "openrouter"
    priority = 15
    supported_types = ["text", "image"]
    supports_vision = True
    description = "OpenRouter LLM (multi-provider, vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        self.base_url = self.config.get('base_url', 'https://openrouter.ai/api/v1')

    @classmethod
    def configured(cls):
        return bool(os.getenv('OPENROUTER_API_KEY'))

    @staticmethod
    def _get_api_key() -> str:
        return os.getenv('OPENROUTER_API_KEY')

    @staticmethod
    def _get_base_url() -> str:
        return 'https://openrouter.ai/api/v1'

    def call_text_model(self, prompt, model=None, **kwargs):
        model = model or self.get_default_model()
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'HTTP-Referer': 'https://your-doc2train-app-url',  # required by OpenRouter for free-tier, else remove
            'X-Title': 'Doc2Train',
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
        """Call OpenRouter vision-capable endpoint."""
        model = model or self.get_default_model()
        content = []
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img
            image_type = 'jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                image_type = 'png'
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/{image_type};base64,{self._encode_image(img_data)}"
                }
            })
        content.append({'type': 'text', 'text': prompt})

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'HTTP-Referer': 'https://your-doc2train-app-url',
            'X-Title': 'Doc2Train',
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
        model = model or OpenRouterPlugin.get_default_model()
        api_key = OpenRouterPlugin._get_api_key()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in environment")
        headers = {
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://your-doc2train-app-url',
            'X-Title': 'Doc2Train',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{OpenRouterPlugin._get_base_url()}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
        return resp.json()['choices'][0]['message']['content']

    @staticmethod
    def call_vision(prompt: str, images: List[Union[str, bytes]], model: Optional[str] = None, **kwargs) -> str:
        model = model or OpenRouterPlugin.get_default_model()
        api_key = OpenRouterPlugin._get_api_key()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in environment")
        content = []
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img
            image_type = 'jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                image_type = 'png'
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/{image_type};base64,{BaseLLMPlugin._encode_image(img_data)}"
                }
            })
        content.append({'type': 'text', 'text': prompt})

        headers = {
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://your-doc2train-app-url',
            'X-Title': 'Doc2Train',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{OpenRouterPlugin._get_base_url()}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
        return resp.json()['choices'][0]['message']['content']

    @staticmethod
    def get_default_model():
        # Change to your preferred OpenRouter model
        return 'openai/gpt-4o'

    @classmethod
    def supported_models(cls):
        return {
            'openai/gpt-4o': {
                'type': 'multimodal', 'max_tokens': 128_000, 'context_window': 128_000,
                'cost': {'input_per_1k': 4.0, 'output_per_1k': 13.0}
            },
            'openai/gpt-4o-mini': {
                'type': 'multimodal', 'max_tokens': 128_000, 'context_window': 128_000,
                'cost': {'input_per_1k': 0.15, 'output_per_1k': 0.6}
            },
            'anthropic/claude-3-haiku': {
                'type': 'multimodal', 'max_tokens': 200_000, 'context_window': 200_000,
                'cost': {'input_per_1k': 0.25, 'output_per_1k': 1.25}
            },
            'google/gemini-pro': {
                'type': 'multimodal', 'max_tokens': 1_000_000, 'context_window': 1_000_000,
                'cost': {'input_per_1k': 0.1, 'output_per_1k': 0.1}
            },
            'google/gemini-2.0-flash-exp:free': {
                'type': 'multimodal', 'max_tokens': 1_000_000, 'context_window': 1_000_000,
                'cost': {'input_per_1k': 0.1, 'output_per_1k': 0.1}
            },
            'deepseek/deepseek-chat-v3-0324:free': {
                'type': 'multimodal', 'max_tokens': 1_000_000, 'context_window': 1_000_000,
                'cost': {'input_per_1k': 0.1, 'output_per_1k': 0.1}
            },
            # ... Add any more OpenRouter-supported models here ...
        }

    def get_available_models(self):
        return self.supported_models()

    def validate_config(self):
        return bool(self.api_key)

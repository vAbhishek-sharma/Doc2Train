# doc2train/plugins/llm_plugins/anthropic_plugin.py
"""
Anthropic Claude plugin for Doc2Train
"""
import os

import requests
from typing import Dict, Any, List, Union, Optional
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin

class AnthropicPlugin(BaseLLMPlugin):
    """Plugin for Anthropic Claude API calls, including text and vision."""
    provider_name = "anthropic"
    priority = 10
    supported_types = ["text", "image"]
    supports_vision = True
    description = "Anthropic Claude LLM (vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.base_url = self.config.get('base_url', 'https://api.anthropic.com/v1')

    @staticmethod
    def configured() -> bool:
        """Return True if API key is set."""
        return bool(os.getenv('ANTHROPIC_API_KEY'))

    @staticmethod
    def get_default_model() -> str:
        """Default Anthropic model"""
        return 'claude-3-5-sonnet-20241022'

    @staticmethod
    def call(prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Call Claude text endpoint."""
        model = model or AnthropicPlugin.get_default_model()
        headers = {
            'Authorization': f'Bearer {AnthropicPlugin._get_api_key()}',
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        payload = {
            'model': model,
            'messages': [ {'role': 'user', 'content': prompt} ],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{AnthropicPlugin._get_base_url()}/messages", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data['content'][0]['text']

    @staticmethod
    def call_vision(prompt: str, images: List[Union[str, bytes]], model: Optional[str] = None, **kwargs) -> str:
        """Call Claude vision endpoint."""
        model = model or AnthropicPlugin.get_default_model()
        content: List[Any] = []
        # Encode images
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img
            media_type = 'image/jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                media_type = 'image/png'
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': media_type,
                    'data': BaseLLMPlugin._encode_image(image_data=img_data)
                }
            })
        # Append text
        content.append({'type': 'text', 'text': prompt})

        headers = {
            'Authorization': f'Bearer {AnthropicPlugin._get_api_key()}',
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        payload = {
            'model': model,
            'messages': [ {'role': 'user', 'content': content} ],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        resp = requests.post(f"{AnthropicPlugin._get_base_url()}/messages", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data['content'][0]['text']

    @classmethod
    def supported_models(cls) -> Dict[str, Any]:
        """Return available models with specs and cost."""
        return {
            'claude-3-5-sonnet-20241022': {
                'type': 'multimodal',
                'max_tokens': 200_000,
                'context_window': 200_000,
                'cost': { 'input_per_1k': 3.00, 'output_per_1k': 15.00 }
            },
            'claude-3-opus-20240229': {
                'type': 'multimodal',
                'max_tokens': 4_096,
                'context_window': 200_000,
                'cost': { 'input_per_1k': 3.00, 'output_per_1k': 15.00 }
            },
            'claude-3-haiku-20240307': {
                'type': 'multimodal',
                'max_tokens': 4_096,
                'context_window': 200_000,
                'cost': { 'input_per_1k': 3.00, 'output_per_1k': 15.00 }
            }
        }

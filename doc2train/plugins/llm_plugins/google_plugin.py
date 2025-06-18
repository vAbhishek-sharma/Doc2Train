# NEW: plugins/llm_plugins/google_plugin.py
"""
Sample Google Gemini plugin implementation
"""

import requests
import json
import os
from typing import Dict, Any, List, Union, Optional
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin

class GooglePlugin(BaseLLMPlugin):
    """
    Google Gemini provider plugin
    """
    provider_name = "google"
    priority = 10
    supported_types = ["text", "image"]
    supports_vision = True
    description = "Google Gemini LLM (vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"
    capabilities = {
                'text': True,
                'vision': True,  # Gemini Pro Vision
                'streaming': False,
                'function_calling': True
            }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.provider_name = 'google'
        self.api_key = self.config.get('api_key') or os.getenv('GOOGLE_API_KEY')
        self.base_url = self.config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')

        self.capabilities = {
            'text': True,
            'vision': True,  # Gemini Pro Vision
            'streaming': False,
            'function_calling': True
        }

        self.supported_models = {
            'gemini-1.5-pro': {
                'type': 'vision',
                'max_tokens': 8192,
                'context_window': 2000000  # 2M tokens
            },
            'gemini-1.5-flash': {
                'type': 'vision',
                'max_tokens': 8192,
                'context_window': 1000000  # 1M tokens
            },
            'gemini-pro': {
                'type': 'text',
                'max_tokens': 2048,
                'context_window': 32000
            }
        }

    def call_text_model(self, prompt: str, model: str = None, **kwargs) -> str:
        """Call Gemini text model"""
        model = model or 'gemini-1.5-flash'

        url = f'{self.base_url}/models/{model}:generateContent?key={self.api_key}'

        data = {
            'contents': [{
                'parts': [{'text': prompt}]
            }],
            'generationConfig': {
                'temperature': kwargs.get('temperature', 0.7),
                'maxOutputTokens': kwargs.get('max_tokens', 2000)
            }
        }

        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise Exception(f"Google API error: {response.status_code} - {response.text}")

        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None,
                         model: str = None, **kwargs) -> str:
        """Call Gemini vision model"""
        model = model or 'gemini-1.5-pro'

        url = f'{self.base_url}/models/{model}:generateContent?key={self.api_key}'

        parts = [{'text': prompt}]

        if images:
            for image in images:
                if isinstance(image, str):
                    with open(image, 'rb') as f:
                        image_data = f.read()
                else:
                    image_data = image

                # Determine MIME type
                mime_type = "image/jpeg"
                if isinstance(image, str):
                    if image.lower().endswith('.png'):
                        mime_type = "image/png"
                    elif image.lower().endswith('.webp'):
                        mime_type = "image/webp"

                parts.append({
                    'inline_data': {
                        'mime_type': mime_type,
                        'data': self._encode_image(image_data)
                    }
                })

        data = {
            'contents': [{'parts': parts}],
            'generationConfig': {
                'temperature': kwargs.get('temperature', 0.7),
                'maxOutputTokens': kwargs.get('max_tokens', 2000)
            }
        }

        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise Exception(f"Google API error: {response.status_code} - {response.text}")

        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']

    def get_available_models(self) -> Dict[str, Dict]:
        """Get available Gemini models"""
        return self.supported_models

    def validate_config(self) -> bool:
        """Validate Google configuration"""
        return bool(self.api_key)

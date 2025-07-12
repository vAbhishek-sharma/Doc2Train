from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
import os
import requests
from typing import Dict, Any, List, Union, Optional

class GooglePlugin(BaseLLMPlugin):
    provider_name = "google"
    priority = 10
    supported_types = ["text", "image"]
    supports_vision = True
    description = "Google Gemini LLM (vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get('api_key') or os.getenv('GOOGLE_API_KEY')
        self.base_url = self.config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta')

    @classmethod
    def configured(cls):
        return bool(os.getenv('GOOGLE_API_KEY'))

    @staticmethod
    def _get_api_key() -> str:
        """Get API key from environment"""
        return os.getenv('GOOGLE_API_KEY')

    @staticmethod
    def _get_base_url() -> str:
        """Get base URL for API"""
        return 'https://generativelanguage.googleapis.com/v1beta'

    def call_text_model(self, prompt, model=None, **kwargs):
        model = model or self.get_default_model()
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'maxOutputTokens': kwargs.get('max_tokens', 2048),
                'temperature': kwargs.get('temperature', 0.7)
            }
        }
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None, model: str = None, **kwargs) -> str:
        """Call Google Gemini vision endpoint."""
        model = model or self.get_default_model()

        parts = []
        # Add images
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img

            # Determine MIME type
            mime_type = 'image/jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                mime_type = 'image/png'

            parts.append({
                'inlineData': {
                    'mimeType': mime_type,
                    'data': self._encode_image(image_data=img_data)
                }
            })

        # Add text prompt
        parts.append({'text': prompt})

        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            'contents': [{'parts': parts}],
            'generationConfig': {
                'maxOutputTokens': kwargs.get('max_tokens', 2048),
                'temperature': kwargs.get('temperature', 0.7)
            }
        }
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

    @staticmethod
    def call(prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Static method to call Google text endpoint."""
        model = model or GooglePlugin.get_default_model()
        api_key = GooglePlugin._get_api_key()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment")

        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'maxOutputTokens': kwargs.get('max_tokens', 2048),
                'temperature': kwargs.get('temperature', 0.7)
            }
        }
        url = f"{GooglePlugin._get_base_url()}/models/{model}:generateContent?key={api_key}"
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Google API error {resp.status_code}: {resp.text}")
        return resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

    @staticmethod
    def call_vision(prompt: str, images: List[Union[str, bytes]], model: Optional[str] = None, **kwargs) -> str:
        """Static method to call Google vision endpoint."""
        model = model or GooglePlugin.get_default_model()
        api_key = GooglePlugin._get_api_key()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment")

        parts = []
        # Add images
        for img in images or []:
            if isinstance(img, str):
                with open(img, 'rb') as f:
                    img_data = f.read()
            else:
                img_data = img

            # Determine MIME type
            mime_type = 'image/jpeg'
            if isinstance(img, str) and img.lower().endswith('.png'):
                mime_type = 'image/png'

            parts.append({
                'inlineData': {
                    'mimeType': mime_type,
                    'data': BaseLLMPlugin._encode_image(image_data=img_data)
                }
            })

        # Add text prompt
        parts.append({'text': prompt})

        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            'contents': [{'parts': parts}],
            'generationConfig': {
                'maxOutputTokens': kwargs.get('max_tokens', 2048),
                'temperature': kwargs.get('temperature', 0.7)
            }
        }
        url = f"{GooglePlugin._get_base_url()}/models/{model}:generateContent?key={api_key}"
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Google API error {resp.status_code}: {resp.text}")
        return resp.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

    @staticmethod
    def get_default_model():
        return 'gemini-pro'

    @classmethod
    def supported_models(cls):
        return {
            'gemini-pro': {
                'type': 'multimodal',
                'max_tokens': 32_000,
                'cost': {'input_per_1k': 1.0, 'output_per_1k': 3.0}
            },
            'gemini-pro-vision': {
                'type': 'multimodal',
                'max_tokens': 32_000,
                'cost': {'input_per_1k': 1.0, 'output_per_1k': 3.0}
            }
        }

    def get_available_models(self):
        return self.supported_models()

    def validate_config(self):
        return bool(self.api_key)

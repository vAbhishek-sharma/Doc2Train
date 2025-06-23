#  plugins/llm_plugins/anthropic_plugin.py
"""
Sample Anthropic Claude plugin implementation
"""

import os
import requests
import json
from typing import Dict, Any, List, Union, Optional
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
class AnthropicPlugin(BaseLLMPlugin):
    """
    Anthropic Claude provider plugin
    """
    provider_name = "anthropic"
    priority = 10
    supported_types = ["text", "image"]
    supports_vision = True
    description = "Anthropic Claude LLM (vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"
    capabilities = {
                'text': True,
                'vision': True,  # Claude 3+ supports vision
                'streaming': True,
                'function_calling': False,

            }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.provider_name = 'anthropic'
        self.api_key = self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.base_url = self.config.get('base_url', 'https://api.anthropic.com/v1')

        self.capabilities = {
            'text': True,
            'vision': True,  # Claude 3+ supports vision
            'streaming': True,
            'function_calling': False
        }

        self.supported_models = {
            'claude-3-5-sonnet-20241022': {
                'type': 'vision',
                'max_tokens': 200000,
                'context_window': 200000
            },
            'claude-3-opus-20240229': {
                'type': 'vision',
                'max_tokens': 4096,
                'context_window': 200000
            },
            'claude-3-haiku-20240307': {
                'type': 'vision',
                'max_tokens': 4096,
                'context_window': 200000
            }
        }

    def call_text_model(self, prompt: str, model: str = None, **kwargs) -> str:
        """Call Claude text model"""
        model = model or 'claude-3-5-sonnet-20241022'

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }

        response = requests.post(
            f'{self.base_url}/messages',
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")

        result = response.json()
        return result['content'][0]['text']

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None,
                         model: str = None, **kwargs) -> str:
        """Call Claude vision model"""
        model = model or 'claude-3-5-sonnet-20241022'

        # Prepare content with images
        content = []

        if images:
            for image in images:
                if isinstance(image, str):
                    # Read image file
                    with open(image, 'rb') as f:
                        image_data = f.read()
                else:
                    image_data = image

                # Determine media type
                media_type = "image/jpeg"  # Default
                if isinstance(image, str):
                    if image.lower().endswith('.png'):
                        media_type = "image/png"
                    elif image.lower().endswith('.webp'):
                        media_type = "image/webp"

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": self._encode_image(image_data)
                    }
                })

        content.append({
            "type": "text",
            "text": prompt
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': content}
            ],
            'max_tokens': kwargs.get('max_tokens', 2000),
            'temperature': kwargs.get('temperature', 0.7)
        }

        response = requests.post(
            f'{self.base_url}/messages',
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")

        result = response.json()
        return result['content'][0]['text']

    def get_available_models(self) -> Dict[str, Dict]:
        """Get available Claude models"""
        return self.supported_models

    def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        return bool(self.api_key)



    @classmethod
    def supported_models(cls) -> Dict[str, Any]:
        """
        Claude family pricing and limits.
        Developers can override this in forks or child classes.
        """
        return {
            'claude-3-5-sonnet-20241022': {
                'type': 'text',
                'max_tokens': 200_000,
                'context_window': 200_000,
                'cost': {
                    'input_per_1k':  3.00,
                    'output_per_1k': 15.00,
                }
            },
            'claude-3-7-sonnet-20241130': {
                'type': 'text',
                'max_tokens': 200_000,
                'context_window': 200_000,
                'cost': {
                    'input_per_1k':  3.00,
                    'output_per_1k': 15.00,
                }
            },
            # …add more Claude variants here…
        }

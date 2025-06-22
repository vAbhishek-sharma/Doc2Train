#  plugins/llm_plugins/base_llm_plugin.py
"""
Base LLM plugin class for extending LLM provider support
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union
import base64
from doc2train.core.plugin_metadata_mixins.llm_plugin_metadata_mixin import LLMPluginMetadataMixin
class BaseLLMPlugin(LLMPluginMetadataMixin,ABC):
    """
    Base class for LLM provider plugins
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LLM plugin

        Args:
            config: Plugin configuration dictionary
        """
        self.config = config or {}
        self.provider_name = self.__class__.__name__.lower().replace('plugin', '')
        self.supported_models = {}
        self.capabilities = {
            'text': True,
            'vision': False,
            'streaming': False,
            'function_calling': False
        }

    @abstractmethod
    def call_text_model(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Call text-only model

        Args:
            prompt: Text prompt
            model: Model name (optional, uses default if None)
            **kwargs: Additional model parameters

        Returns:
            Generated text response
        """
        pass

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None,
                         model: str = None, **kwargs) -> str:
        """
        Call vision model (optional implementation)

        Args:
            prompt: Text prompt
            images: List of image paths or raw image data
            model: Model name (optional)
            **kwargs: Additional model parameters

        Returns:
            Generated text response
        """
        if not self.capabilities['vision']:
            raise NotImplementedError(f"{self.provider_name} does not support vision models")
        return self.call_text_model(prompt, model, **kwargs)

    def process_direct_media(self, media_path: str, prompt: str = None, **kwargs) -> str:
        """
         Process media directly without preprocessing
        Allows users to skip processors and send images/videos directly to LLM

        Args:
            media_path: Path to image or video file
            prompt: Optional prompt to guide processing
            **kwargs: Additional parameters

        Returns:
            Analysis result from LLM
        """
        if not self.capabilities['vision']:
            raise NotImplementedError(f"{self.provider_name} does not support direct media processing")

        # Default implementation reads image and calls vision model
        with open(media_path, 'rb') as f:
            media_data = f.read()

        default_prompt = prompt or "Analyze this image and extract all relevant information, including any text, objects, scenes, and context."
        return self.call_vision_model(default_prompt, [media_data], **kwargs)

    @abstractmethod
    def get_available_models(self) -> Dict[str, Dict]:
        """
        Get available models for this provider

        Returns:
            Dictionary mapping model names to their capabilities
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate plugin configuration

        Returns:
            True if configuration is valid
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider

        Returns:
            Provider information dictionary
        """
        return {
            'name': self.provider_name,
            'capabilities': self.capabilities,
            'models': self.get_available_models(),
            'config_valid': self.validate_config()
        }

    def _encode_image(self, image_data: Union[str, bytes]) -> str:
        """
        Helper method to encode image data as base64

        Args:
            image_data: Image file path or raw bytes

        Returns:
            Base64 encoded image data
        """
        if isinstance(image_data, str):
            with open(image_data, 'rb') as f:
                image_data = f.read()

        return base64.b64encode(image_data).decode('utf-8')



    @classmethod
    def configured(cls):
        """
        Universal check for plugin readiness.
        By default: instantiate and call validate_config().
        Override in subclass if you want custom/static logic.
        """
        try:
            # Try creating an instance and calling validate_config
            instance = cls()
            return instance.validate_config()
        except Exception:
            return False



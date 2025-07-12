from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
from typing import Dict, Any, List, Union, Optional

class LocalPlugin(BaseLLMPlugin):
    provider_name = "local"
    priority = 20
    supported_types = ["text"]
    supports_vision = False
    description = "Local LLM (run on your hardware)"
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config=None):
        super().__init__(config)
        # Add any local model loading logic here
        self.model_path = self.config.get('model_path', './models/local-model')
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 8080)

    @classmethod
    def configured(cls):
        # For local, you may want to check for required files or paths
        return True

    @staticmethod
    def _get_api_key() -> str:
        """Local models don't need API keys"""
        return ""

    @staticmethod
    def _get_base_url() -> str:
        """Get base URL for local API"""
        return 'http://localhost:8080'

    def call_text_model(self, prompt, model=None, **kwargs):
        # Implement your local inference here, e.g. llama.cpp call or similar
        # This is a placeholder implementation
        return f"Local model result for prompt: {prompt[:50]}... (implement your logic here)."

    def call_vision_model(self, prompt: str, images: List[Union[str, bytes]] = None, model: str = None, **kwargs) -> str:
        """Local plugin doesn't support vision by default, raise NotImplementedError"""
        raise NotImplementedError(f"{self.provider_name} does not support vision models")

    @staticmethod
    def call(prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Static method to call local text endpoint."""
        # Implement local model calling logic here
        # This could involve calling a local server, loading a model, etc.
        return f"Local static call result for prompt: {prompt[:50]}... (implement your logic here)."

    @staticmethod
    def call_vision(prompt: str, images: List[Union[str, bytes]], model: Optional[str] = None, **kwargs) -> str:
        """Local plugin doesn't support vision by default, raise NotImplementedError"""
        raise NotImplementedError("Local plugin does not support vision models")

    @staticmethod
    def get_default_model():
        return 'local-default'

    @classmethod
    def supported_models(cls):
        return {
            'local-default': {
                'type': 'text',
                'max_tokens': 4096,
                'cost': {'input_per_1k': 0.0, 'output_per_1k': 0.0}
            },
            'local-llama': {
                'type': 'text',
                'max_tokens': 8192,
                'cost': {'input_per_1k': 0.0, 'output_per_1k': 0.0}
            }
        }

    def get_available_models(self):
        return self.supported_models()

    def validate_config(self):
        # Add your local config validation logic
        # Could check if model files exist, if local server is running, etc.
        return True

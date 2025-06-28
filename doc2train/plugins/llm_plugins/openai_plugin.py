# openai_plugin.py
from typing import Optional, Dict, Any
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
from doc2train.core.llm_client import _call_openai, _call_openai_vision

class OpenAIPlugin(BaseLLMPlugin):
    name = "openai"
    supported_types = ["text"]
    supports_vision = True

    @classmethod
    def configured(cls) -> bool:
        return bool(cls._get_api_key())

    @classmethod
    def get_default_model(cls) -> str:
        return "gpt-3.5-turbo-0613"

    @classmethod
    def supported_models(cls) -> Dict[str, Any]:
        return {
            "gpt-3.5-turbo-0613": {
                "type": "text",
                "max_tokens": 16384,
                "context_window": 16384,
                "cost": {"input_per_1k": 0.003, "output_per_1k": 0.004}
            },
            "gpt-4-1106": {
                "type": "text",
                "max_tokens": 8192,
                "context_window": 8192,
                "cost": {"input_per_1k": 0.03, "output_per_1k": 0.06}
            }
        }

    @classmethod
    def call(cls, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        provider = {
            "config": {"api_key": cls._get_api_key(), "base_url": cls._get_base_url()},
            "model": model or cls.get_default_model()
        }
        return _call_openai(prompt, provider, **kwargs)

    @classmethod
    def call_vision(cls, prompt: str, image_data: Any, model: Optional[str] = None, **kwargs) -> str:
        provider = {
            "config": {"api_key": cls._get_api_key(), "base_url": cls._get_base_url()},
            "model": model or cls.get_default_model()
        }
        return _call_openai_vision(prompt, image_data, provider, **kwargs)

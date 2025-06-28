# deepseek_plugin.py
from typing import Optional, Dict, Any
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
from doc2train.core.llm_client import _call_deepseek

class DeepSeekPlugin(BaseLLMPlugin):
    provider_name = "deepseek"
    priority = 10
    supported_types = ["text", "image"]
    supports_vision = True
    description = "Deepseek LLM (vision & text)"
    version = "1.0.0"
    author = "Doc2Train Team"


    @classmethod
    def configured(cls) -> bool:
        return bool(cls._get_api_key())

    @classmethod
    def get_default_model(cls) -> str:
        return "default"

    @classmethod
    def supported_models(cls) -> Dict[str, Any]:
        return {
            "default": {
                "type": "text",
                "max_tokens": 8000,
                "context_window": 8000,
                "cost": {"input_per_1k": 0.025, "output_per_1k": 0.04}
            }
        }

    @classmethod
    def call(cls, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        provider = {
            "config": {"api_key": cls._get_api_key(), "base_url": cls._get_base_url()},
            "model": model or cls.get_default_model()
        }
        return _call_deepseek(prompt, provider, **kwargs)

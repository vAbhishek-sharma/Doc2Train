# local_plugin.py
from typing import Optional, Dict, Any
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
from doc2train.core.llm_client import _call_local

class LocalPlugin(BaseLLMPlugin):
    name = "local"
    supported_types = ["text"]
    supports_vision = False

    @classmethod
    def configured(cls) -> bool:
        # assume local runtime always available
        return True

    @classmethod
    def get_default_model(cls) -> str:
        return "local"

    @classmethod
    def supported_models(cls) -> Dict[str, Any]:
        return {
            "local": {
                "type": "text",
                "max_tokens": 2000,
                "context_window": 2000,
                "cost": {"input_per_1k": 0.0, "output_per_1k": 0.0}
            }
        }

    @classmethod
    def call(cls, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        provider = {
            "config": {},
            "model": model or cls.get_default_model()
        }
        return _call_local(prompt, provider, **kwargs)

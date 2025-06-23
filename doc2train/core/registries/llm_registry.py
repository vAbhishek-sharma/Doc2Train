from typing import Any, Dict, Optional
from doc2train.core.registries.plugin_registry import PluginRegistry
import ipdb
class LLMRegistry(PluginRegistry):
    def __init__(self):
        super().__init__()
        self._provider_configs = {}

    def register(self, name: str, llm_cls, config: dict = None):
        # Only register highest-priority plugin per name
        existing = self._plugins.get(name)
        new_priority = getattr(llm_cls, "priority", 10)
        if existing:
            old_priority = getattr(existing, "priority", 10)
            if new_priority <= old_priority:
                return
        super().register(name, llm_cls)
        if config is not None:
            self._provider_configs[name] = config

    def get(self, name: str, with_config: bool = False):
        cls = super().get(name)
        if not with_config:
            return cls
        config = self._provider_configs.get(name, {})
        return cls, config

    def all_with_configs(self):
        return {name: (cls, self._provider_configs.get(name, {}))
                for name, cls in self.all().items()}

    def get_available_providers(self):
        available = []
        for name, llm_cls in self.all().items():
            config = self._provider_configs.get(name, {})
            # Use classmethod for config check
            if hasattr(llm_cls, "configured"):
                try:
                    if llm_cls.configured(config=config):
                        available.append(name)
                except Exception:
                    continue
            elif config.get('api_key'):
                available.append(name)
        return available

    def get_supported_types(self):
        return {name: getattr(cls, "supported_types", ["text"]) for name, cls in self.all().items()}

    def get_vision_models(self):
        return [name for name, cls in self.all().items() if getattr(cls, "supports_vision", False)]

    def get_plugin_metadata(self, name: str):
        """Get static metadata for LLM plugin."""
        cls = self.get(name)
        if cls:
            return {
                "provider_name": getattr(cls, "provider_name", name),
                "priority": getattr(cls, "priority", 10),
                "supported_types": getattr(cls, "supported_types", ["text"]),
                "supports_vision": getattr(cls, "supports_vision", False),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }
        return {}

    @staticmethod
    def get_all_models_for(provider_name: str) -> Dict[str, Any]:
        plugin_cls = LLMRegistry._lookup_plugin_class(provider_name)
        return plugin_cls.supported_models()

    @staticmethod
    def get_default_model(provider_name: str) -> str:
        models = LLMRegistry.get_all_models_for(provider_name)
        # pick first key or use some configured default
        return next(iter(models))


# Singleton instance
_LLM_REGISTRY = LLMRegistry()

def register_llm_plugin(name: str, llm_cls, config: dict = None):
    _LLM_REGISTRY.register(name, llm_cls, config)

def get_llm_plugin(name: str):
    return _LLM_REGISTRY.get(name)

def get_llm_plugin_config(name: str):
    # Helper to get config for instantiation
    return _LLM_REGISTRY._provider_configs.get(name, {})

def get_available_providers():
    return _LLM_REGISTRY.get_available_providers()

def list_llm_plugins():
    return list(_LLM_REGISTRY.all().keys())





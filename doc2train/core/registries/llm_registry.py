from typing import Any, Dict, Optional
from doc2train.core.registries.plugin_registry import PluginRegistry

from doc2train.core.plugin_managers.llm_plugin_manager import LLMPluginManager

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

def get_llm_plugin_class(name: str):
    """Get just the LLM plugin class, handling both dict and class formats."""
    result = _LLM_REGISTRY.get(name)
    if isinstance(result, dict) and 'class' in result:
        return result['class']
    return result

def get_llm_plugin_config(name: str):
    # Helper to get config for instantiation
    return _LLM_REGISTRY._provider_configs.get(name, {})

def get_available_providers():
    return _LLM_REGISTRY.get_available_providers()

def list_llm_plugins():
    return list(_LLM_REGISTRY.all().keys())

def get_llm_client(config):
    provider = config.get("provider", "openai")
    plugin_cls = get_llm_plugin_class(provider)
    if not plugin_cls:
        raise ValueError(f"No LLM plugin found for provider: {provider}")
    return plugin_cls(config)

def get_vision_provider(
    provider_name: Optional[str] = None,
    model_name:    Optional[str] = None
) -> Dict[str, Any]:
    """
    Pick a vision-capable LLM provider + model:
      1. Filter to only providers where plugin_cls.supports_vision is True.
      2. If provider_name is in that list, use it (and model_name if supported, else default).
      3. Else if model_name is given, pick the first vision provider that supports it.
      4. Else fall back to the first vision provider + its default model.
    Returns: {"provider": str, "plugin_cls": class, "model": str}
    """
    # 1) only vision-capable names
    vision_candidates = [
        name
        for name in get_available_providers()
        if get_llm_plugin_class(name).supports_vision
    ]
    if not vision_candidates:
        raise RuntimeError("No vision-capable providers registered")

    # 2) explicit provider
    if provider_name in vision_candidates:
        cls = get_llm_plugin_class(provider_name)
        supported = cls.supported_models()
        chosen = model_name if (model_name in supported) else cls.get_default_model()
        return {"provider": provider_name, "plugin_cls": cls, "model": chosen}

    # 3) pick any provider that supports the requested model
    if model_name:
        for name in vision_candidates:
            cls = get_llm_plugin_class(name)
            if model_name in cls.supported_models():
                return {"provider": name, "plugin_cls": cls, "model": model_name}

    # 4) fallback to first vision provider + its default
    fallback = vision_candidates[0]
    cls = get_llm_plugin_class(fallback)
    return {"provider": fallback, "plugin_cls": cls, "model": cls.get_default_model()}




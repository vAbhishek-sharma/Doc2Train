# core/registries/generator_registry.py

from doc2train.core.registries.plugin_registry import PluginRegistry

class GeneratorRegistry(PluginRegistry):
    def __init__(self):
        super().__init__()

    def register(self, name: str, gen_cls):
        super().register(name, gen_cls)

    def get_plugin_metadata(self, name: str):
        cls = self.get(name)
        if cls:
            return {
                "name": getattr(cls, "name", name),
                "priority": getattr(cls, "priority", 10),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }
        return {}

    def get(self, name):
        return super().get(name)

# Singleton instance
_GENERATOR_REGISTRY = GeneratorRegistry()

def register_generator(name: str, gen_cls):
    _GENERATOR_REGISTRY.register(name, gen_cls)

def get_generator(name: str):
    # Check if this is coming from plugin manager (dict format)
    result = _GENERATOR_REGISTRY.get(name)
    if isinstance(result, dict) and 'class' in result:
        return result['class']
    return result

def list_all_generators():
    return _GENERATOR_REGISTRY.all()

# core/registries/generator_registry.py

from doc2train.core.registries.plugin_registry import PluginRegistry

class GeneratorRegistry(PluginRegistry):
    def __init__(self):
        super().__init__()
        self._type_map = {}

    def register(self, name: str, gen_cls):
        super().register(name, gen_cls)
        for t in getattr(gen_cls, "types_supported", []):
            # Only keep the highest-priority class for each type
            existing = self._type_map.get(t)
            new_priority = getattr(gen_cls, "priority", 10)
            if existing:
                old_priority = getattr(existing, "priority", 10)
                if new_priority > old_priority:
                    continue  # skip lower priority
            self._type_map[t] = gen_cls

    def get_by_type(self, gen_type):
        return self._type_map.get(gen_type)

# Singleton instance
_GENERATOR_REGISTRY = GeneratorRegistry()

def register_generator(name: str, gen_cls):
    _GENERATOR_REGISTRY.register(name, gen_cls)

def get_generator(gen_type: str):
    return _GENERATOR_REGISTRY.get_by_type(gen_type)

def list_all_generators():
    return _GENERATOR_REGISTRY.all()

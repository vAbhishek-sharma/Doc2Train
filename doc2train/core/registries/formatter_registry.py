# doc2train/core/registries/formatter_registry.py

from doc2train.core.registries.plugin_registry import PluginRegistry

class FormatterRegistry(PluginRegistry):
    def __init__(self):
        super().__init__()
        self._format_names = set()

    def register(self, name: str, formatter_cls):
        # Only keep the highest-priority formatter for each name
        existing = self._plugins.get(name)
        new_priority = getattr(formatter_cls, "priority", 10)
        if existing:
            old_priority = getattr(existing, "priority", 10)
            if new_priority > old_priority:
                return  # skip lower priority
        super().register(name, formatter_cls)
        self._format_names.add(name.lower())


    def get_supported_formats(self):
        return list(self._format_names)

# Singleton instance
_FORMATTER_REGISTRY = FormatterRegistry()

def register_formatter(name: str, formatter_cls):
    _FORMATTER_REGISTRY.register(name, formatter_cls)

def get_formatter(name: str):
    return _FORMATTER_REGISTRY.get(name)

def list_all_formatters():
    return _FORMATTER_REGISTRY.all()

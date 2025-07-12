# doc2train/core/registries/formatter_registry.py

from doc2train.core.registries.plugin_registry import PluginRegistry
from typing import Dict, Any, Optional

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
            if new_priority <= old_priority:
                return  # skip lower priority
        super().register(name, formatter_cls)
        self._format_names.add(name.lower())

    def get_supported_formats(self):
        return list(self._format_names)

    def get_plugin_metadata(self, name: str) -> Dict[str, Any]:
        """Get static metadata for a formatter plugin."""
        cls = self.get(name)
        if cls:
            return {
                "name": getattr(cls, "name", name),  # Use 'name' attribute
                "priority": getattr(cls, "priority", 10),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }
        return {}

# Singleton instance
_FORMATTER_REGISTRY = FormatterRegistry()

def register_formatter(name: str, formatter_cls):
    _FORMATTER_REGISTRY.register(name, formatter_cls)

def get_formatter(name: str):
    """Get formatter class by name (e.g., 'csv', 'json', 'markdown')"""
    return _FORMATTER_REGISTRY.get(name)

def get_formatter_class(name: str):
    """Get just the formatter class, handling both dict and class formats."""
    result = _FORMATTER_REGISTRY.get(name)
    if isinstance(result, dict) and 'class' in result:
        return result['class']
    return result

def list_all_formatters():
    return _FORMATTER_REGISTRY.all()

def get_supported_formats():
    return _FORMATTER_REGISTRY.get_supported_formats()

def has_formatter(name: str) -> bool:
    """Check if formatter exists."""
    return _FORMATTER_REGISTRY.get(name) is not None

def get_formatter_instance(name: str, config: Optional[Dict[str, Any]] = None):
    """Get an instance of the formatter with optional config."""
    formatter_cls = get_formatter(name)
    if formatter_cls:
        return formatter_cls(config) if config else formatter_cls()
    raise ValueError(f"No formatter found for: {name}")

def get_available_formatters():
    """Get list of available formatter names."""
    return list(_FORMATTER_REGISTRY.all().keys())

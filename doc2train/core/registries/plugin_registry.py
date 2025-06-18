# doc2train/utils/plugin_registry.py

from typing import Dict, Type, Any

class PluginRegistry:
    def __init__(self):
        self._plugins: Dict[str, Type[Any]] = {}

    def register(self, name: str, plugin_cls: Type[Any]):
        self._plugins[name] = plugin_cls

    def get(self, name: str) -> Type[Any]:
        return self._plugins.get(name)

    def all(self) -> Dict[str, Type[Any]]:
        return self._plugins

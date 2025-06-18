from pathlib import Path
from typing import Dict, Any, Optional
from doc2train.plugins.llm_plugins.base_llm_plugin import BaseLLMPlugin
from doc2train.utils.plugin_loader import load_plugins_from_dirs

class LLMPluginManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.plugins: Dict[str, Dict] = {}

        plugin_dirs = [
            Path(__file__).parent.parent / "plugins" / "llm_plugins",
            *self.config.get("llm_plugin_dirs", []),
        ]
        pkg_prefix = "doc2train.plugins.llm_plugins"

        raw = load_plugins_from_dirs(plugin_dirs, BaseLLMPlugin, pkg_prefix)
        for name, cls in raw.items():
            provider_name = getattr(cls, "provider_name", name.lower().replace("plugin", ""))
            # Only keep highest-priority plugin for each provider_name
            if provider_name in self.plugins:
                prev_priority = self.plugins[provider_name]["priority"]
                this_priority = getattr(cls, "priority", 10)
                if this_priority <= prev_priority:
                    continue
            self.plugins[provider_name] = {
                "class": cls,
                "provider_name": provider_name,
                "priority": getattr(cls, "priority", 10),
                "supported_types": getattr(cls, "supported_types", ["text"]),
                "supports_vision": getattr(cls, "supports_vision", False),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }

    def list_llm_plugins(self):
        return list(self.plugins.keys())

    def get_plugin_metadata(self, name):
        return self.plugins.get(name, {})

    def all(self):
        return self.plugins

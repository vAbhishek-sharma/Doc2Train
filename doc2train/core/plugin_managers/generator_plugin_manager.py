# core/plugin_managers/generator_plugin_manager.py

from pathlib import Path
from typing import Dict, Any, List
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.plugins.generator_plugins.base_generator import BaseGenerator
import doc2train

class GeneratorPluginManager:
    def __init__(self, config: Dict[str, Any]):
        dirs = [
            Path(doc2train.__file__).parent / "plugins" / "generator_plugins",
            *config.get('generator_plugin_dirs', [])
        ]
        pkg_prefix = "doc2train.plugins.generator_plugins"

        raw = load_plugins_from_dirs(dirs, BaseGenerator, pkg_prefix)
        self.plugins: Dict[str, Any] = {}
        for name, cls in raw.items():
            generator_name = getattr(cls, "name", name)
            # Priority logic: only highest-priority plugin per generator_name
            if generator_name in self.plugins:
                prev_priority = self.plugins[generator_name]["priority"]
                this_priority = getattr(cls, "priority", 10)
                if this_priority <= prev_priority:
                    continue
            self.plugins[generator_name] = {
                "class": cls,
                "name": generator_name,
                "priority": getattr(cls, "priority", 10),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }

    def list_generators(self) -> List[str]:
        return list(self.plugins.keys())

    def get_plugin(self, name: str) -> Dict:
        return self.plugins.get(name)

    def get_generator_info(self, name: str) -> Dict:
        plugin = self.get_plugin(name)
        if plugin:
            return {
                k: v for k, v in plugin.items()
                if k in ("name", "priority", "description", "version", "author")
            }
        return {}

    def all(self):
        return self.plugins

    def classes(self):
        return [info["class"] for info in self.plugins.values()]

    def has_plugin(self, name: str) -> bool:
        return name in self.plugins

    def list_plugins(self):
        for name, info in self.plugins.items():
            print(f"{name}: {info['description']}")

    def discover_plugins(self, plugin_dirs: List[str]):
        # Logic for runtime plugin discovery if needed
        pass

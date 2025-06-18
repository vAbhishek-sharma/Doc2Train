from pathlib import Path
from typing import Dict, Any, List
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.processors.base_processor import BaseProcessor

class ProcessorPluginManager:
    def __init__(self, config: Dict[str, Any]):
        dirs = [
            Path(__file__).parent.parent / 'plugins' / 'processor_plugins',
            *config.get('processor_plugin_dirs', [])
        ]
        pkg_prefix = "doc2train.plugins.processor_plugins"

        raw = load_plugins_from_dirs(dirs, BaseProcessor, pkg_prefix)
        self.plugins: Dict[str, Dict] = {}
        for name, cls in raw.items():
            processor_name = getattr(cls, "processor_name", name)
            # Priority logic: only highest-priority plugin per processor_name
            if processor_name in self.plugins:
                prev_priority = self.plugins[processor_name]["priority"]
                this_priority = getattr(cls, "priority", 10)
                if this_priority <= prev_priority:
                    continue
            self.plugins[processor_name] = {
                "class": cls,
                "name": processor_name,
                "extensions": getattr(cls, "supported_extensions", []),
                "priority": getattr(cls, "priority", 10),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }

    def list_processors(self) -> List[str]:
        """List all processor names."""
        return list(self.plugins.keys())

    def get_plugin(self, name: str) -> Dict:
        """Get the processor plugin info dict by name."""
        return self.plugins.get(name)

    def get_processor_extensions(self, name: str) -> List[str]:
        """Get supported extensions for a processor."""
        plugin = self.get_plugin(name)
        return plugin["extensions"] if plugin else []

    def get_supported_extensions(self) -> set:
        """Get set of all supported file extensions."""
        exts = set()
        for info in self.plugins.values():
            exts.update([e.lower() for e in info["extensions"]])
        return exts

    def get_supported_extensions_dict(self) -> Dict[str, List[str]]:
        """Get dict mapping processor_name -> [extensions]."""
        return {name: info["extensions"] for name, info in self.plugins.items()}

    def list_plugins(self):
        """Print a human-friendly table of loaded processor plugins."""
        for name, info in self.plugins.items():
            print(f"{name}: {info['description']} (extensions: {', '.join(info['extensions'])})")

    def get_processor_info(self, name: str) -> Dict:
        """Get metadata for a given processor plugin."""
        plugin = self.get_plugin(name)
        if plugin:
            return {
                k: v for k, v in plugin.items()
                if k in ("name", "priority", "extensions", "description", "version", "author")
            }
        return {}

    def all(self):
        """Return full dict of all plugin info."""
        return self.plugins

    def classes(self):
        """Return list of all processor classes."""
        return [info["class"] for info in self.plugins.values()]

    def has_plugin(self, name: str) -> bool:
        """Quick check if processor plugin exists."""
        return name in self.plugins

    def discover_plugins(self, plugin_dirs: List[str]):
        """Dynamically discover and add plugins at runtime (call after init if needed)."""
        # Use same logic as __init__
        pass

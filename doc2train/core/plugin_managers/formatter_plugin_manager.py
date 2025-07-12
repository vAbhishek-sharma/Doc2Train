# outputs/formatter_plugin_manager.py
from pathlib import Path
from typing import Dict, Any, List
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter
import doc2train

class FormatterPluginManager:
    def __init__(self, config: Dict[str, Any]):
        dirs = [
            Path(doc2train.__file__).parent / "plugins" / "formatter_plugins",
            *config.get('formatter_plugin_dirs', [])
        ]
        pkg_prefix = "doc2train.plugins.formatter_plugins"

        raw = load_plugins_from_dirs(dirs, BaseFormatter, pkg_prefix)
        self.plugins: Dict[str, Dict] = {}

        for name, cls in raw.items():
            # Use 'name' attribute from the class, fallback to discovered name
            formatter_name = getattr(cls, "name", name)
            # Priority logic: only highest-priority plugin per formatter_name
            if formatter_name in self.plugins:
                prev_priority = self.plugins[formatter_name]["priority"]
                this_priority = getattr(cls, "priority", 10)
                if this_priority <= prev_priority:
                    continue
            self.plugins[formatter_name] = {
                "class": cls,
                "name": formatter_name,
                "priority": getattr(cls, "priority", 10),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }

    def list_formatters(self) -> List[str]:
        """List all formatter names."""
        return list(self.plugins.keys())

    def get_plugin(self, name: str) -> Dict:
        """Get the formatter plugin info dict by name."""
        return self.plugins.get(name)

    def get(self, name: str):
        """Get the formatter class by name."""
        plugin = self.get_plugin(name)
        return plugin["class"] if plugin else None

    def available(self) -> List[str]:
        """Get list of available formatter names."""
        return self.list_formatters()

    def get_formatter_info(self, name: str) -> Dict:
        """Get metadata for a given formatter plugin."""
        plugin = self.get_plugin(name)
        if plugin:
            return {
                k: v for k, v in plugin.items()
                if k in ("name", "priority", "description", "version", "author")
            }
        return {}

    def all(self):
        """Return full dict of all plugin info."""
        return self.plugins

    def classes(self):
        """Return list of all formatter classes."""
        return [info["class"] for info in self.plugins.values()]

    def has_plugin(self, name: str) -> bool:
        """Quick check if formatter plugin exists."""
        return name in self.plugins

    def list_plugins(self):
        """Print a human-friendly table of loaded formatter plugins."""
        for name, info in self.plugins.items():
            print(f"{name}: {info['description']}")

    def discover_plugins(self, plugin_dirs: List[str]):
        """Dynamically discover and add plugins at runtime (call after init if needed)."""
        # Use same logic as __init__
        pass

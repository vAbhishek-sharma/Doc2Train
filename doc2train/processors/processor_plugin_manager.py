# processors/processor_plugin_manager.py
from pathlib import Path
from typing import Dict
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.processors.base_processor import BaseProcessor

class ProcessorPluginManager:
    def __init__(self, config):
        dirs = [
                Path(__file__).parent.parent / 'plugins' / 'processor_plugins',                    # built-in processors folder
            *config.get('processor_plugin_dirs', [])  # overrides
        ]
        pkg_prefix = "doc2train.plugins.processor_plugins"

        raw = load_plugins_from_dirs(dirs, BaseProcessor, pkg_prefix)
        self.plugins: Dict[str, Dict] = {}
        for name, cls in raw.items():
            inst = cls(config)
            self.plugins[name] = {
                "class": cls,
                "name": getattr(inst, "processor_name", name),
                "extensions": getattr(inst, "supported_extensions", []),
            }

    def get(self, name):
        return self.plugins.get(name)

    def available(self):
        return list(self.plugins)


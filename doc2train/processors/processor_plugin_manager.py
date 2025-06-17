# processors/processor_plugin_manager.py
from pathlib import Path
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.processors.base_processor import BaseProcessor

class ProcessorPluginManager:
    def __init__(self, config):
        dirs = [
            Path(__file__).parent,                    # your built-in processors folder
            *config.get('processor_plugin_dirs', [])  # cli/yaml overrides
        ]
        processor_eps = "plugins.processor_plugins"
        self.plugins = load_plugins_from_dirs(
            dirs,
            BaseProcessor,
            processor_eps
        )

    def get(self, name):       return self.plugins.get(name)
    def available(self):       return list(self.plugins)

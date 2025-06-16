# outputs/formatter_plugin_manager.py
from pathlib import Path
from utils.plugin_loader import load_plugins_from_dirs
from outputs.base_formatters import BaseFormatter

class FormatterPluginManager:
    def __init__(self, config):
        dirs = [
            Path(__file__).parent/'formatter_plugins',
            *config.get('formatter_plugin_dirs', [])
        ]
        self.plugins = load_plugins_from_dirs(
            dirs,
            BaseFormatter,
            entry_point_group="doc2train.formatter_plugins"
        )

    def get(self, name):
        return self.plugins.get(name)

    def available(self):
        return list(self.plugins)

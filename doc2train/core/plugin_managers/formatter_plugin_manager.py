# outputs/formatter_plugin_manager.py
from pathlib import Path
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter

class FormatterPluginManager:
    def __init__(self, config):
        dirs = [
            Path(__file__).parent/'formatter_plugins',
            *config.get('formatter_plugin_dirs', [])
        ]
        formatter_eps = "doc2train.formatter_plugins"
        self.plugins = load_plugins_from_dirs(
            dirs,
            BaseFormatter,
            formatter_eps
        )

    def get(self, name):
        return self.plugins.get(name)

    def available(self):
        return list(self.plugins)

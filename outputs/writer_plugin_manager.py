# outputs/writer_plugin_manager.py
from pathlib import Path
from utils.plugin_loader import load_plugins_from_dirs
from outputs.base_writer import BaseWriter  # define a common base

class WriterPluginManager:
    def __init__(self, config):
        dirs = [
            Path(__file__).parent/'writer_plugins',
            *config.get('writer_plugin_dirs', [])
        ]
        self.plugins = load_plugins_from_dirs(
            dirs,
            BaseWriter,
            entry_point_group="doc2train.writer_plugins"
        )

    def get(self, name):
        return self.plugins.get(name)

    def available(self):
        return list(self.plugins)

from pathlib import Path
from typing import Dict, Any
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.plugins.writer_plugins.base_writer import BaseWriter
import doc2train

class WriterPluginManager:
    def __init__(self, config: Dict[str, Any]):
        dirs = [
            Path(doc2train.__file__).parent / "plugins" / "writer_plugins",
            *config.get('writer_plugin_dirs', [])
        ]
        pkg_prefix = "doc2train.plugins.writer_plugins"

        raw = load_plugins_from_dirs(dirs, BaseWriter, pkg_prefix)
        self.plugins: Dict[str, Any] = {}
        for name, cls in raw.items():
            writer_name = getattr(cls, "writer_name", name)
            self.plugins[writer_name] = cls

    def list_writers(self):
        return list(self.plugins.keys())

    def get(self, name):
        return self.plugins.get(name)

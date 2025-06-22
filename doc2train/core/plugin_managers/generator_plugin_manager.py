# outputs/generator_plugin_manager.py

from pathlib import Path
from doc2train.utils.plugin_loader import load_plugins_from_dirs
from doc2train.plugins.generator_plugins.base_generator import BaseGenerator
import doc2train

class GeneratorPluginManager:
    def __init__(self, config):
        dirs = [
            Path(doc2train.__file__).parent / "plugins" / "generator_plugins",
            *config.get('generator_plugin_dirs', [])
        ]
        pkg_prefix = "doc2train.plugins.generator_plugins"
        raw = load_plugins_from_dirs(dirs, BaseGenerator, pkg_prefix)
        self.plugins = {}
        for name, cls in raw.items():
            gen_name = getattr(cls, "generator_name", name)
            self.plugins[gen_name] = cls

    def list_generators(self):
        return list(self.plugins.keys())

    def get(self, name):
        return self.plugins.get(name)

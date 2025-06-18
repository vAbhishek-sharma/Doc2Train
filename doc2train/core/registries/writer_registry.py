from doc2train.core.registries.plugin_registry import PluginRegistry

class WriterRegistry(PluginRegistry):
    def __init__(self):
        super().__init__()
        self._writer_names = set()

    def register(self, name: str, writer_cls):
        # Only register highest-priority plugin per name
        existing = self._plugins.get(name)
        new_priority = getattr(writer_cls, "priority", 10)
        if existing:
            old_priority = getattr(existing, "priority", 10)
            if new_priority > old_priority:
                return
        super().register(name, writer_cls)
        self._writer_names.add(name.lower())

    def get_supported_writers(self):
        return list(self._writer_names)

# Singleton instance
_WRITER_REGISTRY = WriterRegistry()

def register_writer(name: str, writer_cls):
    _WRITER_REGISTRY.register(name, writer_cls)

def get_writer(name: str):
    return _WRITER_REGISTRY.get(name)

def list_all_writers():
    return _WRITER_REGISTRY.all()

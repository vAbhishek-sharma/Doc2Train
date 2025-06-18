from doc2train.core.registries.plugin_registry import PluginRegistry

class ProcessorRegistry(PluginRegistry):
    def __init__(self):
        super().__init__()
        self._ext_map = {}      # Map extension to processor name
        self._name_to_exts = {} # Map processor name to list of extensions

    def register(self, name: str, extensions, processor_cls):
        # Only register highest-priority plugin per name
        existing = self._plugins.get(name)
        new_priority = getattr(processor_cls, "priority", 10)
        if existing:
            old_priority = getattr(existing, "priority", 10)
            if new_priority <= old_priority:
                return
        super().register(name, processor_cls)
        self._name_to_exts[name] = list(extensions)
        for ext in extensions:
            self._ext_map[ext.lower()] = name

    def get_by_extension(self, ext: str):
        name = self._ext_map.get(ext.lower())
        if name:
            return self.get(name)
        return None

    def get_supported_extensions_dict(self):
        """Returns: {processor_name: [list of extensions]}"""
        return dict(self._name_to_exts)

    def get_supported_extensions(self):
        """Returns: set of all extensions (across all processors)"""
        all_exts = set()
        for exts in self._name_to_exts.values():
            all_exts.update([e.lower() for e in exts])
        return all_exts

    def get_all_extensions(self):
        """Returns a dict: {extension: processor_name}"""
        return dict(self._ext_map)

    def get_plugin_metadata(self, name: str):
        """Get static metadata for a processor plugin."""
        cls = self.get(name)
        if cls:
            return {
                "processor_name": getattr(cls, "processor_name", name),
                "priority": getattr(cls, "priority", 10),
                "supported_extensions": getattr(cls, "supported_extensions", []),
                "description": getattr(cls, "description", ""),
                "version": getattr(cls, "version", ""),
                "author": getattr(cls, "author", ""),
            }
        return {}

# Singleton instance
_PROCESSOR_REGISTRY = ProcessorRegistry()

def register_processor(name: str, extensions, processor_cls):
    _PROCESSOR_REGISTRY.register(name, extensions, processor_cls)

def get_processor_for_file(file_path, config=None):
    from pathlib import Path
    ext = Path(file_path).suffix
    cls = _PROCESSOR_REGISTRY.get_by_extension(ext)
    if cls:
        return cls(config)
    raise ValueError(f"No processor for extension: {ext}")

def list_all_processors():
    return list(_PROCESSOR_REGISTRY.all().keys())

def get_supported_extensions_dict():
    return _PROCESSOR_REGISTRY.get_supported_extensions_dict()

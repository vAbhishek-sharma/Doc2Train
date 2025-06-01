# config/manager.py

from config import settings  # Import at top level

class ConfigManager:
    """Centralized configuration management"""

    def __init__(self):
        self._config = {}
        self._load_from_env()
        self._load_from_files()

    def _load_from_env(self):
        """Load from environment variables or settings module"""
        self._config.update({
            'chunk_size': settings.CHUNK_SIZE,
            'max_workers': settings.MAX_WORKERS,
            'output_dir': settings.OUTPUT_DIR,
            'cache_dir': settings.CACHE_DIR,
            'use_cache': settings.USE_CACHE,
            'llm_providers': settings.LLM_PROVIDERS,
            'default_provider': settings.DEFAULT_PROVIDER,
        })

    def _load_from_files(self):
        """Stub for file-based config overrides"""
        pass  # Add logic later if needed

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value

    def update(self, config_dict):
        self._config.update(config_dict)

# Global instance
config_manager = ConfigManager()

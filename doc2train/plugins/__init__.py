# plugins/__init__.py

# Make subpackages available under the plugins namespace
from . import llm_plugins
from . import processor_plugins
from . import formatter_plugins
from . import writer_plugins

__all__ = [
    "llm_plugins",
    "processor_plugins",
    "formatter_plugins",
    "writer_plugins",
]

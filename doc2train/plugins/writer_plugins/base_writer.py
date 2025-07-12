# outputs/writers/base_writer.py

from abc import ABC, abstractmethod
from typing import Type, Dict, Optional, List
from pathlib import Path

# Registry for writer plugins
_WRITER_PLUGINS: Dict[str, Type["BaseWriter"]] = {}

class BaseWriter(ABC):
    """
    Abstract base class for all writer plugins.
    Subclasses must implement `write` to emit `items` to `output_file`.
    """

    def __init__(self, config: Dict):
        """
        :param config: The global config dict, from which writers can read settings.
        """
        self.config = config

    @abstractmethod
    def write(self, output_file: Path, items: List[Dict], data_type: str) -> None:
        """
        Write the given items to the specified file.

        :param output_file: Path (including filename) where output should be written.
        :param items: A list of dicts/records to serialize.
        :param data_type: Logical type of data (e.g. "conversations", "qapairs").
        """
        ...

def register_writer(name: str, cls: Type[BaseWriter]) -> None:
    """
    Register a writer plugin under the given name.
    Later, get_writer(name) will return this class.
    """
    _WRITER_PLUGINS[name] = cls

def get_writer(name: str) -> Optional[Type[BaseWriter]]:
    """
    Retrieve a previously-registered writer class by name.
    Returns None if no plugin was registered under that name.
    """
    return _WRITER_PLUGINS.get(name)

def list_writers() -> List[str]:
    """
    List all names of registered writer plugins.
    """
    return list(_WRITER_PLUGINS.keys())

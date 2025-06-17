from abc import ABC, abstractmethod
from typing import Optional, Type, Dict,Any,List

_FORMATTER_PLUGINS: Dict[str, Type["BaseFormatter"]] = {}

class BaseFormatter(ABC):
    """Base class for all output formatters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.format_name = "base"

    @abstractmethod
    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversation data"""
        pass

    @abstractmethod
    def format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs data"""
        pass

    @abstractmethod
    def format_embeddings(self, embeddings: List[Dict]) -> str:
        """Format embedding pairs data"""
        pass

    @abstractmethod
    def format_summaries(self, summaries: List[Dict]) -> str:
        """Format summary data"""
        pass

    def get_file_extension(self) -> str:
        """Get file extension for this format"""
        return ".txt"

def register_formatter(name: str, cls: Type[BaseFormatter]):
    _FORMATTER_PLUGINS[name] = cls

def get_formatter(name: str) -> Optional[Type[BaseFormatter]]:
    return _FORMATTER_PLUGINS.get(name)

def list_formatters() -> List[str]:
    return List(_FORMATTER_PLUGINS)

from abc import ABC, abstractmethod
from typing import Optional, Type, Dict,Any,List

_FORMATTER_PLUGINS: Dict[str, Type["BaseFormatter"]] = {}

class BaseFormatter(ABC):
    """Base class for all output formatters"""
    format_name = None
    priority = 10
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

    def format(self, data, data_type):
        """
        Dynamically dispatch to format_{data_type}, or fallback to generic.
        """
        method = getattr(self, f"format_{data_type}", None)
        if callable(method):
            return method(data)
        return self.format_generic(data, data_type)

    def format_generic(self, data, data_type):
        """
        Generic fallback: dump as JSON with type header.
        """
        import json
        return json.dumps({"type": data_type, "data": data}, indent=2, ensure_ascii=False)

    def get_file_extension(self):
        return ".txt"

def register_formatter(name: str, cls: Type[BaseFormatter]):
    _FORMATTER_PLUGINS[name] = cls

def get_formatter(name: str) -> Optional[Type[BaseFormatter]]:
    return _FORMATTER_PLUGINS.get(name)

def list_formatters() -> List[str]:
    return List(_FORMATTER_PLUGINS)
